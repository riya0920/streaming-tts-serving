"""
M10 — export VITS's front half: the stochastic duration predictor and the flow.

This is the piece M4 skipped and M9 proved was the whole problem. The gateway's own
histograms put 100% of the latency tail in this stage (p50 150 ms, p99 1000 ms at 32
sessions, while every other stage stays under 20 ms), and the load test showed the knee
arrives at ~2% GPU utilization — so the system is latency-bound on exactly this code.

Why it was skipped, and how that is solved here
-----------------------------------------------
`VitsStochasticDurationPredictor` draws `torch.randn` *inside* its reverse pass. A module
that samples its own noise is not a pure function of its inputs, so tracing it bakes one
fixed noise draw into the graph as a constant — every utterance would then get identical
durations, and the prosody variation that makes VITS sound natural would vanish.

The fix is to hoist the noise into a graph **input**. Rather than reimplementing the
reverse flow loop — intricate, and it would silently drift when transformers changes —
`torch.randn` is swapped for a function returning the caller-supplied tensor for the
duration of the trace. The tracer records that tensor as an input, so the exported graph
takes noise as an argument and stays a pure function. At serve time the backend draws
fresh noise per request, exactly as PyTorch did.

The flow (`VitsResidualCouplingBlock`) needs none of this: its reverse pass is already
deterministic given `z_p`. The randomness upstream of it lives in `VitsModel.forward`,
which samples `z_p = m_p + randn_like(m_p) * exp(logs_p) * noise_scale`. That one line
stays in Python — it is a single elementwise op, not worth an engine.

  python export/export_frontend_onnx.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import VitsModel

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")
OPSET = 17


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


@contextlib.contextmanager
def randn_returns(tensor: torch.Tensor):
    """Make every torch.randn / randn_like call return `tensor` for the duration.

    Used only while tracing. Because the replacement returns a real tensor that the
    tracer has already seen as a graph input, the resulting ONNX takes noise as an
    argument instead of embedding a constant sample.

    Shape is not checked against the request: VITS asks for exactly [B, 2, T] here, and
    the caller constructs the tensor to match. A mismatch would surface immediately as a
    shape error during export rather than silently producing a wrong graph.
    """
    real_randn, real_randn_like = torch.randn, torch.randn_like

    def fake_randn(*args, **kwargs):
        return tensor

    def fake_randn_like(other, *args, **kwargs):
        return tensor.to(dtype=other.dtype, device=other.device)

    torch.randn, torch.randn_like = fake_randn, fake_randn_like
    try:
        yield
    finally:
        torch.randn, torch.randn_like = real_randn, real_randn_like


class DurationPredictorWrapper(nn.Module):
    """(hidden_states, padding_mask, noise) -> log_duration.

    Flat signature and explicit noise, so the exported graph is pure.
    """

    def __init__(self, dp, noise_holder: dict):
        super().__init__()
        self.dp = dp
        self._noise_holder = noise_holder

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
        self._noise_holder["t"] = noise
        # noise_scale=1.0: the caller scales the noise it supplies, so the scale factor
        # does not need to be a graph input too.
        return self.dp(hidden_states, padding_mask, reverse=True, noise_scale=1.0)


class FlowWrapper(nn.Module):
    """(z_p, padding_mask) -> latents. Already deterministic in reverse."""

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, z_p: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.flow(z_p, padding_mask, reverse=True)


def parity(onnx_path: Path, feeds: dict, ref: torch.Tensor) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)[0]
    r = ref.detach().cpu().numpy()
    diff = np.abs(got - r)
    denom = float(np.sqrt(np.mean(r.astype(np.float64) ** 2))) or 1.0
    return {
        "shape": list(got.shape),
        "max_abs_diff": float(diff.max()),
        "rel_rms": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)) / denom),
        "allclose": bool(np.allclose(got, r, rtol=1e-3, atol=1e-3)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seq", type=int, default=48, help="token length to trace at")
    args = ap.parse_args()

    path = resolve()
    outdir = Path(args.outdir or (Path(MODEL_DIR) / "onnx"))
    outdir.mkdir(parents=True, exist_ok=True)

    model = VitsModel.from_pretrained(path).eval()
    cfg = model.config
    ch = int(getattr(cfg, "flow_size", 192))
    S = args.seq

    manifest: dict = {"model": path, "opset": OPSET, "artifacts": {}}

    # ------------------------------------------------- duration predictor
    dp = model.duration_predictor
    is_stochastic = bool(getattr(cfg, "use_stochastic_duration_prediction", True))
    print(f"duration predictor: {type(dp).__name__} (stochastic={is_stochastic})")

    hidden = torch.randn(1, ch, S)
    pad_mask = torch.ones(1, 1, S)
    # The reverse pass samples [B, 2, T]; supply exactly that so the swap is transparent.
    noise = torch.randn(1, 2, S)

    holder: dict = {}
    wrap_dp = DurationPredictorWrapper(dp, holder).eval()
    dp_path = outdir / "vits_duration.onnx"

    with torch.inference_mode():
        with randn_returns(noise):
            ref_dp = wrap_dp(hidden, pad_mask, noise)

    print(f"exporting duration predictor -> {dp_path}")
    with randn_returns(noise):
        torch.onnx.export(
            wrap_dp, (hidden, pad_mask, noise), str(dp_path),
            input_names=["hidden_states", "padding_mask", "noise"],
            output_names=["log_duration"],
            dynamic_axes={
                "hidden_states": {0: "batch", 2: "seq"},
                "padding_mask": {0: "batch", 2: "seq"},
                "noise": {0: "batch", 2: "seq"},
                "log_duration": {0: "batch", 2: "seq"},
            },
            opset_version=OPSET, do_constant_folding=True,
        )

    rep = parity(dp_path, {"hidden_states": hidden.numpy(),
                           "padding_mask": pad_mask.numpy(),
                           "noise": noise.numpy()}, ref_dp)
    print(f"  parity: {json.dumps(rep)}")

    # The whole point: two different noise draws must give two different durations. If
    # the noise were baked in as a constant, these would be identical and every utterance
    # would get the same prosody.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(dp_path), providers=["CPUExecutionProvider"])
    n1 = torch.randn(1, 2, S).numpy()
    n2 = torch.randn(1, 2, S).numpy()
    d1 = sess.run(None, {"hidden_states": hidden.numpy(),
                         "padding_mask": pad_mask.numpy(), "noise": n1})[0]
    d2 = sess.run(None, {"hidden_states": hidden.numpy(),
                         "padding_mask": pad_mask.numpy(), "noise": n2})[0]
    spread = float(np.abs(d1 - d2).mean())
    noise_is_live = spread > 1e-6
    print(f"  noise is a live input: {noise_is_live} (mean |d1-d2| = {spread:.6f})")
    if not noise_is_live:
        raise RuntimeError(
            "noise got baked in as a constant — every utterance would receive identical "
            "durations. The randn swap did not take effect during tracing."
        )

    manifest["artifacts"]["duration_predictor"] = {
        "file": dp_path.name, "size_mb": round(dp_path.stat().st_size / 1e6, 2),
        "parity": rep, "noise_is_live": noise_is_live,
        "noise_spread": round(spread, 6),
    }

    # ------------------------------------------------------------- flow
    flow_path = outdir / "vits_flow.onnx"
    T = S * 2  # latent frames differ from token count; trace at a representative length
    z_p = torch.randn(1, ch, T)
    out_mask = torch.ones(1, 1, T)
    wrap_flow = FlowWrapper(model.flow).eval()

    with torch.inference_mode():
        ref_flow = wrap_flow(z_p, out_mask)

    print(f"\nexporting flow -> {flow_path}")
    torch.onnx.export(
        wrap_flow, (z_p, out_mask), str(flow_path),
        input_names=["z_p", "padding_mask"], output_names=["latents"],
        dynamic_axes={"z_p": {0: "batch", 2: "frames"},
                      "padding_mask": {0: "batch", 2: "frames"},
                      "latents": {0: "batch", 2: "frames"}},
        opset_version=OPSET, do_constant_folding=True,
    )
    rep_f = parity(flow_path, {"z_p": z_p.numpy(), "padding_mask": out_mask.numpy()},
                   ref_flow)
    print(f"  parity: {json.dumps(rep_f)}")
    manifest["artifacts"]["flow"] = {
        "file": flow_path.name, "size_mb": round(flow_path.stat().st_size / 1e6, 2),
        "parity": rep_f,
    }

    # TensorRT profiles. Token lengths are bucketed to 16 by the gateway; latent frames
    # run much longer, so the two get different ranges.
    manifest["trt_profiles"] = {
        "duration_predictor": {
            "hidden_states": {"min": [1, ch, 16], "opt": [8, ch, 48], "max": [16, ch, 256]},
            "padding_mask": {"min": [1, 1, 16], "opt": [8, 1, 48], "max": [16, 1, 256]},
            "noise": {"min": [1, 2, 16], "opt": [8, 2, 48], "max": [16, 2, 256]},
        },
        "flow": {
            "z_p": {"min": [1, ch, 16], "opt": [8, ch, 400], "max": [16, ch, 2048]},
            "padding_mask": {"min": [1, 1, 16], "opt": [8, 1, 400], "max": [16, 1, 2048]},
        },
    }

    mpath = outdir / "frontend_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {mpath}")


if __name__ == "__main__":
    main()
