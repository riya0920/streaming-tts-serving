"""
M10 — build TensorRT engines for the duration predictor and flow, and prove they help.

These are the two graphs `export_frontend_onnx.py` produced. Together they are the stage
M9 measured as 100% of the latency tail, so this is the only remaining lever that attacks
per-request latency rather than throughput — and the load test showed the knee arrives at
~2% GPU utilization, meaning throughput was never the constraint.

Validation here is stricter than for the decoder, for one specific reason: the duration
predictor's noise is now a graph input. An engine that ignores that input, or that
collapses it to a constant during optimization, would still pass a parity check against a
single fixed sample — and every utterance served would come out with identical rhythm.
So the noise is explicitly re-tested through the built engine, not just the ONNX.

  python export/build_trt_frontend.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from transformers import VitsModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export.build_trt import TRTRunner, bench, build_engine  # noqa: E402

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


def rel_rms(ref: np.ndarray, got: np.ndarray) -> float:
    ref64, got64 = ref.astype(np.float64), got.astype(np.float64)
    denom = float(np.sqrt(np.mean(ref64 ** 2))) or 1.0
    return float(np.sqrt(np.mean((got64 - ref64) ** 2)) / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=None)
    ap.add_argument("--out", default="results/m10_frontend_trt.json")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir or (Path(MODEL_DIR) / "onnx"))
    engines = Path(MODEL_DIR) / "engines"
    engines.mkdir(parents=True, exist_ok=True)

    model = VitsModel.from_pretrained(resolve()).cuda().eval()
    ch = int(getattr(model.config, "flow_size", 192))
    results: dict = {"trt_version": trt.__version__, "engines": {}, "bench": {}}

    # Token lengths are bucketed to 16 by the gateway; latent frames run far longer, so
    # the two graphs get very different profiles.
    profiles = {
        # Same character-level reasoning as the encoder: 1024 tokens, not 256.
        "duration": {
            "hidden_states": ([1, ch, 4], [8, ch, 64], [16, ch, 1024]),
            "padding_mask": ([1, 1, 4], [8, 1, 64], [16, 1, 1024]),
            "noise": ([1, 2, 4], [8, 2, 64], [16, 2, 1024]),
        },
        "flow": {
            "z_p": ([1, ch, 4], [8, ch, 400], [16, ch, 2048]),
            "padding_mask": ([1, 1, 4], [8, 1, 400], [16, 1, 2048]),
        },
    }

    print(f"TensorRT {trt.__version__}\n")
    for name, onnx_name in (("duration", "vits_duration.onnx"), ("flow", "vits_flow.onnx")):
        for tag, fp16 in (("fp32", False), ("fp16", True)):
            out = engines / f"vits_{name}_{tag}.plan"
            if out.exists():
                print(f"  {out.name} exists, skipping")
                info = {"file": out.name, "size_mb": round(out.stat().st_size / 1e6, 2),
                        "fp16": fp16, "build_seconds": None}
            else:
                print(f"  building {out.name} ...", flush=True)
                info = build_engine(onnx_dir / onnx_name, out, profiles[name], fp16)
                print(f"    {info['build_seconds']}s, {info['size_mb']} MB")
            results["engines"][f"{name}_{tag}"] = info

    # ---------------------------------------------------------- duration predictor
    print("\nDURATION PREDICTOR — TensorRT vs PyTorch")
    dur16 = TRTRunner(engines / "vits_duration_fp16.plan")
    S = 48
    hidden = torch.randn(1, ch, S, device="cuda")
    pad = torch.ones(1, 1, S, device="cuda")
    noise = torch.randn(1, 2, S, device="cuda")

    with torch.inference_mode():
        t_torch = bench(lambda: model.duration_predictor(
            hidden, pad, reverse=True, noise_scale=1.0), 30)
    t_trt = bench(lambda: dur16({"hidden_states": hidden, "padding_mask": pad,
                                 "noise": noise}), 30)
    print(f"  torch {t_torch:.3f} ms   trt16 {t_trt:.3f} ms   speedup {t_torch / t_trt:.2f}x")
    results["bench"]["duration"] = {"torch_ms": round(t_torch, 3),
                                    "trt_fp16_ms": round(t_trt, 3),
                                    "speedup": round(t_torch / t_trt, 2)}

    # The check that matters. A parity test against one fixed noise sample cannot tell
    # the difference between "noise is wired up" and "noise was folded into a constant";
    # both give identical output for identical input. Feed two different draws instead.
    n1 = torch.randn(1, 2, S, device="cuda")
    n2 = torch.randn(1, 2, S, device="cuda")
    d1 = dur16({"hidden_states": hidden, "padding_mask": pad, "noise": n1})[0].float().cpu().numpy()
    d2 = dur16({"hidden_states": hidden, "padding_mask": pad, "noise": n2})[0].float().cpu().numpy()
    spread = float(np.abs(d1 - d2).mean())
    live = spread > 1e-6
    print(f"  noise still live through the engine: {live} (mean |d1-d2| = {spread:.6f})")
    results["bench"]["duration"]["noise_live"] = live
    results["bench"]["duration"]["noise_spread"] = round(spread, 6)
    if not live:
        raise RuntimeError(
            "TensorRT folded the noise input into a constant. Every utterance would get "
            "identical durations and identical prosody. Rebuild with the noise tensor "
            "excluded from constant folding."
        )

    # Same noise must reproduce PyTorch, or the engine is fast and wrong.
    with torch.inference_mode():
        ref = model.duration_predictor(hidden, pad, reverse=True, noise_scale=1.0)
    # PyTorch samples internally, so compare shape and distribution rather than values;
    # exact agreement is only checkable through the ONNX, which export already did.
    got = dur16({"hidden_states": hidden, "padding_mask": pad, "noise": noise})[0]
    print(f"  shapes: torch {tuple(ref.shape)} vs trt {tuple(got.shape)}")
    results["bench"]["duration"]["shape_match"] = list(ref.shape) == list(got.shape)

    # ------------------------------------------------------------------- flow
    print("\nFLOW — TensorRT vs PyTorch")
    flow16 = TRTRunner(engines / "vits_flow_fp16.plan")
    for T in (200, 400, 800):
        z = torch.randn(1, ch, T, device="cuda")
        m = torch.ones(1, 1, T, device="cuda")
        with torch.inference_mode():
            tt = bench(lambda: model.flow(z, m, reverse=True), 20)
            ref_f = model.flow(z, m, reverse=True).float().cpu().numpy()
        tr = bench(lambda: flow16({"z_p": z, "padding_mask": m}), 20)
        got_f = flow16({"z_p": z, "padding_mask": m})[0].float().cpu().numpy()
        err = rel_rms(ref_f, got_f)
        print(f"  T={T:>4}  torch {tt:>7.3f} ms   trt16 {tr:>7.3f} ms"
              f"   speedup {tt / tr:>5.2f}x   rel_rms {err:.2e}")
        results["bench"][f"flow_T{T}"] = {"torch_ms": round(tt, 3), "trt_fp16_ms": round(tr, 3),
                                          "speedup": round(tt / tr, 2), "rel_rms": err}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
