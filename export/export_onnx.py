"""M4 — export the two hot submodules to ONNX, ready for TensorRT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import VitsModel, VitsTokenizer

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")
OPSET = 17


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


class EncoderWrapper(nn.Module):
    """Flat signature for ONNX: (input_ids, attention_mask) -> 3 tensors.

    The padding mask VitsTextEncoder wants is derived from attention_mask inside the
    graph, so callers do not have to reproduce that convention.
    """

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        padding_mask = attention_mask.unsqueeze(-1).float()
        out = self.enc(
            input_ids=input_ids,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return out.last_hidden_state, out.prior_means, out.prior_log_variances


class DecoderWrapper(nn.Module):
    """latents [B, C, T] -> waveform [B, 1, T*hop]."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.dec(latents)


def check(onnx_path: Path, torch_out, feeds: dict, rtol=1e-3, atol=1e-3) -> dict:
    """Run the exported graph on CPU EP and compare against PyTorch."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)
    ref = torch_out if isinstance(torch_out, (list, tuple)) else [torch_out]
    report = {}
    for i, (g, r) in enumerate(zip(got, ref)):
        r = r.detach().cpu().numpy()
        diff = np.abs(g - r)
        denom = float(np.sqrt(np.mean(r.astype(np.float64) ** 2))) or 1.0
        report[f"out{i}"] = {
            "shape": list(g.shape),
            "max_abs_diff": float(diff.max()),
            "rel_rms": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)) / denom),
            "allclose": bool(np.allclose(g, r, rtol=rtol, atol=atol)),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None, help="default: $TTS_MODEL_DIR/onnx")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    args = ap.parse_args()

    path = resolve()
    outdir = Path(args.outdir or (Path(MODEL_DIR) / "onnx"))
    outdir.mkdir(parents=True, exist_ok=True)

    tok = VitsTokenizer.from_pretrained(path)
    model = VitsModel.from_pretrained(path).eval()   # export on CPU: deterministic, and
    cfg = model.config                                # TRT builds from the ONNX anyway
    hop = 1
    for r in cfg.upsample_rates:
        hop *= int(r)
    ch = int(getattr(cfg, "flow_size", 192))
    sr = int(cfg.sampling_rate)

    manifest: dict = {"model": path, "opset": OPSET, "hop": hop,
                      "flow_channels": ch, "sampling_rate": sr, "artifacts": {}}

    # ------------------------------------------------------------------ encoder
    enc_path = outdir / "vits_encoder.onnx"
    ids = tok(args.text, return_tensors="pt")["input_ids"]
    attn = torch.ones_like(ids)
    wrap_e = EncoderWrapper(model.text_encoder).eval()
    with torch.inference_mode():
        ref_e = wrap_e(ids, attn)

    print(f"exporting encoder -> {enc_path}")
    torch.onnx.export(
        wrap_e, (ids, attn), str(enc_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden_states", "prior_means", "prior_log_variances"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "hidden_states": {0: "batch", 1: "seq"},
            "prior_means": {0: "batch", 1: "seq"},
            "prior_log_variances": {0: "batch", 1: "seq"},
        },
        opset_version=OPSET, do_constant_folding=True,
    )
    rep = check(enc_path, ref_e,
                {"input_ids": ids.numpy(), "attention_mask": attn.numpy()})
    print(f"  parity: {json.dumps(rep, indent=2)}")
    manifest["artifacts"]["encoder"] = {
        "file": enc_path.name, "size_mb": round(enc_path.stat().st_size / 1e6, 2),
        "parity": rep,
    }

    # ------------------------------------------------------------------ decoder
    # Exported at the steady-state chunk shape (50 kept + 2*13 overlap = 76 frames), so
    # the traced graph matches what actually gets served.
    dec_path = outdir / "vits_decoder.onnx"
    T_export = 76
    z = torch.randn(1, ch, T_export)
    wrap_d = DecoderWrapper(model.decoder).eval()
    with torch.inference_mode():
        ref_d = wrap_d(z)

    print(f"\nexporting decoder -> {dec_path}  (traced at T={T_export})")
    torch.onnx.export(
        wrap_d, (z,), str(dec_path),
        input_names=["latents"], output_names=["waveform"],
        dynamic_axes={"latents": {0: "batch", 2: "frames"},
                      "waveform": {0: "batch", 2: "samples"}},
        opset_version=OPSET, do_constant_folding=True,
    )
    rep = check(dec_path, ref_d, {"latents": z.numpy()})
    print(f"  parity: {json.dumps(rep, indent=2)}")
    manifest["artifacts"]["decoder"] = {
        "file": dec_path.name, "size_mb": round(dec_path.stat().st_size / 1e6, 2),
        "traced_frames": T_export, "parity": rep,
    }

    # Shapes TensorRT needs optimization profiles for. From M3's progressive sizing:
    # first chunk 12 frames, growing to 50, each padded by 13 either side.
    profiles = {
        "decoder": {
            "min": [1, ch, 12 + 2 * 13],
            "opt": [8, ch, 50 + 2 * 13],      # batch 8 from the M2 batching probe
            "max": [32, ch, 50 + 2 * 13],
        },
        "encoder": {"min": [1, 4], "opt": [8, 48], "max": [32, 256]},
    }
    manifest["trt_profiles"] = profiles
    print("\nTensorRT optimization profiles (min/opt/max):")
    print(json.dumps(profiles, indent=2))

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir / 'manifest.json'}")


if __name__ == "__main__":
    main()
