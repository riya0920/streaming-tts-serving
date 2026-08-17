"""
M4 — does FP16 cost audible quality?

Supersedes the quality check that was embedded in build_trt.py, which was wrong: it
decoded each utterance in 76-frame pieces with NO overlap and compared against a
single-pass reference, so it measured chunk-boundary artifacts (M3 put those at ~4 dB
SNR on their own) and attributed them to precision.

The fix is to compare like with like. Every comparison here decodes **the same window**
through different backends and looks only at the valid centre — the region M3 showed is
unaffected by window edges. No chunking, no stitching, nothing to confound the result.

Three comparisons, because "FP16 is worse" is not one claim but two:

  TRT FP32 vs PyTorch FP32   conversion error alone (graph rewrite, fusion, tactics)
  TRT FP16 vs PyTorch FP32   what a user would actually hear
  TRT FP16 vs TRT FP32       precision alone, with conversion held constant

If the first is already large, precision is not the story and the engine build is.

  python export/validate_fp16.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import VitsModel, VitsTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export.build_trt import TRTRunner, log_spectral_distance, snr_db  # noqa: E402
from streaming.chunked import ChunkConfig, ChunkedSynthesizer  # noqa: E402

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")

TEXTS = [
    "Sure, I can help with that.",
    "Your flight leaves at four fifteen from gate B twelve, and boarding starts about forty minutes before that.",
    "The main difference is that the first option charges a flat monthly rate regardless of usage, which is simpler to predict.",
]

WINDOW = 76      # steady-state served window: 50 kept + 13 context each side
TRIM = 13        # measured receptive field; discard the contaminated edges


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


def valid_centre(wav: np.ndarray, hop: int) -> np.ndarray:
    """Drop the edge samples that the window boundary contaminated."""
    pad = TRIM * hop
    return wav[pad:-pad] if wav.shape[0] > 2 * pad else wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m4_fp16_quality.json")
    args = ap.parse_args()

    path = resolve()
    model = VitsModel.from_pretrained(path).cuda().eval()
    tok = VitsTokenizer.from_pretrained(path)
    hop = 1
    for r in model.config.upsample_rates:
        hop *= int(r)

    engines = Path(MODEL_DIR) / "engines"
    r32 = TRTRunner(engines / "vits_decoder_fp32.plan")
    r16 = TRTRunner(engines / "vits_decoder_fp16.plan")

    syn = ChunkedSynthesizer(model, tok, ChunkConfig(), "cuda")
    torch.manual_seed(7)

    # Collect real windows from real latents — random noise would not exercise the
    # decoder's actual dynamic range.
    windows = []
    for t in TEXTS:
        z = syn.latents_for(t)
        T = z.shape[-1]
        for s in range(0, max(1, T - WINDOW), max(1, (T - WINDOW) // 3 or 1)):
            if s + WINDOW <= T:
                windows.append(z[..., s:s + WINDOW])
        if len(windows) > 12:
            break
    print(f"{len(windows)} windows of {WINDOW} frames from {len(TEXTS)} utterances\n")

    pairs = {"trt32_vs_torch": [], "trt16_vs_torch": [], "trt16_vs_trt32": []}
    with torch.inference_mode():
        for z in windows:
            t_ref = valid_centre(model.decoder(z).squeeze().float().cpu().numpy(), hop)
            e32 = valid_centre(r32({"latents": z})[0].float().squeeze().cpu().numpy(), hop)
            e16 = valid_centre(r16({"latents": z})[0].float().squeeze().cpu().numpy(), hop)

            pairs["trt32_vs_torch"].append((snr_db(t_ref, e32), log_spectral_distance(t_ref, e32)))
            pairs["trt16_vs_torch"].append((snr_db(t_ref, e16), log_spectral_distance(t_ref, e16)))
            pairs["trt16_vs_trt32"].append((snr_db(e32, e16), log_spectral_distance(e32, e16)))

    print(f"  {'comparison':<20}{'SNR dB':>10}{'LSD dB':>10}   isolates")
    print("  " + "-" * 62)
    labels = {
        "trt32_vs_torch": "TRT conversion",
        "trt16_vs_torch": "what a user hears",
        "trt16_vs_trt32": "precision alone",
    }
    results = {}
    for k, vals in pairs.items():
        snr = float(np.mean([v[0] for v in vals]))
        lsd = float(np.mean([v[1] for v in vals]))
        results[k] = {"snr_db": round(snr, 2), "lsd_db": round(lsd, 3)}
        print(f"  {k:<20}{snr:>10.2f}{lsd:>10.3f}   {labels[k]}")

    lsd = results["trt16_vs_torch"]["lsd_db"]
    # Under ~1 dB LSD is generally treated as transparent for vocoder output; beyond
    # ~2 dB it tends to be audible as roughness.
    verdict = "transparent" if lsd < 1.0 else ("marginal" if lsd < 2.0 else "AUDIBLE")
    print(f"\n  FP16 end-to-end: LSD {lsd:.3f} dB -> {verdict}")

    conv = results["trt32_vs_torch"]["lsd_db"]
    prec = results["trt16_vs_trt32"]["lsd_db"]
    if conv > prec:
        print(f"  Conversion ({conv:.3f} dB) exceeds precision ({prec:.3f} dB): the engine")
        print("  build is the larger error source, not FP16.")
    else:
        print(f"  Precision ({prec:.3f} dB) exceeds conversion ({conv:.3f} dB): FP16 is the")
        print("  larger error source. Selectively pinning sensitive layers to FP32 is the lever.")

    results["verdict"] = verdict
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
