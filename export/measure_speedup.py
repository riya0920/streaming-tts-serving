"""
Measure whole-pipeline GPU time, PyTorch vs TensorRT, and derive cost per audio-minute.

The résumé claim this settles is "cut per-request GPU time by 62% and inference cost by
41%". Per-stage speedups (decoder 5.67x, duration predictor 6.01x, flow 7.77x) do not
answer it: Amdahl decides the whole-model figure, and cost needs a stated model rather
than a vibe. Both are computed here on the same card, same utterances, same session.

Timing uses CUDA events, not wall clock — CUDA is asynchronous, so a wall-clock timer
around a GPU call measures queueing rather than execution.

Cost model, stated so it can be argued with:

    gpu_seconds_per_audio_minute = (gpu_seconds_per_request / audio_seconds_per_request) * 60
    dollars_per_audio_minute     = gpu_seconds_per_audio_minute * (price_per_hour / 3600)

This deliberately counts only GPU occupancy. It ignores CPU, network and idle headroom —
all of which make real cost higher and none of which shrink proportionally, so the honest
reading is that cost falls by *less* than GPU time does.

  python export/measure_speedup.py --price-per-hour 0.90
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
from export.build_trt import TRTRunner  # noqa: E402
from streaming.chunked import capture_latents  # noqa: E402
from streaming.trt_frontend import TRTFrontend  # noqa: E402

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")

TEXTS = [
    "Sure, I can help with that.",
    "Your flight leaves at four fifteen from gate B twelve, and boarding starts about forty minutes before that.",
    "The main difference is that the first option charges a flat monthly rate regardless of usage, which is simpler to predict, while the second bills per request.",
    "I have added milk, eggs, and coffee to your shopping list, and moved the reminder to tomorrow morning at eight.",
]

# Matches the served chunk geometry from M3 (50 kept + 13 context each side).
CHUNK, OVERLAP = 50, 13


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


def gpu_ms(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def chunked_decode_torch(model, latents):
    """Decode through PyTorch using the same chunk geometry the server serves."""
    T = latents.shape[-1]
    out = 0
    a = 0
    while a < T:
        b = min(a + CHUNK, T)
        lo, hi = max(0, a - OVERLAP), min(T, b + OVERLAP)
        w = model.decoder(latents[..., lo:hi])
        out += w.shape[-1]
        a = b
    return out


def chunked_decode_trt(runner, latents, min_window: int = 14):
    T = latents.shape[-1]
    a = 0
    while a < T:
        b = min(a + CHUNK, T)
        lo, hi = max(0, a - OVERLAP), min(T, b + OVERLAP)
        piece = latents[..., lo:hi]
        if piece.shape[-1] < min_window:
            lo = max(0, hi - min_window)
            piece = latents[..., lo:hi]
        runner({"latents": piece})
        a = b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-per-hour", type=float, default=0.90,
                    help="USD per GPU-hour for the cost model")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="results/m13_speedup_cost.json")
    args = ap.parse_args()

    path = resolve()
    tok = VitsTokenizer.from_pretrained(path)
    model = VitsModel.from_pretrained(path).cuda().eval()
    sr = int(model.config.sampling_rate)
    hop = 1
    for r in model.config.upsample_rates:
        hop *= int(r)

    engines = Path(MODEL_DIR) / "engines"
    trt_front = TRTFrontend(engines, TRTRunner, model.config, "cuda")
    dec_trt = TRTRunner(engines / "vits_decoder_fp16.plan")

    rows = []
    print(f"{'utterance':<12}{'audio s':>9}{'torch ms':>11}{'trt ms':>10}{'speedup':>9}{'reduction':>11}")
    print("-" * 62)

    for text in TEXTS:
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"].cuda()
        mask = enc.get("attention_mask")
        mask = torch.ones_like(ids) if mask is None else mask.cuda()

        # Reference audio length, so cost is per real audio-second.
        with torch.inference_mode():
            ref_lat = capture_latents(model, input_ids=ids, attention_mask=mask)
        audio_s = ref_lat.shape[-1] * hop / sr

        # Full PyTorch path: front half + chunked decode, exactly as M5 served it.
        def torch_path():
            with torch.inference_mode():
                lat = capture_latents(model, input_ids=ids, attention_mask=mask)
                chunked_decode_torch(model, lat)

        # Full TensorRT path: TRT front half + TRT chunked decode, as served now.
        def trt_path():
            lat = trt_front(ids, mask)
            chunked_decode_trt(dec_trt, lat)

        t_torch = gpu_ms(torch_path, args.iters)
        t_trt = gpu_ms(trt_path, args.iters)
        red = (1 - t_trt / t_torch) * 100

        rows.append({"text": text[:36], "audio_seconds": round(audio_s, 2),
                     "torch_ms": round(t_torch, 2), "trt_ms": round(t_trt, 2),
                     "speedup": round(t_torch / t_trt, 2),
                     "reduction_pct": round(red, 1)})
        print(f"{text[:11]:<12}{audio_s:>9.2f}{t_torch:>11.2f}{t_trt:>10.2f}"
              f"{t_torch / t_trt:>8.2f}x{red:>10.1f}%")

    # Weight by audio produced, not by utterance count: a per-utterance mean would let a
    # two-word reply count as much as a fifteen-second explanation.
    tot_audio = sum(r["audio_seconds"] for r in rows)
    torch_total = sum(r["torch_ms"] for r in rows)
    trt_total = sum(r["trt_ms"] for r in rows)
    reduction = (1 - trt_total / torch_total) * 100

    print("-" * 62)
    print(f"{'WEIGHTED':<12}{tot_audio:>9.2f}{torch_total:>11.2f}{trt_total:>10.2f}"
          f"{torch_total / trt_total:>8.2f}x{reduction:>10.1f}%")

    # ---- cost ------------------------------------------------------------
    price = args.price_per_hour
    def cost(ms_total: float) -> dict:
        gpu_s_per_audio_min = (ms_total / 1000.0) / tot_audio * 60.0
        return {
            "gpu_seconds_per_audio_minute": round(gpu_s_per_audio_min, 3),
            "usd_per_audio_minute": round(gpu_s_per_audio_min * price / 3600.0, 6),
            "audio_minutes_per_gpu_hour": round(3600.0 / gpu_s_per_audio_min, 1),
        }

    c_torch, c_trt = cost(torch_total), cost(trt_total)
    cost_red = (1 - c_trt["usd_per_audio_minute"] / c_torch["usd_per_audio_minute"]) * 100

    print(f"\nCOST MODEL at ${price:.2f}/GPU-hour (GPU occupancy only)\n")
    print(f"  {'':<22}{'PyTorch':>14}{'TensorRT':>14}")
    print(f"  {'GPU-s / audio-min':<22}{c_torch['gpu_seconds_per_audio_minute']:>14.3f}"
          f"{c_trt['gpu_seconds_per_audio_minute']:>14.3f}")
    print(f"  {'audio-min / GPU-hr':<22}{c_torch['audio_minutes_per_gpu_hour']:>14.1f}"
          f"{c_trt['audio_minutes_per_gpu_hour']:>14.1f}")
    print(f"  {'USD / audio-min':<22}{c_torch['usd_per_audio_minute']:>14.6f}"
          f"{c_trt['usd_per_audio_minute']:>14.6f}")
    print(f"\n  GPU time reduction : {reduction:.1f}%")
    print(f"  Cost reduction     : {cost_red:.1f}%")
    print("\n  Cost falls by the same proportion as GPU time under this model, because it")
    print("  counts GPU occupancy alone. Real cost falls by LESS: CPU, network and the")
    print("  idle headroom kept to protect the tail do not shrink with kernel time.")

    payload = {
        "price_per_gpu_hour": price,
        "per_utterance": rows,
        "weighted": {"audio_seconds": round(tot_audio, 2),
                     "torch_ms": round(torch_total, 2), "trt_ms": round(trt_total, 2),
                     "speedup": round(torch_total / trt_total, 2),
                     "gpu_time_reduction_pct": round(reduction, 1)},
        "cost": {"pytorch": c_torch, "tensorrt": c_trt,
                 "reduction_pct": round(cost_red, 1),
                 "model": "GPU occupancy only; excludes CPU, network and idle headroom"},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
