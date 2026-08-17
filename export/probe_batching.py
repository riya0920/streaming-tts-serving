"""
The decisive experiment for the chunked-streaming design.

M2 established that the HiFi-GAN decoder is launch-bound below ~192 latent frames: 96x
more work costs 1.1x the time. Two consequences pull in opposite directions.

  Against chunking: decoding a 192 ms chunk costs 4.6 ms while decoding a whole 5 s
  utterance costs 11.5 ms. Per audio-second, chunking is ~8x more expensive.

  For chunking: if the GPU is idle between kernel launches, then batching many sessions'
  chunks into one call should cost barely more than a single chunk — which is exactly
  the regime where Triton's dynamic batching pays for its queue delay.

So the question that decides the architecture is: **at the chunk sizes we would actually
serve, how does cost scale with batch size?** If per-stream cost collapses with batch,
chunked streaming plus dynamic batching wins and the design stands. If it scales
linearly, chunking is a net loss and the design needs rethinking.

Also measures CPU wall time against GPU stream time. A large gap means the Python
dispatch path — not the GPU — is the constraint, which is the quantitative case for
moving the per-chunk loop into C++.

  python export/probe_batching.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from transformers import VitsModel, VitsTokenizer

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")
DEV = "cuda"


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


def gpu_ms(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def wall_ms(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    path = resolve()
    tok = VitsTokenizer.from_pretrained(path)
    model = VitsModel.from_pretrained(path).to(DEV).eval()
    sr = int(model.config.sampling_rate)
    ch = int(getattr(model.config, "flow_size", 192))
    hop = 1
    for r in model.config.upsample_rates:
        hop *= int(r)

    results: dict = {"model": path, "sampling_rate": sr, "hop": hop}

    # Chunk sizes worth serving: 100 / 200 / 400 ms of audio, plus the padded window a
    # chunk actually decodes once overlap context is included (chunk + 2*overlap).
    chunk_specs = {
        "100ms": round(0.100 * sr / hop),
        "200ms": round(0.200 * sr / hop),
        "400ms": round(0.400 * sr / hop),
        "200ms+ovl16": round(0.200 * sr / hop) + 32,
    }
    batches = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f"model {path} | hop={hop} | {sr} Hz\n")
    print("DECODER: per-stream cost vs batch size, at streaming chunk sizes")
    print("(if per-stream cost collapses, chunking + dynamic batching is the right design)\n")

    batch_table: dict = {}
    with torch.inference_mode():
        for label, T in chunk_specs.items():
            row = {}
            print(f"  chunk {label}  (T={T} frames, {T * hop / sr * 1e3:.0f} ms audio)")
            base = None
            for B in batches:
                try:
                    z = torch.randn(B, ch, T, device=DEV)
                    ms = gpu_ms(lambda: model.decoder(z), 30)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"      B={B:>4}  OOM")
                    break
                per = ms / B
                if base is None:
                    base = per
                row[B] = {"total_ms": round(ms, 3), "per_stream_ms": round(per, 4)}
                print(f"      B={B:>4}  total {ms:>8.3f} ms   per-stream {per:>7.4f} ms"
                      f"   speedup {base / per:>5.1f}x")
                del z
            batch_table[label] = row
            print()
    results["decoder_batch_scaling"] = batch_table

    # --------------------------------------------------------------- cpu vs gpu
    # A wide gap here means the dispatch path is the bottleneck, not the device.
    print("CPU wall time vs GPU stream time (gap = dispatch overhead)\n")
    text = "The quick brown fox jumps over the lazy dog, and then it does it again."
    inputs = tok(text, return_tensors="pt").to(DEV)
    gap: dict = {}
    with torch.inference_mode():
        g = gpu_ms(lambda: model(**inputs), 30)
        w = wall_ms(lambda: model(**inputs), 30)
        gap["full_forward"] = {"gpu_ms": round(g, 3), "wall_ms": round(w, 3),
                               "cpu_overhead_ms": round(w - g, 3)}
        print(f"  full forward     gpu {g:>7.3f} ms   wall {w:>7.3f} ms"
              f"   cpu overhead {w - g:>6.3f} ms ({(w - g) / w * 100:.0f}%)")

        T = chunk_specs["200ms+ovl16"]
        z = torch.randn(1, ch, T, device=DEV)
        g = gpu_ms(lambda: model.decoder(z), 50)
        w = wall_ms(lambda: model.decoder(z), 50)
        gap["one_chunk_decode"] = {"gpu_ms": round(g, 3), "wall_ms": round(w, 3),
                                   "cpu_overhead_ms": round(w - g, 3)}
        print(f"  one chunk decode gpu {g:>7.3f} ms   wall {w:>7.3f} ms"
              f"   cpu overhead {w - g:>6.3f} ms ({(w - g) / w * 100:.0f}%)")
    results["cpu_vs_gpu"] = gap

    # ------------------------------------------------------- concurrency estimate
    # How many live streams can one GPU sustain? Each stream needs one chunk of audio
    # decoded per chunk-duration of wall time. With batching, cost per batched call is
    # what matters, not cost per stream.
    print("\nSustainable concurrent streams (decoder only, ignoring frontend)\n")
    est = {}
    T = chunk_specs["200ms+ovl16"]
    chunk_audio_s = round(0.200 * sr / hop) * hop / sr
    for B, v in batch_table["200ms+ovl16"].items():
        # In chunk_audio_s of wall time we can issue chunk_audio_s / call_ms batched calls,
        # each serving B streams.
        calls_per_window = chunk_audio_s * 1e3 / v["total_ms"]
        est[B] = round(calls_per_window * B)
        print(f"  batch {B:>4}  ->  ~{est[B]:>6} concurrent streams"
              f"   (call {v['total_ms']:.2f} ms per {chunk_audio_s * 1e3:.0f} ms window)")
    results["stream_capacity_estimate"] = est
    print("\n  Upper bound only: frontend cost, queueing, and the gateway are not in this.")

    out = Path("docs/batching_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
