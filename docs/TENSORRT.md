# M4 — TensorRT engines

`facebook/mms-tts-eng` decoder and text encoder, TensorRT 10.3.0, **NVIDIA A40**.
Raw data: `results/m4_trt.json`, `results/m4_fp16_quality.json`. Audio: `results/audio/`.

## Engines

| engine | build | size |
|---|---:|---:|
| decoder FP32 | 58.8 s | 59.3 MB |
| decoder FP16 | 210.2 s | 32.4 MB |
| encoder FP32 | 20.7 s | 32.6 MB |
| encoder FP16 | 59.2 s | 20.0 MB |

Optimization profiles come from M3's progressive chunking: frames 14 → 76 (a 1-frame
remainder with left context only, up to 50 kept + 13 context per side), batch 1 → 32 with
opt at 8, which is where the M2 batching probe showed free batching ends.

FP32 engines were built deliberately, not as a leftover. M2 found the decoder is
launch-bound in this regime, so the prediction was that fusion — not precision — would
do the work. Building both is the only way to test that.

## Speed

Decoder, TensorRT vs PyTorch, at served shapes:

| shape | PyTorch | TRT FP32 | TRT FP16 | FP16 speedup |
|---|---:|---:|---:|---:|
| B=1 T=38 | 6.97 ms | 3.42 ms | 1.71 ms | **4.07x** |
| B=1 T=76 | 7.02 ms | 4.15 ms | 1.87 ms | 3.76x |
| B=8 T=76 | 27.61 ms | 16.97 ms | 6.86 ms | 4.02x |
| B=32 T=76 | 102.47 ms | 63.03 ms | 24.83 ms | 4.13x |

The prediction was half right. Fusion alone (FP32 engine) gives **~1.7–2x**, confirming
launch overhead was a real cost. But FP16 adds another **~2x on top**, including at B=1
T=38 where M2 said we were firmly launch-bound and precision "shouldn't" matter much.

Two things explain the extra: TensorRT's FP16 kernels use tensor cores, and halving
activation size lets more layers fuse, which reduces launches again. So precision is not
purely a bandwidth lever here — it buys additional fusion.

## Quality

**The first attempt at this was wrong.** The harness embedded in `build_trt.py` decoded
each utterance in 76-frame pieces with *no overlap* and compared against a single-pass
reference. M3 had already measured overlap-0 chunking at ~4 dB SNR, so that harness was
mostly scoring chunk-boundary artifacts and attributing them to precision. It reported
2.37 dB LSD and "AUDIBLE". `export/validate_fp16.py` replaces it, comparing the same
window through different backends and looking only at the valid centre.

Corrected, over 12 real 76-frame windows:

| comparison | SNR | LSD | isolates |
|---|---:|---:|---|
| TRT FP32 vs PyTorch FP32 | 58.21 dB | **0.116 dB** | graph conversion |
| TRT FP16 vs PyTorch FP32 | 42.34 dB | **1.524 dB** | what a user hears |
| TRT FP16 vs TRT FP32 | 42.09 dB | 1.525 dB | precision alone |

**TensorRT's conversion is lossless for practical purposes.** At 0.116 dB LSD the FP32
engine is indistinguishable from PyTorch — so a 1.7–2x speedup is available at no quality
cost whatsoever, and that is not a trade-off, it is free.

All of the FP16 error is precision (1.525 dB precision-alone vs 0.116 dB conversion).
1.52 dB sits in the "marginal" band — between transparent (<1 dB) and audible (>2 dB).

Those thresholds are heuristics, not measurements, so `export/make_ab_audio.py` renders
identical latents through each backend for listening. Error relative to signal peak:

| utterance | FP16 error below signal |
|---|---:|
| medium | −34.6 dB |
| short | −32.0 dB |
| long | −29.3 dB |
| **numbers** | **−24.3 dB** |

The number-heavy line is consistently worst, which is what you would expect: fricatives
and plosives carry high-frequency energy with low amplitude, exactly where FP16's reduced
mantissa costs the most.

## The decision

Three options, and the data makes the trade explicit rather than assumed:

| option | speedup | quality cost |
|---|---:|---|
| TRT FP32 | ~1.9x | none measurable (0.116 dB) |
| TRT FP16 | ~4.0x | 1.52 dB LSD, marginal |
| Mixed precision | ~3–4x expected | untested — pin sensitive layers to FP32 |

**Pending the listening test.** If FP16 is indistinguishable, take the 4x. If it is not,
mixed precision is the next move: TensorRT allows per-layer precision, and for vocoders
the usual candidates are the first and last convolutions, where the output dynamic range
is widest. That would be a targeted experiment, not a guess — the −24.3 dB result on
fricative-heavy text says where to look.

## Standing limit

The stochastic duration predictor (21.9% of GPU time) and flow (14.2%) remain FP32
PyTorch. Together they are **36% of the forward pass**, so by Amdahl a 4x decoder speedup
yields at most ~1.9x on the model as a whole. Making them engines requires hoisting the
duration predictor's internal `randn` into a graph input; that is tractable and worth
doing, but it is a separate change and the flow is the numerically touchy part in half
precision.
