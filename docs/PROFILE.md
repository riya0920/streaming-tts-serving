# M2 — Profile before optimizing

All numbers: `facebook/mms-tts-eng` (VITS, 36.3M params, 16 kHz, hop 256), PyTorch 2.6
FP32, **NVIDIA A40 48 GB**. Raw data in `docs/stage_profile.json`,
`docs/batching_probe.json`, `docs/receptive_field.json`.

This milestone exists to check the assumptions the design was built on *before* spending
time optimizing. Three of them changed.

---

## 1. Where the time goes

Device time via CUDA events (5.06 s of audio per forward):

| Stage | ms/forward | % |
|---|---:|---:|
| decoder (HiFi-GAN) | 38.02 | **54.5%** |
| duration_predictor (stochastic) | 15.26 | 21.9% |
| flow | 9.91 | 14.2% |
| text_encoder | 5.86 | 8.4% |
| glue | 0.74 | 1.1% |
| **total** | **69.79** | |

Real-time factor 72.4x at batch 1.

The decoder dominating is the one assumption that survived. But the front half is **44%**
combined, and the stochastic duration predictor alone costs 2.6x the text encoder — so
optimizing only the decoder caps the achievable speedup at about 1.8x no matter how good
the engine is.

---

## 2. The decoder is launch-bound, and that reframes everything

| latent frames | audio | decode ms |
|---:|---:|---:|
| 4 | 64 ms | 4.585 |
| 12 | 192 ms | 4.608 |
| 48 | 768 ms | 4.576 |
| 96 | 1536 ms | 5.054 |
| 192 | 3072 ms | 10.151 |
| 384 | 6144 ms | 18.800 |

**96x the work for 1.10x the time**, up to ~96 frames. There is a fixed floor of roughly
**4.5 ms per decoder call** regardless of how little audio it produces — the GPU idles
between kernel launches while the CPU dispatches the next of ~40 small convolutions.

This is the single most consequential measurement in the project, and it cuts both ways.

- **Against chunking:** a 192 ms chunk costs 4.4 ms; the entire 5 s utterance costs
  11.5 ms. Per audio-second, chunking is ~8x more expensive.
- **For chunking:** an idle GPU between launches is exactly the condition under which
  batching is nearly free — which is the whole argument for dynamic batching.

It also relocates the highest-leverage optimization. Against a fixed per-call launch cost,
the wins are kernel fusion, CUDA graphs, and batching — **not** arithmetic precision.
FP16's benefit here is smaller than the framing "memory-bandwidth-bound, so halving bytes
nearly halves time" implies; that framing describes the compute-bound regime this model
only enters above ~192 frames.

---

## 3. Batching is free, but only up to B≈4–8

Per-stream decoder cost against batch size:

| batch | chunk 100 ms | chunk 200 ms | chunk 400 ms |
|---:|---:|---:|---:|
| 1 | 5.29 ms | 4.39 ms | 4.50 ms |
| 4 | 1.37 ms | 1.09 ms | 1.39 ms |
| 8 | 0.69 ms | 0.70 ms | 1.35 ms |
| 32 | 0.31 ms | 0.57 ms | 1.12 ms |
| 128 | 0.27 ms | 0.53 ms | 1.05 ms |

Up to B≈4–8 the batch is absorbed into the idle time and costs almost nothing — a 4–8x
per-stream win for free. Past that the decoder becomes compute-bound and scaling is
roughly linear, so further batching buys little while adding queue delay.

**This sets the dynamic batching policy directly:** a queue window sized to collect ~8
requests, not 32 or 128. Bigger batches trade real latency for negligible throughput.

---

## 4. The receptive field is larger than the chunk

Measured by perturbing one latent frame and finding where the output changes by more
than −60 dB relative to peak:

```
influence spans ~10 frames left, ~12 frames right  ->  effective RF = 13 frames = 208 ms
```

The 200 ms chunk the design assumed is **smaller than the decoder's own receptive field**.
Correct overlap is 13 frames per side, so a 12-frame chunk must decode 38 frames:

| overlap | frames decoded | wasted | sufficient? |
|---:|---:|---:|---|
| 4 | 20 | 40% | no — seams click |
| 8 | 28 | 57% | no — seams click |
| **13** | **38** | **68%** | **yes** |
| 16 | 44 | 73% | yes, needlessly |

At batch 1 that waste is free — the launch floor means 38 frames costs the same as 12. It
stops being free under batching, where the compute-bound regime makes it real: at B=32,
T=12 costs 18.3 ms against T=44 at 61.6 ms.

---

## Consequences for the design

1. **Progressive chunk sizing.** Use a small first chunk to get time-to-first-audio down,
   then grow subsequent chunks — the listener is already hearing audio, so later chunks
   should optimize amortization rather than latency. Small chunks everywhere pay the
   overlap tax on every chunk for a benefit only the first one delivers.
2. **Target batch 8, not 32.** From the table above, not from intuition.
3. **Optimize the front half too.** 44% of the time is encoder + duration + flow, and
   the stochastic duration predictor is the second-biggest single cost.
4. **Prioritize CUDA graphs and kernel fusion over FP16.** The model is launch-bound in
   the regime we serve; that is what a fixed 4.5 ms per-call floor means.
5. **Streaming convolutions are the real fix, if the waste starts to matter.** Caching
   each conv layer's tail state would eliminate the left-context overlap entirely and
   leave only right-side lookahead. That is a larger change and only worth it if the
   68% waste shows up as a real constraint under load.

## Capacity, honestly

Decoder-only upper bound on this A40, from `docs/batching_probe.json`: roughly **100
concurrent streams** at 200 ms chunks. Including the frontend (~31 ms GPU per utterance),
a full stream costs on the order of 16 ms of GPU per audio-second, which puts a realistic
ceiling near **60 continuously-speaking streams**.

An A100 is perhaps 2–2.5x this, and TensorRT with CUDA graphs plausibly another 2–3x
against a launch-bound model — call it 300–450 continuously-speaking streams.

Reaching a number like 3,200 *sessions* therefore depends entirely on **duty cycle**: a
held session in a voice agent is speaking maybe 10–20% of the time. At 10%, 3,200 held
sessions is ~320 concurrently speaking, which is inside that range. That is a legitimate
way to count, and it is what makes the number achievable — but it has to be **stated**,
because "3,200 concurrent sessions" and "3,200 simultaneous synthesis streams" differ by
an order of magnitude. M9 reports both, with the duty cycle spelled out.
