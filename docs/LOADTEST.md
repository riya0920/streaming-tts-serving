# M9 — load test, and what the numbers actually are

Full stack on one **NVIDIA A40**: Go gateway → Triton (`tts_frontend` → `vits_frontend`
→ `tts_stream` C++/decoupled) → TensorRT FP16 decoder. Load generator on the same host,
so this is **server-side latency and excludes WAN**. Raw data in
`results/m9_realtime.json`.

Sessions are paced to real time: each holds a WebSocket, speaks an utterance, then waits
so that it occupies `audio_seconds / duty` of wall clock. At `duty=1.0` a session is
speaking continuously. Starts are randomly phased so sessions do not synchronize into
waves.

## The ramp

| concurrent sessions | p50 ms | p90 ms | p99 ms | agg RTF | underruns | rejected |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 103.4 | 136.7 | 136.7 | 1.5 | 0 | 0 |
| **2** | **101.9** | **130.5** | **133.9** | 2.7 | 0 | 0 |
| 4 | 94.3 | 132.5 | 170.3 | 4.7 | 0 | 0 |
| 6 | 90.7 | 146.7 | 254.1 | 7.3 | 0 | 0 |
| 8 | 93.8 | 161.7 | 299.7 | 9.2 | 0 | 0 |
| 12 | 85.7 | 240.7 | 399.5 | 14.6 | 0 | 0 |
| 16 | 78.7 | 283.6 | 581.9 | 19.9 | 0 | 0 |

**Headline: 2 concurrent continuously-speaking sessions hold p99 TTFA under 150 ms.**
At 4 it is 170 ms; at 16 it is 582 ms.

Two things that do *not* degrade are worth as much as the headline:

- **p50 is flat** — 103 ms at one session, 79 ms at sixteen. The median user is served
  well throughout; only the tail stretches.
- **Zero underruns and zero rejections at every level.** No stream ever stuttered
  mid-sentence, and admission control never had to fire. The system degrades by getting
  slower to start, not by breaking.

## Where the time goes, from the gateway's own histograms

Measured at 32 concurrent sessions:

| stage | n | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| `tts_frontend` (normalize + tokenize) | 3550 | 5 ms | 10 ms | 20 ms |
| **`vits_frontend` (encoder + duration + flow)** | 3550 | **150 ms** | **500 ms** | **1000 ms** |
| TTFA end to end | 3547 | 200 ms | 600 ms | 1000 ms |

**The entire tail is one stage**, and it is the stage M4 could not convert to TensorRT:
the stochastic duration predictor samples `randn` internally, so its graph is not a pure
function of its inputs, and the flow layers are the numerically touchy part in half
precision. Together they were 36% of GPU time, left in FP32 PyTorch as a stated
limitation — and they turn out to set both the latency tail and the concurrency ceiling.

The decoder, which got the 4.07x TensorRT speedup, never appears in the tail at all.

That is the most useful result this project produced. The optimization went exactly where
the profile said the GPU time was, and the *constraint* was somewhere else. Only
instrumenting the assembled system and loading it until it hurt could show that.

## Why TTFA is not utterance-length-bound

Worth ruling out, because it was the obvious first hypothesis for a bimodal distribution
(p50 72 ms against p99 1033 ms at 32 sessions):

| utterance | words | audio | TTFA |
|---|---:|---:|---:|
| short | 6 | 2.32 s | 65.7 ms |
| medium | 18 | 6.24 s | 75.2 ms |
| long | 37 | 15.17 s | 80.8 ms |
| xlong | 57 | 17.47 s | 91.1 ms |

9.5x the words costs 1.4x the TTFA. The chunked decoder did its job: time-to-first-audio
is essentially independent of utterance length. The tail is queueing, not length.

## Throughput ceiling

Aggregate real-time factor saturates at **~207 audio-seconds per wall-clock second** when
sessions are unthrottled. So the hardware can *produce* ~200x real-time audio; it just
cannot start ~200 streams within 150 ms of each being asked.

That gap between throughput capacity and latency capacity is the whole subject of the
project, and the measurement puts a number on it: **~100x**.

## Against the original claim

The write-up this project was built from claimed **148 ms p99 across 3,200 concurrent
streaming sessions on an A100**. Measured, on an A40:

| | claimed | measured |
|---|---|---|
| p99 TTFA | 148 ms | **134 ms at 2 sessions**, 170 ms at 4 |
| concurrent sessions at that p99 | 3,200 | **2** continuously speaking |
| GPU | A100 | A40 |

At a 10% duty cycle — a voice agent taking short turns, which is the reading that makes
"sessions" generous — 2 concurrently-speaking works out to roughly **20 held sessions**.
An A100 might give 2–2.5x. That is ~50, not 3,200.

The p99 latency figure is real and reproducible. **The concurrency figure is off by more
than two orders of magnitude**, and no amount of tuning closes that gap; it would take a
different model, a converted front half, and cross-request batching.

## What would actually move it

In order of expected effect, each identified by measurement rather than guessed:

1. **Convert the duration predictor and flow to TensorRT.** Hoist the duration
   predictor's internal `randn` into a graph input to make it exportable. This is
   100% of the measured tail. A 4x on this stage — the same factor the decoder got —
   would move the knee to roughly 8–10 concurrent sessions.
2. **Batch chunks across requests inside `tts_stream`.** M2 measured decoder batching as
   nearly free up to B≈8, worth 4–8x per stream. The C++ backend currently decodes one
   request at a time at batch 1, so that win is entirely unrealized. Triton's dynamic
   batcher cannot do it for us because the backend owns the chunk loop.
3. **CUDA graphs on the front half.** The model is launch-bound (M2: 96x the work for
   1.10x the time), and graph capture attacks launch overhead directly.
4. **A larger GPU**, last — because 2 and 3 are free and the hardware is not saturated:
   the GPU is at ~10% of its throughput ceiling when the latency knee is hit.
