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

Final configuration — `vits_frontend` with dynamic batching, three instances
(`results/m9_batched.json`):

| concurrent sessions | p50 ms | p90 ms | p99 ms | agg RTF | underruns | rejected |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 138.1 | 163.6 | 163.6 | 1.5 | 0 | 0 |
| 2 | 121.7 | 150.2 | 151.2 | 2.7 | 0 | 0 |
| **4** | **117.1** | **130.6** | **144.0** | 4.7 | 0 | 0 |
| 8 | 101.7 | 160.5 | 221.0 | 9.0 | 0 | 0 |
| 16 | 88.1 | 212.3 | 394.0 | 19.7 | 0 | 0 |
| 32 | 80.9 | 376.3 | 791.7 | 40.0 | 0 | 0 |

**Headline: 4 concurrent continuously-speaking sessions hold p99 TTFA under 150 ms**
(144.0 ms). At 8 it is 221 ms; at 32 it is 792 ms.

### What tuning bought, and what it did not

| config | knee (p99 under 150 ms) | p99 @ 4 | p99 @ 16 |
|---|---:|---:|---:|
| 8 instances, no batching | 2 | 170.3 ms | 581.9 ms |
| **3 instances, batching** | **4** | **144.0 ms** | **394.0 ms** |
| 8 instances, batching | 0 | 163.8 ms | 447.8 ms |

Enabling batching on the bottleneck stage **doubled** the knee and cut the p99 at 16
sessions by a third. Adding instances *on top of* batching made it worse everywhere:
spreading arrivals across eight instances means the 2 ms queue window almost never
collects a second request, so no batch forms and eight processes contend for one GPU.
Instances and batching pull against each other at this arrival rate.

### The number that reframes the ceiling

At the knee, aggregate real-time factor is **4.7 against a ceiling of 207 — about 2% GPU
utilization.** The latency wall arrives at 2% load.

This system is **latency-bound, not throughput-bound.** What limits concurrency is the
~100 ms of mostly-serial work each request spends in `vits_frontend`, not the device's
capacity. The practical consequence: **a bigger GPU buys almost nothing here.** An A100
would raise the 207x ceiling we are nowhere near, and leave the per-request latency that
actually sets the knee essentially untouched.

Two things that do *not* degrade are worth as much as the headline:

- **p50 is flat, and improves** — 138 ms at one session, 81 ms at thirty-two. The median
  user is served well throughout; only the tail stretches.
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
| p99 TTFA | 148 ms | **144 ms at 4 sessions** |
| concurrent sessions at that p99 | 3,200 | **4** continuously speaking |
| GPU | A100 | A40 |

At a 10% duty cycle — a voice agent taking short turns, which is the reading that makes
"sessions" generous — 4 concurrently-speaking is roughly **40 held sessions**.

An A100 would not close the gap either, because the constraint is per-request latency at
2% utilization rather than throughput. Reaching 3,200 at any defensible duty cycle needs
the front half converted and cross-request chunk batching — a different system, not a
bigger card.

The p99 latency figure is real and reproducible. **The concurrency figure is off by more
than two orders of magnitude**, and no amount of tuning closes that gap; it would take a
different model, a converted front half, and cross-request batching.

## What would actually move it

In order of expected effect, each identified by measurement rather than guessed:

1. **Convert the duration predictor and flow to TensorRT.** Hoist the duration
   predictor's internal `randn` into a graph input to make it exportable. This is 100%
   of the measured tail, and with batching already in and the system provably
   latency-bound at 2% utilization, it is the only remaining lever that attacks
   per-request latency rather than throughput. A 4x here, the factor the decoder got,
   would plausibly take the knee to 12–16.
2. **Batch chunks across requests inside `tts_stream`.** M2 measured decoder batching as
   nearly free up to B≈8, worth 4–8x per stream. The C++ backend currently decodes one
   request at a time at batch 1, so that win is entirely unrealized. Triton's dynamic
   batcher cannot do it for us because the backend owns the chunk loop.
3. **CUDA graphs on the front half.** The model is launch-bound (M2: 96x the work for
   1.10x the time), and graph capture attacks launch overhead directly.
4. **A larger GPU — do not bother.** The knee arrives at ~2% of this card's throughput
   ceiling, so more throughput is not the constraint. It is the item most likely to be
   reached for first and least likely to help.


## A bug only sustained load could find

The first attempt at the final ramp returned **zero completions at every level**, and
zero recorded latencies — because there were none to record. The gateway log had it:

```
GoAway with error code ENHANCE_YOUR_CALM and debug data equal to "too_many_pings"
```

The gateway's gRPC keepalive pinged every 30 s with `PermitWithoutStream: true`, below
Triton's tolerated minimum. Triton tore down the transport and every request failed.
Every short test had passed because the connection never lived long enough to be killed.

It is worth noting what would *not* have caught this: a latency metric. A dead transport
produces no slow requests, just no requests, so p50 and p99 stay silent. Only the
completion count and the gateway's own error log showed it. Fixed to 5 minutes, pings
only while streams are active.
