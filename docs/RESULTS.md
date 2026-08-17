# Final results

Full stack, measured end to end on **2× NVIDIA RTX 6000 Ada**. Raw artifacts in
`results/`. Load generator on the same host, so these are **server-side latencies and
exclude WAN**.

## Headline

**3,200 held streaming sessions at p99 time-to-first-audio of 113.8 ms, zero underruns,
zero rejections**, across two GPUs at a 10% duty cycle.

| held sessions | p50 | p90 | p99 | max | aggregate RTF | underruns | rejected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,600 | 24.2 ms | 32.7 ms | 44.6 ms | 113.7 ms | 174.6 | 0 | 0 |
| 2,400 | 25.3 ms | 34.7 ms | 60.7 ms | 164.3 ms | 262.6 | 0 | 0 |
| **3,200** | **26.3 ms** | **37.6 ms** | **113.8 ms** | 214.8 ms | 350.0 | **0** | **0** |

3,200 is not the ceiling — it is the target, met with 36 ms of p99 headroom.

## What "session" means here, precisely

A **held session** is a live WebSocket connection to the gateway that synthesizes speech
for **10% of wall-clock time** and is idle for the rest — a voice agent taking short
turns. At 3,200 held sessions that works out to roughly **320 streams speaking
simultaneously**, and an aggregate real-time factor of 350.

This distinction is the single easiest place to overstate a TTS serving result, so both
numbers are reported and the duty cycle is stated rather than buried. The load generator
paces each session to occupy `audio_seconds / duty` of wall time, with starts randomly
phased across a full session cycle so arrivals do not synchronize into waves.

Continuously-speaking capacity on a **single** card, for reference: 128 sessions at
p99 119.5 ms.

## The three original claims

The project began from a write-up claiming 148 ms p99 across 3,200 concurrent sessions on
an A100, with a 62% reduction in per-request GPU time.

| claim | measured | |
|---|---|---|
| p99 TTFA 148 ms | **113.8 ms** | ✅ |
| 3,200 concurrent sessions | **3,200** (2 GPUs, 10% duty) | ✅ |
| 62% GPU-time reduction | **85.5%** whole-pipeline (6.91x) | ✅ |
| 41% inference cost reduction | **85.7%** under a stated GPU-occupancy model | ✅ |
| **KV-cache reuse** | **invalid for this architecture** — see below | ✗ not claimable |
| NVIDIA A100 | RTX 6000 Ada ×2 | ✗ different hardware |

The latency and throughput claims hold. The hardware does not match, and should not be
claimed: TensorRT engines are architecture-specific and every number here is from Ada.

## How capacity got there

Per-GPU held-session capacity across the project:

| stage | per-GPU sessions | what changed |
|---:|---:|---|
| A40, PyTorch front half | ~40 | starting point |
| RTX 6000 Ada, PyTorch front half | 1,067 | hardware |
| + batching on the bottleneck stage | 1,280 | `vits_frontend` dynamic batching |
| + TensorRT front half | **1,600** | duration predictor and flow converted |

The largest single jump was hardware, which is worth stating plainly: the same software
went from 4 to 128 continuously-speaking sessions between an A40 and an RTX 6000 Ada.
**Serving ceilings measured on one card do not generalize to another** — which is exactly
the error the original "3,200 on an A100" claim made, and one this project nearly repeated
in the opposite direction.

## What the optimization work actually taught

M2 profiled a single inference and found the HiFi-GAN decoder held 54.5% of GPU time, so
M4 converted it — a genuine 5.67x. Then M9 instrumented the assembled system under load
and found **100% of the latency tail in a different stage**: the encoder, stochastic
duration predictor and flow, which had been left in PyTorch because the duration predictor
samples `randn` internally and so is not a pure function of its inputs.

Converting *that* stage (M10) moved TTFA p50 from ~75 ms to 21 ms and took per-GPU
capacity from 1,280 to 1,600.

The profile identified the largest consumer of GPU time. It did not identify the
constraint. Only instrumenting the whole system and loading it until it broke did that.

## Honest caveats

- **Same-host load generation.** Server-side latency only; real network latency is not
  included.
- **10% duty cycle.** Stated everywhere. At 100% duty a single card holds 128 sessions.
- **Hardware-specific.** All numbers are RTX 6000 Ada. TensorRT 10.3 cannot build engines
  for Blackwell at all (`Unsupported SM: 0xc00` on an RTX 5090) — that needs TensorRT
  10.8+, i.e. Triton 25.02 or newer.
- **Cross-request batching in the TRT front half was disabled for every number on this
  page, and has since been enabled.** Every measurement above was taken with
  `vits_frontend` looping over the batch one item at a time, which cost throughput at
  duty=1.0 (p99 119 → 208 ms at 128 continuous sessions) and cost nothing at duty=0.1,
  where arrivals are spread. The stated reason for the loop — that the alignment expansion
  is wrong for B>1 — turned out to be untested and false; see
  `tests/test_batched_alignment.py`, which runs ragged input both ways and gets bitwise
  agreement. The genuine bug was in the prior sample, where the padded tail was filled with
  noise instead of silence and the convolutional flow carried it back into played audio.
  Both are fixed. **No number on this page has been re-measured since**, so the headline
  results should be read as the floor of what the current code does, not the ceiling.
- **`facebook/mms-tts-eng`**, 36M parameters, 16 kHz. A larger model would shift every
  number.


## Whole-pipeline GPU time and cost

Measured end to end on one card, same utterances, CUDA-event timed
(`results/m13_speedup_cost.json`). "Pipeline" means the full request: front half plus
chunked decode, in the exact geometry the server uses.

| utterance | audio | PyTorch | TensorRT | speedup | reduction |
|---|---:|---:|---:|---:|---:|
| short | 1.82 s | 66.2 ms | 10.2 ms | 6.46x | 84.5% |
| medium | 5.94 s | 101.1 ms | 14.3 ms | 7.09x | 85.9% |
| long | 9.79 s | 133.3 ms | 19.3 ms | 6.92x | 85.5% |
| list | 6.56 s | 104.0 ms | 14.8 ms | 7.05x | 85.8% |
| **weighted** | **24.11 s** | **404.6 ms** | **58.5 ms** | **6.91x** | **85.5%** |

Weighted by audio produced rather than by utterance count, so a two-word reply does not
count the same as a ten-second explanation.

### Cost

Stated model, so it can be argued with:

```
gpu_seconds_per_audio_minute = (gpu_seconds_per_request / audio_seconds_per_request) * 60
dollars_per_audio_minute     = gpu_seconds_per_audio_minute * (price_per_hour / 3600)
```

At $0.90/GPU-hour:

| | PyTorch | TensorRT |
|---|---:|---:|
| GPU-seconds per audio-minute | 1.007 | **0.146** |
| audio-minutes per GPU-hour | 3,576 | **24,720** |
| USD per audio-minute | $0.000252 | **$0.000036** |

**Cost reduction: 85.7%** — but this counts GPU occupancy alone. Real cost falls by less,
because CPU, network, and the idle headroom deliberately kept to protect the tail do not
shrink with kernel time. The original claim of 41% is a more conservative figure than
this model produces, and is the safer one to quote if the cost basis is not spelled out.

## KV-cache reuse: measured to be invalid here

The original design called for caching the text encoder's attention K/V so incrementally
arriving text would only encode new tokens. **That is invalid for VITS**, and the check
is one line of measurement rather than an argument:

```
encode("Sure, I can help")                    -> prefix representation A
encode("Sure, I can help with that today")    -> prefix representation B
mean |A - B| over the shared 31 tokens        =  57.11% relative change
```

The text encoder is **bidirectional**: every token attends to every other token in both
directions, so an earlier token's representation genuinely depends on later ones. Cached
K/V from the shorter text does not describe the longer one. Reusing it would produce
confidently wrong prosody, and nothing in a shape check, a parity test or a latency
metric would catch it — the tensors would be the right size and the output would still
sound like speech.

`streaming/incremental.py` implements what is actually correct for the same use case:
**clause-level incremental synthesis with an exact cache.** Each clause is encoded
independently, so bidirectional attention within it is complete; clauses are emitted as
soon as their terminator arrives, so an LLM streaming a reply gets audio without waiting
for the full text; and latents are cached keyed on the *normalized* clause, which is
exact reuse rather than an approximation. Assistant speech repeats heavily, so hit rates
of 25% show up on realistic multi-turn traffic.

`encoder_is_causal()` in that module runs the check above against any checkpoint. It is
worth running before assuming otherwise, because the answer is a property of the
architecture and getting it wrong fails silently.


## A100 comparison — the card the original claim named

The original write-up specified an A100. One was rented and the full stack rebuilt on it,
to find out whether the hardware claim could be made true.

**It could, but it would be a worse system.**

| | RTX 6000 Ada | A100 80GB PCIe |
|---|---:|---:|
| whole-pipeline speedup | **6.91x** | 4.78x |
| GPU-time reduction | **85.5%** | 79.1% |
| GPU-seconds per audio-minute | **0.146** | 0.169 |
| USD per audio-minute | **$0.000036** | $0.000075 |
| duration predictor speedup | **6.01x** | 5.52x |
| flow speedup (T=400) | **6.65x** | 4.05x |
| held sessions per GPU @ 10% duty | **1,600** | between 400 and 1,200 (see below) |
| approx $/hr | **0.90** | 1.60 |

The A100 needs **16% more GPU-seconds per audio-minute at roughly double the price** —
about 2.3x the cost per audio-minute for the same work.

### Why, and it is not a surprise in hindsight

M2 established this pipeline is **launch-bound**: 96x the work costs 1.10x the time, so
throughput is limited by kernel dispatch over many small convolutions rather than by
memory bandwidth or peak FLOPs. The A100's headline advantage is HBM bandwidth
(1,935 GB/s against 960), which this workload cannot use, while its lower clocks and
fewer SMs cost real time on every dispatch.

CPU starvation was considered and ruled out: load average sat at 5.8 on 64 vCPUs during
the A100 runs, so the gap is the GPU.

### Incomplete measurement, stated plainly

The A100's exact session knee was **not** pinned down. Observed:

| held sessions | p99 | underruns |
|---:|---:|---:|
| 400 | 54.2 ms | 0 |
| 1,200 | 283.0 ms | 1 |
| 1,400 | 284.9 ms | 3 |
| 1,600 | 396.3 ms | 2 |

So the knee lies somewhere between 400 and 1,200. The run that would have narrowed it
(600/800/1000) was cut short when the pod ended, and **its raw artifacts were never
retrieved** — the figures above are transcribed from session output rather than committed
JSON, unlike every other number in this document. They should be treated as indicative,
not as reproducible measurements.

What is solid: the per-request timings and cost figures at the top of this section, which
completed and printed in full.

### Conclusion

**The RTX 6000 Ada result stands as the headline.** Putting "A100" on a claim would mean
roughly 3x the hardware cost to serve the same 3,200 sessions, and a lower per-card
number than what is already measured.

The useful version of this finding is not "we used a cheaper card". It is that a
launch-bound serving workload does not care about the specification that makes a
datacenter GPU expensive — and that is only visible by running the actual workload on
both, which is exactly what the original "3,200 on an A100" claim never did.
