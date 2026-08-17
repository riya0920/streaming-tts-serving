# M1 — the naive baseline

`facebook/mms-tts-eng` behind FastAPI. One request, one full `model.forward()`, complete
WAV returned. No batching across users, no streaming, no quantization. **NVIDIA A40.**
Raw data: `results/baseline_threadpool.json`, `results/baseline_async_blocking.json`.

Two modes were measured, because a strawman baseline is worse than no baseline.

| | what it is |
|---|---|
| `async_blocking` | the classic mistake — a blocking torch call inside `async def`, which parks the event loop and serializes every request |
| `threadpool` | the competent-naive version — sync `def`, so FastAPI dispatches to worker threads and requests genuinely overlap |

## Results

**threadpool**

| concurrency | rps | agg RTF | p50 ms | p99 ms |
|---:|---:|---:|---:|---:|
| 1 | 10.3 | 74.7 | 78.6 | 140.1 |
| 2 | **14.3** | 96.5 | 114.5 | 243.0 |
| 4 | 13.7 | 92.1 | 230.3 | 672.5 |
| 8 | 4.1 | 25.2 | 1583.1 | 2866.2 |
| 16 | 3.7 | 23.4 | 3460.2 | 6607.3 |
| 32 | 4.1 | 27.7 | 6083.8 | 11332.9 |
| 64 | 3.3 | 19.9 | 9098.7 | 14159.8 |
| 128 | — | — | **0 requests completed in 20 s** | |

**async_blocking**

| concurrency | rps | agg RTF | p50 ms | p99 ms |
|---:|---:|---:|---:|---:|
| 1 | 11.3 | 79.5 | 76.8 | 129.2 |
| 2 | 12.5 | 86.1 | 129.4 | 237.0 |
| 8 | 12.7 | 80.2 | 525.4 | 1059.4 |
| 32 | 12.4 | 78.5 | 1831.3 | 3757.3 |
| 64 | **13.7** | 84.8 | 3243.4 | 17436.0 |
| 128 | 13.3 | 88.3 | 4751.3 | 17987.7 |

## The mistake beats the competent version

This is the opposite of what the baseline was designed to show, and it is the most useful
thing M1 produced.

`threadpool` peaks at 14.3 rps around concurrency 2, then **throughput collapses by 3.5x**
— to 4.1 rps at concurrency 8, and to *nothing at all* at 128, where not one request
finished inside a 20-second window. `async_blocking` holds ~13 rps flat all the way to
128.

The reason is that `threadpool` lets up to 40 worker threads issue independent forward
passes at the same time. They cannot actually run concurrently — the GPU serializes them
anyway — but they contend for memory, thrash the allocator, and fight over the GIL for
the tokenization and WAV encoding around each call. `async_blocking` accidentally does
the right thing: it serializes at the front door, so the GPU sees one clean stream of
work.

**Uncontrolled concurrency in front of a GPU is worse than a queue.** That is precisely
the argument for the two mechanisms this project is built on:

- **Admission control** — reject past a threshold rather than admit and let everyone
  degrade. `threadpool` at 128 is the concrete picture of what "admit everything" costs:
  zero throughput, not merely slow throughput.
- **Dynamic batching** — a *controlled* queue that also fuses the waiting work, which is
  what turns the serialization from a cost into a benefit.

## The tail is the whole point

`async_blocking` at concurrency 64 sustains 13.7 rps — throughput looks healthy — while
p99 is **17.4 seconds**. Mean latency would report ~3 s and hide it completely. This is
why every latency metric in this project is a histogram.

## Comparison basis

The honest number to beat is the **better** of the two modes at each level, not the worse.

| | value |
|---|---|
| Peak sustained throughput | **13.7 rps** |
| Aggregate real-time factor at peak | **~85–96** audio-seconds generated per wall-clock second |
| Best-case single-request latency | p50 **77 ms**, p99 **129 ms** (concurrency 1) |
| Concurrency at which p99 exceeds 1 s | **8** |
| Full-collapse point | 128 (threadpool), never (async_blocking, but at 18 s p99) |

The aggregate RTF of ~85–96 independently agrees with the M2 estimate of roughly 60–100
concurrently-speaking streams for this model on this GPU — two different measurements,
same ceiling.

Note what the baseline cannot do at all: there is no time-to-first-audio, because nothing
is emitted until the entire utterance is synthesized. At concurrency 8 that means a user
waits 1.6 seconds before hearing anything. Removing that dependency is the point of the
streaming design, and it is a structural change, not a speedup.
