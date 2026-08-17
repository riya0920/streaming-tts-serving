# streaming-tts-serving

Low-latency streaming text-to-speech serving. VITS on Triton Inference Server with
TensorRT FP16 engines, incremental chunked decoding, and a Go control plane in front.

The question driving this: **what does it take to serve a TTS model such that the first
audio chunk arrives in under 150 ms at the p99, not at the mean, under real concurrency?**

Everything here flows from one tension — low latency wants tiny batches served
immediately, GPU efficiency wants big batches — and the whole design is the resolution
of that fight.

---

## Status

**Built and measured end to end.** Every number was produced on real hardware, with the
raw artifact committed under `results/`. Headline figures are on an **RTX 6000 Ada**;
the earlier design and profiling work was done on an **A40** and is labelled as such.

| Metric | Baseline (FastAPI) | Measured (RTX 6000 Ada) | Target |
|---|---|---|---|
| TTFA p50 | n/a — no streaming | **26 ms** | < 80 ms ✅ |
| TTFA p99 | n/a — no streaming | **148 ms** at the knee, **29 ms** unloaded | < 150 ms ✅ |
| **Held sessions @ 10% duty, one GPU** | — | **1,600** | — |
| Concurrent sessions (continuously speaking) | — | **128** | — |
| Underruns | n/a | **0** at every level | 0 ✅ |
| Aggregate real-time factor | 85–96x | **205x** (saturation) | — |
| Decoder speedup (TensorRT FP16) | 1.0x | **5.67x** | — |
| Duration predictor speedup | 1.0x | **6.09x** | — |
| Flow speedup | 1.0x | **6.87x** | — |

**3,200 sessions needs 2 GPUs: 2 × 1,600 = 3,200.** The per-GPU figure is measured;
the multi-GPU routing is built (least-in-flight across N Triton endpoints, one per card)
and awaits a two-GPU run.

Per-GPU capacity across this project: **4 → 128 → 1,067 → 1,600** — hardware, then
batching the bottleneck stage, then converting the front half to TensorRT.

**The p99 latency target is met. The concurrency target is missed by more than two orders
of magnitude,** and [docs/LOADTEST.md](docs/LOADTEST.md) says exactly why: 100% of the
latency tail is `vits_frontend` — the stochastic duration predictor and flow that could
not be exported to TensorRT, because the duration predictor samples `randn` internally
and so is not a pure function of its inputs. The decoder that got the 4x speedup never
appears in the tail at all.

That is the project's most useful result. The optimization went precisely where the
profile said the GPU time was, and the actual constraint was somewhere else — visible
only after instrumenting the assembled system and loading it until it broke.

The second most useful: at the knee the GPU sits at **~2% of its throughput ceiling**.
This system is latency-bound, not throughput-bound — so a bigger card, the first thing
most people reach for, would not help.

### Measurement log

| Milestone | Finding | Where |
|---|---|---|
| M1 | The *naive* async baseline beats the *competent* threadpool one under load — uncontrolled GPU concurrency is worse than a queue | [docs/BASELINE.md](docs/BASELINE.md) |
| M2 | Decoder is launch-bound, not compute-bound: 96x the work costs 1.10x the time | [docs/PROFILE.md](docs/PROFILE.md) |
| M2 | Batching is free only to B≈4–8, which sets the batching window | [docs/PROFILE.md](docs/PROFILE.md) |
| M3 | Receptive field (13 frames) exceeds the 200 ms chunk; overlap alone removes seams | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| M3 | The crossfade was unnecessary, and the equal-power curve was actively harmful | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| M4 | TensorRT conversion is lossless (0.116 dB LSD); all FP16 error is precision (1.52 dB) | [docs/TENSORRT.md](docs/TENSORRT.md) |
| M9 | TTFA is length-independent — 9.5x the words costs 1.4x the latency | [docs/LOADTEST.md](docs/LOADTEST.md) |
| M9 | 100% of the latency tail is the one stage that stayed in PyTorch | [docs/LOADTEST.md](docs/LOADTEST.md) |
| M10 | The stochastic duration predictor *is* exportable — hoist its `randn` into a graph input | [export/export_frontend_onnx.py](export/export_frontend_onnx.py) |
| M11 | Same software, A40 → RTX 6000 Ada: 4 → 128 concurrent sessions. Capacity ceilings do not generalize across cards | [docs/LOADTEST.md](docs/LOADTEST.md) |

---

## Architecture

```
                 WebSocket                     gRPC                 as built
   client ──────────────────► Go gateway ──────────────► Triton
                             ┌──────────────┐        ┌───────────────────────────────┐
                             │ session state│        │ tts_frontend  (Python, CPU)   │ normalize → tokens
                             │ admission ctl│        │ vits_frontend (Python, GPU)   │ encoder+duration+flow → latents
                             │ dual routing │        │ tts_stream    (C++, DECOUPLED)│ chunked decode → PCM chunks
                             │ OTel + Prom  │        │   └─ loads the TensorRT FP16  │
                             └──────────────┘        │      decoder engine directly  │
                                                     └───────────────────────────────┘
                             ◄───────── audio chunks stream back ─────────
```

The decoder engine is loaded **inside** the C++ backend rather than served as its own
Triton model, so the chunk loop, engine invocation, trimming and PCM conversion all live
in one place with no cross-model hop per chunk.

**Two backends, on purpose.** Python owns the text frontend — normalization
(`"Dr."` → `"doctor"`, `"$45"` → `"forty-five dollars"`), G2P, tokenization. It is
messy rule-heavy string logic that changes constantly and runs exactly once per
utterance, so it is not on the hot path. C++ owns the streaming loop — per-chunk engine
invocation, overlap-add crossfade, float32 → int16 PCM, pushing into the decoupled
response queue. That code runs many times per second per session across every session.
The rule: **Python where the code changes, C++ where the latency lives.**

**Chunking is the whole latency trick.** Synthesizing the full utterance and then
streaming the file does not improve time-to-first-audio at all. Instead: run the text
encoder, duration predictor, and flow once up front (cheap), then slice the *decoder* —
the HiFi-GAN generator, where nearly all GPU time lives — and ship the first ~200 ms of
audio immediately while later slices decode behind it. TTFA stops depending on utterance
length.

The decoder is convolutional, so naively chopping latents produces audible clicks at
chunk boundaries — each slice is missing its neighbours' receptive field. Fix: decode with
overlap padding on both sides and trim back to the valid centre.

There is **no crossfade**. The original design treated one as essential; M3 measured the
decoder's receptive field at 13 frames, showed that overlap at or above it already
matches a single-pass decode (seam step-ratio 10.25 vs 10.26), and showed the intended
equal-power curve actively *hurt* — it is the right curve for uncorrelated signals, and
these two decodes are near-identical. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Why the flow runs whole and not chunked:** the residual coupling layers have their own
receptive field and are numerically touchy in FP16. Running them once over the full
utterance in FP32 sidesteps both problems, and they are cheap next to the decoder.

**Go for the control plane** because 3,200 streaming connections means 3,200 things
concurrently blocked on I/O. Goroutines make that nearly free, there is no GIL competing
for the CPU-side work, and it ships as one static binary with well-behaved tail latency.

---

## Repository layout

```
baseline/       deliberate naive FastAPI server — the number to beat
export/         PyTorch → ONNX → TensorRT engines, plus FP16 quality validation
model_repo/     Triton model repository (configs + artifacts)
backends/       C++ streaming backend source
gateway/        Go control plane: WebSocket ↔ gRPC, sessions, admission control
loadgen/        Go load generator — realistic sentence distributions, TTFA histograms
observability/  Prometheus, Grafana dashboards, OTel collector config
docker/         Triton image + compose stack
scripts/        GPU box provisioning and sync
docs/           architecture decisions, measurement methodology
```

## Getting started

This does not run on a laptop. It needs an NVIDIA GPU, and the toolchain
(TensorRT, Triton custom backends) is Linux-only.

```bash
# on the box (started from nvcr.io/nvidia/tritonserver:24.08-py3 — see docs/GPU_BOX.md)
bash scripts/provision.sh          # deps, venv, Go, Triton backend SDK, observability
python export/fetch_model.py       # VITS checkpoint
python export/export_onnx.py       # encoder + decoder → ONNX
python export/build_trt.py         # FP16 + FP32 engines, benchmarked
bash scripts/build_backend.sh      # compile the C++ decoupled backend
bash scripts/services.sh start     # prometheus, grafana, jaeger, otel
bash scripts/services.sh start triton
cd gateway && go build -o /workspace/bin/gateway ./cmd/gateway && /workspace/bin/gateway
```

Then drive it:

```bash
python streaming/client.py --bench 30              # TTFA percentiles
/workspace/bin/loadgen --levels 1,2,4,8,16         # the ramp
```

RunPod pods are containers and cannot run Docker, so `docker/` is for VM deploys only —
see [docker/README.md](docker/README.md). Run the baseline (`scripts/run_baseline.sh`)
before claiming any speedup.

See [BUILD_PLAN.md](BUILD_PLAN.md) for the milestones and [docs/](docs/) for what each
one measured.
