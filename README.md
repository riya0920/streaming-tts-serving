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

**Phase: bring-up.** Nothing below is measured yet. This section is the scoreboard and
it stays honest: a number appears here only after `loadgen` produced it on real hardware,
and the raw run artifact is committed under `results/`.

| Metric | Baseline (FastAPI) | Current | Target |
|---|---|---|---|
| TTFA p50 | — | — | < 80 ms |
| TTFA p99 | — | — | < 150 ms |
| Real-time factor (p01) | — | — | > 1.0 |
| Max concurrent sessions at target p99 | — | — | discover |
| GPU-seconds per audio-minute | — | — | discover |

Targets are targets. If the honest answer turns out to be 210 ms at 900 sessions, that
is what goes in the table.

---

## Architecture

```
                 WebSocket                    gRPC (decoupled)
   client  ────────────────►  Go gateway  ──────────────────────►  Triton
                             ┌──────────────┐                   ┌──────────────────┐
                             │ session state│                   │ tts_frontend  (PY)│  normalize → G2P → tokens
                             │ admission ctl│                   │ vits_encoder  (TRT)│  text → hidden, m_p, logs_p
                             │ dual routing │                   │ [dur + flow]  (FP32)│  → latents z
                             │ OTel spans   │                   │ tts_stream    (C++)│  chunked decode loop
                             └──────────────┘                   │ vits_decoder  (TRT)│  latents → waveform
                                                                └──────────────────┘
                             ◄──────── audio chunks stream back ────────
```

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
chunk boundaries — each slice is missing its neighbors' receptive field. Fix: decode with
overlap padding on both sides, trim to the valid center, and equal-power crossfade the
seams. Chunk size is a measured tradeoff, not a guess (see `docs/ARCHITECTURE.md`).

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

1. Provision a GPU box — see [docs/GPU_BOX.md](docs/GPU_BOX.md).
2. `scripts/provision.sh` on the box installs Docker, the NVIDIA container toolkit, Go,
   and pulls the Triton image.
3. `docker/` brings up Triton + Prometheus + Grafana + Jaeger.
4. Run the baseline first (`baseline/`) — without it, "62% faster" means nothing.

See [BUILD_PLAN.md](BUILD_PLAN.md) for the sequenced milestones.
