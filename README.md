# streaming-tts-serving

Streaming TTS on Triton. VITS, TensorRT FP16, chunked decoding, Go gateway in front.

The goal was a first audio chunk in under 150ms at p99 (not mean) with real concurrency.

## Numbers

Measured on 2x RTX 6000 Ada. Raw JSON in `results/`, caveats in [docs/RESULTS.md](docs/RESULTS.md).

| | baseline (FastAPI) | measured |
|---|---|---|
| held sessions @ 10% duty | n/a | 3,200 (2 GPUs) |
| TTFA p99 | no streaming | 113.8 ms |
| TTFA p50 | no streaming | 26.3 ms |
| underruns / rejections | n/a | 0 / 0 |
| aggregate real-time factor | 85-96x | 350x |
| whole-pipeline GPU time | 1.0x | 6.91x (85% less) |
| cost per audio-minute | $0.000252 | $0.000036 |
| held sessions, 1 GPU | n/a | 1,600 |
| continuously speaking, 1 GPU | n/a | 128 |

A "held session" is a WebSocket that synthesizes 10% of the time, like a voice agent
taking turns. 3,200 held is ~320 speaking at once. Both numbers are here because
conflating them is how you accidentally overstate a TTS benchmark by 10x.

## Architecture

```
client --WebSocket--> Go gateway --gRPC--> Triton
                      sessions             tts_frontend   (python, CPU)  text -> tokens
                      admission            vits_frontend  (python, GPU)  tokens -> latents
                      routing              tts_stream     (C++, DECOUPLED)
                      OTel + Prom            loads the TRT decoder engine,
                                             chunks, pushes each chunk out
                      <---- audio chunks stream back ----
```

Decoupled mode is the whole trick. A normal Triton model is one request, one response.
Decoupled lets one request emit many responses over time, so chunks go out as they're
decoded instead of after the full utterance. TTFA stops depending on how long the
sentence is (9.5x the words costs 1.4x the latency, measured).

The C++ backend loads the TensorRT engine directly rather than calling another Triton
model, so there's no cross-model hop per chunk.

## What I learned building it

Most of these were surprises. Details in `docs/`.

- Profiling one inference said the decoder was 54.5% of GPU time, so I converted that
  first. Then loading the assembled system put 100% of the latency tail in a *different*
  stage. The profile found the biggest consumer, not the constraint.
- The decoder is launch-bound, not compute-bound: 96x the work costs 1.10x the time.
  That changes which optimizations matter (fusion and batching, not FP16 bandwidth).
- The crossfade I designed for chunk boundaries was unnecessary and also harmful.
  I'd picked equal-power, which is right for uncorrelated signals. These are correlated.
  Overlap alone already matches a single-pass decode.
- KV-cache reuse doesn't apply here. VITS is non-autoregressive and its text encoder is
  bidirectional (appending text moves the existing prefix 57%). Built clause-level
  latent caching instead.
- The naive async baseline beat the "competent" threadpool one under load. Uncontrolled
  concurrency in front of a GPU is worse than a queue.
- Same code, A40 -> RTX 6000 Ada: 4 -> 128 concurrent sessions. Capacity numbers don't
  transfer between cards.
- I also tried an A100, since that's the obvious "serious" card. It was 16% worse per
  audio-minute at double the price. The bandwidth advantage does nothing for a
  launch-bound workload.

## Running it

Needs an NVIDIA GPU and Linux. Start the pod from
`nvcr.io/nvidia/tritonserver:24.08-py3` (see [docs/GPU_BOX.md](docs/GPU_BOX.md)).

```bash
bash scripts/provision.sh            # deps, venv, Go, Triton SDK, observability
python export/fetch_model.py
python export/export_onnx.py
python export/export_frontend_onnx.py
python export/build_trt.py           # decoder + encoder engines
python export/build_trt_frontend.py  # duration predictor + flow engines
bash scripts/build_backend.sh        # compile the C++ backend
bash scripts/gen_triton_proto.sh
bash scripts/services.sh start
bash scripts/services.sh start triton    # one Triton per GPU
cd gateway && go build -o /workspace/bin/gateway ./cmd/gateway
```

Then:

```bash
python streaming/client.py --bench 30
/workspace/bin/loadgen --levels 1600,3200 --duty 0.1 --duration 240s
```

RunPod pods are containers and can't run Docker, so `docker/` is for VM deploys only.

## Layout

```
baseline/       naive FastAPI server, the number to beat
export/         ONNX export, TensorRT engine builds, quality checks
model_repo/     Triton models (2 python, 1 C++ decoupled)
backends/       C++ streaming backend
streaming/      chunked decode, TRT frontend, text normalization, client
gateway/        Go control plane
loadgen/        Go load generator
observability/  Prometheus, Grafana, OTel config
docs/           what each milestone measured (NOTES.md is the condensed version)
results/        raw measurement JSON
```

## Known gaps

- Cross-request batching is off in the TensorRT front half. The batched alignment is
  wrong for B>1 since each item predicts its own frame count. Costs throughput at
  duty=1.0, costs nothing at duty=0.1. Top of the list to fix.
- The A100 comparison is incomplete. Its session knee is bracketed between 400 and
  1,200, and I lost the raw artifacts when the pod ended. Marked as indicative in the docs.
- Decoder still decodes 13 frames of left context per chunk and throws it away (34%
  waste at steady state). Streaming convolutions with cached state would remove it.
- Everything is `facebook/mms-tts-eng`, 36M params, 16kHz. A bigger model moves all of this.
