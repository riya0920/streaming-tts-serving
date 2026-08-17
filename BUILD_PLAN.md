# Build plan

Written before any code, kept here unedited in structure so the plan can be compared
against what actually happened. Notes on where it was wrong are at the bottom.

The ordering rule: every milestone has to end in a measurement, and nothing gets
optimized before it is shown to be the bottleneck.

### M0 - Provision and verify

Rent the box, run `scripts/provision.sh`, confirm `--gpus all` reaches the GPU. Bring up
the stack, serve a trivial identity model.

Done when Triton `/v2/health/ready` returns 200 through the tunnel.

### M1 - A deliberately dumb baseline

`baseline/server.py`: stock PyTorch VITS behind FastAPI. One request, one forward pass,
a whole WAV back. No batching, no streaming. Load it until it falls over.

Done when `results/baseline.json` holds the curve. Every later "N% faster" is measured
against that file or it doesn't get claimed.

### M2 - Profile before optimizing

`torch.profiler` plus Nsight on a single utterance. Confirm rather than assume that the
HiFi-GAN decoder dominates, and quantify how much of a re-invoked pipeline is redundant
text-encoder work.

Done when `docs/PROFILE.md` has the per-module split.

### M3 - Chunked streaming, in Python first

Restructure inference so encoder, duration and flow run once and the decoder runs in
overlapping slices. Overlap-add with a crossfade, then check the seams are inaudible
with a discontinuity metric and by listening. Sweep chunk size and overlap.

Done when the chunk and overlap values are chosen with sweep data behind them. Get
correctness settled here, in the language that's fast to iterate in.

### M4 - TensorRT engines

Export the text encoder and decoder to ONNX with dynamic axes. Build FP16 engines with
optimization profiles that cover the chunked decoder shapes. Leave the stochastic
duration predictor and the flow in PyTorch for now, since the duration predictor samples
internally and isn't a pure function of its inputs.

Done when FP16 vs FP32 shows no meaningful degradation and per-chunk decoder latency is
recorded against the PyTorch number.

### M5 - Triton, properly

`tts_frontend` Python backend for normalization and tokenization. `tts_stream` C++
backend in decoupled mode for the streaming loop, PCM conversion and response queue.
Tune the batching window and instance groups.

Done when one gRPC request yields a stream of chunks and the C++ per-chunk overhead is
tighter than the Python prototype's.

### M6 - Incremental encoding

Cache text-encoder attention K/V so text arriving in fragments, an LLM streaming into
TTS, only encodes the new tokens.

Done when the compute saved is measured against utterance length. If it turns out not to
matter at realistic fragment sizes, say so and drop it on the evidence.

### M7 - Go gateway

WebSocket termination, session state, gRPC to the decoupled endpoint. Separate routing
for live streams and offline batch so offline work can't contaminate the live path.
Admission control on in-flight sessions and queue depth: reject fast rather than admit
and let everyone's p99 collapse.

Done when the gateway holds N idle sessions on flat memory and admission control clamps
the knee instead of riding past it.

### M8 - Observability

OTel context propagated gateway to Triton so one trace splits into gateway, network,
queue wait, frontend, first chunk, first byte. Prometheus histograms, never averages,
since an average hides exactly the tail this project is about.

Done when p99 TTFA and its dominant span are visible side by side.

### M9 - The number

`loadgen` driving real WebSocket sessions with a realistic sentence-length distribution
and ramping concurrency. Find the knee, report p99 and the concurrency it holds at.

Done when `results/` has the run and the README scoreboard is filled in with whatever
actually happened.

## Cost discipline

Stop the box between milestones. M0 through M3 only need a GPU for the M1 and M2 runs.

## How it actually went

Four things in the plan above turned out to be wrong, and they're the useful part.

M3 planned an equal-power crossfade. Equal-power is the right curve for uncorrelated
signals, and two decodes of the same latents are nearly identical, so it added about 3 dB
of bump at every seam. Then the sweep showed the crossfade was unnecessary at any curve:
13 frames of overlap alone already matches a single-pass decode.

M6 planned a KV cache. VITS is non-autoregressive, so there's no incremental generation
to cache for, and its text encoder is bidirectional, so a cached prefix doesn't even
describe itself once more text arrives. Measured drift on a shared 31-token prefix was
57%. Replaced with clause-level caching, which is what was actually wanted.

M2 and M4 assumed the profile would name the bottleneck. It named the biggest consumer,
which is not the same thing. The decoder was 54.5% of GPU time and converting it was a
real 5.67x, but when the assembled system was loaded in M9 the entire latency tail sat in
the PyTorch front half. That became M10, which isn't in this plan.

M9 was supposed to be the last milestone. Hitting the actual target took four more:
front-half TensorRT, multi-GPU routing, the 3,200-session run, and a whole-pipeline
speedup and cost measurement.
