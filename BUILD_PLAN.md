# Build plan

Sequenced so that each milestone produces a **measurement**, not just code. The order is
chosen so nothing is optimized before it is proven to be the bottleneck.

---

### M0 — Provision and verify
- Rent the box, run `scripts/provision.sh`, confirm `--gpus all` reaches the GPU.
- Bring up the compose stack; Triton serves a trivial identity model.
- **Exit criterion:** Triton `/v2/health/ready` returns 200 from inside the tunnel.

### M1 — The deliberate baseline
- `baseline/server.py`: stock PyTorch VITS behind FastAPI. One request, one forward pass,
  full WAV returned. No batching, no streaming. This is intentionally the dumb version.
- Load it until it falls over. Record TTFA, throughput, GPU utilization vs. concurrency.
- **Exit criterion:** a committed `results/baseline.json` with the curve. Every later
  "N% faster" claim is measured against this file or it does not get made.

### M2 — Profile before optimizing
- `torch.profiler` + Nsight Systems on a single utterance.
- Confirm (do not assume) that the HiFi-GAN decoder dominates GPU time, and quantify how
  much of a re-invoked pipeline is redundant text-encoder work.
- **Exit criterion:** `docs/PROFILE.md` with the per-module time split.

### M3 — Chunked streaming, in Python first
- Restructure inference: encoder + duration + flow once, decoder in overlapping slices.
- Implement overlap-add with equal-power crossfade; verify seams are inaudible
  (spectral discontinuity metric at boundaries + listening).
- Sweep chunk size and overlap; plot TTFA vs. per-chunk overhead vs. artifact metric.
- **Exit criterion:** chosen chunk/overlap values with the sweep data behind them.
  Correctness settled here, in the language that is fast to iterate in.

### M4 — TensorRT engines
- Export text encoder and HiFi-GAN decoder to ONNX with dynamic axes.
- Build FP16 TRT engines with sane optimization profiles for the chunked decoder shape.
- Keep the stochastic duration predictor and flow outside the engines, in FP32.
- Validate FP16 vs FP32: mel spectral distance + A/B listening on a fixed test set.
- **Exit criterion:** `results/fp16_quality.json` shows no meaningful degradation, and
  per-chunk decoder latency is recorded against the PyTorch number.

### M5 — Triton, properly
- `tts_frontend` Python backend (normalization → G2P → tokens).
- `tts_stream` C++ backend in **decoupled mode** — the streaming loop, crossfade,
  PCM conversion, response queue.
- Tune dynamic batching window and instance groups.
- **Exit criterion:** a single gRPC request yields a stream of audio chunks, and the
  C++ per-chunk overhead distribution is tighter than the Python prototype's.

### M6 — KV-cached incremental encoder
- Cache text-encoder attention K/V so incrementally-arriving text (an LLM streaming
  sentence fragments into TTS) only encodes new tokens instead of re-encoding everything.
- **Exit criterion:** measured compute saved vs. utterance length, showing the quadratic
  term removed. If it turns out not to matter for realistic fragment sizes, say so and
  keep or drop it on the evidence.

### M7 — Go gateway
- WebSocket termination, session state, gRPC to Triton's decoupled endpoint.
- Dual routing: latency-tuned config for live streams, throughput-tuned for offline batch,
  so offline work cannot contaminate the live path.
- Admission control on in-flight sessions + Triton queue depth: reject fast rather than
  admit and let everyone's p99 collapse.
- **Exit criterion:** gateway holds N idle sessions with flat memory; admission control
  demonstrably clamps the p99 knee instead of riding past it.

### M8 — Observability
- OTel trace context propagated gateway → Triton, so one trace decomposes into
  gateway / network / queue wait / frontend / first chunk / first byte.
- Prometheus **histograms** (never averages — averages hide exactly the tail we care
  about): TTFA, real-time factor, queue depth, GPU util via DCGM exporter.
- **Exit criterion:** a Grafana dashboard where the TTFA p99 and its dominant contributing
  span are visible side by side.

### M9 — The number
- `loadgen`: real WebSocket sessions, realistic sentence-length distribution, ramping
  concurrency, TTFA histogram.
- Find the knee. Report p99 and the concurrency it holds at.
- **Exit criterion:** `results/` contains the run, and README's scoreboard is filled in
  with whatever actually happened.

---

## Cost discipline

Stop the box between milestones. M0–M3 need a GPU only for M1/M2 runs. M9 is the only
milestone that wants the big card.
