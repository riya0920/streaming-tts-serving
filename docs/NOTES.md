# Engineering notes

Everything non-obvious about this system, in one place: why each decision was made, what
broke, and what the measurement said. Written so the reasoning survives independently of
the code comments.

---

## 1. Why the architecture is shaped this way

**Decoupled mode is the load-bearing choice.** A normal Triton model is one request, one
response. Decoupled mode lets one request emit a stream of responses over time. Without
it, streaming has to be faked on top of a request/response server, and time-to-first-audio
stays tied to utterance length no matter how fast the model is.

**Python for the text frontend, C++ for the streaming loop.** The split is not about
language preference. The text frontend runs once per utterance, off the hot path, and its
rules change constantly (new abbreviations, new number formats). The streaming loop runs
many times per second per session, across every session. A Python loop costing 0.4ms
average but occasionally 9ms (GC pause, GIL contention) is fine for p50 and fatal for p99.
The rule: **Python where the code changes, C++ where the latency lives.**

**The C++ backend loads the TensorRT engine directly** instead of calling a separate
Triton model. That avoids a cross-model hop per chunk, and there are many chunks.

**Two routes over the same models.** `/v1/stream` is latency-tuned; `/v1/synthesize` is
throughput work that yields whenever the live path is under pressure. Separate so an
offline job can never sit in front of a listener's next chunk.

**Least-in-flight routing, not round-robin.** TTS requests are not equal cost: a 57-word
utterance occupies a backend far longer than a 6-word one. Round-robin keeps feeding long
work to whichever endpoint drew it; least-in-flight self-corrects.

**A request is pinned to one endpoint for all three hops.** Nothing requires it (the
endpoints are identical replicas), but splitting would ship latents between machines for
no benefit and would let one slow endpoint touch every request instead of a share.

---

## 2. Chunked decoding, and the crossfade that was wrong

The naive way to "stream" TTS is to synthesize everything and then send the WAV in pieces.
That does nothing for TTFA. Instead: run encoder + duration + flow once (cheap), then
slice the decoder and ship each slice as it finishes.

**The decoder is convolutional**, so slicing latents and decoding each slice in isolation
makes the seams click: each slice's edges were computed without their neighbours. Fix:
decode with `overlap_frames` of context on both sides and trim back to the valid centre.

**Measured receptive field: 13 frames (208ms).** Perturb one latent frame, decode, and see
how far the output changes by more than -60dB. The M3 sweep shows the cliff exactly there:
seam SNR is 5.1dB at overlap 8, 31.4dB at 11, and flat above.

**The crossfade was a mistake, twice.** The design called for an equal-power (cos/sin)
blend across the seam, on the reasoning that a linear fade dips in loudness at the
midpoint for correlated signals. That reasoning is inverted. Equal-power is correct for
*uncorrelated* signals, where powers add. The two signals here are the same audio decoded
with slightly different context, so they are near-perfectly correlated, amplitudes add,
and `cos(t)x + sin(t)x` peaks at `sqrt(2)*x`. It was adding a +3dB bump to every seam:
measured seam SNR fell from 31.6dB (no fade) to 11.6dB with a 10ms equal-power fade.

Correcting to linear made the fade mathematically neutral. Which raised the real question:
is a fade needed at all? A **discontinuity metric** (first difference at the boundary
against the local median, which SNR-vs-reference cannot see) answered it:

| | step ratio |
|---|---|
| chunked, overlap 13, no crossfade | 10.25 |
| single-pass decode, same positions | 10.26 |

Identical. **Crossfade defaults to 0.** The machinery stays because FP16 and TensorRT
widen the gap between the two decodes, and this is the knob if a seam ever reappears.

**Progressive chunk sizing.** The receptive field (13 frames) is *larger* than a 200ms
chunk (12 frames), so fixed 200ms chunking decodes 38 frames to keep 12: 68% waste, on
every chunk, for a latency benefit only the first chunk delivers. So chunk size grows:
small first chunk to set TTFA, doubling to a cap. Waste drops from 52% to 34%.

---

## 3. The profiling result that redirected the project

M2 profiled a single inference with CUDA events:

| stage | ms | % |
|---|---:|---:|
| decoder (HiFi-GAN) | 38.02 | 54.5% |
| duration predictor | 15.26 | 21.9% |
| flow | 9.91 | 14.2% |
| text encoder | 5.86 | 8.4% |

So the decoder was converted to TensorRT first. Correct call on that evidence.

Then M9 instrumented the *assembled system under load*, and the gateway's own histograms
put the tail somewhere else entirely:

| stage | p50 | p99 |
|---|---:|---:|
| tts_frontend | 5ms | 20ms |
| **vits_frontend** | **150ms** | **1000ms** |
| TTFA end to end | 200ms | 1000ms |

**100% of the latency tail was the stage still in PyTorch.** The decoder, which got the
5.67x speedup, never appeared in the tail.

**The profile found the largest consumer of GPU time. It did not find the constraint.**
Only instrumenting the whole system and loading it until it broke could show that.

Converting that stage took TTFA p50 from ~75ms to 21ms.

---

## 4. Launch-bound, and why that changes everything

Decoder cost against sequence length:

| frames | audio | decode |
|---:|---:|---:|
| 4 | 64ms | 4.585ms |
| 96 | 1536ms | 5.054ms |
| 384 | 6144ms | 18.800ms |

**96x the work costs 1.10x the time.** There is a fixed ~4.5ms floor per decoder call: the
GPU idles between kernel launches while the CPU dispatches the next of ~40 small
convolutions.

Consequences:
- Fusion, CUDA graphs and batching matter. FP16-as-bandwidth-saving matters less than the
  usual framing suggests, because this model only becomes bandwidth-bound above ~192 frames.
- Batching is nearly free up to B≈4-8 (absorbed into idle time), then roughly linear. That
  sets the dynamic batching window at ~8, not 32.
- **A bigger GPU does not help.** At the latency knee the GPU sits at ~2% of its
  throughput ceiling. The A100 test confirmed it: 16% worse per audio-minute at double the
  price, because its advantage is HBM bandwidth and this workload cannot use it.

---

## 5. Silent-failure traps (the ones worth remembering)

Every one of these passed shape checks, parity tests and latency metrics.

**Zero mean and zero variance are not the same zero.** Past an item's own frame count in a
padded batch, `m_p` and `logs_p` are both zero. Zero mean is silence; zero *log* variance
is `exp(0) = 1`, so `m_p + randn * exp(logs_p) * noise_scale` fills the padded tail with
full-scale noise. Trimming happens after the flow, so nothing downstream ever sees it — but
the flow is dilated convolutions, and in a batch built at the longest item that noise sits
inside the receptive field of the last real frames of every *shorter* item. Quiet
corruption at the end of short utterances, only when batched, only alongside a longer one.
Never fired in practice because the code looped over the batch instead, on a stated reason
("the alignment is wrong for B>1") that I had never tested and that turned out to be false:
batched and looped alignment agree bitwise. A wrong explanation kept a real bug alive by
making the workaround look justified.

**`set_input_shape` returns false, it does not raise.** A shape outside the TensorRT
optimization profile leaves the context on its *previous* shape, so the output buffer is
allocated from a stale shape and the caller gets a wrong-length tensor. It surfaced far
away as `tensor a (208) must match tensor b (688)`, naming two unrelated dimensions. Now
raises with the offending shape and the engine's actual bounds, which immediately revealed
the real cause was the profile *minimum*, not the maximum.

**Optimization profiles sized from the wrong intuition.** The tokenizer is
character-level, so a 60-word sentence is ~300 tokens, not ~60. A 256-token ceiling
rejected every long utterance. Widened to 1024. At the other end, "Sure." is ~12 latent
frames, below the flow engine's 16-frame minimum, so short replies failed too.

**FP16 `exp()` overflow.** Engines return half precision; `exp(log_duration)` overflows to
`inf` above ~11. The `inf` survives `ceil()` and `sum()`, and `.long()` then produces
garbage. It appeared as `torch.arange: upper bound and larger bound inconsistent with step
sign`.

**`clamp` does not filter NaN, it propagates it.** A NaN log-duration survived clamp, exp,
ceil and sum, and `.long()` turned it into an integer that slipped past *both* the minimum
and maximum bounds. Hit ~3% of requests. Fixed with `nan_to_num` plus an explicit bound on
`T`, the one value that corrupts everything downstream.

**Hoisted noise can be folded into a constant.** Making the duration predictor exportable
meant swapping `torch.randn` for a graph input. If TensorRT folds that input to a constant,
every utterance gets identical prosody, and *every* shape and parity check still passes.
The export and the engine build both verify that two different noise draws produce
different durations. This is the one place where "outputs match exactly" is the failure
signal, not the success one.

**gRPC keepalive below the server's tolerance.** The gateway pinged every 30s with
`PermitWithoutStream`; Triton answered with `GOAWAY ENHANCE_YOUR_CALM` and killed the
transport. The load test returned zero completions *and zero recorded latencies*, because
a dead transport produces no slow requests, just no requests. Latency metrics were silent;
only the completion count caught it.

**A comment after a line continuation.** In bash, a `#` comment between `\` and the next
line gets folded into the command and silently discards the rest. Jaeger never started, and
`bash -n` reported the file as valid.

**Phantom queue depth on startup.** Triton's counters are cumulative and outlive the
gateway, so the first metrics poll differenced against zero and read the whole lifetime
counter as one 500ms sample: `queue_depth 885` on an idle server against a threshold of 64.
Admission control would have rejected everything for the first half-second after any restart.

**`Path.write_text()` on Windows writes CRLF.** Eleven shell scripts got carriage returns,
and bash failed with `syntax error near unexpected token` pointing at a line that looked
perfectly fine. The fix had to use `write_bytes`; `write_text` would have re-inserted them.

---

## 6. Measurement methodology, and three bugs in my own tooling

**Histograms, never averages.** A mean latency of 180ms can sit on top of a p99 of 4
seconds. Bucket boundaries are chosen around the 150ms target, because Prometheus defaults
are too coarse below 100ms to read.

**Underruns are checked directly**, not inferred from an aggregate real-time factor. A
stream that delivers every chunk late still "succeeds" by a latency metric while sounding
broken. Each chunk is compared against when playback needed it.

Three bugs in the load generator, all the same species (the instrument measuring itself):

1. **Pacing.** Sessions idled for a *fraction* of the audio duration, so `duty=1.0` left no
   think time and each session looped at ~66x real time. Sixteen of those are sixteen batch
   jobs, not sixteen listeners. Reporting that as concurrency would have overstated capacity
   by nearly two orders of magnitude.
2. **Synchronized phases.** Every session started at t=0 and paced identically, so they
   arrived in waves. p50 72ms against p99 1142ms, entirely self-inflicted.
3. **Phase spread that ignored duty cycle.** Jittered over a fixed 6s while a session at
   duty=0.1 has a ~60s cycle, so all 400 sessions fired in the first 6s and then idled. A
   warning now fires when a level is shorter than three session cycles.

**Weighting.** The whole-pipeline speedup is weighted by audio produced, not by utterance
count, so a two-word reply does not count the same as a ten-second explanation.

---

## 7. Why there is no KV cache

Two independent reasons, either one fatal:

**VITS is non-autoregressive.** KV caching exists to skip recomputing tokens 1..n while
generating token n+1. VITS generates the whole utterance in one parallel pass. There is no
sequential loop to cache across.

**The text encoder is bidirectional.** Measured: encode "Sure, I can help", then encode
"Sure, I can help with that today", and compare the shared 31-token prefix. It moves by
**57.11%**. Every token attends to every other token in both directions, so cached K/V from
shorter text does not describe longer text. Reusing it would produce confidently wrong
prosody that no shape check, parity test or latency metric would catch.

`streaming/incremental.py` does what is actually correct for the same use case: clause-level
incremental synthesis with latents cached on the *normalized* clause text. Each clause is
encoded independently, so bidirectional attention within it is complete, and the cache is
exact reuse rather than an approximation. 25% hit rate on multi-turn traffic.
`encoder_is_causal()` ships alongside so the check can be run against any new checkpoint.

**The decoder has no attention at all** (HiFi-GAN is transposed convolutions plus dilated
residual blocks), so there is nothing to cache there either. The convolutional analogue
would be streaming convolutions with cached layer state, which would remove the 13-frame
left context currently decoded and discarded. That is real and unbuilt.

---

## 8. Numbers worth having memorized

| | |
|---|---|
| TTFA p99 at 3,200 held sessions | 113.8ms |
| TTFA p50 | 26.3ms |
| Held sessions per GPU @ 10% duty | 1,600 |
| Continuously-speaking per GPU | 128 |
| Whole-pipeline speedup | 6.91x (85.5% less GPU time) |
| Decoder / duration predictor / flow | 5.67x / 6.01x / 7.77x |
| Aggregate real-time factor at 3,200 | 350x |
| Decoder receptive field | 13 frames (208ms) |
| Batching free up to | B≈4-8 |
| GPU utilization at the latency knee | ~2% |
| Baseline (naive FastAPI) peak | 13.7 rps, p99 past 1s at concurrency 8 |
