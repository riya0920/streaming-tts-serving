# Architecture decisions

Written as decisions with their reasoning, so that when a benchmark later contradicts
one, it is obvious what has to change.

---

## 1. Why VITS

End-to-end: text in, waveform out of a single network. The older two-stage stacks
(Tacotron → vocoder) require orchestrating two models with a mel-spectrogram handoff in
between, which roughly doubles the serving surface — two sets of engines, two batching
policies, an intermediate tensor to move. VITS collapses that.

Internally it is three pieces that matter here:

| Piece | Job | Serving treatment |
|---|---|---|
| Text encoder (transformer) | phonemes → hidden states + prior means/log-vars | TensorRT FP16; KV-cacheable |
| Duration predictor | how long each phoneme lasts → alignment | FP32 PyTorch, outside engines (stochastic sampling) |
| Flow + HiFi-GAN decoder | latents → raw waveform | flow FP32; **decoder is TensorRT FP16 and is the chunked hot loop** |

That internal split is not trivia — the pieces want genuinely different treatment, and
the whole serving design falls out of it.

---

## 2. The chunking decision

**The mistake to avoid:** synthesize the whole utterance, then stream the resulting
file. This does nothing for time-to-first-audio. TTFA still scales with utterance length,
which is the exact problem.

**What we do instead:** run text encoder → duration predictor → flow once for the whole
utterance (cheap relative to the decoder), producing latents `z` of shape
`[B, C, T_frames]`. Then decode `z` in slices, shipping each slice's audio as soon as it
exists.

```
z:  [ .... T_frames ....................................... ]
      |chunk 0|chunk 1|chunk 2|chunk 3| ...
        ↓ ship    ↓ ship   ↓ ship
      user is already listening to chunk 0 while chunk 2 decodes
```

TTFA becomes a function of *chunk* size, not *utterance* length. That is the entire
latency trick.

### Why the flow runs whole, not chunked

The residual coupling layers have WaveNet-style dilated convolutions with their own
receptive field, and they are the numerically touchiest part of the model in half
precision. Chunking them would mean solving the boundary problem twice and fighting FP16
instability at the same time, for a component that is a small fraction of decoder cost.
Run it once, in FP32, and move on. If profiling (M2) shows flow is more expensive than
assumed, revisit — but not before.

### The boundary artifact, and the fix

The HiFi-GAN decoder is convolutional with a large receptive field. Slice the latents and
decode each slice independently and the seams click audibly: each chunk's edge samples
were computed without the neighboring context they need.

Fix, in three steps:

1. **Decode with overlap.** For a chunk covering frames `[a, b)`, feed the decoder
   `[a - P, b + P)` where `P` is a padding of `overlap_frames`.
2. **Trim to the valid center.** Discard `P * hop` samples from each end of the decoded
   waveform — those are the contaminated ones.
3. **Crossfade the seam.** Adjacent chunks still meet at a hard boundary. Overlap the
   final `X` samples of chunk *i* with the first `X` of chunk *i+1* and blend with an
   equal-power (constant-power) curve rather than linear:

   ```
   out[n] = cos(θ)·a[n] + sin(θ)·b[n],   θ = (π/2)·(n/X)
   ```

   Equal-power, not linear, because a linear crossfade dips in perceived loudness at the
   midpoint when the two signals are correlated — which they are, since they are two
   renderings of the same underlying latents.

`P` (decode padding) and `X` (crossfade length) are separate knobs. `P` must be at least
the decoder's effective receptive field in frames to fully remove the artifact; `X` only
needs to be long enough to mask any residual discontinuity, typically far shorter.

### Chunk size is a measured tradeoff

| Smaller chunks | Larger chunks |
|---|---|
| ✅ lower TTFA | ❌ higher TTFA |
| ❌ overlap overhead amortizes worse (`2P/(chunk+2P)` of the decode is thrown away) | ✅ overlap overhead amortizes well |
| ❌ more per-chunk fixed cost: kernel launches, response-queue writes | ✅ fewer fixed costs |
| ❌ more chances to underrun if RTF dips | ✅ more slack |

Starting point: **~200 ms of audio per chunk**, to be confirmed by the M3 sweep. Convert
to frames with `frames = round(0.200 * sampling_rate / hop_length)` — `hop_length` is the
product of the model's `upsample_rates`, and `export/inspect_model.py` prints it rather
than assuming 256.

**The sweep is the deliverable, not the number.** Plot TTFA, wasted-decode fraction, and
a boundary-discontinuity metric against chunk size, and pick from the curve.

---

## 3. Two backends: Python and C++

| | Python backend (`tts_frontend`) | C++ backend (`tts_stream`) |
|---|---|---|
| Owns | normalization, G2P, tokenization | per-chunk decode, crossfade, PCM conversion, response queue |
| Runs | once per utterance | many times per second, per session |
| Why this language | messy rule-heavy string logic that changes constantly; off the hot path; rewriting phonemization in C++ costs weeks and buys no latency | GIL and interpreter overhead add small *variable* delays, and variability is what destroys a p99 |

**The rule: Python where the code changes, C++ where the latency lives.**

The argument for C++ here is specifically about *variance*, not mean. A Python loop that
adds 0.4 ms on average but occasionally 9 ms — because a GC pass or GIL contention landed
badly — is fine for the p50 and fatal for the p99. M5's exit criterion is therefore that
the C++ per-chunk overhead *distribution* is tighter, not merely that its mean is lower.

---

## 4. Why Triton

Three features, in order of how much they mattered:

1. **Decoupled mode** — the dealbreaker. A normal inference server is one request, one
   response. Decoupled mode lets one request emit a stream of responses over time, which
   is exactly the shape of streaming TTS. Without it, streaming has to be bolted on top
   of a request/response server.
2. **Dynamic batching** — Triton holds arriving requests for a small window (a few ms)
   and fuses them into one batched GPU call. GPU efficiency without any client
   coordinating with any other client. The window length is the direct expression of the
   latency-vs-efficiency tension.
3. **Instance groups** — multiple execution instances on separate CUDA streams, so one
   instance launches kernels while another does GPU work. Hides launch latency, keeps the
   device fed.

TorchServe has nothing equivalent to decoupled mode; rolling our own means writing all
three.

---

## 5. Two model configs, one model

Live streaming and offline batch synthesis want opposite things. Serving them from one
config means the offline job's big batches sit in front of a live listener's next chunk.

So: two Triton model configs over the same engines.

| | `tts_stream_live` | `tts_stream_batch` |
|---|---|---|
| max queue delay | ~2–5 ms | 50–200 ms |
| preferred batch sizes | small | large |
| priority | high | low |
| Used by | WebSocket sessions | "synthesize this article to a file" |

The gateway routes on request type. Offline work can never contaminate the live path's
latency because it is never in the same queue.

---

## 6. Admission control

Past some concurrency, admitting one more session does not serve one more user — it
degrades everyone. The gateway tracks in-flight sessions and Triton queue depth, and
past a threshold rejects new sessions immediately.

**Better to fail one user fast than to degrade three thousand.** A rejected session can
retry or be routed elsewhere; a session admitted into a collapsing queue produces audio
that underruns mid-sentence, which is worse than no audio at all.

The threshold is not guessed — it comes from M9's ramp, set just below the observed p99
knee.

---

## 7. Metrics: histograms, never averages

Averages actively hide tail latency, and the tail is the entire subject of this project.
Every latency metric is a Prometheus histogram.

The four that matter:

- **TTFA** — time to first audio chunk. The headline.
- **Real-time factor** — audio-seconds generated per wall-clock second. **Must stay
  above 1.0** or the stream underruns mid-sentence. Tracked at the p01, not the mean:
  the question is "did it ever dip," not "was it usually fine."
- **Triton queue depth** — the leading saturation indicator. It climbs before latency
  does, which makes it the right signal for admission control.
- **GPU utilization** (DCGM) — the denominator for every efficiency claim.

Traces exist to answer *where the time went*. Early on, the expectation is that TTFA is
dominated by queue wait rather than inference — in which case the fix is batching-window
and instance-group tuning, not more model optimization. That distinction is worth weeks,
and only a trace decomposition can make it.
