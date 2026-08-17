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

Fix, in two steps — the third one turned out to be unnecessary:

1. **Decode with overlap.** For a chunk covering frames `[a, b)`, feed the decoder
   `[a - P, b + P)` where `P` is `overlap_frames`.
2. **Trim to the valid center.** Discard `P * hop` samples from each end of the decoded
   waveform — those are the contaminated ones.

`P` must be at least the decoder's effective receptive field. **Measured at 13 frames
(208 ms)** for `mms-tts-eng` by `export/probe_receptive_field.py`. The M3 sweep shows a
sharp cliff exactly there: seam SNR is 5.1 dB at `P=8`, jumps to 31.4 dB at `P=11`, and is
flat beyond — do not guess this for a different checkpoint.

**The crossfade was a mistake, twice over.** The original design called for an equal-power
(cos/sin) blend across the seam, on the reasoning that a linear fade dips in loudness at
the midpoint for correlated signals. That reasoning is inverted: equal-power is the right
choice for *uncorrelated* signals, where powers add. The two signals here are the same
audio decoded with slightly different context — near-perfectly correlated, so amplitudes
add and `cos(θ)x + sin(θ)x` peaks at `√2·x`, putting a **+3 dB bump on every seam**.
Measured seam SNR fell from 31.6 dB with no fade to 11.6 dB with a 10 ms equal-power fade.

Correcting it to linear made the fade *neutral* rather than helpful — `(1-t)x + tx = x`
exactly. Which raised the real question: does the seam need fixing at all once overlap is
sufficient? A discontinuity metric (first difference at the boundary against the local
median, which SNR-vs-reference cannot see) answers it:

| | step ratio |
|---|---|
| chunked, overlap 13, no crossfade | 10.25 |
| single-pass decode, same positions | 10.26 |

Identical. The boundary is indistinguishable from ordinary waveform motion. **Crossfade
defaults to 0.** The machinery stays because FP16 and TensorRT will make the two decodes
diverge more than FP32 does, and M3 established exactly what "no seam" measures like.

### Chunk size: progressive, not fixed

Because the receptive field (13 frames) is *larger* than a 200 ms chunk (12 frames), a
fixed 200 ms chunking decodes 38 frames to keep 12 — 68% waste on every chunk, forever,
to buy a latency benefit only the first chunk actually delivers.

So chunk size grows: small first chunk to set TTFA, doubling up to a cap. Measured on a
14.9 s utterance, overlap 13:

```
chunk 0:  keeps 12, decodes 25   (52% waste — worth it, this is TTFA)
chunk 3+: keeps 50, decodes 76   (34% waste — steady state)
```

Decode cost is ~4.9 ms per chunk regardless of size, because of the launch floor.

### Settled parameters

From `results/m3_sweep.json`, not from intuition:

| Parameter | Value | Why |
|---|---|---|
| `overlap_frames` | **13** | The measured receptive field. Cliff at 11; flat above. |
| `first_chunk_ms` | **200** | TTFA is frontend-dominated, so shrinking below 200 ms buys nothing measurable (decode-side TTFA is ~5 ms at every size tested from 60–800 ms). |
| `max_chunk_ms` | **800** | Amortizes the overlap tax from 52% to 34% once the listener is already hearing audio. |
| `crossfade_ms` | **0** | Unnecessary at sufficient overlap. See above. |

Convert chunk sizes to frames with `frames = round(ms/1000 * sampling_rate / hop_length)`.
`hop_length` is the product of the model's `upsample_rates` — 256 here, but
`export/fetch_model.py` prints it rather than assuming.

One operational consequence: per-chunk decode cost is ~4.9 ms for every shape, but the
**first** decode of a new tensor shape costs ~66 ms while cuDNN picks an algorithm. With
progressive sizing there are only 3–4 distinct shapes, so warm them all at startup. The
same applies to TensorRT optimization profiles in M4.

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
