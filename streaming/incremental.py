"""
Incremental synthesis for text that arrives in fragments — an LLM streaming a reply
clause by clause into TTS.

Why this is not a KV cache
--------------------------
The original design called for "KV-cache reuse": cache the text encoder's attention keys
and values for tokens already seen, so an appended fragment only encodes the new tokens.
That is the standard trick for autoregressive decoding, and it is **invalid here**.

VITS's text encoder is **bidirectional**. Measured on `facebook/mms-tts-eng`: appending
" with that today" to "Sure, I can help" changes the encoding of the *original 31 tokens*
by 57% relative. Every token attends to every other token in both directions, so an
earlier token's representation genuinely depends on later ones. Reusing cached K/V would
produce confidently wrong prosody — and nothing in a shape check, a parity test or a
latency metric would notice, because the tensors would all be the right size and the
audio would still sound like speech.

What works instead
------------------
Synthesize at **clause boundaries**. Each clause is encoded independently, so
bidirectional attention inside it is complete and correct, and no cross-clause reuse is
attempted. Two things fall out of that:

  1. **Latency**: the first clause is synthesized as soon as it is complete, rather than
     waiting for the whole reply. That is the same trick as chunked decoding, applied one
     level up — and it works even for a model whose encoder cannot be cached.

  2. **Reuse**: assistant speech repeats heavily ("Sure.", "One moment.", "Anything
     else?"), and a clause is a pure function of its own text. Caching latents keyed on
     the normalized clause is therefore exact, not approximate — unlike a KV cache, which
     would be approximate and wrong.

  feed = IncrementalSynthesizer(latents_fn)
  for fragment in llm_stream:
      for clause_latents in feed.push(fragment):
          ...decode and ship...
  for clause_latents in feed.flush():
      ...
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming.text_norm import normalize, split_for_streaming  # noqa: E402


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    clauses_emitted: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class ClauseResult:
    text: str
    latents: object          # whatever latents_fn returns
    from_cache: bool
    index: int


class IncrementalSynthesizer:
    """Buffers streaming text, emits synthesized clauses as they complete."""

    def __init__(self, latents_fn: Callable[[str], object],
                 cache_size: int = 512, min_chars: int = 24):
        self.latents_fn = latents_fn
        self.min_chars = min_chars
        self._buf = ""
        self._emitted = 0
        self.stats = CacheStats()
        # LRU keyed on NORMALIZED clause text: "$45" and "45 dollars" normalize to the
        # same speech, so keying on the raw text would miss reuse that is actually there.
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._cache_size = cache_size

    def _synth(self, clause: str) -> tuple[object, bool]:
        key = normalize(clause)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return self._cache[key], True
        latents = self.latents_fn(clause)
        self._cache[key] = latents
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        self.stats.misses += 1
        return latents, False

    def push(self, fragment: str) -> Iterator[ClauseResult]:
        """Add a fragment; yield any clauses that are now complete.

        A clause is only emitted once its terminator has arrived, so a fragment ending
        mid-sentence emits nothing and waits. Emitting early would mean synthesizing
        text whose prosody depends on words that have not arrived — the same mistake the
        KV cache would have made, one level up.
        """
        # Join with a space when neither side supplies one. The tail re-buffered below
        # comes back from split_for_streaming already stripped, so without this a
        # fragment boundary silently welds two words together — "four fifteen." followed
        # by "Boarding starts" became "fifteen.Boarding", which the tokenizer then reads
        # as a single word and speaks as one. Audio stays fluent-sounding, so nothing
        # downstream flags it.
        if self._buf and fragment and not self._buf[-1].isspace() \
                and not fragment[0].isspace():
            self._buf += " "
        self._buf += fragment

        parts = split_for_streaming(self._buf, min_chars=self.min_chars)
        if len(parts) <= 1:
            return

        # The last part may still be growing; hold it back until a terminator arrives.
        complete, self._buf = parts[:-1], parts[-1]
        for clause in complete:
            latents, cached = self._synth(clause)
            self.stats.clauses_emitted += 1
            yield ClauseResult(clause, latents, cached, self._emitted)
            self._emitted += 1

    def flush(self) -> Iterator[ClauseResult]:
        """Emit whatever remains once the stream ends."""
        tail = self._buf.strip()
        self._buf = ""
        if not tail:
            return
        latents, cached = self._synth(tail)
        self.stats.clauses_emitted += 1
        yield ClauseResult(tail, latents, cached, self._emitted)
        self._emitted += 1


def encoder_is_causal(model, tok, device: str = "cuda", tol: float = 1e-4) -> dict:
    """Check whether a KV cache would even be valid for this encoder.

    Encodes a prefix alone, then encodes the prefix plus a continuation, and compares the
    prefix's representation. If it moves, attention is bidirectional and cached K/V from
    the shorter text does not describe the longer one.

    Worth running against any new checkpoint before assuming otherwise: the answer is a
    property of the architecture, not of the framework, and getting it wrong produces
    audio that is subtly wrong rather than obviously broken.
    """
    import torch

    a = tok("Sure, I can help", return_tensors="pt").to(device)
    b = tok("Sure, I can help with that today", return_tensors="pt").to(device)

    def enc(x):
        ids = x["input_ids"]
        return model.text_encoder(
            input_ids=ids,
            padding_mask=torch.ones_like(ids).unsqueeze(-1).float(),
            attention_mask=torch.ones_like(ids),
            return_dict=True,
        ).last_hidden_state

    with torch.inference_mode():
        ha, hb = enc(a), enc(b)
    n = a["input_ids"].shape[1]
    pa, pb = ha[0, :n].float(), hb[0, :n].float()
    drift = (pa - pb).abs().mean().item()
    rel = drift / (pa.abs().mean().item() or 1.0)
    return {
        "prefix_tokens": int(n),
        "mean_abs_change": round(drift, 6),
        "relative_change": round(rel, 4),
        "causal": bool(rel < tol),
        "kv_cache_valid": bool(rel < tol),
    }
