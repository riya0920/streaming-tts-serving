"""Chunked streaming synthesis for VITS — the piece the whole latency claim rests on."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
from torch import nn


@dataclass
class ChunkConfig:
    # Progressive chunk sizing, from the M2 measurements (docs/PROFILE.md).
    #
    # The decoder has a fixed ~4.5 ms launch floor and a 13-frame (208 ms) receptive
    # field, so a 200 ms chunk must decode 38 frames to keep 12 — 68% waste. That waste
    # is free at batch 1 but real once batching makes the decoder compute-bound.
    #
    # Only the FIRST chunk needs to be small: it sets time-to-first-audio. After that the
    # listener is already hearing audio, so later chunks should amortize the overlap tax
    first_chunk_ms: float = 200.0
    max_chunk_ms: float = 800.0
    growth: float = 2.0
    # Context decoded either side and then discarded. Must be >= the decoder's effective
    # receptive field or the seams click. Measured at 13 frames for mms-tts-eng by
    # export/probe_receptive_field.py — do not guess this for a different checkpoint.
    overlap_frames: int = 13
    # Crossfade length. Default 0: measurement (M3) showed it is not needed.
    #
    # With overlap at or above the receptive field, chunk seams already match a
    # single-pass decode — measured seam step-ratio 10.25 chunked against 10.26
    #
    # Kept, at zero, because FP16 and TensorRT will make the two decodes differ more
    # than FP32 does; if a seam ever reappears, this is the knob, and M3's data says
    # exactly what "no seam" looks like.
    crossfade_ms: float = 0.0
    # "linear" (equal-gain) or "equal_power" (cos/sin).
    #
    # Linear is correct here, and this was originally wrong. Equal-power is the right
    # choice when blending UNCORRELATED signals, where powers add and a linear fade dips
    crossfade_shape: str = "linear"
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.8
    speaking_rate: float = 1.0


@dataclass
class AudioChunk:
    index: int
    pcm: bytes                 # 16-bit little-endian mono
    audio_seconds: float
    # Wall-clock from the start of synthesis to this chunk being ready. chunk 0's value
    # is the time-to-first-audio for this utterance.
    elapsed_ms: float
    decode_ms: float
    is_final: bool
    # Diagnostics: how much was decoded versus kept. The gap is the overlap tax.
    frames_decoded: int = 0
    frames_kept: int = 0
    samples: np.ndarray | None = None   # float32, same content as pcm, for analysis


@dataclass
class StreamStats:
    ttfa_ms: float = 0.0
    frontend_ms: float = 0.0          # encoder + duration + flow, paid once
    total_ms: float = 0.0
    audio_seconds: float = 0.0
    chunks: int = 0
    decode_ms: list[float] = field(default_factory=list)
    wasted_frame_fraction: float = 0.0  # overlap frames decoded and thrown away

    @property
    def rtf(self) -> float:
        """Audio-seconds produced per wall-clock second. Below 1.0, a live stream underruns."""
        return self.audio_seconds / (self.total_ms / 1e3) if self.total_ms else 0.0


class _CaptureDecoder(nn.Module):
    """Stands in for model.decoder so a forward pass yields latents without decoding."""

    def __init__(self) -> None:
        super().__init__()
        self.spectrogram: torch.Tensor | None = None
        self.extra_args: tuple = ()

    def forward(self, spectrogram: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.spectrogram = spectrogram
        self.extra_args = args
        # Shape does not matter; VitsModel only squeezes and returns it, and we discard
        # the result. Keeping it 1-sample keeps the allocation trivial.
        return spectrogram.new_zeros((spectrogram.shape[0], 1, 1))


def capture_latents(model, **inputs) -> torch.Tensor:
    """Run VITS's front half (encoder, duration predictor, flow) and return the latents
    the decoder would have consumed, of shape [B, C, T_frames].

    Rather than reimplementing alignment expansion, stochastic duration sampling and
    flow inversion — intricate and sensitive to the transformers version — the model's
    own forward pass runs with the decoder swapped for a capture stub. The stub records
    its input and returns a one-sample dummy, so the expensive decode never happens.
    This tracks upstream exactly and does not drift when internals change.

    Shared by the offline synthesizer and the Triton vits_frontend backend so the served
    path and the measured path cannot diverge.
    """
    real = model.decoder
    capture = _CaptureDecoder()
    model.decoder = capture
    try:
        with torch.inference_mode():
            model(**inputs)
    finally:
        model.decoder = real
    if capture.spectrogram is None:
        raise RuntimeError("decoder was never called — VitsModel internals changed?")
    return capture.spectrogram


class ChunkedSynthesizer:
    def __init__(self, model, tokenizer, cfg: ChunkConfig | None = None, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.cfg = cfg or ChunkConfig()
        self.device = device

        conf = model.config
        self.sr = int(conf.sampling_rate)

        # Samples produced per latent frame = product of the generator's upsample rates.
        # Never hardcode 256: it is 256 for many VITS configs and not for all of them.
        hop = 1
        for r in getattr(conf, "upsample_rates", []):
            hop *= int(r)
        if hop <= 1:
            raise RuntimeError("could not derive hop_length from config.upsample_rates")
        self.hop = hop

        self.first_frames = max(1, round(self.cfg.first_chunk_ms * self.sr / 1e3 / hop))
        self.max_frames = max(self.first_frames,
                              round(self.cfg.max_chunk_ms * self.sr / 1e3 / hop))
        self.crossfade_samples = int(self.cfg.crossfade_ms * self.sr / 1e3)

        # The crossfade tail is taken from the decoded context region, so it cannot be
        # longer than that region.
        max_xf = self.cfg.overlap_frames * hop
        if self.crossfade_samples > max_xf:
            self.crossfade_samples = max_xf

        self._fade_out, self._fade_in = self._fade_curves(
            self.crossfade_samples, self.cfg.crossfade_shape
        )

        model.noise_scale = self.cfg.noise_scale
        model.noise_scale_duration = self.cfg.noise_scale_duration
        model.speaking_rate = self.cfg.speaking_rate

    @staticmethod
    def _fade_curves(n: int, shape: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (fade_out, fade_in). See ChunkConfig.crossfade_shape for why linear."""
        if n <= 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        if shape == "equal_power":
            theta = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
            return np.cos(theta), np.sin(theta)
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        return (1.0 - t), t

    # ------------------------------------------------------------------ front half
    def _latents(self, text: str) -> torch.Tensor:
        """Run encoder + duration + flow once, returning latents [B, C, T_frames]."""
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        return capture_latents(self.model, **inputs)

    # ------------------------------------------------------------------ the stream
    def stream(self, text: str) -> Iterator[AudioChunk]:
        """Synthesize `text`, yielding audio chunks as they are produced."""
        t0 = time.perf_counter()
        latents = self._latents(text)
        if self.device == "cuda":
            torch.cuda.synchronize()
        yield from self._stream_latents(latents, t0, time.perf_counter())

    def stream_from_latents(self, latents: torch.Tensor) -> Iterator[AudioChunk]:
        """Chunk-decode latents that were computed elsewhere.

        Exists so validation can be deterministic. The stochastic duration predictor
        samples fresh noise on every call, so two synthesize() runs of the same text
        produce different alignments and different lengths — there is no way to A/B
        chunked output against a full decode unless both consume identical latents.
        """
        t0 = time.perf_counter()
        yield from self._stream_latents(latents, t0, t0)

    def decode_full(self, latents: torch.Tensor) -> np.ndarray:
        """Single-pass decode of the same latents — the ground truth for artifact tests."""
        with torch.inference_mode():
            wav = self.model.decoder(latents)
        return wav.squeeze().float().cpu().numpy()

    def _stream_latents(
        self, latents: torch.Tensor, t0: float, t_front: float
    ) -> Iterator[AudioChunk]:
        cfg = self.cfg
        hop, pad = self.hop, cfg.overlap_frames
        xf = self.crossfade_samples

        total_frames = latents.shape[-1]

        # Progressive sizing: the first chunk is small because it sets TTFA; later chunks
        # grow so the fixed per-call cost and the overlap tax are amortized while the
        # listener is already hearing audio.
        bounds: list[tuple[int, int]] = []
        a, size = 0, self.first_frames
        while a < total_frames:
            b = min(a + size, total_frames)
            bounds.append((a, b))
            a = b
            size = min(self.max_frames, max(size + 1, int(size * cfg.growth)))

        self.stats = StreamStats(frontend_ms=(t_front - t0) * 1e3)
        carry: np.ndarray | None = None   # tail of the previous chunk, awaiting blend
        decoded_frames = 0

        for i, (a, b) in enumerate(bounds):
            is_final = b >= total_frames

            # Decode with context on both sides. Clamp at the utterance edges, and record
            # how much context we actually got so the trim is correct.
            lo, hi = max(0, a - pad), min(total_frames, b + pad)
            left_ctx = a - lo

            t_dec = time.perf_counter()
            with torch.inference_mode():
                wav = self.model.decoder(latents[..., lo:hi])
            if self.device == "cuda":
                torch.cuda.synchronize()
            dec_ms = (time.perf_counter() - t_dec) * 1e3
            decoded_frames += hi - lo

            audio = wav.squeeze().float().cpu().numpy()

            # Trim to this chunk's own audio: drop the left context entirely, keep the
            # chunk body, and keep xf samples of the right context as the blend tail.
            start = left_ctx * hop
            body_end = start + (b - a) * hop
            body = audio[start:body_end]
            tail = audio[body_end:body_end + xf] if not is_final else np.zeros(0, np.float32)

            # Blend the previous chunk's tail over this chunk's head.
            if carry is not None and len(carry) > 0:
                n = min(len(carry), len(body))
                body = body.copy()
                body[:n] = carry[:n] * self._fade_out[:n] + body[:n] * self._fade_in[:n]
            out = body
            carry = tail if len(tail) > 0 else None

            pcm = np.clip(out, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype("<i2").tobytes()
            secs = len(out) / self.sr

            now = time.perf_counter()
            chunk = AudioChunk(
                index=i,
                pcm=pcm,
                audio_seconds=secs,
                elapsed_ms=(now - t0) * 1e3,
                decode_ms=dec_ms,
                is_final=is_final,
                frames_decoded=hi - lo,
                frames_kept=b - a,
                samples=out,
            )
            if i == 0:
                self.stats.ttfa_ms = chunk.elapsed_ms
            self.stats.chunks += 1
            self.stats.audio_seconds += secs
            self.stats.decode_ms.append(dec_ms)
            yield chunk

        self.stats.total_ms = (time.perf_counter() - t0) * 1e3
        self.stats.wasted_frame_fraction = (
            1.0 - total_frames / decoded_frames if decoded_frames else 0.0
        )

    def synthesize(self, text: str) -> tuple[np.ndarray, StreamStats]:
        """Collect the whole stream into one array."""
        parts = [c.samples for c in self.stream(text) if c.samples is not None]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return audio, self.stats

    def synthesize_from_latents(self, latents: torch.Tensor) -> tuple[np.ndarray, StreamStats]:
        parts = [c.samples for c in self.stream_from_latents(latents) if c.samples is not None]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return audio, self.stats

    def latents_for(self, text: str) -> torch.Tensor:
        """Public handle on the front half, so callers can hold latents fixed."""
        return self._latents(text)

    def seam_positions(self, total_frames: int) -> list[int]:
        """Sample indices where two chunks meet — where artifacts would appear."""
        seams, a, size = [], 0, self.first_frames
        while a < total_frames:
            b = min(a + size, total_frames)
            if b < total_frames:
                seams.append(b * self.hop)
            a = b
            size = min(self.max_frames, max(size + 1, int(size * self.cfg.growth)))
        return seams
