"""
End-to-end client for the Triton streaming pipeline — and the first real measurement of
time-to-first-audio through a server rather than an in-process function call.

Pipeline, three hops:

    text --> tts_frontend  --> input_ids
         --> vits_frontend --> latents [1, C, T]
         --> tts_stream    --> audio chunks, streamed  (decoupled)

The orchestration lives here rather than in Triton because the Go gateway (M7) will own
it in production: it is control-plane work — sessions, routing, admission — not inference.
This module is the reference implementation the gateway is built to match, and the thing
that proves the backends work before any Go exists.

  python streaming/client.py --text "Sure, I can help with that." --out out.wav
  python streaming/client.py --bench 20
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


@dataclass
class StreamResult:
    pcm: np.ndarray
    ttfa_ms: float                       # text in -> first audio chunk out
    frontend_ms: float                   # tts_frontend + vits_frontend
    total_ms: float
    chunk_ms: list[float] = field(default_factory=list)   # arrival time of each chunk
    chunk_samples: list[int] = field(default_factory=list)
    normalized_text: str = ""

    @property
    def audio_seconds(self) -> float:
        return len(self.pcm) / 16000.0


class TTSClient:
    def __init__(self, url: str = "localhost:8001", verbose: bool = False):
        self.client = grpcclient.InferenceServerClient(url=url, verbose=verbose)

    # ------------------------------------------------------------------ hops 1-2
    def _infer(self, model: str, inputs: dict, outputs: list[str]) -> dict:
        ins = []
        for name, arr in inputs.items():
            t = grpcclient.InferInput(name, list(arr.shape), np_to_triton_dtype(arr.dtype))
            t.set_data_from_numpy(arr)
            ins.append(t)
        outs = [grpcclient.InferRequestedOutput(n) for n in outputs]
        res = self.client.infer(model_name=model, inputs=ins, outputs=outs)
        return {n: res.as_numpy(n) for n in outputs}

    def frontend(self, text: str) -> tuple[np.ndarray, np.ndarray, str]:
        # Shape is [1], not [1, 1]: tts_frontend sets max_batch_size 0, so config dims
        # are the complete shape rather than the per-sample shape after a batch axis.
        arr = np.array([text.encode("utf-8")], dtype=object)
        out = self._infer("tts_frontend", {"TEXT": arr},
                          ["INPUT_IDS", "ATTENTION_MASK", "NORMALIZED_TEXT"])
        norm = out["NORMALIZED_TEXT"].reshape(-1)[0]
        if isinstance(norm, bytes):
            norm = norm.decode("utf-8")
        return out["INPUT_IDS"], out["ATTENTION_MASK"], norm

    def latents(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = self._infer("vits_frontend",
                          {"INPUT_IDS": ids.astype(np.int64),
                           "ATTENTION_MASK": mask.astype(np.int64)},
                          ["LATENTS"])
        return out["LATENTS"]

    # -------------------------------------------------------------------- hop 3
    def stream(self, text: str) -> StreamResult:
        """Run the full pipeline, timing from text submission to each chunk's arrival."""
        t0 = time.perf_counter()
        ids, mask, norm = self.frontend(text)
        latents = self.latents(ids, mask)
        t_front = time.perf_counter()

        results: queue.Queue = queue.Queue()

        def callback(result, error):
            # Fires on the gRPC stream thread once per response the backend sends. The
            # timestamp is taken here, not after draining the queue, so queue latency
            # does not contaminate TTFA.
            results.put((time.perf_counter(), result, error))

        chunks: list[np.ndarray] = []
        arrival: list[float] = []

        self.client.start_stream(callback=callback)
        try:
            inp = grpcclient.InferInput("LATENTS", list(latents.shape), "FP32")
            inp.set_data_from_numpy(latents)
            self.client.async_stream_infer(
                model_name="tts_stream",
                inputs=[inp],
                outputs=[grpcclient.InferRequestedOutput(n)
                         for n in ("AUDIO_CHUNK", "CHUNK_INDEX", "IS_FINAL")],
            )

            while True:
                ts, result, error = results.get(timeout=60)
                if error is not None:
                    raise RuntimeError(f"tts_stream: {error}")
                pcm = result.as_numpy("AUDIO_CHUNK").reshape(-1)
                chunks.append(pcm)
                arrival.append((ts - t0) * 1e3)
                if bool(result.as_numpy("IS_FINAL").reshape(-1)[0]):
                    break
        finally:
            self.client.stop_stream()

        total = (time.perf_counter() - t0) * 1e3
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return StreamResult(
            pcm=audio,
            ttfa_ms=arrival[0] if arrival else float("nan"),
            frontend_ms=(t_front - t0) * 1e3,
            total_ms=total,
            chunk_ms=arrival,
            chunk_samples=[len(c) for c in chunks],
            normalized_text=norm,
        )


def write_wav(path: Path, pcm: np.ndarray, sr: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.astype("<i2").tobytes())


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--text", default="Your flight leaves at 4:15 from gate B12, and boarding starts in about 40 min.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", type=int, default=0, help="repeat N times and report percentiles")
    args = ap.parse_args()

    c = TTSClient(args.url)
    if not c.client.is_server_ready():
        print("server not ready", file=sys.stderr)
        sys.exit(1)

    r = c.stream(args.text)
    print(f"normalized: {r.normalized_text!r}\n")
    print(f"  frontend      {r.frontend_ms:>8.1f} ms   (tts_frontend + vits_frontend)")
    print(f"  TTFA          {r.ttfa_ms:>8.1f} ms   <- text in to first audio out")
    print(f"  total         {r.total_ms:>8.1f} ms")
    print(f"  audio         {r.audio_seconds:>8.2f} s   in {len(r.chunk_ms)} chunks")
    print(f"  RTF           {r.audio_seconds / (r.total_ms / 1e3):>8.1f}x")
    print("\n  chunk arrivals (ms):", " ".join(f"{m:.0f}" for m in r.chunk_ms))

    # A chunk must arrive before the audio already sent has finished playing, or the
    # stream underruns and the listener hears a gap mid-sentence. This checks it
    # directly rather than inferring it from an average real-time factor.
    played = 0.0
    underruns = 0
    for arrive, n in zip(r.chunk_ms, r.chunk_samples):
        if arrive > r.ttfa_ms + played * 1e3:
            underruns += 1
        played += n / 16000.0
    print(f"  underruns: {underruns} / {len(r.chunk_ms)} chunks"
          f"{'  <- stream would stutter' if underruns else '  (playback stays ahead)'}")

    if args.out:
        write_wav(Path(args.out), r.pcm)
        print(f"\nwrote {args.out}")

    if args.bench:
        print(f"\nbenchmarking {args.bench} requests...")
        ttfa, total = [], []
        for _ in range(args.bench):
            rr = c.stream(args.text)
            ttfa.append(rr.ttfa_ms)
            total.append(rr.total_ms)
        print(f"  TTFA   p50 {pct(ttfa, 50):>7.1f}   p90 {pct(ttfa, 90):>7.1f}"
              f"   p99 {pct(ttfa, 99):>7.1f} ms")
        print(f"  total  p50 {pct(total, 50):>7.1f}   p90 {pct(total, 90):>7.1f}"
              f"   p99 {pct(total, 99):>7.1f} ms")


if __name__ == "__main__":
    main()
