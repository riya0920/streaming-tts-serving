"""
The deliberate baseline. This is the dumb version, written on purpose.

Request comes in -> tokenize -> one full model.forward() -> return a complete WAV.
No batching across users. No streaming. No quantization. Nothing clever.

The point of this file is to produce a number to beat. Without it, "62% faster" is a
claim about nothing. It is also the thing that demonstrates *why* the rest of the
system exists: watch TTFA explode with concurrency while the GPU sits mostly idle.

Two modes, because a strawman baseline is worse than no baseline:

  async_blocking  The classic mistake — a blocking torch call inside `async def`, which
                  parks the event loop and serializes every request behind every other.
                  This is genuinely what a lot of first-draft services look like.

  threadpool      The competent-naive version — sync `def`, so FastAPI hands it to a
                  worker thread. Torch releases the GIL around CUDA ops so requests do
                  overlap somewhat, but they still each launch their own forward pass
                  with batch size 1, so the GPU is fed tiny work items and stays idle
                  between them.

Report both. The honest comparison for the final system is against `threadpool`.

  BASELINE_MODE=threadpool python baseline/server.py
"""

from __future__ import annotations

import io
import os
import time
import wave
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from transformers import VitsModel, VitsTokenizer

MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")
DEVICE = os.environ.get("TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODE: Literal["async_blocking", "threadpool"] = os.environ.get(  # type: ignore[assignment]
    "BASELINE_MODE", "threadpool"
)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    tokenizer = VitsTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["sr"] = int(model.config.sampling_rate)

    # Warm up. The first CUDA call pays cuBLAS/cuDNN autotuning and kernel JIT; without
    # this the first few load-test samples are garbage outliers that flatter nothing.
    if DEVICE == "cuda":
        with torch.inference_mode():
            warm = tokenizer("warming up the kernels", return_tensors="pt").to(DEVICE)
            for _ in range(3):
                model(**warm)
            torch.cuda.synchronize()

    print(
        f"[baseline] {MODEL_ID} on {DEVICE} @ {_state['sr']} Hz "
        f"mode={MODE} load={time.perf_counter() - t0:.1f}s"
    )
    yield
    _state.clear()


app = FastAPI(title="tts-baseline", lifespan=lifespan)


class SynthRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    speaking_rate: float = Field(1.0, gt=0.25, lt=4.0)


def _to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """float32 [-1, 1] -> 16-bit PCM WAV container."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _synthesize(text: str, speaking_rate: float) -> tuple[bytes, dict]:
    """One request, one full forward pass, batch size 1. The whole problem, in a function."""
    tok = _state["tokenizer"]
    model = _state["model"]
    sr = _state["sr"]

    t_start = time.perf_counter()
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    t_tok = time.perf_counter()

    model.speaking_rate = speaking_rate
    with torch.inference_mode():
        out = model(**inputs)
        wav = out.waveform  # [1, T]
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    t_fwd = time.perf_counter()

    audio = wav.squeeze(0).float().cpu().numpy()
    data = _to_wav_bytes(audio, sr)
    t_end = time.perf_counter()

    audio_seconds = audio.shape[-1] / sr
    wall = t_end - t_start
    return data, {
        "tokenize_ms": round((t_tok - t_start) * 1e3, 2),
        "forward_ms": round((t_fwd - t_tok) * 1e3, 2),
        "encode_ms": round((t_end - t_fwd) * 1e3, 2),
        "total_ms": round(wall * 1e3, 2),
        "audio_seconds": round(audio_seconds, 3),
        # >1.0 means we generate faster than real time. Below 1.0, a stream would
        # underrun mid-sentence — which is the failure this project exists to prevent.
        "rtf": round(audio_seconds / wall, 3) if wall > 0 else None,
    }


if MODE == "async_blocking":

    @app.post("/synthesize")
    async def synthesize(req: SynthRequest) -> Response:  # noqa: D103
        try:
            data, timings = _synthesize(req.text, req.speaking_rate)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(
            content=data,
            media_type="audio/wav",
            headers={"X-Timings": repr(timings), "X-Baseline-Mode": MODE},
        )

else:

    @app.post("/synthesize")
    def synthesize(req: SynthRequest) -> Response:  # noqa: D103
        try:
            data, timings = _synthesize(req.text, req.speaking_rate)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(
            content=data,
            media_type="audio/wav",
            headers={"X-Timings": repr(timings), "X-Baseline-Mode": MODE},
        )


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": "model" in _state,
        "mode": MODE,
        "device": DEVICE,
        "model": MODEL_ID,
        "sampling_rate": _state.get("sr"),
    }


if __name__ == "__main__":
    import uvicorn

    # Single worker on purpose. Multiple workers would each hold their own copy of the
    # model and quietly turn this into a crude form of replication, which is not the
    # baseline we want to measure.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8500")),
        workers=1,
        log_level="warning",
    )
