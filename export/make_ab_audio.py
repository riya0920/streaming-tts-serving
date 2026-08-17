"""Generate A/B audio for the FP16 decision."""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import VitsModel, VitsTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export.build_trt import TRTRunner  # noqa: E402
from streaming.chunked import ChunkConfig, ChunkedSynthesizer  # noqa: E402

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")

WINDOW, KEEP, TRIM = 76, 50, 13

TEXTS = {
    "short": "Sure, I can help with that.",
    "medium": "Your flight leaves at four fifteen from gate B twelve, and boarding starts about forty minutes before that.",
    "long": "The main difference is that the first option charges a flat monthly rate regardless of usage, which is simpler to predict, while the second bills per request and works out cheaper if your traffic is genuinely intermittent.",
    "numbers": "Doctor Chen's invoice came to twelve hundred forty seven dollars and fifty cents, due on March fourteenth.",
}


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def chunked_trt(runner: TRTRunner, z: torch.Tensor, hop: int) -> np.ndarray:
    """Decode latents through TRT using the deployed window geometry."""
    T = z.shape[-1]
    out = []
    a = 0
    while a < T:
        b = min(a + KEEP, T)
        lo, hi = max(0, a - TRIM), min(T, b + TRIM)
        piece = z[..., lo:hi]
        # The engine profile has a minimum frame count; pad short tails on the left,
        # which only adds context that gets trimmed away.
        if piece.shape[-1] < 14:
            lo = max(0, hi - 14)
            piece = z[..., lo:hi]
        wav = runner({"latents": piece})[0].float().squeeze().cpu().numpy()
        left = (a - lo) * hop
        out.append(wav[left:left + (b - a) * hop])
        a = b
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/audio")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    path = resolve()
    model = VitsModel.from_pretrained(path).cuda().eval()
    tok = VitsTokenizer.from_pretrained(path)
    sr = int(model.config.sampling_rate)
    hop = 1
    for r in model.config.upsample_rates:
        hop *= int(r)

    engines = Path(MODEL_DIR) / "engines"
    r32 = TRTRunner(engines / "vits_decoder_fp32.plan")
    r16 = TRTRunner(engines / "vits_decoder_fp16.plan")

    syn = ChunkedSynthesizer(model, tok, ChunkConfig(), "cuda")
    torch.manual_seed(7)

    print(f"writing to {outdir}\n")
    for name, text in TEXTS.items():
        z = syn.latents_for(text)          # one set of latents, three decoders
        with torch.inference_mode():
            ref = model.decoder(z).squeeze().float().cpu().numpy()
        a32 = chunked_trt(r32, z, hop)
        a16 = chunked_trt(r16, z, hop)

        n = min(len(ref), len(a32), len(a16))
        write_wav(outdir / f"{name}_a_torch_fp32.wav", ref[:n], sr)
        write_wav(outdir / f"{name}_b_trt_fp32.wav", a32[:n], sr)
        write_wav(outdir / f"{name}_c_trt_fp16.wav", a16[:n], sr)

        # Amplified difference, so a subtle artifact is actually listenable rather than
        # sitting 40 dB below the speech.
        diff = (a16[:n] - ref[:n])
        peak = float(np.abs(diff).max()) or 1.0
        write_wav(outdir / f"{name}_d_fp16_error_amplified.wav", diff / peak * 0.7, sr)

        print(f"  {name:<8} {n / sr:>5.2f}s   fp16 error peak {peak:.5f} "
              f"({20 * np.log10(peak / (np.abs(ref[:n]).max() or 1)):.1f} dB below signal)")

    print(f"\nListen in order: a (reference) -> c (fp16). If you cannot tell them apart,")
    print("FP16 is free. File d is the difference amplified to audibility — it will")
    print("always be audible on its own; the question is only whether c differs from a.")


if __name__ == "__main__":
    main()
