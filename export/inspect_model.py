"""
Dump the real module structure, tensor shapes, and per-module timing of the loaded VITS.

Run this on the GPU box *before* touching the export script. Every downstream decision —
where to cut the graph, what the decoder's input shape is, how many latent frames a
200 ms chunk needs, whether the decoder actually dominates GPU time — depends on facts
this prints. Guessing them from the architecture diagram is how you spend three evenings
debugging an ONNX export that was always going to be wrong.

  python export/inspect_model.py --text "The quick brown fox jumps over the lazy dog."
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import VitsModel, VitsTokenizer

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")


def resolve_path() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="docs/model_shapes.json")
    args = ap.parse_args()

    path = resolve_path()
    tok = VitsTokenizer.from_pretrained(path)
    model = VitsModel.from_pretrained(path).to(args.device).eval()
    cfg = model.config
    dev = args.device

    inputs = tok(args.text, return_tensors="pt").to(dev)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask", torch.ones_like(input_ids))

    print(f"model: {path} on {dev}")
    print(f"top-level modules: {[n for n, _ in model.named_children()]}")

    shapes: dict = {"model": path, "device": dev, "text": args.text}

    with torch.inference_mode():
        # --- text encoder -------------------------------------------------------
        padding_mask = attn.unsqueeze(-1).float()
        enc = model.text_encoder(
            input_ids=input_ids,
            padding_mask=padding_mask,
            attention_mask=attn,
            return_dict=True,
        )
        hidden = enc.last_hidden_state
        m_p = enc.prior_means
        logs_p = enc.prior_log_variances
        shapes["text_encoder"] = {
            "input_ids": list(input_ids.shape),
            "last_hidden_state": list(hidden.shape),
            "prior_means": list(m_p.shape),
            "prior_log_variances": list(logs_p.shape),
        }
        print(f"\ntext_encoder: ids{tuple(input_ids.shape)} -> hidden{tuple(hidden.shape)}, "
              f"m_p{tuple(m_p.shape)}, logs_p{tuple(logs_p.shape)}")

        # --- full forward, to learn the latent/waveform relationship ------------
        out = model(input_ids=input_ids, attention_mask=attn)
        wav = out.waveform
        shapes["waveform"] = list(wav.shape)

        hop = 1
        for r in getattr(cfg, "upsample_rates", []):
            hop *= int(r)
        sr = int(cfg.sampling_rate)
        n_frames = wav.shape[-1] // hop
        shapes["derived"] = {
            "sampling_rate": sr,
            "hop_length": hop,
            "ms_per_latent_frame": round(hop / sr * 1e3, 3),
            "latent_frames_in_utterance": int(n_frames),
            "audio_seconds": round(wav.shape[-1] / sr, 3),
            "frames_per_200ms_chunk": round(0.200 * sr / hop),
        }
        print(f"waveform{tuple(wav.shape)} = {wav.shape[-1] / sr:.2f}s @ {sr} Hz")
        print(f"hop={hop} -> {hop / sr * 1e3:.2f} ms per latent frame; "
              f"200 ms chunk = {round(0.200 * sr / hop)} frames")

        # --- decoder in isolation, at the shape streaming will actually use -----
        flow_ch = int(getattr(cfg, "flow_size", m_p.shape[1]))
        chunk_frames = shapes["derived"]["frames_per_200ms_chunk"]
        timings = {}
        for label, T in (("chunk", chunk_frames), ("full", max(n_frames, chunk_frames))):
            z = torch.randn(1, flow_ch, T, device=dev)
            for _ in range(3):
                model.decoder(z)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                y = model.decoder(z)
            if dev == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / args.iters * 1e3
            timings[f"decoder_{label}_{T}frames_ms"] = round(ms, 3)
            audio_s = y.shape[-1] / sr
            print(f"decoder[{label}] T={T} frames -> {tuple(y.shape)}  "
                  f"{ms:.2f} ms  (rtf {audio_s / (ms / 1e3):.1f}x)")

        # --- encoder cost, for the "is the encoder worth caching" question ------
        for _ in range(3):
            model.text_encoder(input_ids=input_ids, padding_mask=padding_mask,
                               attention_mask=attn, return_dict=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model.text_encoder(input_ids=input_ids, padding_mask=padding_mask,
                               attention_mask=attn, return_dict=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        timings["text_encoder_ms"] = round((time.perf_counter() - t0) / args.iters * 1e3, 3)

        # --- end to end ---------------------------------------------------------
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(input_ids=input_ids, attention_mask=attn)
        if dev == "cuda":
            torch.cuda.synchronize()
        timings["full_forward_ms"] = round((time.perf_counter() - t0) / args.iters * 1e3, 3)

    shapes["timings_ms"] = timings
    enc_ms = timings["text_encoder_ms"]
    full_ms = timings["full_forward_ms"]
    print(f"\ntext_encoder {enc_ms:.2f} ms | full forward {full_ms:.2f} ms "
          f"| encoder is {enc_ms / full_ms * 100:.1f}% of total")
    print("If the decoder is not the dominant term here, the chunking design needs "
          "rethinking before any TensorRT work.")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(shapes, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
