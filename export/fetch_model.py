"""
Download the VITS checkpoint once, into ./models, so every later step works offline and
against a pinned revision.

  python export/fetch_model.py

Default is facebook/mms-tts-eng: a real VITS, single speaker, permissively licensed,
and loadable straight from `transformers` without a bespoke training repo. Note it runs
at 16 kHz, not 22.05 kHz — read the rate from the config, never hardcode it.

Swap in a 22.05 kHz LJSpeech VITS via TTS_MODEL_ID if you want the higher rate; the
export path is the same, only the hop length and sample rate change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import VitsModel, VitsTokenizer

MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")
REVISION = os.environ.get("TTS_MODEL_REVISION", "main")
OUT = Path(os.environ.get("TTS_MODEL_DIR", "models")) / MODEL_ID.replace("/", "__")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"fetching {MODEL_ID}@{REVISION} -> {OUT}")

    tok = VitsTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = VitsModel.from_pretrained(MODEL_ID, revision=REVISION).eval()

    tok.save_pretrained(OUT)
    model.save_pretrained(OUT)

    cfg = model.config
    hop = 1
    for r in getattr(cfg, "upsample_rates", []):
        hop *= int(r)

    facts = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "sampling_rate": int(cfg.sampling_rate),
        "num_speakers": int(getattr(cfg, "num_speakers", 1)),
        "stochastic_duration_prediction": bool(
            getattr(cfg, "use_stochastic_duration_prediction", True)
        ),
        "flow_size": int(getattr(cfg, "flow_size", 192)),
        "upsample_rates": list(getattr(cfg, "upsample_rates", [])),
        # samples produced per latent frame — the conversion that every chunking
        # calculation in this repo depends on
        "hop_length": hop,
        "vocab_size": int(cfg.vocab_size),
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    }
    (OUT / "model_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

    print(json.dumps(facts, indent=2))
    print(f"\n1 latent frame = {hop} samples = {hop / facts['sampling_rate'] * 1e3:.2f} ms of audio")
    print(f"a 200 ms chunk is therefore ~{round(0.200 * facts['sampling_rate'] / hop)} latent frames")

    with torch.inference_mode():
        out = model(**tok("Checkpoint fetched and runnable.", return_tensors="pt"))
    print(f"smoke test ok: waveform {tuple(out.waveform.shape)}")


if __name__ == "__main__":
    main()
