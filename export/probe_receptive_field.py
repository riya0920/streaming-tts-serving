"""Measure the HiFi-GAN decoder's effective receptive field, in latent frames."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import VitsModel

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")
DEV = "cuda"

# Influence below this, relative to the peak change, is treated as inaudible.
# -60 dB is a conservative floor: well under the noise of 16-bit quantization in
# any real listening condition.
THRESH_DB = -60.0


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


def main() -> None:
    path = resolve()
    model = VitsModel.from_pretrained(path).to(DEV).eval()
    sr = int(model.config.sampling_rate)
    ch = int(getattr(model.config, "flow_size", 192))
    hop = 1
    for r in model.config.upsample_rates:
        hop *= int(r)

    T = 128
    centre = T // 2
    torch.manual_seed(0)

    results = {"model": path, "hop": hop, "sampling_rate": sr, "threshold_db": THRESH_DB}

    with torch.inference_mode():
        z = torch.randn(1, ch, T, device=DEV)
        y0 = model.decoder(z).squeeze().float().cpu().numpy()

        # Perturb one frame. Magnitude is irrelevant to the *extent* of influence in a
        # (locally) linear system; averaging several perturbations guards against a
        # single unlucky draw landing near a dead unit.
        extents = []
        for trial in range(5):
            zp = z.clone()
            zp[:, :, centre] += torch.randn(ch, device=DEV) * 3.0
            y1 = model.decoder(zp).squeeze().float().cpu().numpy()

            diff = np.abs(y1 - y0)
            peak = diff.max()
            if peak <= 0:
                continue
            db = 20 * np.log10(np.maximum(diff, 1e-12) / peak)
            touched = np.where(db > THRESH_DB)[0]
            lo, hi = touched.min(), touched.max()

            centre_sample = centre * hop
            left = (centre_sample - lo) / hop
            right = (hi - centre_sample) / hop
            extents.append((left, right))
            print(f"  trial {trial}: influence spans frames "
                  f"[-{left:.1f}, +{right:.1f}] around the perturbed frame")

        left = max(e[0] for e in extents)
        right = max(e[1] for e in extents)
        rf = int(np.ceil(max(left, right)))

    print(f"\neffective receptive field: {rf} frames "
          f"({rf * hop / sr * 1e3:.1f} ms) at {THRESH_DB:.0f} dB")

    results["receptive_field_frames"] = rf
    results["receptive_field_ms"] = round(rf * hop / sr * 1e3, 2)
    results["per_trial_extent_frames"] = [[round(a, 2), round(b, 2)] for a, b in extents]

    # ---- cost of each candidate overlap ------------------------------------
    chunk = round(0.200 * sr / hop)
    print(f"\noverlap cost at chunk={chunk} frames (200 ms):\n")
    print(f"  {'overlap':>8}{'decoded':>10}{'wasted':>9}   verdict")
    table = {}
    for ovl in (0, 2, 4, 6, 8, 12, 16):
        decoded = chunk + 2 * ovl
        waste = 1 - chunk / decoded
        ok = "sufficient" if ovl >= rf else "SEAMS WILL CLICK"
        table[ovl] = {"decoded_frames": decoded, "wasted_fraction": round(waste, 3),
                      "sufficient": bool(ovl >= rf)}
        print(f"  {ovl:>8}{decoded:>10}{waste * 100:>8.0f}%   {ok}")
    results["overlap_cost"] = table
    results["recommended_overlap"] = rf

    print(f"\nrecommendation: overlap_frames = {rf} "
          f"({(1 - chunk / (chunk + 2 * rf)) * 100:.0f}% of decode wasted, "
          f"vs {(1 - chunk / (chunk + 32)) * 100:.0f}% at the guessed 16)")

    out = Path("docs/receptive_field.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
