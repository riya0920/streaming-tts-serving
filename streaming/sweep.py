"""
M3 — sweep chunk size and overlap, and measure what each costs.

The chunking parameters were guessed in the original design and then corrected once by
measurement (docs/PROFILE.md). This script replaces guessing entirely: for each candidate
setting it reports time-to-first-audio, how much decode is wasted on overlap, and — the
part that actually decides correctness — how badly the chunked output differs from a
single-pass decode.

The comparison is exact because both sides consume **identical latents**. The stochastic
duration predictor samples fresh noise per call, so synthesizing the same text twice gives
different alignments and different lengths; without pinning the latents there is nothing
to diff against and any "artifact metric" would mostly be measuring resampling noise.

Two error figures are reported:

  global SNR      chunked vs single-pass over the whole utterance
  seam SNR        the same, restricted to windows centred on chunk boundaries

Seam SNR is the one that matters. A high global SNR can still hide a click, because a
2 ms discontinuity contributes almost nothing to whole-utterance energy and everything to
what a listener notices.

  python streaming/sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import VitsModel, VitsTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming.chunked import ChunkConfig, ChunkedSynthesizer  # noqa: E402

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")
DEV = "cuda"

TEXTS = [
    "Sure, I can help with that.",
    "Your flight leaves at four fifteen from gate B twelve, and boarding starts about forty minutes before that.",
    "The main difference is that the first option charges a flat monthly rate regardless of usage, which is simpler to predict, while the second bills per request and works out cheaper if your traffic is genuinely intermittent.",
]


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    if n == 0:
        return float("-inf")
    err = test[:n] - ref[:n]
    p_sig = float(np.sum(ref[:n].astype(np.float64) ** 2))
    p_err = float(np.sum(err.astype(np.float64) ** 2))
    if p_err <= 0:
        return float("inf")
    if p_sig <= 0:
        return float("-inf")
    return 10.0 * np.log10(p_sig / p_err)


def seam_step_ratio(test: np.ndarray, seams: list[int], half_window: int) -> float:
    """How far the sample-to-sample jump at a seam stands out from its neighbourhood.

    SNR against a reference cannot judge a crossfade: blending two signals that differ
    by a little noise leaves the SNR unchanged while removing the *step* between them,
    and it is the step a listener hears as a click. This measures the step directly —
    the first difference at the boundary against the median first difference nearby.

    1.0 means the seam is indistinguishable from ordinary waveform motion. Large values
    mean a discontinuity sitting on top of the signal.
    """
    if not seams:
        return 1.0
    d = np.abs(np.diff(test.astype(np.float64)))
    worst = 0.0
    for s in seams:
        if s <= 1 or s >= len(d) - 1:
            continue
        lo, hi = max(0, s - half_window), min(len(d), s + half_window)
        local = np.median(d[lo:hi])
        if local <= 0:
            continue
        # Peak jump within a couple of samples of the boundary: trimming is exact, but
        # off-by-one is precisely the bug this is meant to catch.
        worst = max(worst, float(np.max(d[s - 2:s + 2]) / local))
    return worst


def seam_snr_db(ref: np.ndarray, test: np.ndarray, seams: list[int], half_window: int) -> float:
    """Worst local SNR across all seam neighbourhoods — the audible-click metric."""
    if not seams:
        return float("inf")
    worst = float("inf")
    for s in seams:
        lo, hi = max(0, s - half_window), min(len(ref), s + half_window)
        if hi - lo < 16:
            continue
        worst = min(worst, snr_db(ref[lo:hi], test[lo:hi]))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m3_sweep.json")
    ap.add_argument("--seam-window-ms", type=float, default=20.0)
    args = ap.parse_args()

    path = resolve()
    tok = VitsTokenizer.from_pretrained(path)
    model = VitsModel.from_pretrained(path).to(DEV).eval()
    sr = int(model.config.sampling_rate)
    half_window = int(args.seam_window_ms * sr / 1e3)

    # Pin the latents once per text. Everything below decodes these same tensors.
    base = ChunkedSynthesizer(model, tok, ChunkConfig(), DEV)
    torch.manual_seed(1234)
    fixtures = []
    for t in TEXTS:
        z = base.latents_for(t)
        fixtures.append((t, z, base.decode_full(z)))
        print(f"fixture: {z.shape[-1]:>4} frames "
              f"({z.shape[-1] * base.hop / sr:.2f}s)  \"{t[:48]}...\"")

    # ---------------------------------------------------------------- overlap sweep
    print(f"\nOVERLAP SWEEP (first chunk 200 ms, seam window +/-{args.seam_window_ms:.0f} ms)")
    print("measured receptive field is 13 frames — expect a cliff below it\n")
    print(f"  {'ovl':>4}{'decoded/kept':>14}{'waste':>8}{'global SNR':>12}{'seam SNR':>11}   verdict")
    overlap_rows = {}
    # Crossfade off, so this isolates the effect of decode context. With a fade active,
    # any error the fade itself introduces floors the result and hides the overlap trend.
    for ovl in (0, 2, 4, 8, 11, 13, 16, 20, 26, 32):
        cfg = ChunkConfig(first_chunk_ms=200.0, max_chunk_ms=800.0,
                          overlap_frames=ovl, crossfade_ms=0.0)
        syn = ChunkedSynthesizer(model, tok, cfg, DEV)
        g_snr, s_snr, waste = [], [], []
        for _, z, ref in fixtures:
            test, stats = syn.synthesize_from_latents(z)
            g_snr.append(snr_db(ref, test))
            s_snr.append(seam_snr_db(ref, test, syn.seam_positions(z.shape[-1]), half_window))
            waste.append(stats.wasted_frame_fraction)
        g, s, w = float(np.mean(g_snr)), float(np.min(s_snr)), float(np.mean(waste))
        # ~40 dB seam SNR is roughly where a discontinuity stops being audible against
        # speech; below ~20 dB it is a click.
        verdict = "clean" if s > 40 else ("marginal" if s > 20 else "AUDIBLE CLICK")
        overlap_rows[ovl] = {"global_snr_db": round(g, 1), "seam_snr_db": round(s, 1),
                             "wasted_fraction": round(w, 3)}
        print(f"  {ovl:>4}{'':>14}{w * 100:>7.0f}%{g:>12.1f}{s:>11.1f}   {verdict}")

    # ------------------------------------------------------------- chunk size sweep
    print("\nFIRST-CHUNK SIZE SWEEP (overlap 13, the measured receptive field)\n")
    print(f"  {'first ms':>9}{'TTFA ms':>10}{'chunks':>8}{'waste':>8}{'seam SNR':>11}")
    chunk_rows = {}
    for first_ms in (60, 100, 150, 200, 300, 400, 800):
        cfg = ChunkConfig(first_chunk_ms=float(first_ms), max_chunk_ms=800.0, overlap_frames=13)
        syn = ChunkedSynthesizer(model, tok, cfg, DEV)
        ttfa, nch, waste, s_snr = [], [], [], []
        for _, z, ref in fixtures:
            syn.synthesize_from_latents(z)      # warm the shape
            test, stats = syn.synthesize_from_latents(z)
            ttfa.append(stats.ttfa_ms)
            nch.append(stats.chunks)
            waste.append(stats.wasted_frame_fraction)
            s_snr.append(seam_snr_db(ref, test, syn.seam_positions(z.shape[-1]), half_window))
        row = {"ttfa_ms": round(float(np.mean(ttfa)), 2),
               "chunks": round(float(np.mean(nch)), 1),
               "wasted_fraction": round(float(np.mean(waste)), 3),
               "seam_snr_db": round(float(np.min(s_snr)), 1)}
        chunk_rows[first_ms] = row
        print(f"  {first_ms:>9}{row['ttfa_ms']:>10.2f}{row['chunks']:>8.1f}"
              f"{row['wasted_fraction'] * 100:>7.0f}%{row['seam_snr_db']:>11.1f}")
    print("\n  TTFA here excludes the frontend (latents are precomputed) — it is the")
    print("  decode-side cost only. Add ~114 ms of frontend for end-to-end TTFA.")

    # ------------------------------------------------------------- crossfade sweep
    # Both curves, because the first run showed equal-power actively hurting: it is the
    # right choice for uncorrelated signals, and these two decodes are near-identical.
    print("\nCROSSFADE LENGTH AND SHAPE (overlap 13, first chunk 200 ms)")
    print("  SNR is blind to a step; step-ratio is the click metric (1.0 = invisible)\n")
    print(f"  {'xfade ms':>9}{'lin SNR':>10}{'lin step':>10}{'eqp SNR':>10}{'eqp step':>10}")
    xfade_rows = {}
    for xf_ms in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        vals = {}
        for shape in ("linear", "equal_power"):
            cfg = ChunkConfig(first_chunk_ms=200.0, overlap_frames=13,
                              crossfade_ms=xf_ms, crossfade_shape=shape)
            syn = ChunkedSynthesizer(model, tok, cfg, DEV)
            s_snr, steps = [], []
            for _, z, ref in fixtures:
                test, _ = syn.synthesize_from_latents(z)
                seams = syn.seam_positions(z.shape[-1])
                s_snr.append(seam_snr_db(ref, test, seams, half_window))
                steps.append(seam_step_ratio(test, seams, half_window))
            vals[shape] = {"snr_db": round(float(np.min(s_snr)), 1),
                           "step_ratio": round(float(np.max(steps)), 2)}
        xfade_rows[xf_ms] = vals
        L, E = vals["linear"], vals["equal_power"]
        print(f"  {xf_ms:>9.1f}{L['snr_db']:>10.1f}{L['step_ratio']:>10.2f}"
              f"{E['snr_db']:>10.1f}{E['step_ratio']:>10.2f}")

    # Reference: the same metric on a single-pass decode, which by construction has no
    # seams. Anything at or below this is indistinguishable from an unchunked decode.
    ref_steps = []
    for _, z, ref in fixtures:
        ref_steps.append(seam_step_ratio(ref, base.seam_positions(z.shape[-1]), half_window))
    print(f"\n  single-pass decode at the same positions: step ratio "
          f"{float(np.max(ref_steps)):.2f}  <- the floor to beat")
    xfade_rows["reference_step_ratio"] = round(float(np.max(ref_steps)), 2)

    # ------------------------------------------------------------ per-chunk detail
    # Chases the anomaly seen in the M2 smoke test: the final chunk cost ~66 ms against
    # ~7 ms for the others.
    print("\nPER-CHUNK DETAIL (first 200 ms, overlap 13, longest fixture)\n")
    cfg = ChunkConfig(first_chunk_ms=200.0, max_chunk_ms=800.0, overlap_frames=13)
    syn = ChunkedSynthesizer(model, tok, cfg, DEV)
    z = fixtures[-1][1]
    list(syn.stream_from_latents(z))     # warm every shape first
    print(f"  {'i':>3}{'kept':>7}{'decoded':>9}{'ms':>9}{'final':>7}")
    for c in syn.stream_from_latents(z):
        print(f"  {c.index:>3}{c.frames_kept:>7}{c.frames_decoded:>9}"
              f"{c.decode_ms:>9.2f}{str(c.is_final):>7}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": path, "sampling_rate": sr,
        "seam_window_ms": args.seam_window_ms,
        "overlap_sweep": overlap_rows,
        "first_chunk_sweep": chunk_rows,
        "crossfade_sweep": xfade_rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
