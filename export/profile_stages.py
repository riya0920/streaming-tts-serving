"""Per-stage timing of a VITS forward pass, plus a launch-overhead test."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from transformers import VitsModel, VitsTokenizer

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "models")
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-eng")

STAGES = ["text_encoder", "duration_predictor", "flow", "decoder", "posterior_encoder"]


def resolve() -> str:
    local = Path(MODEL_DIR) / MODEL_ID.replace("/", "__")
    return str(local) if local.exists() else MODEL_ID


class StageTimer:
    """Records device time per submodule using CUDA events."""

    def __init__(self, model, names: list[str]):
        self.pairs: dict[str, list[tuple]] = defaultdict(list)
        self.totals: dict[str, list[float]] = defaultdict(list)
        self.calls: dict[str, int] = defaultdict(int)
        self._handles = []
        self._open: dict[str, torch.cuda.Event] = {}

        for name in names:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            self._handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self._handles.append(mod.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(_m, _inp):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._open[name] = ev
        return hook

    def _post(self, name):
        def hook(_m, _inp, _out):
            start = self._open.pop(name, None)
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.pairs[name].append((start, end))
            self.calls[name] += 1
        return hook

    def collect(self) -> None:
        torch.cuda.synchronize()
        for name, evs in self.pairs.items():
            for s, e in evs:
                self.totals[name].append(s.elapsed_time(e))
        self.pairs.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog, and then it does it again.")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="docs/stage_profile.json")
    args = ap.parse_args()

    dev = "cuda"
    path = resolve()
    tok = VitsTokenizer.from_pretrained(path)
    model = VitsModel.from_pretrained(path).to(dev).eval()
    sr = int(model.config.sampling_rate)
    inputs = tok(args.text, return_tensors="pt").to(dev)

    with torch.inference_mode():
        for _ in range(5):
            model(**inputs)
        torch.cuda.synchronize()

        # ---- per-stage device time -------------------------------------------
        timer = StageTimer(model, STAGES)
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(args.iters):
            out = model(**inputs)
        ev1.record()
        timer.collect()
        timer.remove()
        torch.cuda.synchronize()
        total_ms = ev0.elapsed_time(ev1) / args.iters

    audio_s = out.waveform.shape[-1] / sr
    print(f"model {path}  |  {audio_s:.2f}s of audio per forward  |  {args.iters} iters\n")

    rows = []
    accounted = 0.0
    for name in STAGES:
        vals = timer.totals.get(name)
        if not vals:
            continue
        per_iter = sum(vals) / args.iters
        calls = timer.calls[name] / args.iters
        accounted += per_iter
        rows.append((name, per_iter, calls))

    print(f"{'stage':<22}{'ms/forward':>12}{'% total':>10}{'calls':>8}")
    print("-" * 52)
    for name, ms, calls in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<22}{ms:>12.3f}{ms / total_ms * 100:>9.1f}%{calls:>8.0f}")
    other = total_ms - accounted
    print(f"{'(unhooked//glue)':<22}{other:>12.3f}{other / total_ms * 100:>9.1f}%")
    print("-" * 52)
    print(f"{'TOTAL':<22}{total_ms:>12.3f}{100.0:>9.1f}%")
    print(f"\nreal-time factor: {audio_s / (total_ms / 1e3):.1f}x")

    # ---- launch-bound test ------------------------------------------------
    # If time barely grows with sequence length, the model is dominated by per-kernel
    print("\ndecoder scaling (is it launch-bound?)")
    flow_ch = int(getattr(model.config, "flow_size", 192))
    scale = {}
    with torch.inference_mode():
        for T in (4, 8, 12, 24, 48, 96, 192, 384):
            z = torch.randn(1, flow_ch, T, device=dev)
            for _ in range(3):
                model.decoder(z)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(args.iters):
                model.decoder(z)
            e1.record()
            torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / args.iters
            scale[T] = round(ms, 3)
            print(f"  T={T:>4} frames ({T * 256 / sr * 1e3:>7.1f} ms audio)  ->  {ms:>7.3f} ms")

    lo, hi = scale[4], scale[384]
    print(f"\n  96x more work costs {hi / lo:.2f}x the time "
          f"-> {'LAUNCH-BOUND' if hi / lo < 8 else 'compute-bound'}")

    # ---- batch scaling ------------------------------------------------------
    # The other side of the same question: if the GPU is idle between launches, batching
    # should be nearly free, which is what makes dynamic batching worth the queue delay.
    print("\ndecoder batch scaling at T=233")
    batch = {}
    with torch.inference_mode():
        for B in (1, 2, 4, 8, 16, 32):
            z = torch.randn(B, flow_ch, 233, device=dev)
            for _ in range(3):
                model.decoder(z)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(max(5, args.iters // 4)):
                model.decoder(z)
            e1.record()
            torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / max(5, args.iters // 4)
            batch[B] = round(ms, 3)
            print(f"  B={B:>3}  ->  {ms:>7.3f} ms  ({ms / B:>6.3f} ms/stream)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": path,
        "audio_seconds_per_forward": round(audio_s, 3),
        "total_ms": round(total_ms, 3),
        "stages_ms": {n: round(ms, 3) for n, ms, _ in rows},
        "stage_calls_per_forward": {n: c for n, _, c in rows},
        "unaccounted_ms": round(other, 3),
        "decoder_length_scaling_ms": scale,
        "decoder_batch_scaling_ms": batch,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
