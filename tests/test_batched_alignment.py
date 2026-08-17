"""Does the TRT front half give the same answer for a batch as it does one at a time?

`vits_frontend` currently loops over the batch instead of making one batched call,
because the alignment expansion was assumed to be wrong for B>1: every item predicts its
own durations, so every item wants a different number of latent frames, and the batch has
to be built at the longest of them.

The alignment is plain PyTorch, so it can be checked without a GPU. The three engines are
replaced with stand-ins that are deterministic and batch-equivariant (row i of the output
depends only on row i of the input), which is the property a real TensorRT engine with a
batch dimension has. Noise scales are set to zero so the two paths are comparable at all.

What this does NOT cover: the flow is a stack of dilated convolutions, so in a padded
batch its receptive field can reach into another item's padding near a boundary. The
stand-in here is pointwise and cannot show that. Settling it needs the real engine on a
GPU.

    python tests/test_batched_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming.trt_frontend import TRTFrontend  # noqa: E402

CH = 192


class FakeRunner:
    """Stands in for TRTRunner. Deterministic, and each row is independent of the others."""

    conv_flow = False
    last_flow_input = None

    def __init__(self, path):
        self.kind = Path(path).stem.split("_")[1]

    def __call__(self, feed: dict):
        if self.kind == "encoder":
            ids = feed["input_ids"].float()                     # [B, S]
            c = torch.arange(CH, dtype=torch.float32)
            base = torch.sin(ids.unsqueeze(-1) * 0.1 + c * 0.01)  # [B, S, CH]
            return base, base * 0.5 + 0.1, base * 0.25 - 0.2

        if self.kind == "duration":
            h = feed["hidden_states"]                            # [B, CH, S]
            # Per-token log duration, so different items end up with different frame
            # counts. That difference is the whole point of the test.
            d = torch.tanh(h.mean(dim=1, keepdim=True)) * 0.6 + 0.9
            return (d,)

        if self.kind == "flow":
            z, mask = feed["z_p"], feed["padding_mask"]           # [B, CH, T], [B, 1, T]
            FakeRunner.last_flow_input = z.clone()
            if self.conv_flow:
                # Dilated convolution along time, which is what the real flow is. A
                # pointwise stand-in cannot show padding leaking across a boundary.
                w = torch.ones(1, 1, 3, dtype=z.dtype) / 3.0
                pad = torch.nn.functional.pad(z, (2, 2))
                z = torch.nn.functional.conv1d(
                    pad.reshape(-1, 1, pad.shape[-1]), w, dilation=2
                ).reshape(z.shape)
            return (torch.tanh(z * 0.7) * mask,)

        raise AssertionError(f"unexpected engine {self.kind}")


def build(**kw) -> TRTFrontend:
    cfg = SimpleNamespace(flow_size=CH)
    return TRTFrontend(Path("."), FakeRunner, cfg, device="cpu",
                       noise_scale=0.0, noise_scale_duration=0.0, **kw)


def make_batch():
    """Four utterances of deliberately different lengths, right-padded like the server."""
    lengths = [7, 23, 41, 12]
    S = max(lengths)
    ids = torch.zeros(len(lengths), S, dtype=torch.long)
    mask = torch.zeros(len(lengths), S, dtype=torch.long)
    for i, n in enumerate(lengths):
        ids[i, :n] = torch.arange(3, 3 + n) * (i + 2) % 97 + 1
        mask[i, :n] = 1
    return ids, mask, lengths


def main() -> int:
    torch.manual_seed(0)
    front = build()
    ids, mask, lengths = make_batch()
    B = ids.shape[0]

    one_at_a_time = [front(ids[i:i + 1], mask[i:i + 1]) for i in range(B)]
    together = front(ids, mask)

    print(f"{'item':>5}{'tokens':>8}{'looped T':>11}{'batched T':>11}"
          f"{'max abs diff':>15}   verdict")
    print("-" * 62)

    worst = 0.0
    failures = 0
    for i, solo in enumerate(one_at_a_time):
        t_solo = solo.shape[-1]
        # The batched tensor is built at the longest item, so compare on the real region
        # and require the remainder to be exactly zero.
        got = together[i:i + 1, :, :t_solo]
        diff = (got - solo).abs().max().item() if t_solo else 0.0
        tail = together[i:i + 1, :, t_solo:].abs().max().item() \
            if together.shape[-1] > t_solo else 0.0
        ok = diff < 1e-5 and tail < 1e-5
        failures += not ok
        worst = max(worst, diff, tail)
        print(f"{i:>5}{lengths[i]:>8}{t_solo:>11}{together.shape[-1]:>11}"
              f"{max(diff, tail):>15.2e}   {'ok' if ok else 'MISMATCH'}")

    frames = [o.shape[-1] for o in one_at_a_time]
    print(f"\nper-item frame counts: {frames}")
    if len(set(frames)) == 1:
        print("all items produced the same frame count, so this run did not actually")
        print("exercise the ragged case. Change the fixture lengths.")
        return 1

    print(f"worst deviation: {worst:.2e}")
    if failures:
        print(f"\nFAIL: {failures}/{B} items differ when batched.")
        return 1
    print("\nPASS: batching the alignment matches looping, on ragged input.")

    failures += check_padding_is_silent(frames)
    return 1 if failures else 0


def check_padding_is_silent(frames: list[int]) -> int:
    """The flow must not see anything past an item's own frame count.

    Past that point m_p and logs_p are zero, so exp(logs_p) is 1 and the prior sample is
    full-scale noise unless it is masked off. The flow is convolutional, so that noise
    does not stay in the padding: it reaches back into frames that get played.
    """
    print("\n\nPADDING HYGIENE (noise on, convolutional flow)")
    print("-" * 62)

    FakeRunner.conv_flow = True
    torch.manual_seed(7)
    front = build()
    front.noise_scale = 0.667          # the value the server runs with
    ids, mask, _ = make_batch()
    front(ids, mask)

    z = FakeRunner.last_flow_input     # what the flow was actually handed
    print(f"{'item':>5}{'frames':>9}{'padded':>9}{'max |z_p| in padding':>24}   verdict")
    bad = 0
    for i, n in enumerate(frames):
        tail = z[i, :, n:]
        m = tail.abs().max().item() if tail.numel() else 0.0
        ok = m == 0.0
        bad += not ok
        print(f"{i:>5}{n:>9}{z.shape[-1] - n:>9}{m:>24.4f}   {'silent' if ok else 'NOISE'}")

    # How much that noise would have been worth, had it been left in.
    dirty = z.clone()
    torch.manual_seed(11)
    for i, n in enumerate(frames):
        if z.shape[-1] > n:
            dirty[i, :, n:] = torch.randn_like(dirty[i, :, n:]) * 0.667
    w = torch.ones(1, 1, 3, dtype=z.dtype) / 3.0

    def conv(x):
        p = torch.nn.functional.pad(x, (2, 2))
        return torch.nn.functional.conv1d(
            p.reshape(-1, 1, p.shape[-1]), w, dilation=2).reshape(x.shape)

    clean_out, dirty_out = conv(z), conv(dirty)
    worst = 0.0
    for i, n in enumerate(frames):
        if n:
            worst = max(worst, (clean_out[i, :, :n] - dirty_out[i, :, :n]).abs().max().item())
    print(f"\nleft unmasked, the padding changes the played region by up to {worst:.4f}")
    print("through a single 3-tap dilated conv. The real flow is a stack of them.")

    if bad:
        print(f"\nFAIL: {bad} items handed the flow live noise past their frame count.")
        return 1
    print("\nPASS: every item's padding reaches the flow as silence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
