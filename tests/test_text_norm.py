"""
Tests for text normalization.

Runs anywhere — no GPU, no Triton, no model. That is the point of keeping the rules in
a plain module: the part of the system most likely to change is also the part that is
cheapest to check.

  python -m pytest tests/test_text_norm.py -q
  python tests/test_text_norm.py          # no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming.text_norm import normalize, split_for_streaming  # noqa: E402

# Hyphens are normalized to spaces throughout: speech has no hyphens, and the
# character-level tokenizer this feeds may not carry one in its vocabulary.
CASES = [
    # currency — must beat the bare-number rule to the digits
    ("$45", "forty five dollars"),
    ("$1,247.50", "one thousand, two hundred forty seven dollars and fifty cents"),
    ("$1", "one dollar"),
    ("$0.99", "zero dollars and ninety nine cents"),
    # ".5" is fifty cents, not five
    ("$3.5", "three dollars and fifty cents"),

    # abbreviations — must run before '.' reads as a sentence end
    ("Dr. Chen", "Doctor Chen"),
    ("Mr. and Mrs. Smith", "Mister and Missus Smith"),
    ("approx. 5 min.", "approximately five minutes"),

    # time
    ("11:45", "eleven forty five"),
    ("9:00", "nine o'clock"),
    ("4:05", "four oh five"),

    # dates
    ("3/14/2026", "March fourteenth twenty twenty six"),

    # years read in pairs, not as cardinals
    ("in 1985", "in nineteen eighty five"),
    ("in 2026", "in twenty twenty six"),

    # ordinals
    ("the 3rd time", "the third time"),

    # units
    ("5.5 km", "five point five kilometers"),
    ("38%", "thirty eight percent"),

    # decimals are read digit by digit after the point
    ("3.14", "three point one four"),

    # symbols
    ("R&D", "R and D"),
    ("5-10", "five to ten"),

    # digits glued to letters must not hide from the number rule
    ("apt. 3B", "apartment three B"),
    # a leading zero marks an identifier, not a quantity
    ("ends in 0942", "ends in zero nine four two"),
]


def check(text: str, expected: str) -> bool:
    got = normalize(text)
    ok = got == expected
    print(f"  {'ok ' if ok else 'FAIL'}  {text!r}")
    if not ok:
        print(f"        expected {expected!r}")
        print(f"        got      {got!r}")
    return ok


def test_cases() -> None:
    for text, expected in CASES:
        assert normalize(text) == expected, f"{text!r} -> {normalize(text)!r}"


def test_no_digits_survive() -> None:
    """Any digit reaching the tokenizer is a normalization bug — it would be spoken
    character by character or dropped entirely."""
    corpus = Path(__file__).resolve().parents[1] / "loadgen" / "corpus.txt"
    if not corpus.exists():
        return
    for line in corpus.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out = normalize(line)
        assert not any(c.isdigit() for c in out), f"digits survived: {line!r} -> {out!r}"


def test_split_merges_short_fragments() -> None:
    # A bare "Okay." is too short to carry prosody on its own, so it should glue on.
    parts = split_for_streaming("Okay. Here is the full explanation of what happened next.")
    assert len(parts) == 1
    parts = split_for_streaming(
        "Here is the first complete thought. And here is a second complete thought."
    )
    assert len(parts) == 2


def test_idempotent() -> None:
    """Normalizing twice must not change anything — the pipeline may retry."""
    for text, _ in CASES:
        once = normalize(text)
        assert normalize(once) == once, f"not idempotent: {text!r}"


if __name__ == "__main__":
    print("normalization cases:")
    failures = sum(0 if check(t, e) else 1 for t, e in CASES)

    print("\nno digits survive the corpus:")
    try:
        test_no_digits_survive()
        print("  ok")
    except AssertionError as exc:
        failures += 1
        print(f"  FAIL  {exc}")

    for fn in (test_split_merges_short_fragments, test_idempotent):
        try:
            fn()
            print(f"\n{fn.__name__}:\n  ok")
        except AssertionError as exc:
            failures += 1
            print(f"\n{fn.__name__}:\n  FAIL  {exc}")

    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
