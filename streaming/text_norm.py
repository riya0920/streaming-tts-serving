"""Text normalization for TTS — turning written English into speakable English."""

from __future__ import annotations

import re
from functools import lru_cache

try:
    import inflect
    _INFLECT = inflect.engine()
except ImportError:  # pragma: no cover - the backend installs it; tests may not have it
    _INFLECT = None


# Expanded before anything else touches the '.', so the period does not read as a
# sentence boundary. Value is the spoken form.
_ABBREVIATIONS = [
    (r"\bMr\.", "Mister"),
    (r"\bMrs\.", "Missus"),
    (r"\bMs\.", "Miss"),
    (r"\bDr\.", "Doctor"),
    (r"\bProf\.", "Professor"),
    (r"\bSt\.", "Saint"),
    (r"\bJr\.", "Junior"),
    (r"\bSr\.", "Senior"),
    (r"\bvs\.?\b", "versus"),
    (r"\betc\.", "etcetera"),
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\bapprox\.", "approximately"),
    (r"\bapt\.", "apartment"),
    (r"\bave\.", "avenue"),
    (r"\bblvd\.", "boulevard"),
    (r"\bmin\.", "minutes"),
    (r"\bhr\.", "hours"),
    (r"\bNo\.", "number"),
]

_UNITS = [
    (r"\b(\d+(?:\.\d+)?)\s?km\b", r"\1 kilometers"),
    (r"\b(\d+(?:\.\d+)?)\s?cm\b", r"\1 centimeters"),
    (r"\b(\d+(?:\.\d+)?)\s?mm\b", r"\1 millimeters"),
    (r"\b(\d+(?:\.\d+)?)\s?kg\b", r"\1 kilograms"),
    (r"\b(\d+(?:\.\d+)?)\s?lbs?\b", r"\1 pounds"),
    (r"\b(\d+(?:\.\d+)?)\s?ft\b", r"\1 feet"),
    (r"\b(\d+(?:\.\d+)?)\s?mph\b", r"\1 miles per hour"),
    (r"\b(\d+(?:\.\d+)?)\s?%", r"\1 percent"),
    (r"\b(\d+(?:\.\d+)?)\s?°C\b", r"\1 degrees Celsius"),
    (r"\b(\d+(?:\.\d+)?)\s?°F\b", r"\1 degrees Fahrenheit"),
]

_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"),
             "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}

_MONTHS = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
           7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
           12: "December"}

_WHITESPACE = re.compile(r"\s+")


def _despan(s: str) -> str:
    """inflect emits 'forty-five'; speech has no hyphens, and a character-level
    tokenizer may not even have one in its vocabulary. Normalize them to spaces here,
    once, rather than with a global rule that would also mangle real compounds."""
    return s.replace("-", " ")


def _num_to_words(n: int) -> str:
    if _INFLECT is not None:
        return _despan(_INFLECT.number_to_words(n, andword=""))
    return str(n)


def _ordinal(n: int) -> str:
    if _INFLECT is not None:
        return _despan(_INFLECT.number_to_words(_INFLECT.ordinal(n)))
    return str(n)


def _expand_abbreviations(text: str) -> str:
    for pat, rep in _ABBREVIATIONS:
        text = re.sub(pat, rep, text)
    return text


def _expand_currency(text: str) -> str:
    """$1,247.50 -> one thousand two hundred forty seven dollars and fifty cents"""
    def repl(m: re.Match) -> str:
        sym, whole, frac = m.group(1), m.group(2).replace(",", ""), m.group(3)
        one, many, sub_one, sub_many = _CURRENCY[sym]
        w = int(whole)
        parts = [f"{_num_to_words(w)} {one if w == 1 else many}"]
        if frac:
            # ".5" means fifty cents, not five. Pad before converting.
            c = int(frac.ljust(2, "0")[:2])
            if c:
                parts.append(f"and {_num_to_words(c)} {sub_one if c == 1 else sub_many}")
        return " ".join(parts)

    syms = "".join(re.escape(s) for s in _CURRENCY)
    return re.sub(rf"([{syms}])\s?(\d[\d,]*)(?:\.(\d{{1,2}}))?", repl, text)


def _expand_time(text: str) -> str:
    """11:45 -> eleven forty five;  9:00 -> nine o'clock"""
    def repl(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            return m.group(0)
        if mi == 0:
            return f"{_num_to_words(h)} o'clock"
        if mi < 10:
            return f"{_num_to_words(h)} oh {_num_to_words(mi)}"
        return f"{_num_to_words(h)} {_num_to_words(mi)}"

    return re.sub(r"\b(\d{1,2}):(\d{2})\b", repl, text)


def _expand_dates(text: str) -> str:
    """3/14/2026 -> March fourteenth twenty twenty six"""
    def repl(m: re.Match) -> str:
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        out = f"{_MONTHS[mo]} {_ordinal(d)}"
        if y:
            out += " " + _expand_year(int(y) if len(y) == 4 else 2000 + int(y))
        return out

    return re.sub(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", repl, text)


def _expand_year(y: int) -> str:
    """Years are read in pairs: 1985 -> nineteen eighty five, not one thousand..."""
    if 1100 <= y <= 1999 or 2010 <= y <= 2099:
        hi, lo = divmod(y, 100)
        if lo == 0:
            return f"{_num_to_words(hi)} hundred"
        return f"{_num_to_words(hi)} {_num_to_words(lo)}"
    return _num_to_words(y)


def _expand_ordinals(text: str) -> str:
    return re.sub(r"\b(\d+)(st|nd|rd|th)\b",
                  lambda m: _ordinal(int(m.group(1))), text)


def _expand_units(text: str) -> str:
    for pat, rep in _UNITS:
        text = re.sub(pat, rep, text)
    return text


def _split_alphanumeric(text: str) -> str:
    """'3B' -> '3 B', 'B12' -> 'B 12'.

    Without this the digit is glued to a letter, so the word-boundary in the number
    rule never matches and the digit reaches the tokenizer verbatim — where a
    character-level model reads it as an unknown symbol or drops it.
    """
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    return text


def _expand_numbers(text: str) -> str:
    """Whatever bare numerals survive the more specific rules above."""
    def repl(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        if "." in raw:
            whole, frac = raw.split(".", 1)
            spoken = _num_to_words(int(whole)) if whole else "zero"
            # Decimals are read digit by digit: 3.14 -> three point one four
            return spoken + " point " + " ".join(_num_to_words(int(d)) for d in frac)
        # A leading zero means the digits are an identifier, not a quantity — phone
        # numbers, room numbers, zero-padded codes. Read them out individually.
        if len(raw) > 1 and raw[0] == "0":
            return " ".join(_num_to_words(int(d)) for d in raw)
        n = int(raw)
        # Four-digit values in this range are overwhelmingly years in speech.
        if 1100 <= n <= 2099 and len(raw) == 4:
            return _expand_year(n)
        return _num_to_words(n)

    return re.sub(r"\d[\d,]*(?:\.\d+)?", repl, text)


def _expand_symbols_pre(text: str) -> str:
    """Runs BEFORE numbers, because these rules need to see the digits.

    The digit-range rule in particular: once 5-10 has become 'five-ten' there is no
    way left to tell a range from a compound word.
    """
    text = re.sub(r"\s?&\s?", " and ", text)
    text = re.sub(r"\s?\+\s?", " plus ", text)
    text = re.sub(r"\s?=\s?", " equals ", text)
    text = re.sub(r"\s?@\s?", " at ", text)
    text = re.sub(r"(?<=\d)\s?-\s?(?=\d)", " to ", text)
    return text


def _expand_symbols_post(text: str) -> str:
    """Runs AFTER numbers: a hyphen between words is a pause, not a word."""
    return re.sub(r"(?<=[a-zA-Z])-(?=[a-zA-Z])", " ", text)


@lru_cache(maxsize=4096)
def normalize(text: str) -> str:
    """Written English -> speakable English.

    Cached because in a streaming pipeline the same short phrases recur constantly
    (acknowledgements, confirmations) and this is pure.
    """
    if not text:
        return ""
    t = text.strip()
    t = _expand_abbreviations(t)   # before '.' can be read as a sentence end
    t = _expand_currency(t)        # before bare numbers eat the digits
    t = _expand_dates(t)           # before time, so 3/14 is not parsed as a ratio
    t = _expand_time(t)
    t = _expand_units(t)           # before bare numbers, so "5 km" keeps its unit
    t = _expand_ordinals(t)
    t = _expand_symbols_pre(t)     # ranges need to see digits
    t = _split_alphanumeric(t)     # so "3B" cannot hide a digit from the next step
    t = _expand_numbers(t)
    t = _expand_symbols_post(t)
    t = _WHITESPACE.sub(" ", t).strip()
    return t


# Sentence-ish boundaries. Used for incremental synthesis: an LLM streaming text into
# TTS should have each clause synthesized as it completes rather than waiting for the
# whole reply.
_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[;:])\s+")


def split_for_streaming(text: str, min_chars: int = 24) -> list[str]:
    """Split into synthesis units, merging fragments too short to sound natural.

    Very short units make prosody choppy — the model has no context to place stress —
    so anything under `min_chars` is glued to its neighbour rather than synthesized alone.
    """
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    if not parts:
        return []
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) < min_chars:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out
