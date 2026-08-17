"""
Load driver for the naive baseline.

Ramps concurrency, holds each level long enough to reach steady state, and records the
full latency distribution — percentiles only, never means. A mean latency of 180 ms can
hide a p99 of 4 seconds, and the p99 is the only number a voice interface cares about.

For the baseline there is no streaming, so time-to-first-audio *is* total latency. That
equivalence is exactly what the rest of the project breaks.

  python baseline/loadtest.py --url http://localhost:8500 --levels 1,4,16,64,256

Writes results/baseline_<mode>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp

CORPUS = Path(__file__).resolve().parents[1] / "loadgen" / "corpus.txt"


@dataclass
class Sample:
    latency_ms: float
    audio_seconds: float
    ok: bool
    status: int


def load_corpus() -> list[str]:
    if CORPUS.exists():
        lines = [ln.strip() for ln in CORPUS.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]
    return ["The quick brown fox jumps over the lazy dog."]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    # Nearest-rank. With a few thousand samples this is within noise of interpolation,
    # and it never invents a value that was not observed.
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * len(xs) + 0.5)) - 1))
    return xs[k]


async def one_request(session: aiohttp.ClientSession, url: str, text: str) -> Sample:
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{url}/synthesize",
            json={"text": text},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            body = await resp.read()
            dt = (time.perf_counter() - t0) * 1e3
            # 44-byte WAV header, 16-bit mono.
            audio_s = max(0, len(body) - 44) / 2 / 16000
            return Sample(dt, audio_s, resp.status == 200, resp.status)
    except Exception:  # noqa: BLE001
        return Sample((time.perf_counter() - t0) * 1e3, 0.0, False, 0)


async def run_level(
    url: str, concurrency: int, duration_s: float, corpus: list[str], rng: random.Random
) -> dict:
    samples: list[Sample] = []
    stop_at = time.perf_counter() + duration_s
    connector = aiohttp.TCPConnector(limit=concurrency + 16)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Warm the level for 3 seconds before recording, so queue depth reaches steady
        # state and we do not measure the ramp.
        async def worker(record: list[Sample], until: float) -> None:
            while time.perf_counter() < until:
                s = await one_request(session, url, rng.choice(corpus))
                record.append(s)

        warm: list[Sample] = []
        await asyncio.gather(
            *[worker(warm, time.perf_counter() + 3.0) for _ in range(concurrency)]
        )
        await asyncio.gather(*[worker(samples, stop_at) for _ in range(concurrency)])

    ok = [s for s in samples if s.ok]
    lat = [s.latency_ms for s in ok]
    audio_total = sum(s.audio_seconds for s in ok)
    wall = duration_s

    return {
        "concurrency": concurrency,
        "requests": len(samples),
        "errors": len(samples) - len(ok),
        "throughput_rps": round(len(ok) / wall, 2),
        # Aggregate real-time factor: audio-seconds produced per wall-clock second across
        # the whole system. Must exceed the number of live listeners you intend to serve.
        "aggregate_rtf": round(audio_total / wall, 2),
        "latency_ms": {
            "p50": round(pct(lat, 50), 1),
            "p90": round(pct(lat, 90), 1),
            "p99": round(pct(lat, 99), 1),
            "max": round(max(lat), 1) if lat else None,
            # Recorded for completeness and to make the point that it is misleading.
            "mean": round(statistics.fmean(lat), 1) if lat else None,
        },
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8500")
    ap.add_argument("--levels", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds per level")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = load_corpus()
    levels = [int(x) for x in args.levels.split(",")]

    async with aiohttp.ClientSession() as s:
        async with s.get(f"{args.url}/healthz") as r:
            health = await r.json()
    mode = health.get("mode", "unknown")
    print(f"target: {health}")

    results = []
    for c in levels:
        print(f"  level {c:>4} ...", end="", flush=True)
        r = await run_level(args.url, c, args.duration, corpus, rng)
        results.append(r)
        print(
            f" p50={r['latency_ms']['p50']:>8.1f}ms  p99={r['latency_ms']['p99']:>9.1f}ms"
            f"  rps={r['throughput_rps']:>6.1f}  errs={r['errors']}"
        )
        # Stop climbing once the level is clearly broken; no point burning GPU minutes
        # measuring how much worse a already-collapsed server can get.
        if r["errors"] > 0.2 * r["requests"]:
            print("  >20% errors — stopping ramp")
            break

    out = Path(args.out or f"results/baseline_{mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"target": health, "levels": results}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
