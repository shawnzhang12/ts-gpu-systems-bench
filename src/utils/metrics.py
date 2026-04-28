from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class TimingResult:
    mean_ms: float
    p50_ms: float
    p95_ms: float


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    w = pos - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def benchmark_callable(fn, warmup: int = 5, iters: int = 30) -> TimingResult:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    return TimingResult(
        mean_ms=sum(timings) / max(len(timings), 1),
        p50_ms=_quantile(timings, 0.5),
        p95_ms=_quantile(timings, 0.95),
    )


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024.0 / 1024.0


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
