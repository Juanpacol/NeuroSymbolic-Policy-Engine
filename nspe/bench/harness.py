"""Timing protocol shared by every benchmark in this project.

Fair timing on GPU-like backends requires an explicit device sync
before and after every timed region -- otherwise the launch is timed,
not the work. This module also fixes the rest of the protocol so every
benchmark in the project reports comparable numbers: warmup reps
discarded, median/p95/p99 over many timed reps, and throughput derived
from the median rather than the mean (mean is skewed by rare stalls).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class TimingStats:
    """Summary statistics for one benchmarked configuration.

    Attributes:
        device: device string the benchmark ran on.
        batch_size: batch size used to compute throughput.
        warmup: number of discarded warmup reps.
        reps: number of timed reps the statistics are computed over.
        median_ms: median wall-clock time per call, in milliseconds.
        p95_ms: 95th percentile wall-clock time, in milliseconds.
        p99_ms: 99th percentile wall-clock time, in milliseconds.
        mean_ms: mean wall-clock time, in milliseconds.
        throughput_per_sec: ``batch_size / (median_ms / 1000)``.
    """

    device: str
    batch_size: int
    warmup: int
    reps: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_per_sec: float


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark(
    fn: Callable[[], None],
    *,
    device: str = "cpu",
    warmup: int = 20,
    reps: int = 200,
    batch_size: int = 1,
) -> TimingStats:
    """Times ``fn`` under the project's standard protocol.

    Args:
        fn: a zero-argument callable that performs the work to time. It
            should not itself synchronize the device; this function
            handles that.
        device: ``"cpu"``, ``"mps"``, or ``"cuda"``.
        warmup: number of untimed calls to ``fn`` before timing starts.
        reps: number of timed calls to ``fn``.
        batch_size: batch size represented by one call to ``fn``, used
            only to compute throughput.

    Returns:
        The computed :class:`TimingStats`.
    """
    for _ in range(warmup):
        fn()
    _synchronize(device)

    times_ms = []
    for _ in range(reps):
        _synchronize(device)
        start = time.perf_counter_ns()
        fn()
        _synchronize(device)
        end = time.perf_counter_ns()
        times_ms.append((end - start) / 1e6)

    times = torch.tensor(times_ms)
    median = times.median().item()
    throughput = (batch_size / (median / 1000.0)) if median > 0 else float("inf")
    return TimingStats(
        device=device,
        batch_size=batch_size,
        warmup=warmup,
        reps=reps,
        median_ms=median,
        p95_ms=torch.quantile(times, 0.95).item(),
        p99_ms=torch.quantile(times, 0.99).item(),
        mean_ms=times.mean().item(),
        throughput_per_sec=throughput,
    )
