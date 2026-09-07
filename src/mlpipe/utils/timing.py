"""Timing utilities for ML execution benchmarking."""

import time
from contextlib import contextmanager
from typing import Generator


class Timer:
    """Timer utility to record elapsed execution time."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed


@contextmanager
def time_block() -> Generator[Timer, None, None]:
    """Context manager to measure block execution time."""
    timer = Timer().start()
    try:
        yield timer
    finally:
        timer.stop()
