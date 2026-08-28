from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import psutil


class PeakMemoryMonitor:
    """Small cross-platform RSS sampler for long-running experiments."""

    def __init__(self, interval_seconds: float = 0.05):
        self.process = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.peak_bytes = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            self._stop.wait(self.interval_seconds)

    def start(self) -> "PeakMemoryMonitor":
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        return self.peak_bytes


def measure(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, dict[str, float]]:
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss
    started = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - started
    after = process.memory_info().rss
    return result, {"execution_time_seconds": elapsed, "rss_delta_bytes": max(0, after - before), "rss_bytes": after}


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
