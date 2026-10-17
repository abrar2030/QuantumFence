import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

import numpy as np


class InferenceMetrics:
    def __init__(self, window: int = 512):
        self._window = window
        self._lock = threading.Lock()
        self._samples: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        self._stamps: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        self._counters: Dict[str, int] = defaultdict(int)

    def record(self, name: str, seconds: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._samples[name].append(seconds)
            self._stamps[name].append(now)
            self._counters[f"{name}_total"] += 1

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._stamps.clear()
            self._counters.clear()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            timings = {}
            for name, values in self._samples.items():
                if not values:
                    continue
                arr = np.array(values, dtype=np.float64) * 1000.0
                stamps = self._stamps[name]
                span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
                rate = (len(stamps) - 1) / span if span > 0 else 0.0
                timings[name] = {
                    "count": int(self._counters[f"{name}_total"]),
                    "avg_ms": round(float(arr.mean()), 3),
                    "p50_ms": round(float(np.percentile(arr, 50)), 3),
                    "p95_ms": round(float(np.percentile(arr, 95)), 3),
                    "max_ms": round(float(arr.max()), 3),
                    "rate_per_s": round(rate, 2),
                }
            counters = {
                k: v for k, v in self._counters.items() if not k.endswith("_total")
            }
            return {"timings": timings, "counters": counters}
