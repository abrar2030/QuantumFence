import time
from typing import Dict, Optional

import numpy as np

from ..manager import ModelManager


def benchmark(
    manager: ModelManager,
    frame_shape=(720, 1280, 3),
    runs: int = 50,
    warmup: int = 3,
    seed: Optional[int] = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, size=frame_shape, dtype=np.uint8)
    for _ in range(warmup):
        manager.infer(frame)
    durations = []
    for _ in range(runs):
        start = time.perf_counter()
        manager.infer(frame)
        durations.append(time.perf_counter() - start)
    arr = np.array(durations) * 1000.0
    return {
        "runs": runs,
        "avg_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "max_ms": round(float(arr.max()), 3),
        "fps": round(1000.0 / float(arr.mean()), 2),
    }
