import threading
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.base import BaseDetector
from ..core.types import BoundingBox, RawDetection


class MockDetector(BaseDetector):
    def __init__(
        self,
        class_names: Optional[Dict[int, str]] = None,
        interval: int = 30,
        seed: int = 0,
        name: str = "mock",
    ):
        self.name = name
        self._names = dict(class_names or {0: "person", 2: "car", 4: "airplane"})
        self.interval = max(1, interval)
        self._rng = np.random.default_rng(seed)
        self._calls = 0
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._names)

    def predict(
        self,
        frame: np.ndarray,
        classes: Optional[Sequence[int]] = None,
        conf: Optional[float] = None,
    ) -> List[RawDetection]:
        with self._lock:
            self._calls += 1
            if self._calls % self.interval != 0:
                return []
            pool = classes if classes is not None else list(self._names)
            candidates = [c for c in pool if c in self._names]
            if not candidates:
                return []
            class_id = int(self._rng.choice(candidates))
            h, w = frame.shape[:2]
            bw = float(self._rng.uniform(0.04, 0.15)) * w
            bh = float(self._rng.uniform(0.06, 0.2)) * h
            x = float(self._rng.uniform(0, max(w - bw, 1)))
            y = float(self._rng.uniform(0, max(h * 0.6 - bh, 1)))
            confidence = float(self._rng.uniform(0.7, 0.95))
        return [
            RawDetection(
                class_id=class_id,
                label=self._names[class_id],
                confidence=confidence,
                bbox=BoundingBox(x, y, bw, bh),
            )
        ]
