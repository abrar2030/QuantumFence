import math
from typing import Set

from ..tracking.tracker import Track


class LoiteringDetector:
    def __init__(self, min_seconds: float = 60.0, radius_px: float = 60.0):
        self.min_seconds = min_seconds
        self.radius_px = radius_px
        self._flagged: Set[int] = set()

    def reset(self) -> None:
        self._flagged.clear()

    def forget(self, track_id: int) -> None:
        self._flagged.discard(track_id)

    def check(self, track: Track) -> bool:
        if track.track_id in self._flagged or not track.confirmed:
            return False
        if track.duration_seconds < self.min_seconds or len(track.history) < 2:
            return False
        end = track.history[-1][0]
        window = [h for h in track.history if end - h[0] <= self.min_seconds]
        if not window or end - window[0][0] < self.min_seconds * 0.5:
            return False
        cx = sum(h[1] for h in window) / len(window)
        cy = sum(h[2] for h in window) / len(window)
        spread = max(math.hypot(h[1] - cx, h[2] - cy) for h in window)
        if spread <= self.radius_px:
            self._flagged.add(track.track_id)
            return True
        return False
