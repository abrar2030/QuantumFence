import math
from typing import Tuple

from ..config import ModelConfig
from ..tracking.tracker import Track

FAST_SPEED_FRAME_WIDTHS_PER_S = 0.04


class ThreatScorer:
    def __init__(self, config: ModelConfig):
        self.config = config

    def score(
        self, track: Track, frame_shape: Tuple[int, ...], is_authorized: bool = False
    ) -> float:
        height, width = frame_shape[:2]
        score = 0.3
        if track.is_approaching:
            score += 0.25
        if track.speed > FAST_SPEED_FRAME_WIDTHS_PER_S * max(width, 1):
            score += 0.15
        duration = track.duration_seconds
        if duration > 60:
            score += 0.20
        elif duration > 30:
            score += 0.10
        if is_authorized:
            score -= 0.40
        box = track.bbox
        cx_n = box.cx / max(width, 1)
        cy_n = box.cy / max(height, 1)
        centre_dist = math.hypot(cx_n - 0.5, cy_n - 0.5)
        proximity = 1.0 - min(centre_dist / 0.707, 1.0)
        score += proximity * 0.10
        return min(1.0, max(0.0, score))

    def is_high(self, score: float) -> bool:
        return score >= self.config.high_threat_score
