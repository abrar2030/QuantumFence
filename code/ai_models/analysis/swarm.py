import math
from typing import Iterable, List


def circular_variance(directions_deg: Iterable[float]) -> float:
    rads: List[float] = [math.radians(d) for d in directions_deg]
    if not rads:
        return 1.0
    c = sum(math.cos(r) for r in rads) / len(rads)
    s = sum(math.sin(r) for r in rads) / len(rads)
    return 1.0 - math.sqrt(c * c + s * s)


def detect_swarm(
    directions_deg: Iterable[float],
    min_count: int = 3,
    max_circular_variance: float = 0.15,
) -> bool:
    directions = list(directions_deg)
    if len(directions) < min_count:
        return False
    return circular_variance(directions) < max_circular_variance
