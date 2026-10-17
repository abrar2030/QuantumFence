from typing import List, Sequence, Tuple

from ..core.types import Detection


def filter_min_size(
    detections: Sequence[Detection], min_width: float = 2.0, min_height: float = 2.0
) -> List[Detection]:
    return [
        d
        for d in detections
        if d.bbox.width >= min_width and d.bbox.height >= min_height
    ]


def filter_drone_candidates(
    detections: Sequence[Detection],
    frame_shape: Tuple[int, ...],
    max_relative_area: float,
    max_relative_y: float = 1.0,
) -> List[Detection]:
    height, width = frame_shape[:2]
    frame_area = max(width * height, 1)
    kept = []
    for det in detections:
        relative_area = det.bbox.area / frame_area
        relative_y = det.bbox.cy / max(height, 1)
        if relative_area <= max_relative_area and relative_y <= max_relative_y:
            kept.append(det)
    return kept
