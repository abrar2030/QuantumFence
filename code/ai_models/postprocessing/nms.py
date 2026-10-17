from typing import List, Sequence

from ..core.types import RawDetection


def non_max_suppression(
    detections: Sequence[RawDetection], iou_threshold: float = 0.45
) -> List[RawDetection]:
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[RawDetection] = []
    for det in ordered:
        if all(
            k.class_id != det.class_id or k.bbox.iou(det.bbox) < iou_threshold
            for k in kept
        ):
            kept.append(det)
    return kept
