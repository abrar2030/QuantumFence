from typing import Optional

from ..constants import DRONE_REAL_WIDTH_M

MICRO_RELATIVE_AREA = 0.0005
FIXED_WING_ASPECT = 2.0


def classify_drone_type(
    width: float,
    height: float,
    relative_area: float,
    label: Optional[str] = None,
) -> str:
    if label:
        lowered = label.lower().replace("-", "_")
        if "fixed" in lowered or "wing" in lowered:
            return "fixed_wing"
        if "quad" in lowered or "multirotor" in lowered:
            return "quadcopter"
    if relative_area < MICRO_RELATIVE_AREA:
        return "micro_drone"
    aspect = width / max(height, 1.0)
    if aspect > FIXED_WING_ASPECT:
        return "fixed_wing"
    return "quadcopter"


def real_width_for(drone_type: Optional[str]) -> float:
    return DRONE_REAL_WIDTH_M.get(
        drone_type or "unknown", DRONE_REAL_WIDTH_M["unknown"]
    )
