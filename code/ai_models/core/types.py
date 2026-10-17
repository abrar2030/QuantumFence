from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


class ObjectKind(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    DRONE = "drone"


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        return cls(float(x1), float(y1), float(x2 - x1), float(y2 - y1))

    @classmethod
    def from_center(cls, cx: float, cy: float, w: float, h: float) -> "BoundingBox":
        return cls(cx - w / 2.0, cy - h / 2.0, w, h)

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def cx(self) -> float:
        return self.x + self.width / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def shifted(self, dx: float, dy: float) -> "BoundingBox":
        return BoundingBox(self.x + dx, self.y + dy, self.width, self.height)

    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        union = self.area + other.area - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def to_list(self, ndigits: int = 2) -> List[float]:
        return [
            round(self.x, ndigits),
            round(self.y, ndigits),
            round(self.width, ndigits),
            round(self.height, ndigits),
        ]


@dataclass
class RawDetection:
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox


@dataclass
class Detection:
    kind: ObjectKind
    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int = -1
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": self.kind.value,
            "class": self.label,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_list(),
        }
        data.update(self.attributes)
        return data


@dataclass
class InferenceOutput:
    persons: List[Detection] = field(default_factory=list)
    vehicles: List[Detection] = field(default_factory=list)
    drones: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    def all(self) -> List[Detection]:
        return [*self.persons, *self.vehicles, *self.drones]


FrameShape = Tuple[int, ...]
