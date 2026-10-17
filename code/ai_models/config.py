from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .exceptions import ConfigurationError

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = PACKAGE_DIR / "weights"

LOW_LIGHT_MODES = ("off", "auto", "on")


@dataclass(frozen=True)
class AuthorizedZone:
    lat: float
    lng: float
    radius_m: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AuthorizedZone":
        try:
            return cls(
                lat=float(data["lat"]),
                lng=float(data["lng"]),
                radius_m=float(data["radius_m"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid authorized zone: {data!r}") from exc


@dataclass(frozen=True)
class ModelConfig:
    yolo_weights: Path = DEFAULT_WEIGHTS_DIR / "yolov8n.pt"
    drone_weights: Path = DEFAULT_WEIGHTS_DIR / "drone_detector.pt"
    device: str = "auto"
    half_precision: bool = False
    image_size: int = 640
    confidence: float = 0.5
    iou_threshold: float = 0.45
    drone_confidence: float = 0.4
    max_detections: int = 100
    warmup: bool = True
    auto_download: bool = True
    allow_mock_fallback: bool = False

    drone_proxy_enabled: bool = True
    drone_proxy_class_ids: Tuple[int, ...] = (4, 33)
    drone_max_relative_area: float = 0.35
    proxy_max_relative_area: float = 0.15
    proxy_max_relative_y: float = 0.7
    drone_tiling: bool = False
    drone_tile_size: int = 640
    drone_tile_overlap: float = 0.2

    low_light_mode: str = "off"
    low_light_threshold: float = 60.0

    track_max_age_seconds: float = 3.0
    track_min_hits: int = 2
    track_match_iou: float = 0.2
    track_max_distance_px: float = 150.0

    swarm_min_drones: int = 3
    swarm_circular_variance: float = 0.15
    high_threat_score: float = 0.7
    loiter_seconds: float = 60.0
    loiter_radius_px: float = 60.0

    camera_mount_height_m: float = 4.0
    camera_tilt_deg: float = 15.0
    camera_max_range_m: float = 300.0
    authorized_zones: Tuple[AuthorizedZone, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "yolo_weights", Path(self.yolo_weights))
        object.__setattr__(self, "drone_weights", Path(self.drone_weights))
        zones = tuple(
            z if isinstance(z, AuthorizedZone) else AuthorizedZone.from_mapping(z)
            for z in self.authorized_zones
        )
        object.__setattr__(self, "authorized_zones", zones)
        self.validate()

    def validate(self) -> None:
        for name in ("confidence", "iou_threshold", "drone_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be within [0, 1], got {value}")
        if self.image_size < 32:
            raise ConfigurationError("image_size must be at least 32")
        if self.low_light_mode not in LOW_LIGHT_MODES:
            raise ConfigurationError(
                f"low_light_mode must be one of {LOW_LIGHT_MODES}, got {self.low_light_mode}"
            )
        if self.track_min_hits < 1:
            raise ConfigurationError("track_min_hits must be at least 1")
        if self.track_max_age_seconds <= 0:
            raise ConfigurationError("track_max_age_seconds must be positive")
        if not 0.0 <= self.drone_tile_overlap < 1.0:
            raise ConfigurationError("drone_tile_overlap must be within [0, 1)")
        if self.drone_tile_size < 64:
            raise ConfigurationError("drone_tile_size must be at least 64")

    def with_overrides(self, **overrides: Any) -> "ModelConfig":
        return replace(self, **overrides)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"Unknown model config keys: {sorted(unknown)}")
        return cls(**dict(data))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = str(value)
            elif f.name == "authorized_zones":
                value = [
                    {"lat": z.lat, "lng": z.lng, "radius_m": z.radius_m} for z in value
                ]
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out
