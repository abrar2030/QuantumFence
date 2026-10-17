import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def destination_point(
    lat: float, lng: float, bearing_deg: float, distance_m: float
) -> Tuple[float, float]:
    delta = distance_m / EARTH_RADIUS_M
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lng)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), (math.degrees(lam2) + 540.0) % 360.0 - 180.0


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lng: float
    range_m: float
    bearing_deg: float


@dataclass(frozen=True)
class CameraModel:
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    heading_deg: float = 0.0
    hfov_deg: float = 90.0
    mount_height_m: float = 4.0
    tilt_deg: float = 15.0
    max_range_m: float = 300.0

    @property
    def has_position(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def with_updates(self, **changes) -> "CameraModel":
        return replace(self, **changes)

    def focal_px(self, frame_width: int) -> float:
        half = math.radians(min(max(self.hfov_deg, 5.0), 170.0)) / 2.0
        return (frame_width / 2.0) / math.tan(half)

    def azimuth(self, cx: float, frame_width: int) -> float:
        f = self.focal_px(frame_width)
        offset = math.degrees(math.atan((cx - frame_width / 2.0) / f))
        return (self.heading_deg + offset) % 360.0

    def depression_angle(self, cy: float, frame_shape: Tuple[int, ...]) -> float:
        height, width = frame_shape[:2]
        f = self.focal_px(width)
        return math.radians(self.tilt_deg) + math.atan((cy - height / 2.0) / f)

    def ground_range(self, cy: float, frame_shape: Tuple[int, ...]) -> float:
        angle = self.depression_angle(cy, frame_shape)
        if angle <= 0.005:
            return self.max_range_m
        return min(self.mount_height_m / math.tan(angle), self.max_range_m)

    def locate_ground(
        self, cx: float, cy: float, frame_shape: Tuple[int, ...]
    ) -> Optional[GeoPoint]:
        if not self.has_position:
            return None
        width = frame_shape[1]
        distance = self.ground_range(cy, frame_shape)
        bearing = self.azimuth(cx, width)
        lat, lng = destination_point(self.latitude, self.longitude, bearing, distance)
        return GeoPoint(lat, lng, distance, bearing)

    def object_range(
        self, width_px: float, frame_width: int, real_width_m: float
    ) -> float:
        if width_px <= 0:
            return self.max_range_m
        return min(
            real_width_m * self.focal_px(frame_width) / width_px, self.max_range_m
        )

    def locate_aerial(
        self,
        cx: float,
        cy: float,
        width_px: float,
        frame_shape: Tuple[int, ...],
        real_width_m: float,
    ) -> Tuple[Optional[GeoPoint], float]:
        slant = self.object_range(width_px, frame_shape[1], real_width_m)
        elevation = -self.depression_angle(cy, frame_shape)
        altitude = max(self.mount_height_m + slant * math.sin(elevation), 0.0)
        horizontal = slant * math.cos(elevation)
        if not self.has_position:
            return None, altitude
        bearing = self.azimuth(cx, frame_shape[1])
        lat, lng = destination_point(self.latitude, self.longitude, bearing, horizontal)
        return GeoPoint(lat, lng, horizontal, bearing), altitude

    def pixel_speed_to_ms(
        self, speed_px_s: float, range_m: float, frame_width: int
    ) -> float:
        return speed_px_s * range_m / self.focal_px(frame_width)
