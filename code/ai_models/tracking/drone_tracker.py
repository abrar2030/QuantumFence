import itertools
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..analysis.drone import real_width_for
from ..analysis.geometry import CameraModel, haversine_m
from ..analysis.swarm import detect_swarm
from ..analysis.threat import ThreatScorer
from ..config import AuthorizedZone, ModelConfig
from ..core.types import Detection, ObjectKind
from .tracker import MultiObjectTracker, Track


class DroneDetector:
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        camera: Optional[CameraModel] = None,
        id_source: Optional[Iterator[int]] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.config = config or ModelConfig()
        self.camera = camera
        self.confidence_threshold = (
            self.config.drone_confidence
            if confidence_threshold is None
            else confidence_threshold
        )
        self.tracker = MultiObjectTracker(
            ObjectKind.DRONE,
            max_age_seconds=self.config.track_max_age_seconds,
            min_hits=self.config.track_min_hits,
            iou_threshold=self.config.track_match_iou,
            max_distance_px=self.config.track_max_distance_px,
            id_source=id_source if id_source is not None else itertools.count(1),
        )
        self.scorer = ThreatScorer(self.config)
        self._authorized_zones: List[AuthorizedZone] = list(
            self.config.authorized_zones
        )

    @property
    def active_tracks(self) -> Dict[int, Track]:
        return self.tracker.tracks

    def add_authorized_zone(self, lat: float, lng: float, radius_m: float) -> None:
        self._authorized_zones.append(AuthorizedZone(lat, lng, radius_m))

    @property
    def authorized_zones(self) -> Tuple[AuthorizedZone, ...]:
        return tuple(self._authorized_zones)

    def _is_authorized(self, lat: Optional[float], lng: Optional[float]) -> bool:
        if lat is None or lng is None:
            return False
        return any(
            haversine_m(lat, lng, z.lat, z.lng) <= z.radius_m
            for z in self._authorized_zones
        )

    def process_frame(
        self,
        frame_shape: Tuple[int, ...],
        detections: Sequence[Detection],
        timestamp: Optional[datetime] = None,
    ):
        shape = frame_shape if len(frame_shape) >= 2 else (720, 1280, 3)
        valid = [
            d
            for d in detections
            if d.kind == ObjectKind.DRONE and d.confidence >= self.confidence_threshold
        ]
        update = self.tracker.update(valid, timestamp)
        for track in update.matched:
            self._annotate(track, shape)
        return update

    def _annotate(self, track: Track, frame_shape: Tuple[int, ...]) -> None:
        box = track.bbox
        drone_type = track.attributes.get("drone_type", "unknown")
        lat = lng = None
        altitude = track.attributes.get("altitude_m")
        speed_ms = None
        if self.camera is not None:
            point, altitude = self.camera.locate_aerial(
                box.cx, box.cy, box.width, frame_shape, real_width_for(drone_type)
            )
            if point is not None:
                lat, lng = point.lat, point.lng
                track.attributes["range_m"] = round(point.range_m, 1)
                track.attributes["bearing_deg"] = round(point.bearing_deg, 1)
            slant = self.camera.object_range(
                box.width, frame_shape[1], real_width_for(drone_type)
            )
            speed_ms = self.camera.pixel_speed_to_ms(track.speed, slant, frame_shape[1])
        authorized = self._is_authorized(lat, lng)
        track.attributes["is_authorized"] = authorized
        track.attributes["threat_score"] = round(
            self.scorer.score(track, frame_shape, authorized), 3
        )
        if altitude is not None:
            track.attributes["altitude_m"] = round(float(altitude), 1)
        if speed_ms is not None:
            track.attributes["speed_ms"] = round(speed_ms, 2)
        if lat is not None:
            track.attributes["latitude"] = round(lat, 7)
            track.attributes["longitude"] = round(lng, 7)

    def detect_swarm(self) -> bool:
        directions = [
            t.direction
            for t in self.tracker.confirmed_tracks()
            if t.direction is not None
        ]
        return detect_swarm(
            directions,
            self.config.swarm_min_drones,
            self.config.swarm_circular_variance,
        )

    def get_summary(self) -> Dict:
        tracks = self.tracker.confirmed_tracks()
        scores = [t.attributes.get("threat_score", 0.0) for t in tracks]
        return {
            "active_drones": len(tracks),
            "high_threat_drones": sum(
                1 for s in scores if s >= self.config.high_threat_score
            ),
            "swarm_detected": self.detect_swarm(),
            "max_threat_score": max(scores, default=0.0),
            "tracks": [t.to_dict() for t in tracks],
        }
