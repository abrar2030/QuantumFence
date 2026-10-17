import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..analysis.geometry import CameraModel
from ..analysis.loitering import LoiteringDetector
from ..config import ModelConfig
from ..core.types import Detection, ObjectKind
from ..preprocessing.frame import prepare_frame
from ..tracking.drone_tracker import DroneDetector
from ..tracking.tracker import MultiObjectTracker, Track


@dataclass(frozen=True)
class DetectionFlags:
    persons: bool = True
    vehicles: bool = True
    drones: bool = True

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "DetectionFlags":
        data = data or {}
        return cls(
            persons=bool(data.get("detect_persons", True)),
            vehicles=bool(data.get("detect_vehicles", True)),
            drones=bool(data.get("detect_drones", True)),
        )


@dataclass
class CameraContext:
    camera_id: int
    name: str = ""
    camera_model: Optional[CameraModel] = None
    night_vision: bool = False


@dataclass
class FrameResult:
    camera_id: int
    timestamp: datetime
    frame_shape: Tuple[int, ...]
    inference_ms: float = 0.0
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    new_tracks: List[Track] = field(default_factory=list)
    lost_tracks: List[Track] = field(default_factory=list)
    loitering: List[Track] = field(default_factory=list)
    swarm: bool = False
    swarm_started: bool = False

    def by_kind(self, kind: ObjectKind) -> List[Track]:
        return [t for t in self.tracks if t.kind == kind]

    def new_by_kind(self, kind: ObjectKind) -> List[Track]:
        return [t for t in self.new_tracks if t.kind == kind]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "inference_ms": round(self.inference_ms, 2),
            "tracks": [t.to_dict() for t in self.tracks],
            "swarm": self.swarm,
        }


class DetectionPipeline:
    def __init__(self, manager, context: Optional[CameraContext] = None):
        self.manager = manager
        self.context = context or CameraContext(camera_id=0)
        self.config: ModelConfig = manager.config
        self._ids = itertools.count(1)
        self.person_tracker = self._make_tracker(ObjectKind.PERSON)
        self.vehicle_tracker = self._make_tracker(ObjectKind.VEHICLE)
        self.drone_detector = DroneDetector(
            self.config, self.context.camera_model, id_source=self._ids
        )
        self.loitering = LoiteringDetector(
            self.config.loiter_seconds, self.config.loiter_radius_px
        )
        self._swarm_active = False
        self.frames_processed = 0

    def _make_tracker(self, kind: ObjectKind) -> MultiObjectTracker:
        return MultiObjectTracker(
            kind,
            max_age_seconds=self.config.track_max_age_seconds,
            min_hits=self.config.track_min_hits,
            iou_threshold=self.config.track_match_iou,
            max_distance_px=self.config.track_max_distance_px,
            id_source=self._ids,
        )

    def update_context(self, context: CameraContext) -> None:
        self.context = context
        self.drone_detector.camera = context.camera_model

    def reset(self) -> None:
        self.person_tracker.reset()
        self.vehicle_tracker.reset()
        self.drone_detector.tracker.reset()
        self.loitering.reset()
        self._swarm_active = False

    def process(
        self,
        frame: np.ndarray,
        flags: Optional[DetectionFlags] = None,
        timestamp: Optional[datetime] = None,
    ) -> FrameResult:
        flags = flags or DetectionFlags()
        ts = timestamp or datetime.now(timezone.utc)
        prepared = prepare_frame(
            frame,
            self.config.low_light_mode,
            self.config.low_light_threshold,
            force_enhance=self.context.night_vision
            and self.config.low_light_mode != "off",
        )
        output = self.manager.infer(
            prepared,
            persons=flags.persons,
            vehicles=flags.vehicles,
            drones=flags.drones,
        )
        shape = prepared.shape
        result = FrameResult(
            camera_id=self.context.camera_id,
            timestamp=ts,
            frame_shape=shape,
            inference_ms=output.inference_ms,
            detections=output.all(),
        )

        if flags.persons:
            update = self.person_tracker.update(output.persons, ts)
            self._collect(result, update, shape)
            for track in update.matched:
                if track.confirmed and self.loitering.check(track):
                    result.loitering.append(track)
            for lost in update.lost:
                self.loitering.forget(lost.track_id)
        if flags.vehicles:
            self._collect(
                result, self.vehicle_tracker.update(output.vehicles, ts), shape
            )
        if flags.drones:
            self._collect(
                result,
                self.drone_detector.process_frame(shape, output.drones, ts),
                shape,
            )
            swarm = self.drone_detector.detect_swarm()
            result.swarm = swarm
            result.swarm_started = swarm and not self._swarm_active
            self._swarm_active = swarm

        self.frames_processed += 1
        return result

    def _collect(self, result: FrameResult, update, shape: Tuple[int, ...]) -> None:
        model = self.context.camera_model
        for track in update.matched:
            if track.kind != ObjectKind.DRONE and model is not None:
                box = track.bbox
                point = model.locate_ground(box.cx, box.y2, shape)
                if point is not None:
                    track.attributes["latitude"] = round(point.lat, 7)
                    track.attributes["longitude"] = round(point.lng, 7)
                    track.attributes["range_m"] = round(point.range_m, 1)
                    track.attributes["bearing_deg"] = round(point.bearing_deg, 1)
            if track.confirmed:
                result.tracks.append(track)
        result.new_tracks.extend(update.new_confirmed)
        result.lost_tracks.extend(update.lost)

    @property
    def swarm_active(self) -> bool:
        return self._swarm_active

    def active_tracks(self) -> List[Track]:
        tracks: List[Track] = []
        tracks.extend(self.person_tracker.confirmed_tracks())
        tracks.extend(self.vehicle_tracker.confirmed_tracks())
        tracks.extend(self.drone_detector.tracker.confirmed_tracks())
        return tracks
