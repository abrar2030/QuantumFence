import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..core.types import BoundingBox, Detection, ObjectKind
from .kalman import KalmanBoxFilter

MIN_DIRECTION_SPEED_PX_S = 1.0
APPROACH_GROWTH_RATIO = 1.05
APPROACH_MIN_SAMPLES = 5


@dataclass
class Track:
    track_id: int
    kind: ObjectKind
    label: str
    filter: KalmanBoxFilter
    first_seen: datetime
    last_seen: datetime
    confidence: float
    hits: int = 1
    confirmed: bool = False
    missed_seconds: float = 0.0
    updated_this_frame: bool = True
    history: Deque[Tuple[float, float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    confidences: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> BoundingBox:
        return self.filter.bbox

    @property
    def velocity(self) -> Tuple[float, float]:
        return self.filter.velocity

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return math.hypot(vx, vy)

    @property
    def direction(self) -> Optional[float]:
        if self.speed < MIN_DIRECTION_SPEED_PX_S:
            return None
        vx, vy = self.velocity
        return math.degrees(math.atan2(vy, vx))

    @property
    def is_approaching(self) -> bool:
        if len(self.history) < APPROACH_MIN_SAMPLES:
            return False
        recent = list(self.history)[-APPROACH_MIN_SAMPLES:]
        first = recent[0][3] * recent[0][4]
        last = recent[-1][3] * recent[-1][4]
        return bool(first > 0 and last > first * APPROACH_GROWTH_RATIO)

    @property
    def duration_seconds(self) -> float:
        return max((self.last_seen - self.first_seen).total_seconds(), 0.0)

    @property
    def avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return float(np.mean(self.confidences))

    @property
    def key(self) -> str:
        return f"{self.kind.value}:{self.track_id}"

    def trajectory(self, limit: int = 30) -> List[List[float]]:
        points = list(self.history)[-limit:]
        return [[round(p[1], 1), round(p[2], 1)] for p in points]

    def to_dict(self) -> Dict[str, Any]:
        box = self.bbox
        data: Dict[str, Any] = {
            "track_id": self.track_id,
            "kind": self.kind.value,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "bbox": box.to_list(),
            "center": [round(box.cx, 2), round(box.cy, 2)],
            "speed_px_s": round(self.speed, 2),
            "direction_deg": (
                round(self.direction, 1) if self.direction is not None else None
            ),
            "is_approaching": self.is_approaching,
            "duration_seconds": round(self.duration_seconds, 2),
            "hits": self.hits,
            "confirmed": self.confirmed,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "trajectory": self.trajectory(),
        }
        data.update(self.attributes)
        return data


@dataclass
class TrackerUpdate:
    matched: List[Track] = field(default_factory=list)
    new_confirmed: List[Track] = field(default_factory=list)
    lost: List[Track] = field(default_factory=list)


class MultiObjectTracker:
    def __init__(
        self,
        kind: ObjectKind,
        max_age_seconds: float = 3.0,
        min_hits: int = 2,
        iou_threshold: float = 0.2,
        max_distance_px: float = 150.0,
        id_source: Optional[Iterator[int]] = None,
    ):
        self.kind = kind
        self.max_age_seconds = max_age_seconds
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance_px = max_distance_px
        self._ids = id_source if id_source is not None else itertools.count(1)
        self.tracks: Dict[int, Track] = {}
        self._last_ts: Optional[float] = None

    def reset(self) -> None:
        self.tracks.clear()
        self._last_ts = None

    def _cost(self, predicted: BoundingBox, det: BoundingBox) -> Optional[float]:
        iou = predicted.iou(det)
        if iou >= self.iou_threshold:
            return 1.0 - iou
        dist = math.hypot(predicted.cx - det.cx, predicted.cy - det.cy)
        if dist <= self.max_distance_px:
            return 1.0 + dist / self.max_distance_px
        return None

    def update(
        self, detections: List[Detection], timestamp: Optional[datetime] = None
    ) -> TrackerUpdate:
        ts = timestamp or datetime.now(timezone.utc)
        now = ts.timestamp()
        dt = 0.0 if self._last_ts is None else max(now - self._last_ts, 0.0)
        self._last_ts = now

        track_list = list(self.tracks.values())
        for track in track_list:
            track.filter.predict(dt)
            track.missed_seconds += dt
            track.updated_this_frame = False

        pairs = []
        for ti, track in enumerate(track_list):
            predicted = track.filter.bbox
            for di, det in enumerate(detections):
                cost = self._cost(predicted, det.bbox)
                if cost is not None:
                    pairs.append((cost, ti, di))
        pairs.sort()

        used_tracks = set()
        used_dets = set()
        result = TrackerUpdate()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            track = track_list[ti]
            was_confirmed = track.confirmed
            self._apply(track, detections[di], ts)
            result.matched.append(track)
            if track.confirmed and not was_confirmed:
                result.new_confirmed.append(track)

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            track = self._spawn(det, ts)
            result.matched.append(track)
            if track.confirmed:
                result.new_confirmed.append(track)

        for track in list(self.tracks.values()):
            limit = (
                self.max_age_seconds
                if track.confirmed
                else min(self.max_age_seconds, 1.0)
            )
            if track.missed_seconds > limit:
                del self.tracks[track.track_id]
                result.lost.append(track)
        return result

    def _apply(self, track: Track, det: Detection, ts: datetime) -> None:
        track.filter.update(det.bbox)
        track.hits += 1
        track.missed_seconds = 0.0
        track.updated_this_frame = True
        track.last_seen = ts
        track.confidence = det.confidence
        track.label = det.label
        track.confidences.append(det.confidence)
        box = track.filter.bbox
        track.history.append((ts.timestamp(), box.cx, box.cy, box.width, box.height))
        track.attributes.update(
            {k: v for k, v in det.attributes.items() if v is not None}
        )
        if track.hits >= self.min_hits:
            track.confirmed = True

    def _spawn(self, det: Detection, ts: datetime) -> Track:
        track = Track(
            track_id=next(self._ids),
            kind=self.kind,
            label=det.label,
            filter=KalmanBoxFilter(det.bbox),
            first_seen=ts,
            last_seen=ts,
            confidence=det.confidence,
            confirmed=self.min_hits <= 1,
        )
        track.confidences.append(det.confidence)
        track.history.append(
            (ts.timestamp(), det.bbox.cx, det.bbox.cy, det.bbox.width, det.bbox.height)
        )
        track.attributes.update(
            {k: v for k, v in det.attributes.items() if v is not None}
        )
        self.tracks[track.track_id] = track
        return track

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.confirmed]
