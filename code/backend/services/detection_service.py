import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from ai_models import (
    CameraContext,
    CameraModel,
    DetectionFlags,
    DetectionPipeline,
    FrameResult,
    ModelManager,
    ObjectKind,
    Track,
)
from api.websocket import manager
from config.model_config import build_model_config
from config.settings import settings
from database.database import SessionLocal
from services.ai_analysis_service import AIAnalysisService
from services.notification_service import NotificationService
from services.perimeter_service import PerimeterService

logger = logging.getLogger("quantumfence.detection")

DRONE_ROW_REFRESH_SECONDS = 5.0
MAX_SNAPSHOT_AGE_CHECK_SECONDS = 3600.0


def geofence_to_dict(fence) -> Optional[Dict[str, Any]]:
    if fence is None:
        return None
    return {
        "id": fence.id,
        "name": fence.name,
        "is_active": fence.is_active,
        "fence_type": fence.fence_type,
        "center_lat": fence.center_lat,
        "center_lng": fence.center_lng,
        "radius_meters": fence.radius_meters,
        "coordinates": fence.coordinates or [],
        "buffer_meters": fence.buffer_meters,
        "alert_on_entry": fence.alert_on_entry,
    }


def camera_runtime_config(camera) -> Dict[str, Any]:
    return {
        "name": camera.name,
        "detect_persons": camera.detect_persons,
        "detect_vehicles": camera.detect_vehicles,
        "detect_drones": camera.detect_drones,
        "latitude": camera.latitude,
        "longitude": camera.longitude,
        "direction_degrees": camera.direction_degrees,
        "fov_degrees": camera.fov_degrees,
        "night_vision": camera.night_vision,
        "geofence": geofence_to_dict(camera.geofence) if camera.geofence_id else None,
    }


class AlertGate:
    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self._last: Dict[str, float] = {}

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        current = time.monotonic() if now is None else now
        last = self._last.get(key)
        if last is not None and current - last < self.cooldown_seconds:
            return False
        self._last[key] = current
        return True

    def clear(self) -> None:
        self._last.clear()


class CameraProcessor:
    def __init__(self, camera_id: int, stream_url: str, config: dict):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_count = 0
        self.consecutive_failures = 0

    @property
    def is_simulated(self) -> bool:
        return self.stream_url in ("simulated", "", None)

    async def start(self):
        self.is_running = True
        await asyncio.to_thread(self._open_stream)

    def _open_stream(self) -> bool:
        if self.is_simulated:
            logger.info(f"Camera {self.camera_id}: using simulated feed")
            return True
        try:
            cap = cv2.VideoCapture(self.stream_url)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, settings.STREAM_BUFFER_SIZE)
            if cap.isOpened():
                self.cap = cap
                self.consecutive_failures = 0
                logger.info(f"Camera {self.camera_id}: stream opened")
                return True
            cap.release()
            logger.warning(f"Camera {self.camera_id}: could not open {self.stream_url}")
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: stream error - {e}")
        self.cap = None
        return False

    def reconnect(self) -> bool:
        if self.cap:
            self.cap.release()
            self.cap = None
        return self._open_stream()

    def get_frame(self) -> Optional[np.ndarray]:
        if self.is_simulated:
            return self._generate_test_frame()
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_count += 1
                self.consecutive_failures = 0
                return frame
        self.consecutive_failures += 1
        return None

    def _generate_test_frame(self) -> np.ndarray:
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (18, 22, 30)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.putText(
            frame,
            f"CAM {self.camera_id:02d}  -  {ts}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 140),
            2,
        )
        cv2.putText(
            frame,
            "SIMULATED FEED",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 80, 200),
            1,
        )
        cy = int(h * 0.55)
        cv2.line(frame, (0, cy), (w, cy), (0, 180, 0), 2)
        cv2.putText(
            frame,
            "PERIMETER FENCE",
            (w // 2 - 100, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 0),
            1,
        )
        self.frame_count += 1
        return frame

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None


@dataclass
class CameraState:
    pipeline: DetectionPipeline
    config: Dict[str, Any]
    drone_rows: Dict[int, int] = field(default_factory=dict)
    drone_row_updated: Dict[int, float] = field(default_factory=dict)


class DetectionService:
    def __init__(self):
        self.camera_processors: Dict[int, CameraProcessor] = {}
        self._processing_tasks: Dict[int, asyncio.Task] = {}
        self._states: Dict[int, CameraState] = {}
        self.model_manager: Optional[ModelManager] = None
        self.ai_analysis = None
        self.notifier: Optional[NotificationService] = None
        self.perimeter = PerimeterService()
        self.db_session_factory = None
        self._last_touch: Dict[int, datetime] = {}
        self._gate = AlertGate(settings.DETECTION_ALERT_COOLDOWN_SECONDS)
        self._background: set = set()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self):
        try:
            self.model_manager = ModelManager(build_model_config())
            await self.model_manager.load_all_models()
            self.ai_analysis = AIAnalysisService()
            self.notifier = NotificationService()
            self.db_session_factory = SessionLocal
            logger.info(
                f"Detection service ready: models {self.model_manager.health()}"
            )
        except Exception as e:
            logger.error(f"Detection service init error: {e}")
            self.model_manager = None
            return
        if settings.MAX_SNAPSHOT_AGE_DAYS > 0:
            self._cleanup_task = asyncio.create_task(self._snapshot_cleanup_loop())

    def model_health(self) -> str:
        if not self.model_manager:
            return "unavailable"
        return self.model_manager.health()

    async def start_camera(self, camera_id: int, stream_url: str, config: dict):
        if camera_id in self.camera_processors:
            await self.stop_camera(camera_id)

        proc = CameraProcessor(camera_id, stream_url, config)
        await proc.start()
        self.camera_processors[camera_id] = proc
        self._ensure_state(camera_id, config)

        task = asyncio.create_task(
            self._process_loop(camera_id),
            name=f"cam_{camera_id}",
        )
        self._processing_tasks[camera_id] = task

        status = "online" if proc.is_simulated or proc.cap else "offline"
        await self._set_camera_status(camera_id, status)
        await manager.broadcast_camera_status(str(camera_id), status)
        logger.info(f"Camera {camera_id}: detection started")

    async def stop_camera(self, camera_id: int):
        task = self._processing_tasks.pop(camera_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        proc = self.camera_processors.pop(camera_id, None)
        if proc:
            proc.stop()

        self._states.pop(camera_id, None)
        self._last_touch.pop(camera_id, None)
        await self._set_camera_status(camera_id, "disabled")
        await manager.broadcast_camera_status(str(camera_id), "disabled")
        logger.info(f"Camera {camera_id}: detection stopped")

    async def reconfigure_camera(self, camera_id: int, stream_url: str, config: dict):
        proc = self.camera_processors.get(camera_id)
        if proc is None:
            return
        if (stream_url or "simulated") != (proc.stream_url or "simulated"):
            await self.start_camera(camera_id, stream_url or "simulated", config)
            return
        proc.config = config
        state = self._states.get(camera_id)
        if state:
            state.config = config
            state.pipeline.update_context(self._build_context(camera_id, config))

    async def shutdown(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        for cam_id in list(self.camera_processors.keys()):
            await self.stop_camera(cam_id)
        for task in list(self._background):
            task.cancel()
        if self.model_manager:
            self.model_manager.unload()
        logger.info("Detection service: shutdown complete")

    def _build_context(self, camera_id: int, config: dict) -> CameraContext:
        cfg = self.model_manager.config if self.model_manager else build_model_config()
        model = CameraModel(
            latitude=config.get("latitude"),
            longitude=config.get("longitude"),
            heading_deg=float(config.get("direction_degrees") or 0.0),
            hfov_deg=float(config.get("fov_degrees") or 90.0),
            mount_height_m=cfg.camera_mount_height_m,
            tilt_deg=cfg.camera_tilt_deg,
            max_range_m=cfg.camera_max_range_m,
        )
        return CameraContext(
            camera_id=camera_id,
            name=config.get("name") or f"Camera {camera_id:02d}",
            camera_model=model,
            night_vision=bool(config.get("night_vision", False)),
        )

    def _ensure_state(self, camera_id: int, config: dict) -> Optional[CameraState]:
        state = self._states.get(camera_id)
        if state is not None:
            return state
        if not self.model_manager:
            return None
        pipeline = self.model_manager.create_pipeline(
            self._build_context(camera_id, config)
        )
        state = CameraState(pipeline=pipeline, config=config)
        self._states[camera_id] = state
        return state

    def swarm_cameras(self) -> List[int]:
        return [cid for cid, st in self._states.items() if st.pipeline.swarm_active]

    def get_live_tracks(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        tracks: List[Dict[str, Any]] = []
        for camera_id, state in self._states.items():
            for track in state.pipeline.active_tracks():
                if kind and track.kind.value != kind:
                    continue
                tracks.append(self._track_payload(camera_id, track))
        return tracks

    async def _process_loop(self, camera_id: int):
        proc = self.camera_processors[camera_id]
        frame_count = 0
        skip = max(1, settings.FRAME_SKIP)
        pace = 1.0 / max(1, settings.PROCESSING_FPS)

        while proc.is_running:
            try:
                frame = await asyncio.to_thread(proc.get_frame)
                if frame is None:
                    if proc.consecutive_failures >= settings.STREAM_RECONNECT_ATTEMPTS:
                        await self._recover_stream(camera_id, proc)
                    else:
                        await asyncio.sleep(0.05)
                    continue

                frame_count += 1
                if frame_count % skip != 0:
                    await asyncio.sleep(pace / skip if proc.is_simulated else 0)
                    continue

                state = self._states.get(camera_id)
                await self._run_detections(
                    camera_id, frame, state.config if state else proc.config
                )
                await asyncio.sleep(pace if proc.is_simulated else 0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Camera {camera_id} processing error: {e}")
                await asyncio.sleep(1.0)

        logger.info(f"Camera {camera_id}: processing loop exited")

    async def _recover_stream(self, camera_id: int, proc: CameraProcessor):
        await self._set_camera_status(camera_id, "offline")
        await manager.broadcast_camera_status(str(camera_id), "offline")
        attempt = 0
        while proc.is_running:
            attempt += 1
            await asyncio.sleep(min(30.0, 2.0**attempt))
            reopened = await asyncio.to_thread(proc.reconnect)
            if reopened:
                await self._set_camera_status(camera_id, "online")
                await manager.broadcast_camera_status(str(camera_id), "online")
                logger.info(f"Camera {camera_id}: stream recovered")
                return

    async def _run_detections(self, camera_id: int, frame: np.ndarray, config: dict):
        if not self.model_manager:
            return
        state = self._ensure_state(camera_id, config)
        if state is None:
            return
        flags = DetectionFlags.from_mapping(config)
        try:
            result = await asyncio.to_thread(state.pipeline.process, frame, flags)
            await self._handle_result(camera_id, frame, result, state)
            self._touch_camera_throttled(camera_id, result.timestamp)
        except Exception as e:
            logger.error(f"Detection pipeline error camera {camera_id}: {e}")

    async def _handle_result(
        self,
        camera_id: int,
        frame: np.ndarray,
        result: FrameResult,
        state: CameraState,
    ):
        timestamp = result.timestamp
        ts_iso = timestamp.isoformat()

        if result.tracks:
            payloads = [
                self._track_payload(camera_id, t, ts_iso) for t in result.tracks
            ]
            await manager.broadcast_detection(
                {
                    "camera_id": camera_id,
                    "detections": payloads,
                    "swarm": result.swarm,
                    "inference_ms": round(result.inference_ms, 2),
                    "timestamp": ts_iso,
                }
            )

        for lost in result.lost_tracks:
            if lost.kind == ObjectKind.DRONE:
                await self._finalize_drone_row(camera_id, lost, state)

        for track in result.by_kind(ObjectKind.DRONE):
            await self._refresh_drone_row(camera_id, track, state)

        new_persons = result.new_by_kind(ObjectKind.PERSON)
        new_vehicles = result.new_by_kind(ObjectKind.VEHICLE)
        new_drones = result.new_by_kind(ObjectKind.DRONE)
        if not (
            new_persons
            or new_vehicles
            or new_drones
            or result.loitering
            or result.swarm_started
        ):
            return

        primary_kind = "drone" if new_drones else "person" if new_persons else "vehicle"
        snap = await self._save_snapshot(camera_id, frame, primary_kind, timestamp)
        camera_name = state.config.get("name") or f"Camera {camera_id:02d}"

        for track in new_drones:
            await self._handle_drone_track(
                camera_id, camera_name, track, snap, timestamp, state
            )

        detection_ids = await self._persist_detections(
            camera_id, new_persons + new_vehicles, snap
        )
        for kind_name, tracks in (("person", new_persons), ("vehicle", new_vehicles)):
            if tracks and self._gate.allow(f"{camera_id}:{kind_name}"):
                await self._create_alert(
                    camera_id, tracks, snap, timestamp, detection_ids
                )

        for track in result.loitering:
            await self._raise_loitering_alert(camera_id, track, snap, timestamp)

        if result.swarm_started and self._gate.allow(f"{camera_id}:swarm"):
            await self._raise_swarm_alert(
                camera_id,
                camera_name,
                result.by_kind(ObjectKind.DRONE),
                snap,
                timestamp,
            )

        fence = state.config.get("geofence")
        if fence:
            await self._check_geofence(
                camera_id,
                fence,
                new_persons + new_vehicles + new_drones,
                snap,
                timestamp,
            )

    def _track_payload(
        self, camera_id: int, track: Track, timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        data = track.to_dict()
        data["type"] = track.kind.value
        data["class"] = track.label
        data["camera_id"] = camera_id
        if timestamp:
            data["timestamp"] = timestamp
        return data

    def _analysis_for_alert(self, primary: Track) -> Dict[str, Any]:
        conf = primary.confidence
        return {
            "threat_level": "high" if conf > 0.8 else "medium",
            "summary": f"{primary.kind.value.title()} detected near perimeter.",
            "recommended_action": "Review camera feed immediately.",
        }

    async def _handle_drone_track(
        self,
        camera_id: int,
        camera_name: str,
        track: Track,
        snap: Optional[str],
        timestamp: datetime,
        state: CameraState,
    ):
        authorized = bool(track.attributes.get("is_authorized"))
        analysis: Dict[str, Any] = {
            "threat_level": "low" if authorized else "high",
            "risk_score": track.attributes.get("threat_score", 0.8),
            "summary": (
                "Authorized drone inside a permitted zone."
                if authorized
                else "Drone detected near perimeter."
            ),
            "recommended_action": (
                "Monitor." if authorized else "Alert security and track trajectory."
            ),
        }
        analyze = not authorized and self._gate.allow(f"{camera_id}:drone")
        if analyze and self.ai_analysis:
            analysis = await self.ai_analysis.analyze_drone_threat(
                confidence=track.confidence,
                drone_type=track.attributes.get("drone_type"),
                altitude_m=track.attributes.get("altitude_m"),
                speed_ms=track.attributes.get("speed_ms"),
                camera_name=camera_name,
                snapshot_path=snap,
            )

        row_id, detection_id = await self._persist_drone(
            camera_id, track, snap, analysis
        )
        if row_id is not None:
            state.drone_rows[track.track_id] = row_id
            state.drone_row_updated[track.track_id] = time.monotonic()

        await manager.broadcast_drone_detection(
            {
                "camera_id": camera_id,
                "track_id": track.track_id,
                "confidence": track.confidence,
                "drone_type": track.attributes.get("drone_type"),
                "altitude_m": track.attributes.get("altitude_m"),
                "speed_ms": track.attributes.get("speed_ms"),
                "threat_score": track.attributes.get("threat_score"),
                "is_authorized": authorized,
                "threat_level": analysis.get("threat_level", "high"),
                "summary": analysis.get("summary"),
                "timestamp": timestamp.isoformat(),
            }
        )

        if analyze:
            await self._store_alert(
                camera_id=camera_id,
                alert_type="drone_detected",
                title=f"Drone detected - Camera {camera_id:02d}",
                analysis=analysis,
                snap=snap,
                timestamp=timestamp,
                detection_payload=[self._track_payload(camera_id, track)],
                detection_id=detection_id,
                alert_kind="drone",
                camera_name=camera_name,
            )

    async def _persist_drone(
        self,
        camera_id: int,
        track: Track,
        snap: Optional[str],
        analysis: Dict[str, Any],
    ):
        from database.models import Detection, DroneDetection, ThreatLevel

        risk_map = {
            "low": ThreatLevel.CAUTION,
            "medium": ThreatLevel.WARNING,
            "high": ThreatLevel.THREAT,
            "critical": ThreatLevel.CRITICAL,
        }
        level = risk_map.get(
            analysis.get("threat_level", "medium"), ThreatLevel.WARNING
        )

        def write(db):
            det = Detection(
                camera_id=camera_id,
                detection_type="drone",
                confidence=track.confidence,
                bounding_box=self._bbox_json(track),
                snapshot_path=snap,
                threat_level=level,
                estimated_lat=track.attributes.get("latitude"),
                estimated_lng=track.attributes.get("longitude"),
                ai_analysis=analysis.get("summary"),
                risk_score=float(
                    analysis.get("risk_score")
                    or track.attributes.get("threat_score")
                    or 0.0
                ),
            )
            db.add(det)
            row = DroneDetection(
                camera_id=camera_id,
                confidence=track.confidence,
                bounding_box=self._bbox_json(track),
                snapshot_path=snap,
                drone_type=track.attributes.get("drone_type", "unknown"),
                estimated_altitude_m=track.attributes.get("altitude_m"),
                estimated_speed_ms=track.attributes.get("speed_ms"),
                trajectory_data=track.trajectory(),
                risk_level=level,
                ai_analysis=analysis.get("summary"),
                is_authorized=bool(track.attributes.get("is_authorized")),
            )
            db.add(row)
            db.flush()
            return row.id, det.id

        result = await self._db_call(write)
        return result if result else (None, None)

    @staticmethod
    def _bbox_json(track: Track) -> Dict[str, Any]:
        return {
            "bbox": track.bbox.to_list(),
            "track_id": track.track_id,
            "label": track.label,
        }

    async def _refresh_drone_row(
        self, camera_id: int, track: Track, state: CameraState
    ):
        row_id = state.drone_rows.get(track.track_id)
        if row_id is None:
            return
        now = time.monotonic()
        if (
            now - state.drone_row_updated.get(track.track_id, 0.0)
            < DRONE_ROW_REFRESH_SECONDS
        ):
            return
        state.drone_row_updated[track.track_id] = now
        await self._update_drone_row(row_id, track)

    async def _finalize_drone_row(
        self, camera_id: int, track: Track, state: CameraState
    ):
        row_id = state.drone_rows.pop(track.track_id, None)
        state.drone_row_updated.pop(track.track_id, None)
        if row_id is not None:
            await self._update_drone_row(row_id, track)

    async def _update_drone_row(self, row_id: int, track: Track):
        from database.models import DroneDetection

        def write(db):
            row = db.query(DroneDetection).filter(DroneDetection.id == row_id).first()
            if row is None:
                return
            row.trajectory_data = track.trajectory(60)
            row.estimated_speed_ms = track.attributes.get("speed_ms")
            row.estimated_altitude_m = track.attributes.get("altitude_m")
            row.is_authorized = bool(track.attributes.get("is_authorized"))

        await self._db_call(write)

    async def _create_alert(
        self,
        camera_id: int,
        tracks: List[Track],
        snap: Optional[str],
        timestamp: datetime,
        detection_ids: Optional[Dict[int, int]] = None,
    ):
        if not tracks:
            return
        primary = max(tracks, key=lambda t: t.confidence)
        payloads = [self._track_payload(camera_id, t) for t in tracks]
        camera_name = f"Camera {camera_id:02d}"

        analysis = self._analysis_for_alert(primary)
        if self.ai_analysis:
            analysis = await self.ai_analysis.analyze_threat(
                detection_type=primary.kind.value,
                confidence=primary.confidence,
                camera_name=camera_name,
                camera_location=None,
                additional_context={"detections": payloads},
                snapshot_path=snap,
            )

        await self._store_alert(
            camera_id=camera_id,
            alert_type=f"{primary.kind.value}_detected",
            title=f"{primary.kind.value.title()} detected - Camera {camera_id:02d}",
            analysis=analysis,
            snap=snap,
            timestamp=timestamp,
            detection_payload=payloads,
            detection_id=(detection_ids or {}).get(primary.track_id),
            alert_kind=primary.kind.value,
            camera_name=camera_name,
            latitude=primary.attributes.get("latitude"),
            longitude=primary.attributes.get("longitude"),
        )

    async def _persist_detections(
        self, camera_id: int, tracks: List[Track], snap: Optional[str]
    ) -> Dict[int, int]:
        if not tracks:
            return {}
        from database.models import Detection, ThreatLevel

        def write(db):
            rows = []
            for track in tracks:
                row = Detection(
                    camera_id=camera_id,
                    detection_type=track.kind.value,
                    confidence=track.confidence,
                    bounding_box=self._bbox_json(track),
                    snapshot_path=snap,
                    threat_level=ThreatLevel.CAUTION,
                    estimated_lat=track.attributes.get("latitude"),
                    estimated_lng=track.attributes.get("longitude"),
                    risk_score=float(track.confidence),
                )
                db.add(row)
                rows.append((track.track_id, row))
            db.flush()
            return {track_id: row.id for track_id, row in rows}

        return await self._db_call(write, {}) or {}

    async def _store_alert(
        self,
        camera_id: int,
        alert_type: str,
        title: str,
        analysis: Dict[str, Any],
        snap: Optional[str],
        timestamp: datetime,
        detection_payload: list,
        alert_kind: str,
        camera_name: str,
        detection_id: Optional[int] = None,
        geofence_id: Optional[int] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Optional[int]:
        if not self.db_session_factory:
            return None
        from database.models import Alert, AlertSeverity, AlertStatus, AlertType

        try:
            type_enum = AlertType(alert_type)
        except ValueError:
            type_enum = AlertType.UNKNOWN_OBJECT
        sev_map = {
            "low": AlertSeverity.LOW,
            "medium": AlertSeverity.MEDIUM,
            "high": AlertSeverity.HIGH,
            "critical": AlertSeverity.CRITICAL,
        }
        level = analysis.get("threat_level", "medium")
        severity = sev_map.get(level, AlertSeverity.MEDIUM)

        def write(db):
            alert = Alert(
                camera_id=camera_id,
                geofence_id=geofence_id,
                detection_id=detection_id,
                alert_type=type_enum,
                severity=severity,
                status=AlertStatus.ACTIVE,
                title=title,
                description=analysis.get("summary"),
                ai_summary=analysis.get("summary"),
                recommended_action=analysis.get("recommended_action"),
                latitude=latitude,
                longitude=longitude,
                snapshot_path=snap,
                detection_data={"detections": detection_payload},
            )
            db.add(alert)
            db.flush()
            return alert.id

        alert_id = await self._db_call(write)
        if alert_id is None:
            return None

        await manager.broadcast_alert(
            {
                "id": alert_id,
                "camera_id": camera_id,
                "type": alert_kind,
                "severity": level,
                "title": title,
                "summary": analysis.get("summary"),
                "timestamp": timestamp.isoformat(),
            }
        )
        self._notify(alert_type, level, title, analysis, camera_name, snap)
        return alert_id

    def _notify(
        self,
        alert_type: str,
        severity: str,
        title: str,
        analysis: Dict[str, Any],
        camera_name: str,
        snap: Optional[str],
    ):
        if not self.notifier:
            return
        coro = self.notifier.notify_alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=analysis.get("summary") or title,
            camera_name=camera_name,
            snapshot_path=snap,
            ai_summary=analysis.get("summary"),
            recommended_action=analysis.get("recommended_action"),
        )
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _raise_loitering_alert(
        self, camera_id: int, track: Track, snap: Optional[str], timestamp: datetime
    ):
        analysis = {
            "threat_level": "medium",
            "summary": (
                f"Person has remained in the same area for "
                f"{int(track.duration_seconds)} seconds."
            ),
            "recommended_action": "Review the feed and verify the person's purpose.",
        }
        await self._store_alert(
            camera_id=camera_id,
            alert_type="loitering",
            title=f"Loitering detected - Camera {camera_id:02d}",
            analysis=analysis,
            snap=snap,
            timestamp=timestamp,
            detection_payload=[self._track_payload(camera_id, track)],
            alert_kind="loitering",
            camera_name=f"Camera {camera_id:02d}",
            latitude=track.attributes.get("latitude"),
            longitude=track.attributes.get("longitude"),
        )

    async def _raise_swarm_alert(
        self,
        camera_id: int,
        camera_name: str,
        tracks: List[Track],
        snap: Optional[str],
        timestamp: datetime,
    ):
        payloads = [self._track_payload(camera_id, t) for t in tracks]
        analysis: Dict[str, Any] = {
            "threat_level": "critical",
            "summary": f"{len(tracks)} drones moving in a coordinated pattern.",
            "recommended_action": "Escalate to security lead and notify authorities.",
        }
        if self.ai_analysis:
            analysis = await self.ai_analysis.analyze_multiple_threats(
                payloads, camera_name
            )
            analysis["threat_level"] = "critical"
        await self._store_alert(
            camera_id=camera_id,
            alert_type="multiple_threats",
            title=f"Possible drone swarm - Camera {camera_id:02d}",
            analysis=analysis,
            snap=snap,
            timestamp=timestamp,
            detection_payload=payloads,
            alert_kind="drone",
            camera_name=camera_name,
        )

    async def _check_geofence(
        self,
        camera_id: int,
        fence: Dict[str, Any],
        tracks: List[Track],
        snap: Optional[str],
        timestamp: datetime,
    ):
        for track in tracks:
            lat = track.attributes.get("latitude")
            lng = track.attributes.get("longitude")
            if lat is None or lng is None:
                continue
            breaches = await self.perimeter.check_detection_against_geofences(
                lat, lng, track.kind.value, camera_id, [fence]
            )
            for breach in breaches:
                analysis = {
                    "threat_level": breach["severity"],
                    "summary": (
                        f"{track.kind.value.title()} entered geofence "
                        f"{breach['geofence_name']}."
                    ),
                    "recommended_action": "Dispatch security to the affected zone.",
                }
                await self._store_alert(
                    camera_id=camera_id,
                    alert_type="perimeter_breach",
                    title=f"Perimeter breach - {breach['geofence_name']}",
                    analysis=analysis,
                    snap=snap,
                    timestamp=timestamp,
                    detection_payload=[self._track_payload(camera_id, track)],
                    alert_kind=track.kind.value,
                    camera_name=f"Camera {camera_id:02d}",
                    geofence_id=breach["geofence_id"],
                    latitude=lat,
                    longitude=lng,
                )

    async def _save_snapshot(
        self,
        camera_id: int,
        frame: np.ndarray,
        label: str,
        timestamp: datetime,
    ) -> Optional[str]:
        try:
            Path(settings.SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
            name = (
                f"cam{camera_id:02d}_{label}_"
                f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            )
            path = str(Path(settings.SNAPSHOTS_DIR) / name)
            ok = await asyncio.to_thread(
                cv2.imwrite, path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            return path if ok else None
        except Exception as e:
            logger.error(f"Snapshot save error: {e}")
            return None

    async def _snapshot_cleanup_loop(self):
        while True:
            try:
                await asyncio.to_thread(self._purge_old_snapshots)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Snapshot cleanup error: {e}")
            await asyncio.sleep(MAX_SNAPSHOT_AGE_CHECK_SECONDS)

    def _purge_old_snapshots(self) -> int:
        directory = Path(settings.SNAPSHOTS_DIR)
        if not directory.is_dir():
            return 0
        cutoff = time.time() - settings.MAX_SNAPSHOT_AGE_DAYS * 86400
        removed = 0
        for path in directory.glob("*.jpg"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed

    async def _db_call(self, fn: Callable, default: Any = None) -> Any:
        if not self.db_session_factory:
            return default
        factory = self.db_session_factory

        def run():
            db = factory()
            try:
                result = fn(db)
                db.commit()
                return result
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        try:
            return await asyncio.to_thread(run)
        except Exception as e:
            logger.error(f"Database write error: {e}")
            return default

    async def _set_camera_status(self, camera_id: int, status: str):
        from database.models import Camera, CameraStatus

        status_map = {
            "online": CameraStatus.ONLINE,
            "offline": CameraStatus.OFFLINE,
            "disabled": CameraStatus.DISABLED,
            "initializing": CameraStatus.INITIALIZING,
            "error": CameraStatus.ERROR,
        }

        def write(db):
            cam = db.query(Camera).filter(Camera.id == camera_id).first()
            if cam:
                cam.status = status_map.get(status, CameraStatus.OFFLINE)

        await self._db_call(write)

    def _touch_camera_throttled(self, camera_id: int, timestamp: datetime):
        interval = timedelta(seconds=settings.TOUCH_CAMERA_INTERVAL_SECONDS)
        last = self._last_touch.get(camera_id)
        if last is not None and (timestamp - last) < interval:
            return
        self._last_touch[camera_id] = timestamp
        task = asyncio.create_task(self._touch_camera(camera_id, timestamp))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _touch_camera(self, camera_id: int, timestamp: datetime):
        from database.models import Camera

        def write(db):
            cam = db.query(Camera).filter(Camera.id == camera_id).first()
            if cam:
                cam.last_seen = timestamp

        await self._db_call(write)
