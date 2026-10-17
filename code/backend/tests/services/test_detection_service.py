import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from ai_models import (
    BoundingBox,
    Detection,
    FrameResult,
    ModelConfig,
    ModelManager,
    MultiObjectTracker,
    ObjectKind,
)
from services.detection_service import (
    AlertGate,
    CameraProcessor,
    DetectionService,
    camera_runtime_config,
    geofence_to_dict,
)

pytestmark = pytest.mark.unit


class _NoCloseSession:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def close(self):
        pass

    def rollback(self):
        pass


class _FakeBoxes:
    def __init__(self, rows):
        arr = np.array(rows, dtype=np.float32).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = arr[:, :4], arr[:, 4], arr[:, 5]


class _FakeResult:
    def __init__(self, rows):
        self.boxes = _FakeBoxes(rows)
        self.names = {0: "person", 2: "car", 4: "airplane"}


class _FakeYolo:
    names = {0: "person", 2: "car", 4: "airplane"}

    def __init__(self, rows):
        self.rows = rows

    def predict(self, frame, **kw):
        classes = kw.get("classes")
        conf = kw.get("conf", 0.0)
        kept = [
            r
            for r in self.rows
            if (classes is None or int(r[5]) in classes) and r[4] >= conf
        ]
        return [_FakeResult(kept)]


def make_manager(tmp_path, rows):
    weights = tmp_path / "w"
    weights.mkdir(exist_ok=True)
    (weights / "yolov8n.pt").write_bytes(b"x")
    cfg = ModelConfig(
        yolo_weights=weights / "yolov8n.pt",
        drone_weights=weights / "drone.pt",
        warmup=False,
        auto_download=False,
        device="cpu",
        track_min_hits=1,
    )
    mm = ModelManager(cfg, model_factory=lambda p: _FakeYolo(rows))
    mm.load()
    return mm


@pytest.fixture
def ws():
    with patch("services.detection_service.manager") as mock_ws:
        mock_ws.broadcast_detection = AsyncMock()
        mock_ws.broadcast_alert = AsyncMock()
        mock_ws.broadcast_drone_detection = AsyncMock()
        mock_ws.broadcast_camera_status = AsyncMock()
        yield mock_ws


@pytest.fixture
def svc(tmp_path, db_session, ws, monkeypatch):
    monkeypatch.setattr(
        "services.detection_service.settings.SNAPSHOTS_DIR", str(tmp_path / "snaps")
    )
    s = DetectionService()
    s.model_manager = make_manager(
        tmp_path,
        [
            (100, 400, 160, 560, 0.9, 0),
            (500, 300, 700, 420, 0.85, 2),
            (600, 80, 640, 110, 0.8, 4),
        ],
    )
    s.ai_analysis = None
    s.notifier = None
    s.db_session_factory = lambda: _NoCloseSession(db_session)
    return s


def build_result(kind, count=1, camera_id=1, **attrs):
    tracker = MultiObjectTracker(kind, min_hits=1)
    now = datetime.now(timezone.utc)
    dets = [
        Detection(
            kind,
            kind.value,
            0.9,
            BoundingBox(100 + i * 200, 300, 40, 90),
            attributes=attrs,
        )
        for i in range(count)
    ]
    update = tracker.update(dets, now)
    return FrameResult(
        camera_id=camera_id,
        timestamp=now,
        frame_shape=(720, 1280, 3),
        tracks=list(update.matched),
        new_tracks=list(update.new_confirmed),
    )


def state_for(svc, camera_id=1, config=None):
    return svc._ensure_state(camera_id, config or {"name": "Gate"})


class TestCameraProcessor:
    def test_simulated_and_empty_urls_skip_cv2(self):
        for url in ("simulated", ""):
            proc = CameraProcessor(1, url, {})
            assert proc._open_stream() is True
            assert proc.cap is None and proc.is_simulated

    def test_test_frame_shape_and_counter(self):
        proc = CameraProcessor(1, "simulated", {})
        frame = proc._generate_test_frame()
        assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
        assert proc.frame_count == 1

    def test_simulated_get_frame(self):
        assert CameraProcessor(1, "simulated", {}).get_frame().shape == (720, 1280, 3)

    def test_real_stream_failure_returns_none_and_counts(self):
        proc = CameraProcessor(1, "rtsp://invalid", {})
        assert proc.get_frame() is None
        assert proc.get_frame() is None
        assert proc.consecutive_failures == 2

    def test_successful_read_resets_failures(self):
        proc = CameraProcessor(1, "rtsp://x", {})
        proc.consecutive_failures = 4
        proc.cap = MagicMock()
        proc.cap.isOpened.return_value = True
        proc.cap.read.return_value = (True, np.zeros((4, 4, 3), dtype=np.uint8))
        assert proc.get_frame() is not None
        assert proc.consecutive_failures == 0 and proc.frame_count == 1

    def test_open_stream_failure(self):
        with patch("services.detection_service.cv2.VideoCapture") as vc:
            vc.return_value.isOpened.return_value = False
            proc = CameraProcessor(1, "rtsp://bad", {})
            assert proc._open_stream() is False and proc.cap is None

    def test_open_stream_success_and_reconnect(self):
        with patch("services.detection_service.cv2.VideoCapture") as vc:
            vc.return_value.isOpened.return_value = True
            proc = CameraProcessor(1, "rtsp://ok", {})
            assert proc._open_stream() is True
            assert proc.reconnect() is True
            assert vc.return_value.release.called

    def test_stop_releases_capture(self):
        proc = CameraProcessor(1, "rtsp://x", {})
        proc.is_running = True
        proc.cap = MagicMock()
        cap = proc.cap
        proc.stop()
        cap.release.assert_called_once()
        assert proc.is_running is False and proc.cap is None

    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        proc = CameraProcessor(1, "simulated", {})
        await proc.start()
        assert proc.is_running


class TestAlertGate:
    def test_blocks_within_cooldown_and_allows_after(self):
        gate = AlertGate(30)
        assert gate.allow("a", now=100)
        assert not gate.allow("a", now=110)
        assert gate.allow("a", now=131)

    def test_keys_independent(self):
        gate = AlertGate(30)
        assert gate.allow("a", now=0) and gate.allow("b", now=1)

    def test_clear(self):
        gate = AlertGate(30)
        gate.allow("a", now=0)
        gate.clear()
        assert gate.allow("a", now=1)


class TestConfigBuilders:
    def test_camera_runtime_config(self, make_camera, make_geofence):
        fence = make_geofence()
        cam = make_camera(geofence_id=fence.id, night_vision=True)
        cfg = camera_runtime_config(cam)
        assert cfg["detect_drones"] is True and cfg["night_vision"] is True
        assert cfg["latitude"] == cam.latitude and cfg["geofence"]["id"] == fence.id

    def test_runtime_config_without_geofence(self, make_camera):
        assert camera_runtime_config(make_camera())["geofence"] is None

    def test_geofence_to_dict_none(self):
        assert geofence_to_dict(None) is None


class TestInitAndLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_loads_manager(self):
        s = DetectionService()
        mm = MagicMock()
        mm.load_all_models = AsyncMock()
        mm.health.return_value = "loaded"
        with patch("services.detection_service.ModelManager", return_value=mm), patch(
            "services.detection_service.AIAnalysisService"
        ), patch("services.detection_service.NotificationService"), patch(
            "services.detection_service.settings.MAX_SNAPSHOT_AGE_DAYS", 0
        ):
            await s.initialize()
        mm.load_all_models.assert_awaited_once()
        assert s.model_manager is mm and s.model_health() == "loaded"

    @pytest.mark.asyncio
    async def test_initialize_failure_leaves_service_usable(self):
        s = DetectionService()
        with patch(
            "services.detection_service.ModelManager", side_effect=ImportError("x")
        ):
            await s.initialize()
        assert s.model_manager is None and s.model_health() == "unavailable"

    @pytest.mark.asyncio
    async def test_start_and_stop_camera(self, svc, ws):
        svc.db_session_factory = None
        await svc.start_camera(1, "simulated", {"detect_persons": True})
        assert (
            1 in svc.camera_processors
            and 1 in svc._processing_tasks
            and 1 in svc._states
        )
        ws.broadcast_camera_status.assert_any_await("1", "online")
        await svc.stop_camera(1)
        assert 1 not in svc.camera_processors and 1 not in svc._states
        ws.broadcast_camera_status.assert_any_await("1", "disabled")

    @pytest.mark.asyncio
    async def test_restart_replaces_task(self, svc):
        svc.db_session_factory = None
        await svc.start_camera(1, "simulated", {})
        first = svc._processing_tasks[1]
        await svc.start_camera(1, "simulated", {})
        assert svc._processing_tasks[1] is not first
        await svc.stop_camera(1)

    @pytest.mark.asyncio
    async def test_stop_unknown_camera_is_safe(self, svc):
        svc.db_session_factory = None
        await svc.stop_camera(404)

    @pytest.mark.asyncio
    async def test_reconfigure_updates_flags_without_restart(self, svc):
        svc.db_session_factory = None
        await svc.start_camera(1, "simulated", {"detect_persons": True})
        task = svc._processing_tasks[1]
        await svc.reconfigure_camera(
            1,
            "simulated",
            {"detect_persons": False, "latitude": 10.0, "longitude": 20.0},
        )
        assert svc._processing_tasks[1] is task
        assert svc._states[1].config["detect_persons"] is False
        assert svc._states[1].pipeline.context.camera_model.latitude == 10.0
        await svc.stop_camera(1)

    @pytest.mark.asyncio
    async def test_reconfigure_restarts_when_stream_changes(self, svc):
        svc.db_session_factory = None
        await svc.start_camera(1, "simulated", {})
        task = svc._processing_tasks[1]
        with patch.object(svc, "start_camera", new=AsyncMock()) as start:
            await svc.reconfigure_camera(1, "rtsp://other", {})
        start.assert_awaited_once()
        await svc.stop_camera(1)
        assert task is not None

    @pytest.mark.asyncio
    async def test_reconfigure_unknown_camera_noop(self, svc):
        await svc.reconfigure_camera(5, "simulated", {})

    @pytest.mark.asyncio
    async def test_shutdown_stops_everything_and_unloads(self, svc):
        svc.db_session_factory = None
        await svc.start_camera(1, "simulated", {})
        await svc.start_camera(2, "simulated", {})
        await svc.shutdown()
        assert svc.camera_processors == {}
        assert not svc.model_manager.models_loaded

    @pytest.mark.asyncio
    async def test_process_loop_runs_detections_for_simulated_camera(self, svc):
        svc.db_session_factory = None
        calls = []

        async def fake_run(camera_id, frame, config):
            calls.append(camera_id)

        svc._run_detections = fake_run
        with patch("services.detection_service.settings.FRAME_SKIP", 1), patch(
            "services.detection_service.settings.PROCESSING_FPS", 200
        ):
            await svc.start_camera(7, "simulated", {})
            await asyncio.sleep(0.15)
            await svc.stop_camera(7)
        assert calls and set(calls) == {7}

    @pytest.mark.asyncio
    async def test_recover_stream_reopens_and_marks_online(self, svc, ws):
        svc.db_session_factory = None
        proc = CameraProcessor(1, "rtsp://x", {})
        proc.is_running = True
        proc.reconnect = MagicMock(side_effect=[False, True])
        with patch("services.detection_service.asyncio.sleep", new=AsyncMock()):
            await svc._recover_stream(1, proc)
        assert proc.reconnect.call_count == 2
        ws.broadcast_camera_status.assert_any_await("1", "offline")
        ws.broadcast_camera_status.assert_any_await("1", "online")


class TestRunDetections:
    @pytest.mark.asyncio
    async def test_no_model_manager_is_noop(self, ws, blank_frame):
        s = DetectionService()
        await s._run_detections(1, blank_frame, {})
        ws.broadcast_detection.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_error_is_swallowed(self, svc, blank_frame):
        state = state_for(svc)
        state.pipeline.process = MagicMock(side_effect=RuntimeError("boom"))
        await svc._run_detections(1, blank_frame, {"name": "Gate"})

    @pytest.mark.asyncio
    async def test_detections_broadcast_with_track_fields(self, svc, ws, blank_frame):
        svc.db_session_factory = None
        for _ in range(3):
            await svc._run_detections(1, blank_frame, {"name": "Gate"})
        payload = ws.broadcast_detection.await_args.args[0]
        assert payload["camera_id"] == 1
        det = payload["detections"][0]
        for key in ("type", "confidence", "bbox", "track_id", "timestamp", "class"):
            assert key in det
        assert {d["type"] for d in payload["detections"]} == {
            "person",
            "vehicle",
            "drone",
        }

    @pytest.mark.asyncio
    async def test_flags_disable_detection_types(self, svc, ws, blank_frame):
        svc.db_session_factory = None
        cfg = {"detect_persons": False, "detect_vehicles": False, "detect_drones": True}
        for _ in range(3):
            await svc._run_detections(1, blank_frame, cfg)
        payload = ws.broadcast_detection.await_args.args[0]
        assert {d["type"] for d in payload["detections"]} == {"drone"}

    @pytest.mark.asyncio
    async def test_alerts_and_rows_created_once_per_track(
        self, svc, ws, db_session, make_camera, blank_frame
    ):
        from database.models import Alert
        from database.models import Detection as DetRow
        from database.models import DroneDetection

        cam = make_camera()
        cfg = {"name": cam.name, "latitude": cam.latitude, "longitude": cam.longitude}
        for _ in range(6):
            await svc._run_detections(cam.id, blank_frame, cfg)
        assert (
            db_session.query(DetRow).filter(DetRow.detection_type == "person").count()
            == 1
        )
        assert (
            db_session.query(DetRow).filter(DetRow.detection_type == "vehicle").count()
            == 1
        )
        assert db_session.query(DroneDetection).count() == 1
        kinds = {a.alert_type.value for a in db_session.query(Alert).all()}
        assert kinds == {"person_detected", "vehicle_detected", "drone_detected"}
        person_row = (
            db_session.query(DetRow).filter(DetRow.detection_type == "person").first()
        )
        assert person_row.estimated_lat is not None

    @pytest.mark.asyncio
    async def test_snapshot_written_on_event(
        self, svc, db_session, make_camera, blank_frame, tmp_path
    ):
        cam = make_camera()
        for _ in range(3):
            await svc._run_detections(cam.id, blank_frame, {"name": "x"})
        assert list((tmp_path / "snaps").glob("*.jpg"))


class TestHandleResult:
    @pytest.mark.asyncio
    async def test_person_alert_persisted_with_geo_and_broadcast(
        self, svc, ws, db_session, make_camera, blank_frame
    ):
        from database.models import Alert

        cam = make_camera()
        result = build_result(
            ObjectKind.PERSON, camera_id=cam.id, latitude=33.1, longitude=73.2
        )
        await svc._handle_result(cam.id, blank_frame, result, state_for(svc, cam.id))
        alert = db_session.query(Alert).first()
        assert alert.latitude == 33.1 and alert.detection_id is not None
        assert alert.detection_data["detections"][0]["type"] == "person"
        ws.broadcast_alert.assert_awaited()

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_repeat_alerts_but_keeps_rows(
        self, svc, db_session, make_camera, blank_frame
    ):
        from database.models import Alert
        from database.models import Detection as DetRow

        cam = make_camera()
        state = state_for(svc, cam.id)
        for _ in range(3):
            await svc._handle_result(
                cam.id,
                blank_frame,
                build_result(ObjectKind.PERSON, camera_id=cam.id),
                state,
            )
        assert db_session.query(Alert).count() == 1
        assert db_session.query(DetRow).count() == 3

    @pytest.mark.asyncio
    async def test_alert_severity_follows_ai_analysis(
        self, svc, db_session, make_camera, blank_frame
    ):
        from database.models import Alert, AlertSeverity

        cam = make_camera()
        svc.ai_analysis = MagicMock()
        svc.ai_analysis.analyze_threat = AsyncMock(
            return_value={
                "threat_level": "critical",
                "summary": "bad",
                "recommended_action": "act",
                "risk_score": 0.99,
            }
        )
        await svc._handle_result(
            cam.id,
            blank_frame,
            build_result(ObjectKind.PERSON, camera_id=cam.id),
            state_for(svc, cam.id),
        )
        alert = db_session.query(Alert).first()
        assert alert.severity == AlertSeverity.CRITICAL and alert.ai_summary == "bad"

    @pytest.mark.asyncio
    async def test_unknown_threat_level_defaults_to_medium(
        self, svc, db_session, make_camera, blank_frame
    ):
        from database.models import Alert, AlertSeverity

        cam = make_camera()
        svc.ai_analysis = MagicMock()
        svc.ai_analysis.analyze_threat = AsyncMock(
            return_value={"threat_level": "weird", "summary": "s"}
        )
        await svc._handle_result(
            cam.id,
            blank_frame,
            build_result(ObjectKind.VEHICLE, camera_id=cam.id),
            state_for(svc, cam.id),
        )
        assert db_session.query(Alert).first().severity == AlertSeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_no_db_factory_means_no_alert_broadcast(self, svc, ws, blank_frame):
        svc.db_session_factory = None
        await svc._handle_result(
            1, blank_frame, build_result(ObjectKind.PERSON), state_for(svc)
        )
        ws.broadcast_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_drone_track_persists_row_and_alert(
        self, svc, ws, db_session, make_camera, blank_frame
    ):
        from database.models import Alert, DroneDetection

        cam = make_camera()
        result = build_result(
            ObjectKind.DRONE,
            camera_id=cam.id,
            drone_type="quadcopter",
            altitude_m=40.0,
            threat_score=0.8,
            is_authorized=False,
        )
        await svc._handle_result(cam.id, blank_frame, result, state_for(svc, cam.id))
        row = db_session.query(DroneDetection).first()
        assert row.drone_type == "quadcopter" and row.estimated_altitude_m == 40.0
        assert (
            db_session.query(Alert).filter(Alert.alert_type == "drone_detected").count()
            == 1
        )
        payload = ws.broadcast_drone_detection.await_args.args[0]
        assert (
            payload["track_id"] == result.new_tracks[0].track_id
            and payload["is_authorized"] is False
        )

    @pytest.mark.asyncio
    async def test_authorized_drone_stored_without_alert(
        self, svc, ws, db_session, make_camera, blank_frame
    ):
        from database.models import Alert, DroneDetection

        cam = make_camera()
        result = build_result(
            ObjectKind.DRONE,
            camera_id=cam.id,
            drone_type="quadcopter",
            is_authorized=True,
        )
        await svc._handle_result(cam.id, blank_frame, result, state_for(svc, cam.id))
        assert db_session.query(DroneDetection).first().is_authorized is True
        assert db_session.query(Alert).count() == 0

    @pytest.mark.asyncio
    async def test_drone_ai_analysis_used_when_available(
        self, svc, db_session, make_camera, blank_frame
    ):
        from database.models import DroneDetection

        cam = make_camera()
        svc.ai_analysis = MagicMock()
        svc.ai_analysis.analyze_drone_threat = AsyncMock(
            return_value={
                "threat_level": "critical",
                "summary": "hostile",
                "recommended_action": "x",
                "risk_score": 0.95,
            }
        )
        result = build_result(
            ObjectKind.DRONE, camera_id=cam.id, drone_type="quadcopter"
        )
        await svc._handle_result(cam.id, blank_frame, result, state_for(svc, cam.id))
        svc.ai_analysis.analyze_drone_threat.assert_awaited_once()
        assert db_session.query(DroneDetection).first().ai_analysis == "hostile"

    @pytest.mark.asyncio
    async def test_drone_row_refreshed_and_finalized(
        self, svc, db_session, make_camera, blank_frame
    ):
        from database.models import DroneDetection

        cam = make_camera()
        state = state_for(svc, cam.id)
        result = build_result(
            ObjectKind.DRONE, camera_id=cam.id, drone_type="quadcopter"
        )
        await svc._handle_result(cam.id, blank_frame, result, state)
        track = result.tracks[0]
        assert track.track_id in state.drone_rows
        track.attributes["speed_ms"] = 12.5
        state.drone_row_updated[track.track_id] = 0.0
        await svc._refresh_drone_row(cam.id, track, state)
        db_session.expire_all()
        assert db_session.query(DroneDetection).first().estimated_speed_ms == 12.5
        await svc._finalize_drone_row(cam.id, track, state)
        assert track.track_id not in state.drone_rows

    @pytest.mark.asyncio
    async def test_refresh_throttled(self, svc, make_camera, blank_frame):
        cam = make_camera()
        state = state_for(svc, cam.id)
        result = build_result(ObjectKind.DRONE, camera_id=cam.id)
        await svc._handle_result(cam.id, blank_frame, result, state)
        track = result.tracks[0]
        with patch.object(svc, "_update_drone_row", new=AsyncMock()) as upd:
            await svc._refresh_drone_row(cam.id, track, state)
        upd.assert_not_called()

    @pytest.mark.asyncio
    async def test_loitering_alert(self, svc, db_session, make_camera, blank_frame):
        from database.models import Alert

        cam = make_camera()
        result = build_result(ObjectKind.PERSON, camera_id=cam.id)
        result.loitering = list(result.tracks)
        result.new_tracks = []
        await svc._handle_result(cam.id, blank_frame, result, state_for(svc, cam.id))
        assert (
            db_session.query(Alert).filter(Alert.alert_type == "loitering").count() == 1
        )

    @pytest.mark.asyncio
    async def test_swarm_alert_critical_and_gated(
        self, svc, db_session, make_camera, blank_frame
    ):
        from database.models import Alert, AlertSeverity

        cam = make_camera()
        state = state_for(svc, cam.id)
        for _ in range(2):
            result = build_result(
                ObjectKind.DRONE, count=3, camera_id=cam.id, is_authorized=True
            )
            result.new_tracks = []
            result.swarm = result.swarm_started = True
            await svc._handle_result(cam.id, blank_frame, result, state)
        alerts = (
            db_session.query(Alert).filter(Alert.alert_type == "multiple_threats").all()
        )
        assert len(alerts) == 1 and alerts[0].severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_geofence_breach_alert(
        self, svc, db_session, make_camera, make_geofence, blank_frame
    ):
        from database.models import Alert

        fence = make_geofence()
        cam = make_camera(geofence_id=fence.id)
        config = camera_runtime_config(cam)
        result = build_result(
            ObjectKind.PERSON, camera_id=cam.id, latitude=33.684, longitude=73.048
        )
        state = state_for(svc, cam.id, config)
        await svc._handle_result(cam.id, blank_frame, result, state)
        breach = (
            db_session.query(Alert).filter(Alert.alert_type == "perimeter_breach").all()
        )
        assert len(breach) == 1 and breach[0].geofence_id == fence.id

    @pytest.mark.asyncio
    async def test_geofence_ignores_tracks_without_position(
        self, svc, db_session, make_camera, make_geofence, blank_frame
    ):
        from database.models import Alert

        fence = make_geofence()
        cam = make_camera(geofence_id=fence.id)
        result = build_result(ObjectKind.PERSON, camera_id=cam.id)
        await svc._handle_result(
            cam.id,
            blank_frame,
            result,
            state_for(svc, cam.id, camera_runtime_config(cam)),
        )
        assert (
            db_session.query(Alert)
            .filter(Alert.alert_type == "perimeter_breach")
            .count()
            == 0
        )

    @pytest.mark.asyncio
    async def test_high_severity_alert_triggers_notifier(
        self, svc, make_camera, blank_frame
    ):
        cam = make_camera()
        svc.notifier = MagicMock()
        svc.notifier.notify_alert = AsyncMock()
        svc.ai_analysis = MagicMock()
        svc.ai_analysis.analyze_threat = AsyncMock(
            return_value={"threat_level": "high", "summary": "s"}
        )
        await svc._handle_result(
            cam.id,
            blank_frame,
            build_result(ObjectKind.PERSON, camera_id=cam.id),
            state_for(svc, cam.id),
        )
        await asyncio.sleep(0)
        svc.notifier.notify_alert.assert_called_once()
        assert svc.notifier.notify_alert.call_args.kwargs["severity"] == "high"

    @pytest.mark.asyncio
    async def test_nothing_new_means_no_events(self, svc, ws, db_session, blank_frame):
        from database.models import Alert

        result = build_result(ObjectKind.PERSON)
        result.new_tracks = []
        await svc._handle_result(1, blank_frame, result, state_for(svc))
        assert db_session.query(Alert).count() == 0
        ws.broadcast_detection.assert_awaited()


class TestLiveTracksAndSnapshots:
    @pytest.mark.asyncio
    async def test_get_live_tracks_filters_by_kind(self, svc, blank_frame):
        svc.db_session_factory = None
        for _ in range(3):
            await svc._run_detections(1, blank_frame, {"name": "x"})
        assert {t["type"] for t in svc.get_live_tracks()} == {
            "person",
            "vehicle",
            "drone",
        }
        assert {t["type"] for t in svc.get_live_tracks("drone")} == {"drone"}
        assert svc.get_live_tracks()[0]["camera_id"] == 1
        assert svc.swarm_cameras() == []

    @pytest.mark.asyncio
    async def test_save_snapshot_writes_file(self, svc, tmp_path, blank_frame):
        path = await svc._save_snapshot(
            3, blank_frame, "person", datetime.now(timezone.utc)
        )
        assert path and Path(path).exists() and "cam03_person_" in path

    @pytest.mark.asyncio
    async def test_snapshot_names_unique_within_second(self, svc, blank_frame):
        ts = datetime.now(timezone.utc)
        a = await svc._save_snapshot(1, blank_frame, "person", ts)
        b = await svc._save_snapshot(
            1, blank_frame, "person", ts + timedelta(milliseconds=5)
        )
        assert a != b

    @pytest.mark.asyncio
    async def test_save_snapshot_returns_none_on_error(self, svc, blank_frame):
        with patch(
            "services.detection_service.Path.mkdir", side_effect=OSError("denied")
        ):
            assert (
                await svc._save_snapshot(
                    1, blank_frame, "person", datetime.now(timezone.utc)
                )
                is None
            )

    @pytest.mark.asyncio
    async def test_save_snapshot_returns_none_when_write_fails(self, svc, blank_frame):
        with patch("services.detection_service.cv2.imwrite", return_value=False):
            assert (
                await svc._save_snapshot(
                    1, blank_frame, "person", datetime.now(timezone.utc)
                )
                is None
            )

    def test_purge_old_snapshots(self, svc, tmp_path):
        directory = tmp_path / "snaps"
        directory.mkdir()
        old = directory / "old.jpg"
        fresh = directory / "fresh.jpg"
        old.write_bytes(b"x")
        fresh.write_bytes(b"x")
        past = time.time() - 40 * 86400
        os.utime(old, (past, past))
        with patch("services.detection_service.settings.MAX_SNAPSHOT_AGE_DAYS", 30):
            assert svc._purge_old_snapshots() == 1
        assert fresh.exists() and not old.exists()

    def test_purge_missing_directory(self, svc, tmp_path):
        with patch(
            "services.detection_service.settings.SNAPSHOTS_DIR", str(tmp_path / "none")
        ):
            assert svc._purge_old_snapshots() == 0


class TestDbHelpers:
    @pytest.mark.asyncio
    async def test_set_camera_status(self, svc, db_session, make_camera):
        from database.models import CameraStatus

        cam = make_camera(status=CameraStatus.OFFLINE)
        await svc._set_camera_status(cam.id, "online")
        db_session.refresh(cam)
        assert cam.status == CameraStatus.ONLINE
        await svc._set_camera_status(cam.id, "disabled")
        db_session.refresh(cam)
        assert cam.status == CameraStatus.DISABLED

    @pytest.mark.asyncio
    async def test_unknown_camera_is_safe(self, svc):
        await svc._set_camera_status(99999, "online")
        await svc._touch_camera(99999, datetime.now(timezone.utc))

    @pytest.mark.asyncio
    async def test_touch_camera_updates_last_seen(self, svc, db_session, make_camera):
        cam = make_camera()
        await svc._touch_camera(cam.id, datetime(2024, 1, 1, 12, 0, 0))
        db_session.refresh(cam)
        assert cam.last_seen is not None

    @pytest.mark.asyncio
    async def test_touch_throttled(self, svc):
        with patch.object(svc, "_touch_camera", new=AsyncMock()) as touch:
            now = datetime.now(timezone.utc)
            svc._touch_camera_throttled(1, now)
            svc._touch_camera_throttled(1, now + timedelta(seconds=1))
            await asyncio.sleep(0)
        assert touch.call_count == 1

    @pytest.mark.asyncio
    async def test_no_factory_skips_writes(self):
        s = DetectionService()
        s.db_session_factory = None
        await s._set_camera_status(1, "online")
        await s._touch_camera(1, datetime.now(timezone.utc))
        assert await s._db_call(lambda db: 1, default="d") == "d"

    @pytest.mark.asyncio
    async def test_db_error_returns_default(self, svc):
        def explode(db):
            raise RuntimeError("db down")

        assert await svc._db_call(explode, default="fallback") == "fallback"
