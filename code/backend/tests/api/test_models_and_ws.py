from unittest.mock import AsyncMock, MagicMock

import pytest
from ai_models import ModelConfig, ModelManager

pytestmark = pytest.mark.api


class _NoCloseSession:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def close(self):
        pass


class _Fake:
    names = {0: "person"}

    def predict(self, frame, **kw):
        return []


@pytest.fixture
def loaded_manager(tmp_path):
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"x")
    mm = ModelManager(
        ModelConfig(
            yolo_weights=weights,
            drone_weights=tmp_path / "none.pt",
            warmup=False,
            auto_download=False,
            device="cpu",
        ),
        model_factory=lambda p: _Fake(),
    )
    mm.load()
    return mm


@pytest.fixture
def service(client, loaded_manager):
    from main import app

    ds = MagicMock()
    ds.model_manager = loaded_manager
    ds.model_health = MagicMock(return_value=loaded_manager.health())
    ds.camera_processors = {}
    ds.start_camera = AsyncMock()
    ds.stop_camera = AsyncMock()
    ds.reconfigure_camera = AsyncMock()
    ds.get_live_tracks = MagicMock(
        return_value=[{"track_id": 1, "type": "drone", "threat_score": 0.9}]
    )
    ds.swarm_cameras = MagicMock(return_value=[2])
    previous = app.state.detection_service
    app.state.detection_service = ds
    yield ds
    app.state.detection_service = previous


class TestModelRoutes:
    def test_status_requires_auth(self, client):
        assert client.get("/api/models/status").status_code == 401

    def test_status_reports_manager_state(self, client, service, auth_headers):
        res = client.get("/api/models/status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["loaded"] is True and data["health"] == "degraded"
        assert data["drone_mode"] == "proxy"
        assert "config" not in data

    def test_status_503_when_manager_missing(self, client, service, auth_headers):
        service.model_manager = None
        assert client.get("/api/models/status", headers=auth_headers).status_code == 503

    def test_status_503_when_service_missing(self, client, auth_headers):
        from main import app

        previous = app.state.detection_service
        app.state.detection_service = None
        try:
            assert (
                client.get("/api/models/status", headers=auth_headers).status_code
                == 503
            )
        finally:
            app.state.detection_service = previous

    def test_metrics(self, client, service, auth_headers, loaded_manager):
        import numpy as np

        loaded_manager.infer(np.zeros((64, 64, 3), dtype=np.uint8))
        res = client.get("/api/models/metrics", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["timings"]["inference"]["count"] == 1

    def test_config_admin_only(self, client, service, auth_headers, operator_headers):
        assert (
            client.get("/api/models/config", headers=operator_headers).status_code
            == 403
        )
        res = client.get("/api/models/config", headers=auth_headers)
        assert res.status_code == 200 and res.json()["image_size"] == 640

    def test_reload_admin_only_and_reloads(
        self, client, service, auth_headers, operator_headers, loaded_manager
    ):
        assert (
            client.post("/api/models/reload", headers=operator_headers).status_code
            == 403
        )
        before = loaded_manager.general_detector
        res = client.post("/api/models/reload", headers=auth_headers)
        assert res.status_code == 200 and res.json()["health"] == "degraded"
        assert loaded_manager.general_detector is not before


class TestDroneTrackRoute:
    def test_requires_auth(self, client):
        assert client.get("/api/drones/tracks").status_code == 401

    def test_returns_live_tracks(self, client, service, auth_headers):
        res = client.get("/api/drones/tracks", headers=auth_headers)
        data = res.json()
        assert res.status_code == 200
        assert data["active_tracks"] == 1 and data["swarm_detected"] is True
        assert data["swarm_cameras"] == [2]
        service.get_live_tracks.assert_called_with("drone")

    def test_empty_without_service(self, client, auth_headers):
        from main import app

        previous = app.state.detection_service
        app.state.detection_service = None
        try:
            data = client.get("/api/drones/tracks", headers=auth_headers).json()
            assert data == {
                "active_tracks": 0,
                "swarm_detected": False,
                "swarm_cameras": [],
                "tracks": [],
            }
        finally:
            app.state.detection_service = previous


class TestHealthAndCameraIntegration:
    def test_health_reports_model_health(self, client, service):
        data = client.get("/health").json()
        assert data["components"]["ai_models"] == "degraded"

    def test_health_unavailable_without_service(self, client):
        from main import app

        previous = app.state.detection_service
        app.state.detection_service = None
        try:
            assert (
                client.get("/health").json()["components"]["ai_models"] == "unavailable"
            )
        finally:
            app.state.detection_service = previous

    def test_camera_update_reconfigures_running_detection(
        self, client, service, make_camera, auth_headers
    ):
        cam = make_camera()
        res = client.put(
            f"/api/cameras/{cam.id}",
            headers=auth_headers,
            json={"detect_drones": False},
        )
        assert res.status_code == 200
        service.reconfigure_camera.assert_awaited_once()
        args = service.reconfigure_camera.await_args.args
        assert args[0] == cam.id and args[2]["detect_drones"] is False
        assert args[2]["latitude"] == cam.latitude

    def test_camera_create_passes_geometry_to_detection(
        self, client, service, auth_headers
    ):
        res = client.post(
            "/api/cameras",
            headers=auth_headers,
            json={
                "name": "Gate",
                "camera_type": "simulated",
                "latitude": 33.1,
                "longitude": 73.2,
                "direction_degrees": 90,
                "fov_degrees": 70,
            },
        )
        assert res.status_code in (200, 201)
        config = service.start_camera.await_args.args[2]
        assert config["latitude"] == 33.1 and config["direction_degrees"] == 90


class TestWebSocketAuth:
    @pytest.fixture
    def ws_db(self, monkeypatch, db_session):
        monkeypatch.setattr("main.SessionLocal", lambda: _NoCloseSession(db_session))

    def test_rejects_missing_token(self, client, ws_db):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/c1"):
                pass

    def test_rejects_invalid_token(self, client, ws_db):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/c1?token=garbage"):
                pass

    def test_ping_pong_with_valid_token(self, client, ws_db, admin_token):
        with client.websocket_connect(f"/ws/c1?token={admin_token}") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

    def test_subscribe_and_unsubscribe(self, client, ws_db, admin_token):
        with client.websocket_connect(f"/ws/c1?token={admin_token}") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json({"type": "subscribe_camera", "camera_id": 4})
            assert ws.receive_json() == {"type": "subscribed", "camera_id": 4}
            ws.send_json({"type": "unsubscribe_camera", "camera_id": 4})
            assert ws.receive_json() == {"type": "unsubscribed", "camera_id": 4}

    def test_start_camera_loads_config_from_database(
        self, client, service, ws_db, admin_token, make_camera
    ):
        cam = make_camera(stream_url="rtsp://cam/1")
        with client.websocket_connect(f"/ws/c1?token={admin_token}") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json(
                {
                    "type": "start_camera",
                    "camera_id": cam.id,
                    "stream_url": "file:///etc/passwd",
                }
            )
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
        service.start_camera.assert_awaited_once()
        args = service.start_camera.await_args.args
        assert args[0] == cam.id and args[1] == "rtsp://cam/1"

    def test_start_camera_ignores_unknown_camera(
        self, client, service, ws_db, admin_token
    ):
        with client.websocket_connect(f"/ws/c1?token={admin_token}") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json({"type": "start_camera", "camera_id": 9999})
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
        service.start_camera.assert_not_awaited()


class TestMetricsEndpoint:
    def test_metrics_exposed_without_auth_and_counts_requests(self, client):
        client.get("/health")
        res = client.get("/metrics")
        assert res.status_code == 200
        assert "http_request" in res.text
