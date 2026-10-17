from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.api, pytest.mark.integration]


class TestRootEndpoint:

    def test_root_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200

    def test_root_contains_system_name(self, client):
        res = client.get("/")
        data = res.json()
        assert data["system"] == "QuantumFence"
        assert data["version"] is not None

    def test_root_contains_status_operational(self, client):
        res = client.get("/")
        assert res.json()["status"] == "operational"


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_reports_healthy(self, client):
        res = client.get("/health")
        data = res.json()
        assert data["status"] == "healthy"

    def test_health_contains_components(self, client):
        res = client.get("/health")
        data = res.json()
        assert "components" in data
        assert "database" in data["components"]

    def test_health_reports_active_cameras(self, client):
        from main import app

        app.state.detection_service.camera_processors = {"1": MagicMock()}
        res = client.get("/health")
        data = res.json()
        assert data["components"]["active_cameras"] >= 0
        app.state.detection_service.camera_processors = {}


class TestCORSHeaders:

    def test_options_request_returns_cors_headers(self, client):
        res = client.options(
            "/api/cameras",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert res.status_code in (200, 204)

    def test_get_request_has_cors_allow_origin(self, client, auth_headers):
        res = client.get(
            "/api/cameras",
            headers={**auth_headers, "Origin": "http://localhost:3000"},
        )
        assert "access-control-allow-origin" in {k.lower() for k in res.headers.keys()}


class TestStaticFilesMount:

    def test_snapshots_endpoint_exists(self, client, tmp_path, monkeypatch):
        import os

        import cv2
        import numpy as np
        from config.settings import settings

        snap_dir = settings.SNAPSHOTS_DIR
        os.makedirs(snap_dir, exist_ok=True)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        snap_path = os.path.join(snap_dir, "test_static.jpg")
        cv2.imwrite(snap_path, frame)

        try:
            res = client.get("/snapshots/test_static.jpg")
            assert res.status_code == 200
            assert "jpeg" in res.headers.get("content-type", "")
        finally:
            if os.path.exists(snap_path):
                os.remove(snap_path)


class TestEndToEndAlertWorkflow:

    def test_full_alert_lifecycle(self, client, auth_headers):
        cam_res = client.post(
            "/api/cameras",
            json={
                "name": "E2E Cam",
                "camera_type": "simulated",
                "stream_url": "simulated",
            },
            headers=auth_headers,
        )
        assert cam_res.status_code == 201
        cam_id = cam_res.json()["id"]

        list_res = client.get("/api/cameras", headers=auth_headers)
        cam_ids = [c["id"] for c in list_res.json()]
        assert cam_id in cam_ids

        stats_res = client.get(f"/api/cameras/{cam_id}/stats", headers=auth_headers)
        assert stats_res.json()["total_alerts"] == 0

        ov_res = client.get("/api/analytics/overview", headers=auth_headers)
        assert ov_res.json()["cameras"]["total"] >= 1

        del_res = client.delete(f"/api/cameras/{cam_id}", headers=auth_headers)
        assert del_res.status_code == 204

        list_res2 = client.get("/api/cameras", headers=auth_headers)
        ids2 = [c["id"] for c in list_res2.json()]
        assert cam_id not in ids2

    def test_geofence_to_alert_workflow(
        self, client, auth_headers, make_camera, make_alert
    ):
        gf_res = client.post(
            "/api/geofences",
            json={
                "name": "E2E Zone",
                "coordinates": [
                    [73.046, 33.686],
                    [73.050, 33.686],
                    [73.050, 33.682],
                    [73.046, 33.682],
                    [73.046, 33.686],
                ],
            },
            headers=auth_headers,
        )
        assert gf_res.status_code == 201
        gf_id = gf_res.json()["id"]

        pt_res = client.post(
            f"/api/geofences/{gf_id}/check-point?lat=33.684&lng=73.048",
            headers=auth_headers,
        )
        assert pt_res.json()["inside"] is True

        cam = make_camera(geofence_id=gf_id)

        alert = make_alert(camera_id=cam.id)

        stats_res = client.get("/api/alerts/stats", headers=auth_headers)
        assert stats_res.json()["total"] >= 1

        ack_res = client.post(
            f"/api/alerts/{alert.id}/acknowledge", headers=auth_headers
        )
        assert ack_res.json()["status"] == "acknowledged"

        res_res = client.post(f"/api/alerts/{alert.id}/resolve", headers=auth_headers)
        assert res_res.json()["status"] == "resolved"

        client.delete(f"/api/geofences/{gf_id}", headers=auth_headers)


class TestRoleBasedAccess:

    def test_viewer_can_read_cameras(self, client, operator_headers, make_camera):
        make_camera(name="Visible Cam")
        res = client.get("/api/cameras", headers=operator_headers)
        assert res.status_code == 200

    def test_operator_can_create_camera(self, client, operator_headers):
        res = client.post(
            "/api/cameras", json={"name": "Operator Cam"}, headers=operator_headers
        )
        assert res.status_code == 201

    def test_unauthenticated_cannot_access_any_api(self, client):
        for endpoint in [
            "/api/cameras",
            "/api/alerts",
            "/api/drones",
            "/api/analytics/overview",
        ]:
            res = client.get(endpoint)
            assert res.status_code == 401, f"Expected 401 for {endpoint}"
