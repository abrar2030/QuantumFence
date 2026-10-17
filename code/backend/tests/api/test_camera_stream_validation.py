import pytest

pytestmark = pytest.mark.api


class TestStreamUrlValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "simulated",
            "rtsp://192.168.1.10:554/stream",
            "rtsps://host/live",
            "rtmp://host/app/key",
            "http://host/mjpeg",
            "https://host/hls.m3u8",
        ],
    )
    def test_allowed_urls_accepted(self, client, auth_headers, url):
        res = client.post(
            "/api/cameras",
            headers=auth_headers,
            json={"name": "Cam", "camera_type": "rtsp", "stream_url": url},
        )
        assert res.status_code == 201

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "/etc/passwd",
            "../../secret.mp4",
            "ftp://host/x",
            "gopher://x",
        ],
    )
    def test_local_and_unsupported_urls_rejected(self, client, auth_headers, url):
        res = client.post(
            "/api/cameras",
            headers=auth_headers,
            json={"name": "Cam", "camera_type": "rtsp", "stream_url": url},
        )
        assert res.status_code == 422

    def test_update_rejects_bad_url(self, client, auth_headers, make_camera):
        cam = make_camera()
        res = client.put(
            f"/api/cameras/{cam.id}",
            headers=auth_headers,
            json={"stream_url": "file:///etc/hosts"},
        )
        assert res.status_code == 422

    def test_update_accepts_valid_url(self, client, auth_headers, make_camera):
        cam = make_camera()
        res = client.put(
            f"/api/cameras/{cam.id}",
            headers=auth_headers,
            json={"stream_url": "rtsp://10.0.0.5/live"},
        )
        assert res.status_code == 200
