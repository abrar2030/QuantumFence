import pytest

pytestmark = pytest.mark.unit


class TestSettings:

    def test_settings_loads_without_error(self):
        from config.settings import settings

        assert settings is not None

    def test_default_database_url_contains_sqlite_in_test(self):
        from config.settings import settings

        assert (
            "sqlite" in settings.DATABASE_URL or "postgresql" in settings.DATABASE_URL
        )

    def test_jwt_algorithm_is_hs256(self):
        from config.settings import settings

        assert settings.ALGORITHM == "HS256"

    def test_access_token_expire_positive(self):
        from config.settings import settings

        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES > 0

    def test_detection_confidence_between_0_and_1(self):
        from config.settings import settings

        assert 0.0 < settings.DETECTION_CONFIDENCE < 1.0

    def test_frame_skip_at_least_1(self):
        from config.settings import settings

        assert settings.FRAME_SKIP >= 1

    def test_max_cameras_positive(self):
        from config.settings import settings

        assert settings.MAX_CAMERAS > 0

    def test_allowed_origins_is_list(self):
        from config.settings import settings

        assert isinstance(settings.ALLOWED_ORIGINS, list)

    def test_snapshots_dir_string(self):
        from config.settings import settings

        assert isinstance(settings.SNAPSHOTS_DIR, str)
        assert len(settings.SNAPSHOTS_DIR) > 0

    def test_secret_key_non_empty(self):
        from config.settings import settings

        assert len(settings.SECRET_KEY) > 10

    def test_ai_confidence_threshold_between_0_and_1(self):
        from config.settings import settings

        assert 0.0 < settings.AI_CONFIDENCE_THRESHOLD <= 1.0


class TestPerimeterEdgeCases:

    @pytest.fixture
    def svc(self):
        from services.perimeter_service import PerimeterService

        return PerimeterService()

    @pytest.mark.parametrize(
        "lat,lng,inside",
        [
            (0.5, 0.5, True),
            (0.1, 0.1, True),
            (0.9, 0.9, True),
            (2.0, 0.5, False),
            (-1.0, 0.5, False),
            (0.5, 2.0, False),
            (0.5, -1.0, False),
        ],
    )
    def test_point_in_unit_square(self, svc, lat, lng, inside):
        coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        assert svc._point_in_polygon(lat, lng, coords) is inside

    @pytest.mark.parametrize(
        "det_type,expected_sev",
        [
            ("drone", "high"),
            ("vehicle", "high"),
            ("person", "medium"),
            ("unknown", "medium"),
        ],
    )
    def test_severity_by_detection_type(self, svc, det_type, expected_sev):
        fence = {"name": "Regular Zone"}
        assert svc._calculate_severity(det_type, fence) == expected_sev

    @pytest.mark.parametrize(
        "lat1,lng1,lat2,lng2,expected_km",
        [
            (51.5074, -0.1278, 48.8566, 2.3522, 340),
            (33.6844, 73.0479, 33.6844, 73.0479, 0),
            (40.7128, -74.0060, 34.0522, -118.2437, 3940),
        ],
    )
    def test_haversine_distance_parametrized(
        self, svc, lat1, lng1, lat2, lng2, expected_km
    ):
        dist_m = svc._haversine_distance(lat1, lng1, lat2, lng2)
        dist_km = dist_m / 1000
        if expected_km == 0:
            assert dist_km < 0.001
        else:
            assert abs(dist_km - expected_km) / expected_km < 0.05


class TestAIAnalysisEdgeCases:

    @pytest.fixture
    def svc(self):
        from services.ai_analysis_service import AIAnalysisService

        s = AIAnalysisService()
        s.enabled = False
        return s

    @pytest.mark.parametrize(
        "det_type,confidence,expected_level",
        [
            ("drone", 0.95, "high"),
            ("drone", 0.70, "medium"),
            ("drone", 0.40, "low"),
            ("person", 0.91, "high"),
            ("person", 0.65, "medium"),
            ("vehicle", 0.50, "low"),
            ("unknown", 0.85, "high"),
        ],
    )
    @pytest.mark.asyncio
    async def test_threat_level_matrix(self, svc, det_type, confidence, expected_level):
        result = await svc.analyze_threat(det_type, confidence, "Cam", None)
        assert result["threat_level"] == expected_level

    @pytest.mark.parametrize(
        "bad_json",
        [
            "",
            "null",
            "[]",
            "{'not': 'valid json'}",
            "undefined",
            "true",
        ],
    )
    def test_parse_handles_all_bad_inputs(self, svc, bad_json):
        result = svc._parse_analysis_response(bad_json, "person", 0.7)
        assert isinstance(result, dict)
        assert "threat_level" in result

    @pytest.mark.parametrize(
        "risk_score_str,expected_float",
        [
            ("0.85", 0.85),
            ("1", 1.0),
            ("0", 0.0),
            ("0.123", 0.123),
        ],
    )
    def test_risk_score_string_parsed_to_float(
        self, svc, risk_score_str, expected_float
    ):
        import json

        raw = json.dumps(
            {
                "threat_level": "medium",
                "risk_score": risk_score_str,
                "summary": "Test",
                "recommended_action": "Test",
            }
        )
        result = svc._parse_analysis_response(raw, "person", 0.7)
        assert isinstance(result["risk_score"], float)
        assert result["risk_score"] == pytest.approx(expected_float)


class TestModelManagerEdgeCases:

    @pytest.fixture
    def manager(self, tmp_path):
        from ai_models import ModelConfig, ModelManager

        weights = tmp_path / "yolov8n.pt"
        weights.write_bytes(b"x")

        class Fake:
            names = {0: "person"}

            def predict(self, frame, **kw):
                return []

        m = ModelManager(
            ModelConfig(
                yolo_weights=weights,
                drone_weights=tmp_path / "none.pt",
                warmup=False,
                auto_download=False,
                device="cpu",
            ),
            model_factory=lambda p: Fake(),
        )
        m.load()
        return m

    @pytest.mark.parametrize(
        "w,h,size,expected_type",
        [
            (200, 60, 0.02, "fixed_wing"),
            (80, 70, 0.02, "quadcopter"),
            (60, 50, 0.02, "quadcopter"),
            (10, 10, 0.0001, "micro_drone"),
            (50, 40, 0.015, "quadcopter"),
        ],
    )
    def test_drone_classification_parametrized(self, w, h, size, expected_type):
        from ai_models.analysis import classify_drone_type

        assert classify_drone_type(w, h, size) == expected_type

    @pytest.mark.parametrize(
        "frame_shape",
        [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3), (2160, 3840, 3)],
    )
    def test_detect_persons_accepts_various_resolutions(self, manager, frame_shape):
        import numpy as np

        frame = np.zeros(frame_shape, dtype=np.uint8)
        assert manager.detect_persons(frame) == []

    @pytest.mark.parametrize("bad", [None, "frame", [1, 2, 3]])
    def test_invalid_frames_do_not_raise_from_legacy_api(self, manager, bad):
        assert manager.detect_persons(bad) == []


class TestDroneTrackerEdgeCases:

    SHAPE = (1080, 1920, 3)

    @pytest.fixture
    def detector(self):
        from ai_models import DroneDetector, ModelConfig

        return DroneDetector(ModelConfig(track_min_hits=1), confidence_threshold=0.45)

    def _det(self, x, confidence=0.88, size=30):
        from ai_models import BoundingBox, Detection, ObjectKind

        return Detection(
            ObjectKind.DRONE, "drone", confidence, BoundingBox(x, 150, size, size)
        )

    @pytest.mark.parametrize(
        "confidence,expected_count",
        [(0.90, 1), (0.45, 1), (0.44, 0), (0.10, 0)],
    )
    def test_confidence_threshold_boundary(self, detector, confidence, expected_count):
        update = detector.process_frame(self.SHAPE, [self._det(600, confidence)])
        assert len(update.matched) == expected_count

    @pytest.mark.parametrize("num_drones", [1, 2, 3, 5])
    def test_multiple_drones_tracked_independently(self, detector, num_drones):
        dets = [self._det(i * 300) for i in range(num_drones)]
        update = detector.process_frame(self.SHAPE, dets)
        assert len(update.matched) == num_drones
        assert len({t.track_id for t in update.matched}) == num_drones

    @pytest.mark.parametrize(
        "is_authorized,expect_lower_score",
        [(True, True), (False, False)],
    )
    def test_authorization_status_affects_threat_score(
        self, detector, is_authorized, expect_lower_score
    ):
        from ai_models import ModelConfig
        from ai_models.analysis import ThreatScorer

        scorer = ThreatScorer(ModelConfig())
        track = detector.process_frame(self.SHAPE, [self._det(500, size=25)]).matched[0]
        reference = scorer.score(track, self.SHAPE, False)
        tested = scorer.score(track, self.SHAPE, is_authorized)
        if expect_lower_score:
            assert tested < reference
        else:
            assert tested == pytest.approx(reference)


class TestModelSettings:

    def test_model_paths_resolved_to_absolute(self):
        from pathlib import Path

        from config.settings import settings

        assert Path(settings.YOLO_MODEL_PATH).is_absolute()
        assert Path(settings.DRONE_MODEL_PATH).is_absolute()

    def test_relative_path_resolves_against_code_dir(self):
        from config.settings import CODE_DIR, Settings

        s = Settings(YOLO_MODEL_PATH="ai_models/weights/custom.pt")
        assert s.YOLO_MODEL_PATH == str(CODE_DIR / "ai_models/weights/custom.pt")

    def test_absolute_path_untouched(self):
        from config.settings import Settings

        assert (
            Settings(DRONE_MODEL_PATH="/opt/models/drone.pt").DRONE_MODEL_PATH
            == "/opt/models/drone.pt"
        )

    def test_build_model_config_maps_settings(self):
        from config.model_config import build_model_config
        from config.settings import Settings

        s = Settings(
            DETECTION_CONFIDENCE=0.6,
            DRONE_CONFIDENCE=0.3,
            TRACK_MIN_HITS=3,
            MODEL_ALLOW_MOCK_FALLBACK=True,
            AUTHORIZED_DRONE_ZONES=[{"lat": 1.0, "lng": 2.0, "radius_m": 30}],
        )
        cfg = build_model_config(s)
        assert cfg.confidence == 0.6 and cfg.drone_confidence == 0.3
        assert cfg.track_min_hits == 3 and cfg.allow_mock_fallback is True
        assert cfg.authorized_zones[0].radius_m == 30.0

    def test_defaults_do_not_enable_mock_fallback(self):
        from config.settings import Settings

        assert Settings().MODEL_ALLOW_MOCK_FALLBACK is False

    def test_ensure_directories_creates_weights_dir(self, tmp_path, monkeypatch):
        from config import settings as settings_module

        monkeypatch.setattr(
            settings_module.settings, "YOLO_MODEL_PATH", str(tmp_path / "w1" / "a.pt")
        )
        monkeypatch.setattr(
            settings_module.settings, "DRONE_MODEL_PATH", str(tmp_path / "w2" / "b.pt")
        )
        monkeypatch.setattr(
            settings_module.settings, "SNAPSHOTS_DIR", str(tmp_path / "s")
        )
        monkeypatch.setattr(
            settings_module.settings, "RECORDINGS_DIR", str(tmp_path / "r")
        )
        monkeypatch.chdir(tmp_path)
        settings_module.ensure_directories()
        assert (tmp_path / "w1").is_dir() and (tmp_path / "w2").is_dir()


class TestGoogleEarthEdgeCases:

    @pytest.fixture
    def geo(self):
        from config.settings import settings
        from integrations.google_earth import GoogleEarthIntegration

        settings.GOOGLE_MAPS_API_KEY = ""
        return GoogleEarthIntegration()

    @pytest.mark.parametrize(
        "direction_deg,fov_deg,range_m",
        [
            (0, 90, 100),
            (90, 45, 200),
            (180, 120, 50),
            (270, 60, 300),
            (360, 90, 100),
        ],
    )
    def test_fov_polygon_all_directions(self, geo, direction_deg, fov_deg, range_m):
        polygon = geo.calculate_camera_fov_polygon(
            lat=33.6844,
            lng=73.0479,
            direction_deg=direction_deg,
            fov_deg=fov_deg,
            range_meters=float(range_m),
        )
        assert len(polygon) >= 3
        for lat, lng in polygon:
            assert -90 <= lat <= 90
            assert -180 <= lng <= 180

    @pytest.mark.parametrize(
        "bbox_x,bbox_y",
        [
            (0.0, 0.5),
            (0.5, 0.5),
            (1.0, 0.5),
            (0.5, 0.0),
            (0.5, 1.0),
        ],
    )
    def test_location_estimation_valid_for_all_bbox_positions(
        self, geo, bbox_x, bbox_y
    ):
        lat, lng = geo.estimate_detection_location(
            camera_lat=33.6844,
            camera_lng=73.0479,
            camera_direction=0.0,
            camera_fov=90.0,
            bbox_center_x=bbox_x,
            bbox_center_y=bbox_y,
            estimated_range_m=100.0,
        )
        assert -90 <= lat <= 90
        assert -180 <= lng <= 180
