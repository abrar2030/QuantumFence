import pytest
from ai_models import (
    AuthorizedZone,
    BoundingBox,
    ConfigurationError,
    Detection,
    ModelConfig,
    ObjectKind,
)

pytestmark = pytest.mark.unit


class TestBoundingBox:
    def test_from_xyxy_and_properties(self):
        box = BoundingBox.from_xyxy(10, 20, 50, 100)
        assert (box.x, box.y, box.width, box.height) == (10, 20, 40, 80)
        assert (box.cx, box.cy) == (30, 60)
        assert box.area == 3200
        assert box.x2 == 50 and box.y2 == 100

    def test_iou_identical_and_disjoint(self):
        a = BoundingBox(0, 0, 10, 10)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(BoundingBox(50, 50, 10, 10)) == 0.0

    def test_iou_partial_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 0, 10, 10)
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_iou_zero_area(self):
        assert BoundingBox(0, 0, 0, 0).iou(BoundingBox(0, 0, 0, 0)) == 0.0

    def test_shifted_and_to_list(self):
        assert BoundingBox(1, 2, 3, 4).shifted(10, 20).to_list() == [11, 22, 3, 4]

    def test_from_center(self):
        box = BoundingBox.from_center(50, 50, 20, 10)
        assert (box.x, box.y) == (40, 45)


class TestDetection:
    def test_to_dict_contains_expected_keys(self):
        det = Detection(
            ObjectKind.DRONE,
            "drone",
            0.87654,
            BoundingBox(1, 2, 3, 4),
            4,
            {"drone_type": "quadcopter"},
        )
        data = det.to_dict()
        assert data["type"] == "drone"
        assert data["class"] == "drone"
        assert data["confidence"] == 0.8765
        assert data["bbox"] == [1, 2, 3, 4]
        assert data["drone_type"] == "quadcopter"


class TestModelConfig:
    def test_defaults_are_valid(self):
        cfg = ModelConfig()
        assert cfg.image_size == 640
        assert cfg.yolo_weights.name == "yolov8n.pt"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"confidence": 1.5},
            {"iou_threshold": -0.1},
            {"drone_confidence": 2},
            {"image_size": 8},
            {"low_light_mode": "sometimes"},
            {"track_min_hits": 0},
            {"track_max_age_seconds": 0},
            {"drone_tile_overlap": 1.0},
            {"drone_tile_size": 10},
        ],
    )
    def test_invalid_values_rejected(self, kwargs):
        with pytest.raises(ConfigurationError):
            ModelConfig(**kwargs)

    def test_with_overrides_returns_new_instance(self):
        cfg = ModelConfig()
        other = cfg.with_overrides(confidence=0.7)
        assert other.confidence == 0.7 and cfg.confidence == 0.5

    def test_from_mapping_rejects_unknown_keys(self):
        with pytest.raises(ConfigurationError):
            ModelConfig.from_mapping({"nope": 1})

    def test_authorized_zones_from_mappings(self):
        cfg = ModelConfig(authorized_zones=({"lat": 1.0, "lng": 2.0, "radius_m": 50},))
        assert cfg.authorized_zones == (AuthorizedZone(1.0, 2.0, 50.0),)

    def test_invalid_zone_rejected(self):
        with pytest.raises(ConfigurationError):
            ModelConfig(authorized_zones=({"lat": 1},))

    def test_to_dict_is_json_friendly(self):
        import json

        cfg = ModelConfig(authorized_zones=({"lat": 1.0, "lng": 2.0, "radius_m": 50},))
        json.dumps(cfg.to_dict())
