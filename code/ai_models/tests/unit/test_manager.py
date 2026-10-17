import numpy as np
import pytest
from ai_models import ModelManager, ModelNotLoadedError
from ai_models.detectors import MockDetector

pytestmark = pytest.mark.unit

SCRIPT = [
    (100, 400, 160, 560, 0.9, 0),
    (500, 400, 700, 520, 0.85, 2),
    (600, 80, 640, 110, 0.7, 4),
    (900, 600, 940, 630, 0.7, 4),
    (50, 50, 90, 90, 0.2, 0),
]


@pytest.fixture
def manager(config, fake_yolo_factory):
    m = ModelManager(config, model_factory=fake_yolo_factory(SCRIPT))
    m.load()
    return m


class TestLoading:
    def test_unloaded_manager_state(self, config):
        m = ModelManager(config)
        assert not m.models_loaded and m.health() == "unavailable"
        with pytest.raises(ModelNotLoadedError):
            m.infer(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_loads_general_and_uses_proxy_for_drones(self, manager):
        assert manager.models_loaded
        assert manager.drone_mode == "proxy"
        assert manager.degraded and manager.health() == "degraded"

    def test_custom_drone_model_selected_when_weights_exist(
        self, config, fake_yolo_factory
    ):
        config.drone_weights.write_bytes(b"fake")
        names = {0: "drone", 1: "bird"}
        calls = {}

        def factory(path):
            from ..conftest import FakeYolo

            names_for = names if path.endswith("drone_detector.pt") else None
            model = FakeYolo(
                path, [(10, 10, 30, 30, 0.9, 0), (60, 60, 80, 80, 0.9, 1)], names_for
            )
            calls[path] = model
            return model

        m = ModelManager(config, model_factory=factory)
        m.load()
        assert m.drone_mode == "custom" and m.health() == "loaded"
        out = m.infer(
            np.zeros((720, 1280, 3), dtype=np.uint8), persons=False, vehicles=False
        )
        assert len(out.drones) == 1
        assert out.drones[0].attributes["detector_mode"] == "custom"

    def test_missing_general_weights_disables_without_mock(self, tmp_path):
        from ai_models import ModelConfig

        cfg = ModelConfig(
            yolo_weights=tmp_path / "x.pt", auto_download=False, warmup=False
        )
        m = ModelManager(cfg)
        m.load()
        assert not m.models_loaded
        assert m.load_errors and m.drone_mode == "disabled"

    def test_mock_fallback_when_enabled(self, tmp_path):
        from ai_models import ModelConfig

        cfg = ModelConfig(
            yolo_weights=tmp_path / "x.pt",
            auto_download=False,
            warmup=False,
            allow_mock_fallback=True,
        )
        m = ModelManager(cfg)
        m.load()
        assert m.models_loaded and m.drone_mode == "mock"
        assert isinstance(m.general_detector, MockDetector)

    def test_corrupt_custom_drone_falls_back_to_proxy(self, config, fake_yolo_factory):
        config.drone_weights.write_bytes(b"x")
        good = fake_yolo_factory(SCRIPT)

        def factory(path):
            if path.endswith("drone_detector.pt"):
                raise RuntimeError("bad weights")
            return good(path)

        m = ModelManager(config, model_factory=factory)
        m.load()
        assert m.drone_mode == "proxy"
        assert any(e.startswith("drone:") for e in m.load_errors)

    def test_reload_swaps_state(self, manager):
        old = manager.general_detector
        manager.reload()
        assert manager.general_detector is not old and manager.models_loaded

    def test_unload(self, manager):
        manager.unload()
        assert not manager.models_loaded

    @pytest.mark.asyncio
    async def test_async_load(self, config, fake_yolo_factory):
        m = ModelManager(config, model_factory=fake_yolo_factory([]))
        await m.load_all_models()
        assert m.models_loaded


class TestInference:
    def test_single_forward_pass_for_all_kinds(
        self, config, fake_yolo_factory, blank_frame
    ):
        factory = fake_yolo_factory(SCRIPT)
        m = ModelManager(config, model_factory=factory)
        m.load()
        m.infer(blank_frame)
        assert len(factory.created[0].calls) == 1

    def test_splits_kinds_and_applies_thresholds(self, manager, blank_frame):
        out = manager.infer(blank_frame)
        assert len(out.persons) == 1
        assert len(out.vehicles) == 1 and out.vehicles[0].label == "car"
        assert len(out.drones) == 1
        assert out.drones[0].bbox.x == 600
        assert out.drones[0].attributes["detector_mode"] == "proxy"
        assert out.inference_ms >= 0

    def test_proxy_drone_in_lower_frame_filtered(self, manager, blank_frame):
        out = manager.infer(blank_frame, persons=False, vehicles=False)
        assert all(d.bbox.cy < 720 * 0.7 for d in out.drones)

    def test_flags_disable_kinds(self, manager, blank_frame):
        out = manager.infer(blank_frame, persons=False, vehicles=False, drones=False)
        assert out.all() == []

    def test_flags_restrict_classes_sent_to_model(
        self, config, fake_yolo_factory, blank_frame
    ):
        factory = fake_yolo_factory(SCRIPT)
        m = ModelManager(config, model_factory=factory)
        m.load()
        m.infer(blank_frame, persons=True, vehicles=False, drones=False)
        assert factory.created[0].calls[-1]["classes"] == [0]

    def test_legacy_detect_methods(self, manager, blank_frame):
        persons = manager.detect_persons(blank_frame)
        assert persons[0]["type"] == "person" and persons[0]["bbox"][2] == 60
        assert manager.detect_vehicles(blank_frame)[0]["class"] == "car"
        drones = manager.detect_drones(blank_frame)
        assert drones[0]["type"] == "drone" and "drone_type" in drones[0]

    def test_legacy_methods_swallow_errors(
        self, config, fake_yolo_factory, blank_frame
    ):
        def explode(frame):
            raise RuntimeError("gpu fault")

        m = ModelManager(config, model_factory=fake_yolo_factory(explode))
        m.load()
        assert m.detect_persons(blank_frame) == []
        assert m.metrics.snapshot()["counters"]["inference_errors"] == 1

    def test_legacy_methods_when_unloaded(self, config, blank_frame):
        assert ModelManager(config).detect_drones(blank_frame) == []

    def test_metrics_recorded(self, manager, blank_frame):
        manager.infer(blank_frame)
        snap = manager.metrics.snapshot()
        assert snap["timings"]["inference"]["count"] == 1
        assert snap["counters"]["persons_detected"] == 1

    def test_status_payload(self, manager):
        status = manager.status()
        assert status["loaded"] and status["drone_mode"] == "proxy"
        assert status["general_model"]["name"] == "general"
        assert {w["name"] for w in status["weights"]} == {"yolov8n.pt"}

    def test_tiled_custom_inference_merges_duplicates(self, config, fake_yolo_factory):
        config.drone_weights.write_bytes(b"x")
        cfg = config.with_overrides(drone_tiling=True, drone_tile_size=640)
        from ..conftest import FakeYolo

        counter = {"n": 0}

        def script(frame):
            counter["n"] += 1
            return [(10, 10, 40, 40, 0.9, 0)] if counter["n"] <= 2 else []

        def factory(path):
            if path.endswith("drone_detector.pt"):
                return FakeYolo(path, script, {0: "drone"})
            return FakeYolo(path, [])

        m = ModelManager(cfg, model_factory=factory)
        m.load()
        counter["n"] = 0
        out = m.infer(
            np.zeros((720, 1280, 3), dtype=np.uint8), persons=False, vehicles=False
        )
        assert counter["n"] > 2
        assert len(out.drones) == 1
