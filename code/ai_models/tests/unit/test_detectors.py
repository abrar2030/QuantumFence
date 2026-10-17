import pytest
from ai_models import ModelConfig, ModelLoadError, ModelNotLoadedError
from ai_models.detectors import MockDetector, YoloDetector

pytestmark = pytest.mark.unit


class TestYoloDetector:
    def test_load_parses_names_and_predicts(
        self, config, fake_yolo_factory, blank_frame
    ):
        factory = fake_yolo_factory(
            [(10, 10, 50, 90, 0.9, 0), (200, 200, 260, 260, 0.8, 2)]
        )
        det = YoloDetector(config.yolo_weights, config, model_factory=factory)
        det.load()
        assert det.is_loaded and det.class_names[0] == "person"
        raws = det.predict(blank_frame)
        assert {r.label for r in raws} == {"person", "car"}
        assert raws[0].bbox.width == 40

    def test_class_and_conf_forwarded(self, config, fake_yolo_factory, blank_frame):
        factory = fake_yolo_factory(
            [(10, 10, 50, 90, 0.9, 0), (10, 10, 50, 90, 0.3, 0)]
        )
        det = YoloDetector(config.yolo_weights, config, model_factory=factory)
        det.load()
        raws = det.predict(blank_frame, classes=[0], conf=0.5)
        assert len(raws) == 1
        call = factory.created[0].calls[-1]
        assert (
            call["classes"] == [0] and call["conf"] == 0.5 and call["verbose"] is False
        )

    def test_predict_before_load_raises(self, config, blank_frame):
        det = YoloDetector(config.yolo_weights, config)
        with pytest.raises(ModelNotLoadedError):
            det.predict(blank_frame)

    def test_missing_weights_without_download_raises(self, tmp_path):
        cfg = ModelConfig(
            yolo_weights=tmp_path / "none.pt", auto_download=False, warmup=False
        )
        with pytest.raises(ModelLoadError):
            YoloDetector(cfg.yolo_weights, cfg, model_factory=lambda p: None).load()

    def test_factory_failure_wrapped(self, config):
        def boom(path):
            raise RuntimeError("corrupt")

        with pytest.raises(ModelLoadError):
            YoloDetector(config.yolo_weights, config, model_factory=boom).load()

    def test_missing_ultralytics_reports_load_error(self, config):
        det = YoloDetector(config.yolo_weights, config)
        try:
            pass
        except ImportError:
            with pytest.raises(ModelLoadError):
                det.load()
        else:
            pytest.skip("ultralytics installed")

    def test_warmup_runs_once(self, config, fake_yolo_factory):
        cfg = config.with_overrides(warmup=True)
        factory = fake_yolo_factory([])
        YoloDetector(cfg.yolo_weights, cfg, model_factory=factory).load()
        assert len(factory.created[0].calls) == 1

    def test_degenerate_boxes_skipped(self, config, fake_yolo_factory, blank_frame):
        factory = fake_yolo_factory([(50, 50, 50, 90, 0.9, 0)])
        det = YoloDetector(config.yolo_weights, config, model_factory=factory)
        det.load()
        assert det.predict(blank_frame) == []

    def test_unload(self, config, fake_yolo_factory):
        det = YoloDetector(
            config.yolo_weights, config, model_factory=fake_yolo_factory([])
        )
        det.load()
        det.unload()
        assert not det.is_loaded

    def test_list_names_supported(self, config, blank_frame):
        class ListNamed:
            names = ["person", "bicycle"]

            def predict(self, frame, **kw):
                return []

        det = YoloDetector(
            config.yolo_weights, config, model_factory=lambda p: ListNamed()
        )
        det.load()
        assert det.class_names == {0: "person", 1: "bicycle"}


class TestMockDetector:
    def test_emits_on_interval(self, blank_frame):
        mock = MockDetector(interval=3)
        mock.load()
        results = [mock.predict(blank_frame) for _ in range(6)]
        assert [bool(r) for r in results] == [False, False, True, False, False, True]

    def test_respects_requested_classes(self, blank_frame):
        mock = MockDetector({0: "person", 2: "car"}, interval=1)
        mock.load()
        for _ in range(10):
            assert mock.predict(blank_frame, classes=[2])[0].class_id == 2

    def test_unsupported_classes_return_nothing(self, blank_frame):
        mock = MockDetector({0: "person"}, interval=1)
        assert mock.predict(blank_frame, classes=[99]) == []

    def test_deterministic_with_seed(self, blank_frame):
        a = MockDetector(interval=1, seed=7).predict(blank_frame)[0].bbox
        b = MockDetector(interval=1, seed=7).predict(blank_frame)[0].bbox
        assert a == b
