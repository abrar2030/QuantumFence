import json
import time

import pytest
from ai_models import BoundingBox, ModelLoadError
from ai_models.core import ModelRegistry
from ai_models.evaluation import (
    ScoredBox,
    average_precision,
    benchmark,
    evaluate_detections,
    match_predictions,
)
from ai_models.export import SUPPORTED_FORMATS, export_model
from ai_models.monitoring import InferenceMetrics
from ai_models.training import train_drone_detector
from ai_models.utils import resolve_device, supports_half

pytestmark = pytest.mark.unit


class TestMetrics:
    def test_records_percentiles(self):
        m = InferenceMetrics()
        for i in range(1, 11):
            m.record("inference", i / 1000)
        snap = m.snapshot()["timings"]["inference"]
        assert snap["count"] == 10
        assert snap["avg_ms"] == pytest.approx(5.5)
        assert snap["max_ms"] == pytest.approx(10.0)
        assert snap["p95_ms"] >= snap["p50_ms"]

    def test_counters_and_reset(self):
        m = InferenceMetrics()
        m.increment("errors", 2)
        assert m.snapshot()["counters"]["errors"] == 2
        m.reset()
        assert m.snapshot() == {"timings": {}, "counters": {}}

    def test_rate_positive_with_spread_samples(self):
        m = InferenceMetrics()
        m.record("x", 0.001)
        time.sleep(0.02)
        m.record("x", 0.001)
        assert m.snapshot()["timings"]["x"]["rate_per_s"] > 0

    def test_window_bounded(self):
        m = InferenceMetrics(window=5)
        for _ in range(20):
            m.record("x", 0.001)
        assert m.snapshot()["timings"]["x"]["count"] == 20


class TestRegistry:
    def test_lists_only_weight_files(self, tmp_path):
        (tmp_path / "a.pt").write_bytes(b"abc")
        (tmp_path / "b.onnx").write_bytes(b"def")
        (tmp_path / "notes.txt").write_text("x")
        names = [m.name for m in ModelRegistry(tmp_path).list_models()]
        assert names == ["a.pt", "b.onnx"]

    def test_digest_stable_and_cached(self, tmp_path):
        f = tmp_path / "a.pt"
        f.write_bytes(b"abc")
        reg = ModelRegistry(tmp_path)
        first = reg.describe(f)
        assert first.sha256 == reg.describe(f).sha256
        assert first.size_bytes == 3
        json.dumps(first.to_dict())

    def test_digest_changes_with_content(self, tmp_path):
        f = tmp_path / "a.pt"
        f.write_bytes(b"abc")
        reg = ModelRegistry(tmp_path)
        before = reg.describe(f).sha256
        f.write_bytes(b"abcd")
        assert reg.describe(f).sha256 != before

    def test_missing_directory_and_file(self, tmp_path):
        reg = ModelRegistry(tmp_path / "nope")
        assert reg.list_models() == []
        assert reg.describe(tmp_path / "nope.pt") is None


class TestEvaluation:
    def _box(self, x, conf=0.9):
        return ScoredBox(BoundingBox(x, 0, 10, 10), conf, "drone")

    def test_perfect_detection(self):
        result = evaluate_detections([[self._box(0)]], [[BoundingBox(0, 0, 10, 10)]])
        assert result.precision == result.recall == result.f1 == 1.0
        assert result.average_precision == pytest.approx(1.0)

    def test_false_positive_and_miss(self):
        result = evaluate_detections([[self._box(100)]], [[BoundingBox(0, 0, 10, 10)]])
        assert result.true_positives == 0
        assert result.false_positives == 1 and result.false_negatives == 1
        assert result.f1 == 0.0

    def test_confidence_threshold_filters(self):
        result = evaluate_detections(
            [[self._box(0, 0.3)]],
            [[BoundingBox(0, 0, 10, 10)]],
            confidence_threshold=0.5,
        )
        assert result.true_positives == 0 and result.false_negatives == 1

    def test_matching_each_gt_once(self):
        outcomes = match_predictions(
            [self._box(0, 0.9), self._box(1, 0.8)], [BoundingBox(0, 0, 10, 10)]
        )
        assert [hit for _, hit in outcomes] == [True, False]

    def test_ap_edge_cases(self):
        assert average_precision([], 3) == 0.0
        assert average_precision([(0.9, True)], 0) == 0.0

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(ValueError):
            evaluate_detections([[]], [])

    def test_to_dict(self):
        result = evaluate_detections([[self._box(0)]], [[BoundingBox(0, 0, 10, 10)]])
        assert result.to_dict()["precision"] == 1.0

    def test_benchmark_reports_stats(self, config, fake_yolo_factory):
        from ai_models import ModelManager

        m = ModelManager(config, model_factory=fake_yolo_factory([]))
        m.load()
        stats = benchmark(m, frame_shape=(64, 64, 3), runs=5, warmup=1)
        assert stats["runs"] == 5 and stats["fps"] > 0


class TestUtils:
    def test_explicit_device_passthrough(self):
        assert resolve_device("cuda:1") == "cuda:1"

    def test_auto_resolves_to_known_device(self):
        assert resolve_device("auto").split(":")[0] in ("cpu", "cuda", "mps")

    def test_half_only_on_cuda(self):
        assert (
            supports_half("cuda:0")
            and not supports_half("cpu")
            and not supports_half(None)
        )


class TestTrainAndExportGuards:
    def test_export_rejects_bad_format(self, tmp_path):
        with pytest.raises(ValueError):
            export_model(tmp_path / "w.pt", fmt="bogus")
        assert "onnx" in SUPPORTED_FORMATS

    def test_export_missing_weights(self, tmp_path):
        with pytest.raises(ModelLoadError):
            export_model(tmp_path / "missing.pt")

    def test_train_requires_ultralytics(self, tmp_path):
        try:
            pass
        except ImportError:
            with pytest.raises(ModelLoadError):
                train_drone_detector(output=tmp_path / "out.pt")
        else:
            pytest.skip("ultralytics installed")

    def test_cli_parsers(self):
        from ai_models.training.train import build_parser

        args = build_parser().parse_args(["--epochs", "3", "--batch", "4"])
        assert args.epochs == 3 and args.batch == 4
