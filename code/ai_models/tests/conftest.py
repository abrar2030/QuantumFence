import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ai_models import BoundingBox, Detection, ModelConfig, ObjectKind


class FakeBoxes:
    def __init__(self, rows):
        arr = np.array(rows, dtype=np.float32).reshape(-1, 6)
        self.xyxy = arr[:, :4]
        self.conf = arr[:, 4]
        self.cls = arr[:, 5]


class FakeResult:
    def __init__(self, rows, names):
        self.boxes = FakeBoxes(rows)
        self.names = names


class FakeYolo:
    def __init__(self, path, script=None, names=None):
        self.path = path
        self.names = names or {0: "person", 2: "car", 4: "airplane", 33: "kite"}
        self.script = script if script is not None else []
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        rows = self.script(frame) if callable(self.script) else self.script
        classes = kwargs.get("classes")
        conf = kwargs.get("conf", 0.0)
        kept = [
            r
            for r in rows
            if (classes is None or int(r[5]) in classes) and r[4] >= conf
        ]
        return [FakeResult(kept, self.names)]


@pytest.fixture
def fake_yolo_factory():
    created = []

    def make(script=None, names=None):
        def factory(path):
            model = FakeYolo(path, script, names)
            created.append(model)
            return model

        factory.created = created
        return factory

    return make


@pytest.fixture
def config(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "yolov8n.pt").write_bytes(b"fake")
    return ModelConfig(
        yolo_weights=weights / "yolov8n.pt",
        drone_weights=weights / "drone_detector.pt",
        warmup=False,
        auto_download=False,
        device="cpu",
    )


@pytest.fixture
def blank_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def noise_frame():
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def t0():
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def make_detection():
    def make(kind=ObjectKind.PERSON, x=100, y=100, w=40, h=80, conf=0.9, **attrs):
        return Detection(
            kind=kind,
            label=kind.value,
            confidence=conf,
            bbox=BoundingBox(x, y, w, h),
            attributes=attrs,
        )

    return make


@pytest.fixture
def step():
    def advance(base, seconds):
        return base + timedelta(seconds=seconds)

    return advance
