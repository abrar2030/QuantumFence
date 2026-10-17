import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..config import ModelConfig
from ..core.base import BaseDetector
from ..core.types import BoundingBox, RawDetection
from ..exceptions import ModelLoadError, ModelNotLoadedError
from ..utils.device import resolve_device, supports_half

logger = logging.getLogger("quantumfence.models.yolo")

ModelFactory = Callable[[str], Any]


def _default_factory(path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError("ultralytics is not installed") from exc
    return YOLO(path)


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YoloDetector(BaseDetector):
    def __init__(
        self,
        weights: Path,
        config: ModelConfig,
        name: str = "yolo",
        model_factory: Optional[ModelFactory] = None,
    ):
        self.name = name
        self.weights = Path(weights)
        self.config = config
        self.device = "cpu"
        self._factory = model_factory or _default_factory
        self._model: Any = None
        self._names: Dict[int, str] = {}
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._names)

    def load(self) -> None:
        self.device = resolve_device(self.config.device)
        target = self.weights
        if not target.exists():
            if not self.config.auto_download:
                raise ModelLoadError(f"Weights not found: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Weights %s missing, attempting download", target.name)
        try:
            model = self._factory(str(target))
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Failed to load {target}: {exc}") from exc
        names = getattr(model, "names", None) or {}
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        self._names = {int(k): str(v) for k, v in dict(names).items()}
        self._model = model
        if self.config.warmup:
            self._warmup()
        logger.info("%s loaded from %s on %s", self.name, target, self.device)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._names = {}

    def _warmup(self) -> None:
        size = self.config.image_size
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        try:
            self.predict(blank, conf=0.99)
        except Exception as exc:
            logger.warning("Warmup failed for %s: %s", self.name, exc)

    def predict(
        self,
        frame: np.ndarray,
        classes: Optional[Sequence[int]] = None,
        conf: Optional[float] = None,
    ) -> List[RawDetection]:
        if self._model is None:
            raise ModelNotLoadedError(f"{self.name} is not loaded")
        kwargs: Dict[str, Any] = {
            "conf": self.config.confidence if conf is None else conf,
            "iou": self.config.iou_threshold,
            "imgsz": self.config.image_size,
            "max_det": self.config.max_detections,
            "device": self.device,
            "half": self.config.half_precision and supports_half(self.device),
            "verbose": False,
        }
        if classes is not None:
            kwargs["classes"] = [int(c) for c in classes]
        with self._lock:
            results = self._model.predict(frame, **kwargs)
        return self._parse(results)

    def _parse(self, results: Any) -> List[RawDetection]:
        detections: List[RawDetection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
            confs = _to_numpy(boxes.conf).reshape(-1)
            cls = _to_numpy(boxes.cls).reshape(-1)
            names = getattr(result, "names", None) or self._names
            for i in range(min(len(xyxy), len(confs))):
                class_id = int(cls[i]) if i < len(cls) else -1
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                if x2 <= x1 or y2 <= y1:
                    continue
                label = (
                    names.get(class_id)
                    if isinstance(names, dict)
                    else self._names.get(class_id)
                )
                detections.append(
                    RawDetection(
                        class_id=class_id,
                        label=str(label or class_id),
                        confidence=float(confs[i]),
                        bbox=BoundingBox.from_xyxy(x1, y1, x2, y2),
                    )
                )
        return detections

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "loaded": self.is_loaded,
            "weights": str(self.weights),
            "device": self.device,
            "classes": len(self._names),
        }
