import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analysis.drone import classify_drone_type
from .config import ModelConfig
from .constants import (
    DRONE_LABELS,
    DRONE_PROXY_CLASS_NAMES,
    PERSON_CLASS_ID,
    VEHICLE_CLASS_NAMES,
)
from .core.base import BaseDetector
from .core.registry import ModelRegistry
from .core.types import Detection, InferenceOutput, ObjectKind, RawDetection
from .detectors.mock import MockDetector
from .detectors.yolo import ModelFactory, YoloDetector
from .exceptions import ModelLoadError, ModelNotLoadedError
from .monitoring.metrics import InferenceMetrics
from .postprocessing.filters import filter_drone_candidates
from .postprocessing.nms import non_max_suppression
from .preprocessing.frame import ensure_bgr
from .preprocessing.tiling import crop, generate_tiles

logger = logging.getLogger("quantumfence.models")

MODE_CUSTOM = "custom"
MODE_PROXY = "proxy"
MODE_MOCK = "mock"
MODE_DISABLED = "disabled"


@dataclass(frozen=True)
class _State:
    general: Optional[BaseDetector] = None
    drone: Optional[BaseDetector] = None
    drone_mode: str = MODE_DISABLED
    drone_class_ids: Optional[Tuple[int, ...]] = None
    device: str = "cpu"
    errors: Tuple[str, ...] = ()
    loaded_at: Optional[float] = None

    @property
    def loaded(self) -> bool:
        return self.general is not None or self.drone is not None


class ModelManager:
    PERSON_CLASS_ID = PERSON_CLASS_ID
    VEHICLE_CLASS_IDS = dict(VEHICLE_CLASS_NAMES)

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_factory: Optional[ModelFactory] = None,
    ):
        self.config = config or ModelConfig()
        self._factory = model_factory
        self.metrics = InferenceMetrics()
        self.registry = ModelRegistry(self.config.yolo_weights.parent)
        self._state = _State()
        self._load_lock = threading.Lock()

    @property
    def general_detector(self) -> Optional[BaseDetector]:
        return self._state.general

    @property
    def drone_detector(self) -> Optional[BaseDetector]:
        return self._state.drone

    @property
    def drone_mode(self) -> str:
        return self._state.drone_mode

    @property
    def device(self) -> str:
        return self._state.device

    @property
    def load_errors(self) -> List[str]:
        return list(self._state.errors)

    @property
    def models_loaded(self) -> bool:
        return self._state.loaded

    @property
    def degraded(self) -> bool:
        state = self._state
        return state.loaded and (
            state.drone_mode in (MODE_DISABLED, MODE_PROXY, MODE_MOCK)
            or bool(state.errors)
        )

    async def load_all_models(self) -> None:
        await asyncio.to_thread(self.load)

    def load(self) -> None:
        with self._load_lock:
            errors: List[str] = []
            general, device = self._build_general(errors)
            drone, mode, class_ids = self._build_drone(general, errors)
            previous = self._state
            self._state = _State(
                general=general,
                drone=drone,
                drone_mode=mode,
                drone_class_ids=class_ids,
                device=device,
                errors=tuple(errors),
                loaded_at=time.time(),
            )
            self._release(previous, keep=(general, drone))
            logger.info(
                "Models loaded: general=%s drone_mode=%s device=%s",
                general.name if general else None,
                mode,
                device,
            )

    def reload(self) -> None:
        self.load()

    def unload(self) -> None:
        with self._load_lock:
            previous = self._state
            self._state = _State()
            self._release(previous, keep=())

    @staticmethod
    def _release(state: _State, keep: Tuple[Optional[BaseDetector], ...]) -> None:
        for det in {id(d): d for d in (state.general, state.drone) if d}.values():
            if not any(det is k for k in keep):
                det.unload()

    def _new_yolo(self, weights, name: str) -> YoloDetector:
        return YoloDetector(
            weights, self.config, name=name, model_factory=self._factory
        )

    def _build_general(self, errors: List[str]) -> Tuple[Optional[BaseDetector], str]:
        detector = self._new_yolo(self.config.yolo_weights, "general")
        try:
            detector.load()
            return detector, detector.device
        except ModelLoadError as exc:
            logger.error("General model failed to load: %s", exc)
            errors.append(f"general: {exc}")
        if self.config.allow_mock_fallback:
            logger.warning("Using mock general detector")
            mock = MockDetector(
                {
                    PERSON_CLASS_ID: "person",
                    **VEHICLE_CLASS_NAMES,
                    **DRONE_PROXY_CLASS_NAMES,
                },
                name="mock-general",
            )
            mock.load()
            return mock, "cpu"
        return None, "cpu"

    def _build_drone(
        self, general: Optional[BaseDetector], errors: List[str]
    ) -> Tuple[Optional[BaseDetector], str, Optional[Tuple[int, ...]]]:
        weights = self.config.drone_weights
        if weights.exists():
            detector = self._new_yolo(weights, "drone")
            try:
                detector.load()
                classes = self._select_drone_classes(detector.class_names)
                return detector, MODE_CUSTOM, classes
            except ModelLoadError as exc:
                logger.error("Drone model failed to load: %s", exc)
                errors.append(f"drone: {exc}")
        if general is not None and self.config.drone_proxy_enabled:
            if isinstance(general, MockDetector):
                return general, MODE_MOCK, None
            logger.warning(
                "No custom drone model, using generic proxy classes %s",
                self.config.drone_proxy_class_ids,
            )
            return general, MODE_PROXY, None
        return None, MODE_DISABLED, None

    @staticmethod
    def _select_drone_classes(names: Dict[int, str]) -> Optional[Tuple[int, ...]]:
        matches = tuple(
            cid
            for cid, name in names.items()
            if name.lower().replace("-", "_") in DRONE_LABELS
        )
        return matches or None

    def _require_loaded(self) -> _State:
        state = self._state
        if not state.loaded:
            raise ModelNotLoadedError("Models are not loaded")
        return state

    def _general_classes(self, persons: bool, vehicles: bool, proxy: bool) -> List[int]:
        ids = set()
        if persons:
            ids.add(PERSON_CLASS_ID)
        if vehicles:
            ids.update(VEHICLE_CLASS_NAMES)
        if proxy:
            ids.update(self.config.drone_proxy_class_ids)
        return sorted(ids)

    def infer(
        self,
        frame: np.ndarray,
        persons: bool = True,
        vehicles: bool = True,
        drones: bool = True,
    ) -> InferenceOutput:
        state = self._require_loaded()
        started = time.perf_counter()
        frame = ensure_bgr(frame)
        out = InferenceOutput()
        general = state.general
        proxy = drones and state.drone_mode in (MODE_PROXY, MODE_MOCK)
        custom = drones and state.drone_mode == MODE_CUSTOM

        general_classes = self._general_classes(persons, vehicles, proxy)
        if general is not None and general_classes:
            min_conf = min(
                [self.config.confidence]
                + ([self.config.drone_confidence] if proxy else [])
            )
            raws = general.predict(frame, classes=general_classes, conf=min_conf)
            self._split_general(raws, out, persons, vehicles, proxy, frame.shape)

        if custom and state.drone is not None:
            raws = self._predict_drone_custom(state.drone, state.drone_class_ids, frame)
            out.drones.extend(self._to_drone_detections(raws, frame.shape, proxy=False))

        out.inference_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.record("inference", out.inference_ms / 1000.0)
        self.metrics.increment("persons_detected", len(out.persons))
        self.metrics.increment("vehicles_detected", len(out.vehicles))
        self.metrics.increment("drones_detected", len(out.drones))
        return out

    def _split_general(
        self,
        raws: Sequence[RawDetection],
        out: InferenceOutput,
        persons: bool,
        vehicles: bool,
        proxy: bool,
        frame_shape: Tuple[int, ...],
    ) -> None:
        proxy_raws: List[RawDetection] = []
        for raw in raws:
            if raw.class_id == PERSON_CLASS_ID and persons:
                if raw.confidence >= self.config.confidence:
                    out.persons.append(
                        self._detection(ObjectKind.PERSON, "person", raw)
                    )
            elif raw.class_id in VEHICLE_CLASS_NAMES and vehicles:
                if raw.confidence >= self.config.confidence:
                    out.vehicles.append(
                        self._detection(
                            ObjectKind.VEHICLE, VEHICLE_CLASS_NAMES[raw.class_id], raw
                        )
                    )
            elif proxy and raw.class_id in self.config.drone_proxy_class_ids:
                if raw.confidence >= self.config.drone_confidence:
                    proxy_raws.append(raw)
        if proxy_raws:
            out.drones.extend(
                self._to_drone_detections(proxy_raws, frame_shape, proxy=True)
            )

    @staticmethod
    def _detection(kind: ObjectKind, label: str, raw: RawDetection) -> Detection:
        return Detection(
            kind=kind,
            label=label,
            confidence=raw.confidence,
            bbox=raw.bbox,
            class_id=raw.class_id,
        )

    def _predict_drone_custom(
        self,
        detector: BaseDetector,
        classes: Optional[Tuple[int, ...]],
        frame: np.ndarray,
    ) -> List[RawDetection]:
        conf = self.config.drone_confidence
        raws = detector.predict(frame, classes=classes, conf=conf)
        if self.config.drone_tiling:
            tiles = generate_tiles(
                frame.shape, self.config.drone_tile_size, self.config.drone_tile_overlap
            )
            if len(tiles) > 1:
                for tile in tiles:
                    sub = crop(frame, tile)
                    for raw in detector.predict(sub, classes=classes, conf=conf):
                        raws.append(
                            RawDetection(
                                raw.class_id,
                                raw.label,
                                raw.confidence,
                                raw.bbox.shifted(tile[0], tile[1]),
                            )
                        )
                raws = non_max_suppression(raws, self.config.iou_threshold)
        return raws

    def _to_drone_detections(
        self, raws: Sequence[RawDetection], frame_shape: Tuple[int, ...], proxy: bool
    ) -> List[Detection]:
        height, width = frame_shape[:2]
        frame_area = max(width * height, 1)
        dets = []
        for raw in raws:
            relative = raw.bbox.area / frame_area
            label = None if proxy else raw.label
            dets.append(
                Detection(
                    kind=ObjectKind.DRONE,
                    label="drone",
                    confidence=raw.confidence,
                    bbox=raw.bbox,
                    class_id=raw.class_id,
                    attributes={
                        "drone_type": classify_drone_type(
                            raw.bbox.width, raw.bbox.height, relative, label
                        ),
                        "source_class": raw.label,
                        "detector_mode": MODE_PROXY if proxy else MODE_CUSTOM,
                    },
                )
            )
        if proxy:
            return filter_drone_candidates(
                dets,
                frame_shape,
                self.config.proxy_max_relative_area,
                self.config.proxy_max_relative_y,
            )
        return filter_drone_candidates(
            dets, frame_shape, self.config.drone_max_relative_area
        )

    def _run(self, frame: np.ndarray, **flags: bool) -> InferenceOutput:
        try:
            return self.infer(frame, **flags)
        except ModelNotLoadedError:
            return InferenceOutput()
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            self.metrics.increment("inference_errors")
            return InferenceOutput()

    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        out = self._run(frame, persons=True, vehicles=False, drones=False)
        return [d.to_dict() for d in out.persons]

    def detect_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        out = self._run(frame, persons=False, vehicles=True, drones=False)
        return [d.to_dict() for d in out.vehicles]

    def detect_drones(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        out = self._run(frame, persons=False, vehicles=False, drones=True)
        return [d.to_dict() for d in out.drones]

    def create_pipeline(self, context=None):
        from .pipeline.detection_pipeline import DetectionPipeline

        return DetectionPipeline(self, context)

    def status(self) -> Dict[str, Any]:
        state = self._state
        return {
            "loaded": state.loaded,
            "degraded": self.degraded,
            "device": state.device,
            "drone_mode": state.drone_mode,
            "general_model": state.general.describe() if state.general else None,
            "drone_model": state.drone.describe() if state.drone else None,
            "errors": list(state.errors),
            "loaded_at": state.loaded_at,
            "weights": [m.to_dict() for m in self.registry.list_models()],
            "config": self.config.to_dict(),
            "metrics": self.metrics.snapshot(),
        }

    def health(self) -> str:
        if not self._state.loaded:
            return "unavailable"
        return "degraded" if self.degraded else "loaded"
