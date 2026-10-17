from ai_models import ModelConfig
from config.settings import Settings
from config.settings import settings as default_settings


def build_model_config(source: Settings = None) -> ModelConfig:
    s = source or default_settings
    return ModelConfig(
        yolo_weights=s.YOLO_MODEL_PATH,
        drone_weights=s.DRONE_MODEL_PATH,
        device=s.MODEL_DEVICE,
        half_precision=s.MODEL_HALF_PRECISION,
        image_size=s.MODEL_IMAGE_SIZE,
        confidence=s.DETECTION_CONFIDENCE,
        iou_threshold=s.DETECTION_IOU_THRESHOLD,
        drone_confidence=s.DRONE_CONFIDENCE,
        max_detections=s.MODEL_MAX_DETECTIONS,
        warmup=s.MODEL_WARMUP,
        auto_download=s.MODEL_AUTO_DOWNLOAD,
        allow_mock_fallback=s.MODEL_ALLOW_MOCK_FALLBACK,
        drone_proxy_enabled=s.DRONE_PROXY_ENABLED,
        drone_proxy_class_ids=tuple(s.DRONE_PROXY_CLASS_IDS),
        drone_tiling=s.DRONE_TILED_INFERENCE,
        low_light_mode=s.LOW_LIGHT_MODE,
        track_max_age_seconds=s.TRACK_MAX_AGE_SECONDS,
        track_min_hits=s.TRACK_MIN_HITS,
        loiter_seconds=s.LOITER_SECONDS,
        swarm_min_drones=s.SWARM_MIN_DRONES,
        camera_mount_height_m=s.CAMERA_MOUNT_HEIGHT_METERS,
        camera_tilt_deg=s.CAMERA_TILT_DEGREES,
        authorized_zones=tuple(s.AUTHORIZED_DRONE_ZONES),
    )
