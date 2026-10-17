import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings

CODE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    APP_NAME: str = "QuantumFence"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    SECRET_KEY: str = "quantum-fence-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    DATABASE_URL: str = "sqlite:///./quantumfence.db"

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:80",
        "http://127.0.0.1:3000",
    ]

    ANTHROPIC_API_KEY: str = ""
    AI_MODEL: str = "claude-sonnet-4-20250514"
    THREAT_ANALYSIS_ENABLED: bool = True
    AI_CONFIDENCE_THRESHOLD: float = 0.65

    YOLO_MODEL_PATH: str = "ai_models/weights/yolov8n.pt"
    DRONE_MODEL_PATH: str = "ai_models/weights/drone_detector.pt"
    DETECTION_CONFIDENCE: float = 0.5
    DETECTION_IOU_THRESHOLD: float = 0.45
    DRONE_CONFIDENCE: float = 0.4
    FRAME_SKIP: int = 3
    PROCESSING_FPS: int = 15
    MAX_CAMERAS: int = 64
    TOUCH_CAMERA_INTERVAL_SECONDS: int = 30

    MODEL_DEVICE: str = "auto"
    MODEL_HALF_PRECISION: bool = False
    MODEL_IMAGE_SIZE: int = 640
    MODEL_MAX_DETECTIONS: int = 100
    MODEL_WARMUP: bool = True
    MODEL_AUTO_DOWNLOAD: bool = True
    MODEL_ALLOW_MOCK_FALLBACK: bool = False
    DRONE_PROXY_ENABLED: bool = True
    DRONE_PROXY_CLASS_IDS: List[int] = [4, 33]
    DRONE_TILED_INFERENCE: bool = False
    LOW_LIGHT_MODE: str = "off"
    TRACK_MAX_AGE_SECONDS: float = 3.0
    TRACK_MIN_HITS: int = 2
    LOITER_SECONDS: float = 60.0
    SWARM_MIN_DRONES: int = 3
    CAMERA_MOUNT_HEIGHT_METERS: float = 4.0
    CAMERA_TILT_DEGREES: float = 15.0
    AUTHORIZED_DRONE_ZONES: List[Dict[str, Any]] = []
    DETECTION_ALERT_COOLDOWN_SECONDS: int = 30
    PROMETHEUS_ENABLED: bool = True

    DEFAULT_FENCE_BUFFER_METERS: float = 10.0
    GEOFENCE_ALERT_COOLDOWN_SECONDS: int = 30

    GOOGLE_MAPS_API_KEY: str = ""
    GOOGLE_EARTH_ENGINE_PROJECT: str = ""
    DEFAULT_MAP_CENTER_LAT: float = 33.6844
    DEFAULT_MAP_CENTER_LNG: float = 73.0479

    SNAPSHOTS_DIR: str = "snapshots"
    RECORDINGS_DIR: str = "recordings"
    MAX_SNAPSHOT_AGE_DAYS: int = 30

    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    ALERT_EMAIL_RECIPIENTS: List[str] = []

    RTSP_TIMEOUT_SECONDS: int = 10
    STREAM_RECONNECT_ATTEMPTS: int = 3
    STREAM_BUFFER_SIZE: int = 5

    @field_validator("YOLO_MODEL_PATH", "DRONE_MODEL_PATH")
    @classmethod
    def _resolve_model_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = CODE_DIR / path
        return str(path)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


settings = Settings()


def ensure_directories() -> None:
    for d in [
        settings.SNAPSHOTS_DIR,
        settings.RECORDINGS_DIR,
        str(Path(settings.YOLO_MODEL_PATH).parent),
        str(Path(settings.DRONE_MODEL_PATH).parent),
        "logs",
    ]:
        os.makedirs(d, exist_ok=True)
