import numpy as np

from ..exceptions import InvalidFrameError


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise InvalidFrameError("frame must be a numpy array")
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise InvalidFrameError(f"unsupported frame shape {frame.shape}")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise InvalidFrameError("frame is too small")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def mean_luminance(frame: np.ndarray) -> float:
    step = max(1, min(frame.shape[:2]) // 64)
    sample = frame[::step, ::step]
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    return float((sample.astype(np.float32) @ weights).mean())


def is_low_light(frame: np.ndarray, threshold: float = 60.0) -> bool:
    return mean_luminance(frame) < threshold


def enhance_low_light(
    frame: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8
) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        mean = max(mean_luminance(frame), 1.0)
        gamma = float(np.clip(np.log(0.45) / np.log(mean / 255.0), 0.4, 1.0))
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        return lut[frame]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def prepare_frame(
    frame: np.ndarray,
    mode: str = "off",
    threshold: float = 60.0,
    force_enhance: bool = False,
) -> np.ndarray:
    frame = ensure_bgr(frame)
    if force_enhance or mode == "on":
        return enhance_low_light(frame)
    if mode == "auto" and is_low_light(frame, threshold):
        return enhance_low_light(frame)
    return frame
