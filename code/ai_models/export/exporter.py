import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

from ..config import DEFAULT_WEIGHTS_DIR
from ..exceptions import ModelLoadError

logger = logging.getLogger("quantumfence.models.export")

SUPPORTED_FORMATS = ("onnx", "torchscript", "engine", "openvino")


def export_model(
    weights: Path,
    fmt: str = "onnx",
    image_size: int = 640,
    half: bool = False,
    dynamic: bool = False,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported export format {fmt}, choose from {SUPPORTED_FORMATS}"
        )
    weights = Path(weights)
    if not weights.is_file():
        raise ModelLoadError(f"Weights not found: {weights}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError("ultralytics is required for export") from exc
    kwargs = {"format": fmt, "imgsz": image_size, "half": half, "dynamic": dynamic}
    if device:
        kwargs["device"] = device
    exported = Path(YOLO(str(weights)).export(**kwargs))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / exported.name
        if exported.is_dir():
            shutil.copytree(exported, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(exported, target)
        exported = target
    logger.info("Exported %s to %s", weights.name, exported)
    return exported


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ai_models.export.exporter")
    parser.add_argument(
        "weights", type=Path, nargs="?", default=DEFAULT_WEIGHTS_DIR / "yolov8n.pt"
    )
    parser.add_argument("--format", default="onnx", choices=SUPPORTED_FORMATS)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    export_model(
        args.weights,
        args.format,
        args.image_size,
        args.half,
        args.dynamic,
        args.device,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
