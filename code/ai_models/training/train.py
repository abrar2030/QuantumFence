import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

from ..config import DEFAULT_WEIGHTS_DIR
from ..exceptions import ModelLoadError

logger = logging.getLogger("quantumfence.models.training")

DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "drone.yaml"


def train_drone_detector(
    data: Path = DEFAULT_DATASET,
    base_weights: str = "yolov8n.pt",
    epochs: int = 100,
    image_size: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    project: Path = Path("runs/drone"),
    name: str = "train",
    output: Path = DEFAULT_WEIGHTS_DIR / "drone_detector.pt",
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError("ultralytics is required for training") from exc
    model = YOLO(base_weights)
    kwargs = {
        "data": str(data),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch,
        "project": str(project),
        "name": name,
        "exist_ok": True,
    }
    if device:
        kwargs["device"] = device
    results = model.train(**kwargs)
    save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise ModelLoadError(f"Training finished without best weights at {best}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, output)
    logger.info("Drone detector weights written to %s", output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ai_models.training.train")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=Path("runs/drone"))
    parser.add_argument("--name", default="train")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_WEIGHTS_DIR / "drone_detector.pt"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    train_drone_detector(
        data=args.data,
        base_weights=args.base_weights,
        epochs=args.epochs,
        image_size=args.image_size,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
