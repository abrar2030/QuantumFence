# ai_models

Detection, tracking and threat analysis package used by the QuantumFence backend. The package has no dependency on the backend. The backend builds a `ModelConfig` from its settings and hands it to `ModelManager`.

## Layout

```
ai_models/
  config.py                 ModelConfig and AuthorizedZone
  constants.py              COCO class ids, drone labels, drone real world widths
  exceptions.py             Error hierarchy
  manager.py                ModelManager, the single entry point
  core/                     Types (BoundingBox, Detection), BaseDetector, ModelRegistry
  detectors/                YoloDetector (ultralytics), MockDetector (explicit opt in)
  preprocessing/            Frame validation, low light enhancement, tiling
  postprocessing/           NMS, size and drone candidate filters
  tracking/                 Kalman filter, multi object tracker, DroneDetector
  analysis/                 Camera geometry, threat scoring, swarm and loitering detection
  pipeline/                 DetectionPipeline, one instance per camera
  monitoring/               Latency and throughput metrics
  evaluation/               Precision, recall, AP and latency benchmark
  training/                 Drone detector fine tuning CLI and dataset definition
  export/                   ONNX, TorchScript, TensorRT and OpenVINO export CLI
  utils/                    Device resolution
  weights/                  Model weights (not tracked)
  tests/                    Unit and integration tests
```

## Runtime behaviour

- One forward pass per frame serves persons, vehicles and proxy drones. Class filtering is done by the detector, so a camera with only drone detection enabled never pays for the other classes.
- Drone detection mode is reported by `ModelManager.drone_mode`:
  - `custom`: a dedicated drone model exists at `drone_weights`.
  - `proxy`: no custom model, the general model is used with the classes in `drone_proxy_class_ids` (airplane and kite by default, birds excluded). Accuracy is limited and the manager reports a degraded health state.
  - `mock`: only when `allow_mock_fallback` is enabled and no weights could be loaded.
  - `disabled`: no usable model.
- Mock detections are never generated unless `allow_mock_fallback` is set.
- `ModelManager.reload()` builds the new detectors first and swaps them in atomically, so running cameras keep working during a reload.
- Inference on a shared model is serialised by a lock inside `YoloDetector`.

## Tracking and analysis

Each camera owns a `DetectionPipeline` with separate trackers for persons, vehicles and drones. Tracks use a constant velocity Kalman filter with IoU and distance association, and a track is confirmed after `track_min_hits` matches. Confirmation, not raw detection, is what raises events, which removes the per frame alert flood.

Per track outputs: id, speed, heading, approach flag, trajectory, duration. Drone tracks additionally get drone type, estimated range, altitude and speed in metres, threat score and authorization state. Pipelines also report drone swarm start and person loitering.

## Camera geometry

`CameraModel` converts pixels to positions with a pinhole model and a flat ground assumption. Inputs are camera latitude and longitude, heading, horizontal field of view, mounting height and downward tilt. Ground objects use the bottom centre of their box, drones use apparent width and a nominal real width per drone type. These are estimates intended for heatmaps, geofence checks and authorized zone matching. Accuracy depends on how well mounting height and tilt match the real installation.

## Usage

```python
from ai_models import CameraContext, CameraModel, ModelConfig, ModelManager

manager = ModelManager(ModelConfig())
manager.load()

pipeline = manager.create_pipeline(
    CameraContext(camera_id=1, camera_model=CameraModel(33.68, 73.04, heading_deg=90))
)
result = pipeline.process(frame)
for track in result.new_tracks:
    print(track.key, track.to_dict())
```

## Training and export

```
python -m ai_models.training.train --data ai_models/training/datasets/drone.yaml --epochs 100
python -m ai_models.export.exporter ai_models/weights/drone_detector.pt --format onnx
```

Training writes the best checkpoint to `weights/drone_detector.pt`, which the manager picks up on the next load or reload.

## Tests

```
cd code/ai_models
python -m pytest
```
