from datetime import timedelta

import numpy as np
import pytest
from ai_models import (
    CameraContext,
    CameraModel,
    DetectionFlags,
    ModelManager,
    ObjectKind,
)

pytestmark = pytest.mark.integration


def moving_script(state):
    def script(frame):
        i = state["i"]
        rows = [(100 + i * 6, 400, 160 + i * 6, 560, 0.9, 0)]
        rows.append((500, 300, 700, 420, 0.88, 2))
        if state.get("drone"):
            rows.append((600 + i * 12, 80, 624 + i * 12, 100, 0.8, 4))
        return rows

    return script


@pytest.fixture
def setup(config, fake_yolo_factory):
    state = {"i": 0, "drone": True}
    manager = ModelManager(
        config, model_factory=fake_yolo_factory(moving_script(state))
    )
    manager.load()
    ctx = CameraContext(
        camera_id=3,
        name="North Gate",
        camera_model=CameraModel(33.0, 73.0, heading_deg=0, mount_height_m=5),
    )
    return manager, manager.create_pipeline(ctx), state


def run(pipeline, state, t0, frames=12, step=0.2, flags=None):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    results = []
    for i in range(frames):
        state["i"] = i
        results.append(pipeline.process(frame, flags, t0 + timedelta(seconds=i * step)))
    return results


class TestPipeline:
    def test_tracks_confirmed_and_new_flagged_once(self, setup, t0):
        _, pipeline, state = setup
        results = run(pipeline, state, t0)
        new_counts = {
            k: sum(len(r.new_by_kind(k)) for r in results) for k in ObjectKind
        }
        assert new_counts == {
            ObjectKind.PERSON: 1,
            ObjectKind.VEHICLE: 1,
            ObjectKind.DRONE: 1,
        }
        assert results[0].tracks == []
        assert len(results[-1].tracks) == 3

    def test_track_ids_unique_across_kinds(self, setup, t0):
        _, pipeline, state = setup
        run(pipeline, state, t0)
        ids = [t.track_id for t in pipeline.active_tracks()]
        assert len(ids) == len(set(ids)) == 3

    def test_persons_get_ground_position(self, setup, t0):
        _, pipeline, state = setup
        results = run(pipeline, state, t0)
        person = results[-1].by_kind(ObjectKind.PERSON)[0]
        assert "latitude" in person.attributes and person.attributes["range_m"] > 0
        assert 0 <= person.attributes["bearing_deg"] < 360

    def test_drone_attributes_populated(self, setup, t0):
        _, pipeline, state = setup
        drone = run(pipeline, state, t0)[-1].by_kind(ObjectKind.DRONE)[0]
        for key in (
            "drone_type",
            "threat_score",
            "altitude_m",
            "speed_ms",
            "is_authorized",
            "bearing_deg",
        ):
            assert key in drone.attributes
        assert drone.speed > 0

    def test_flags_disable_tracking(self, setup, t0):
        _, pipeline, state = setup
        results = run(
            pipeline,
            state,
            t0,
            flags=DetectionFlags(persons=False, vehicles=False, drones=True),
        )
        assert {t.kind for r in results for t in r.tracks} == {ObjectKind.DRONE}

    def test_tracks_expire_when_objects_leave(self, setup, t0):
        _, pipeline, state = setup
        run(pipeline, state, t0)
        state["drone"] = False
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        later = t0 + timedelta(seconds=30)
        state["i"] = 0
        result = pipeline.process(frame, None, later)
        assert any(t.kind == ObjectKind.DRONE for t in result.lost_tracks)

    def test_frame_result_serializes(self, setup, t0):
        import json

        _, pipeline, state = setup
        json.dumps(run(pipeline, state, t0)[-1].to_dict())

    def test_swarm_event_fires_once(self, config, fake_yolo_factory, t0):
        state = {"i": 0}

        def script(frame):
            i = state["i"]
            return [
                (100 + i * 10, 40 + k * 120, 124 + i * 10, 64 + k * 120, 0.8, 4)
                for k in range(3)
            ]

        manager = ModelManager(config, model_factory=fake_yolo_factory(script))
        manager.load()
        pipeline = manager.create_pipeline(CameraContext(camera_id=1))
        results = run(pipeline, state, t0, frames=15)
        assert sum(r.swarm_started for r in results) == 1
        assert results[-1].swarm and pipeline.swarm_active

    def test_loitering_flagged(self, config, fake_yolo_factory, t0):
        cfg = config.with_overrides(loiter_seconds=10, track_max_age_seconds=30)
        manager = ModelManager(
            cfg, model_factory=fake_yolo_factory([(100, 300, 140, 380, 0.9, 0)])
        )
        manager.load()
        pipeline = manager.create_pipeline(CameraContext(camera_id=1))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        flagged = []
        for i in range(30):
            result = pipeline.process(frame, None, t0 + timedelta(seconds=i))
            flagged.extend(result.loitering)
        assert len(flagged) == 1

    def test_reset_clears_everything(self, setup, t0):
        _, pipeline, state = setup
        run(pipeline, state, t0)
        pipeline.reset()
        assert pipeline.active_tracks() == []

    def test_night_vision_context_enhances_when_enabled(
        self, config, fake_yolo_factory, t0
    ):
        cfg = config.with_overrides(low_light_mode="auto")
        manager = ModelManager(cfg, model_factory=fake_yolo_factory([]))
        manager.load()
        pipeline = manager.create_pipeline(
            CameraContext(camera_id=1, night_vision=True)
        )
        result = pipeline.process(np.zeros((128, 128, 3), dtype=np.uint8), None, t0)
        assert result.frame_shape == (128, 128, 3)
