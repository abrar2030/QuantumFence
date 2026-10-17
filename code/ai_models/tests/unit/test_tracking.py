import math

import pytest
from ai_models import (
    BoundingBox,
    DroneDetector,
    ModelConfig,
    MultiObjectTracker,
    ObjectKind,
)
from ai_models.tracking import KalmanBoxFilter

pytestmark = pytest.mark.unit


class TestKalman:
    def test_converges_to_constant_velocity(self):
        kf = KalmanBoxFilter(BoundingBox(0, 0, 20, 20))
        for i in range(1, 30):
            kf.predict(0.1)
            kf.update(BoundingBox(i * 5, 0, 20, 20))
        vx, vy = kf.velocity
        assert vx == pytest.approx(50, rel=0.15)
        assert abs(vy) < 5

    def test_predict_zero_dt_noop(self):
        kf = KalmanBoxFilter(BoundingBox(10, 10, 20, 20))
        before = kf.x.copy()
        kf.predict(0.0)
        assert (kf.x == before).all()

    def test_size_never_collapses(self):
        kf = KalmanBoxFilter(BoundingBox(0, 0, 2, 2))
        for _ in range(20):
            kf.predict(1.0)
            kf.update(BoundingBox(0, 0, 1, 1))
        assert kf.bbox.width >= 1 and kf.bbox.height >= 1


def _tracker(**kw):
    return MultiObjectTracker(ObjectKind.PERSON, **kw)


class TestMultiObjectTracker:
    def test_new_track_tentative_until_min_hits(self, make_detection, t0, step):
        tr = _tracker(min_hits=2)
        u1 = tr.update([make_detection()], t0)
        assert u1.new_confirmed == []
        u2 = tr.update([make_detection(x=102)], step(t0, 0.2))
        assert len(u2.new_confirmed) == 1
        assert u2.new_confirmed[0].confirmed

    def test_min_hits_one_confirms_immediately(self, make_detection, t0):
        tr = _tracker(min_hits=1)
        assert len(tr.update([make_detection()], t0).new_confirmed) == 1

    def test_identity_persists_across_frames(self, make_detection, t0, step):
        tr = _tracker(min_hits=1)
        ids = set()
        for i in range(10):
            u = tr.update([make_detection(x=100 + i * 4)], step(t0, i * 0.2))
            ids.update(t.track_id for t in u.matched)
        assert len(ids) == 1

    def test_two_objects_tracked_separately(self, make_detection, t0, step):
        tr = _tracker(min_hits=1)
        for i in range(5):
            tr.update(
                [make_detection(x=100 + i * 3), make_detection(x=800 + i * 3)],
                step(t0, i * 0.2),
            )
        assert len(tr.tracks) == 2

    def test_track_pruned_after_max_age(self, make_detection, t0, step):
        tr = _tracker(min_hits=1, max_age_seconds=2.0)
        tr.update([make_detection()], t0)
        tr.update([], step(t0, 1.0))
        assert len(tr.tracks) == 1
        u = tr.update([], step(t0, 3.5))
        assert len(tr.tracks) == 0 and len(u.lost) == 1

    def test_tentative_track_dropped_quickly(self, make_detection, t0, step):
        tr = _tracker(min_hits=3, max_age_seconds=5.0)
        tr.update([make_detection()], t0)
        tr.update([], step(t0, 1.5))
        assert len(tr.tracks) == 0

    def test_far_detection_creates_new_track(self, make_detection, t0, step):
        tr = _tracker(min_hits=1, max_distance_px=50)
        tr.update([make_detection(x=100)], t0)
        tr.update([make_detection(x=900)], step(t0, 0.2))
        assert len(tr.tracks) == 2

    def test_small_object_matched_by_distance(self, make_detection, t0, step):
        tr = MultiObjectTracker(ObjectKind.DRONE, min_hits=1, max_distance_px=100)
        tr.update([make_detection(ObjectKind.DRONE, x=100, y=100, w=6, h=6)], t0)
        tr.update(
            [make_detection(ObjectKind.DRONE, x=140, y=100, w=6, h=6)], step(t0, 0.2)
        )
        assert len(tr.tracks) == 1

    def test_speed_and_direction(self, make_detection, t0, step):
        tr = _tracker(min_hits=1)
        for i in range(12):
            tr.update([make_detection(x=100 + i * 10)], step(t0, i * 0.2))
        track = next(iter(tr.tracks.values()))
        assert track.speed > 20
        assert abs(track.direction) < 20

    def test_stationary_has_no_direction(self, make_detection, t0, step):
        tr = _tracker(min_hits=1)
        for i in range(5):
            tr.update([make_detection()], step(t0, i * 0.2))
        assert next(iter(tr.tracks.values())).direction is None

    def test_approaching_when_box_grows(self, make_detection, t0, step):
        tr = _tracker(min_hits=1)
        for i in range(8):
            tr.update(
                [make_detection(x=300, y=200, w=20 + i * 6, h=20 + i * 6)],
                step(t0, i * 0.2),
            )
        assert next(iter(tr.tracks.values())).is_approaching

    def test_not_approaching_with_few_samples(self, make_detection, t0):
        tr = _tracker(min_hits=1)
        tr.update([make_detection()], t0)
        assert not next(iter(tr.tracks.values())).is_approaching

    def test_to_dict_shape(self, make_detection, t0, step):
        tr = _tracker(min_hits=1)
        for i in range(3):
            tr.update([make_detection(x=100 + i)], step(t0, i * 0.2))
        data = next(iter(tr.tracks.values())).to_dict()
        for key in (
            "track_id",
            "kind",
            "bbox",
            "speed_px_s",
            "trajectory",
            "confirmed",
            "duration_seconds",
        ):
            assert key in data
        assert data["kind"] == "person"

    def test_reset_clears_state(self, make_detection, t0):
        tr = _tracker(min_hits=1)
        tr.update([make_detection()], t0)
        tr.reset()
        assert tr.tracks == {}

    def test_shared_id_source_unique(self, make_detection, t0):
        import itertools

        ids = itertools.count(1)
        a = MultiObjectTracker(ObjectKind.PERSON, min_hits=1, id_source=ids)
        b = MultiObjectTracker(ObjectKind.VEHICLE, min_hits=1, id_source=ids)
        a.update([make_detection()], t0)
        b.update([make_detection(ObjectKind.VEHICLE)], t0)
        assert set(a.tracks) | set(b.tracks) == {1, 2}

    def test_attributes_copied_from_detection(self, make_detection, t0):
        tr = _tracker(min_hits=1)
        tr.update([make_detection(drone_type="quadcopter")], t0)
        assert next(iter(tr.tracks.values())).attributes["drone_type"] == "quadcopter"


class TestDroneDetector:
    def _drone(self, make_detection, x, conf=0.9, **kw):
        return make_detection(ObjectKind.DRONE, x=x, y=100, w=20, h=20, conf=conf, **kw)

    def test_low_confidence_ignored(self, make_detection, t0):
        dd = DroneDetector(ModelConfig(track_min_hits=1))
        update = dd.process_frame(
            (720, 1280, 3), [self._drone(make_detection, 100, conf=0.1)], t0
        )
        assert update.matched == [] and dd.active_tracks == {}

    def test_other_kinds_ignored(self, make_detection, t0):
        dd = DroneDetector(ModelConfig(track_min_hits=1))
        dd.process_frame((720, 1280, 3), [make_detection(ObjectKind.PERSON)], t0)
        assert dd.active_tracks == {}

    def test_threat_score_in_range(self, make_detection, t0, step):
        dd = DroneDetector(ModelConfig(track_min_hits=1))
        for i in range(10):
            dd.process_frame(
                (720, 1280, 3),
                [self._drone(make_detection, 100 + i * 8)],
                step(t0, i * 0.2),
            )
        track = next(iter(dd.active_tracks.values()))
        assert 0.0 <= track.attributes["threat_score"] <= 1.0

    def test_authorized_zone_lowers_score(self, make_detection, t0, step):
        from ai_models import CameraModel

        cam = CameraModel(latitude=33.0, longitude=73.0, heading_deg=0.0)
        base = DroneDetector(ModelConfig(track_min_hits=1), camera=cam)
        authorized = DroneDetector(ModelConfig(track_min_hits=1), camera=cam)
        authorized.add_authorized_zone(33.0, 73.0, 5000)
        for dd in (base, authorized):
            for i in range(5):
                dd.process_frame(
                    (720, 1280, 3),
                    [self._drone(make_detection, 600 + i)],
                    step(t0, i * 0.2),
                )
        a = next(iter(base.active_tracks.values())).attributes
        b = next(iter(authorized.active_tracks.values())).attributes
        assert b["is_authorized"] and not a["is_authorized"]
        assert b["threat_score"] < a["threat_score"]

    def test_camera_adds_geo_and_altitude(self, make_detection, t0):
        from ai_models import CameraModel

        dd = DroneDetector(
            ModelConfig(track_min_hits=1), camera=CameraModel(33.0, 73.0)
        )
        dd.process_frame(
            (720, 1280, 3),
            [self._drone(make_detection, 600, drone_type="quadcopter")],
            t0,
        )
        attrs = next(iter(dd.active_tracks.values())).attributes
        assert (
            "latitude" in attrs and "altitude_m" in attrs and attrs["altitude_m"] >= 0
        )

    def test_swarm_detected_for_parallel_drones(self, make_detection, t0, step):
        dd = DroneDetector(ModelConfig(track_min_hits=1))
        for i in range(10):
            dets = [self._drone(make_detection, 100 + i * 10 + k * 5) for k in range(3)]
            for k, d in enumerate(dets):
                d.bbox = BoundingBox(100 + i * 10, 100 + k * 120, 20, 20)
            dd.process_frame((720, 1280, 3), dets, step(t0, i * 0.2))
        assert dd.detect_swarm()
        assert dd.get_summary()["swarm_detected"]

    def test_no_swarm_with_two_drones(self, make_detection, t0, step):
        dd = DroneDetector(ModelConfig(track_min_hits=1))
        for i in range(10):
            dets = [self._drone(make_detection, 0) for _ in range(2)]
            for k, d in enumerate(dets):
                d.bbox = BoundingBox(100 + i * 10, 100 + k * 200, 20, 20)
            dd.process_frame((720, 1280, 3), dets, step(t0, i * 0.2))
        assert not dd.detect_swarm()

    def test_summary_when_empty(self):
        summary = DroneDetector().get_summary()
        assert summary["active_drones"] == 0 and summary["max_threat_score"] == 0.0

    def test_math_helper_sanity(self):
        assert math.isclose(1.0, 1.0)
