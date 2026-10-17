from datetime import timedelta

import pytest
from ai_models import CameraModel, ModelConfig, MultiObjectTracker, ObjectKind
from ai_models.analysis import (
    LoiteringDetector,
    ThreatScorer,
    circular_variance,
    classify_drone_type,
    destination_point,
    detect_swarm,
    haversine_m,
    real_width_for,
)

pytestmark = pytest.mark.unit


class TestDroneClassification:
    def test_micro_checked_before_aspect_ratio(self):
        assert classify_drone_type(10, 2, 0.0001) == "micro_drone"

    def test_wide_is_fixed_wing(self):
        assert classify_drone_type(100, 30, 0.01) == "fixed_wing"

    def test_square_is_quadcopter(self):
        assert classify_drone_type(40, 40, 0.01) == "quadcopter"

    def test_label_overrides_geometry(self):
        assert classify_drone_type(40, 40, 0.01, "fixed-wing") == "fixed_wing"
        assert classify_drone_type(100, 20, 0.01, "quadcopter") == "quadcopter"

    def test_real_width_lookup(self):
        assert real_width_for("fixed_wing") > real_width_for("micro_drone")
        assert real_width_for("nonsense") == real_width_for(None)


class TestSwarm:
    def test_same_direction_is_swarm(self):
        assert detect_swarm([10, 12, 8, 11])

    def test_wraparound_headings_are_similar(self):
        assert detect_swarm([350, 355, 5, 10])

    def test_opposite_directions_not_swarm(self):
        assert not detect_swarm([0, 180, 90, 270])

    def test_too_few_drones(self):
        assert not detect_swarm([10, 11])

    def test_circular_variance_bounds(self):
        assert circular_variance([45, 45, 45]) == pytest.approx(0.0, abs=1e-9)
        assert circular_variance([]) == 1.0


class TestGeometry:
    def test_haversine_zero_and_known(self):
        assert haversine_m(10, 10, 10, 10) == 0
        assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=0.01)

    def test_destination_roundtrip(self):
        lat, lng = destination_point(33.0, 73.0, 90, 1000)
        assert haversine_m(33.0, 73.0, lat, lng) == pytest.approx(1000, rel=0.01)
        assert lng > 73.0

    def test_azimuth_follows_heading_and_offset(self):
        cam = CameraModel(33, 73, heading_deg=90, hfov_deg=90)
        assert cam.azimuth(640, 1280) == pytest.approx(90)
        assert cam.azimuth(1280, 1280) == pytest.approx(135, abs=0.5)
        assert cam.azimuth(0, 1280) == pytest.approx(45, abs=0.5)

    def test_lower_in_frame_is_closer_on_ground(self):
        cam = CameraModel(33, 73, tilt_deg=15, mount_height_m=5)
        shape = (720, 1280, 3)
        assert cam.ground_range(700, shape) < cam.ground_range(450, shape)

    def test_above_horizon_clamped_to_max_range(self):
        cam = CameraModel(33, 73, tilt_deg=0, max_range_m=250)
        assert cam.ground_range(10, (720, 1280, 3)) == 250

    def test_locate_ground_requires_position(self):
        assert CameraModel().locate_ground(640, 500, (720, 1280, 3)) is None
        point = CameraModel(33, 73, heading_deg=0).locate_ground(
            640, 600, (720, 1280, 3)
        )
        assert point.lat > 33 and point.range_m > 0

    def test_aerial_altitude_and_range(self):
        cam = CameraModel(33, 73, tilt_deg=0, mount_height_m=4)
        point, altitude = cam.locate_aerial(640, 100, 20, (720, 1280, 3), 0.35)
        assert altitude > 4
        assert point.range_m > 0
        near_point, near_alt = cam.locate_aerial(640, 100, 100, (720, 1280, 3), 0.35)
        assert near_point.range_m < point.range_m

    def test_aerial_without_position(self):
        point, altitude = CameraModel().locate_aerial(
            640, 100, 20, (720, 1280, 3), 0.35
        )
        assert point is None and altitude >= 0

    def test_object_range_clamped(self):
        assert CameraModel(max_range_m=100).object_range(0, 1280, 0.35) == 100

    def test_pixel_speed_conversion(self):
        cam = CameraModel(hfov_deg=90)
        assert cam.pixel_speed_to_ms(640, 10, 1280) == pytest.approx(10)

    def test_with_updates(self):
        assert CameraModel().with_updates(heading_deg=45).heading_deg == 45


def _confirmed_track(make_detection, t0, seconds=0.0, samples=5, grow=0.0):
    tr = MultiObjectTracker(ObjectKind.DRONE, min_hits=1)
    for i in range(samples):
        ts = t0 + timedelta(seconds=seconds * i / max(samples - 1, 1))
        tr.update(
            [
                make_detection(
                    ObjectKind.DRONE, x=600, y=300, w=20 + grow * i, h=20 + grow * i
                )
            ],
            ts,
        )
    return next(iter(tr.tracks.values()))


class TestThreatScorer:
    def test_score_bounded(self, make_detection, t0):
        scorer = ThreatScorer(ModelConfig())
        track = _confirmed_track(make_detection, t0, seconds=90, grow=5)
        assert 0.0 <= scorer.score(track, (720, 1280, 3)) <= 1.0

    def test_longer_presence_raises_score(self, make_detection, t0):
        scorer = ThreatScorer(ModelConfig())
        short = _confirmed_track(make_detection, t0, seconds=1)
        long = _confirmed_track(make_detection, t0, seconds=90)
        assert scorer.score(long, (720, 1280, 3)) > scorer.score(short, (720, 1280, 3))

    def test_approaching_raises_score(self, make_detection, t0):
        scorer = ThreatScorer(ModelConfig())
        flat = _confirmed_track(make_detection, t0, seconds=2)
        grow = _confirmed_track(make_detection, t0, seconds=2, grow=6)
        assert scorer.score(grow, (720, 1280, 3)) > scorer.score(flat, (720, 1280, 3))

    def test_authorized_lowers_score(self, make_detection, t0):
        scorer = ThreatScorer(ModelConfig())
        track = _confirmed_track(make_detection, t0, seconds=2)
        assert scorer.score(track, (720, 1280, 3), True) < scorer.score(
            track, (720, 1280, 3), False
        )

    def test_is_high(self):
        scorer = ThreatScorer(ModelConfig(high_threat_score=0.7))
        assert scorer.is_high(0.8) and not scorer.is_high(0.5)


class TestLoitering:
    def _person_track(self, make_detection, t0, duration, moving):
        tr = MultiObjectTracker(ObjectKind.PERSON, min_hits=1, max_age_seconds=1000)
        steps = int(duration / 2) + 1
        for i in range(steps):
            x = 100 + (i * 40 if moving else 0)
            tr.update([make_detection(x=x, y=300)], t0 + timedelta(seconds=i * 2))
        return next(iter(tr.tracks.values()))

    def test_stationary_long_presence_flagged_once(self, make_detection, t0):
        det = LoiteringDetector(min_seconds=30, radius_px=60)
        track = self._person_track(make_detection, t0, 40, moving=False)
        assert det.check(track)
        assert not det.check(track)

    def test_moving_person_not_flagged(self, make_detection, t0):
        det = LoiteringDetector(min_seconds=30, radius_px=60)
        assert not det.check(self._person_track(make_detection, t0, 40, moving=True))

    def test_short_presence_not_flagged(self, make_detection, t0):
        det = LoiteringDetector(min_seconds=30, radius_px=60)
        assert not det.check(self._person_track(make_detection, t0, 10, moving=False))

    def test_forget_allows_reflag(self, make_detection, t0):
        det = LoiteringDetector(min_seconds=30, radius_px=60)
        track = self._person_track(make_detection, t0, 40, moving=False)
        assert det.check(track)
        det.forget(track.track_id)
        assert det.check(track)
