import numpy as np
import pytest
from ai_models import BoundingBox, InvalidFrameError, ObjectKind, RawDetection
from ai_models.core.types import Detection
from ai_models.postprocessing import (
    filter_drone_candidates,
    filter_min_size,
    non_max_suppression,
)
from ai_models.preprocessing import (
    crop,
    enhance_low_light,
    ensure_bgr,
    generate_tiles,
    is_low_light,
    mean_luminance,
    prepare_frame,
)

pytestmark = pytest.mark.unit


class TestEnsureBgr:
    def test_grayscale_expanded(self):
        out = ensure_bgr(np.zeros((10, 10), dtype=np.uint8))
        assert out.shape == (10, 10, 3)

    def test_single_channel_expanded(self):
        assert ensure_bgr(np.zeros((10, 10, 1), dtype=np.uint8)).shape == (10, 10, 3)

    def test_alpha_dropped(self):
        assert ensure_bgr(np.zeros((10, 10, 4), dtype=np.uint8)).shape == (10, 10, 3)

    def test_float_converted_to_uint8(self):
        out = ensure_bgr(np.full((10, 10, 3), 300.0))
        assert out.dtype == np.uint8 and out.max() == 255

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "frame",
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((4, 4, 5), dtype=np.uint8),
        ],
    )
    def test_invalid_rejected(self, bad):
        with pytest.raises(InvalidFrameError):
            ensure_bgr(bad)


class TestLowLight:
    def test_luminance_and_flag(self):
        dark = np.full((64, 64, 3), 10, dtype=np.uint8)
        bright = np.full((64, 64, 3), 200, dtype=np.uint8)
        assert mean_luminance(dark) < mean_luminance(bright)
        assert is_low_light(dark) and not is_low_light(bright)

    def test_enhance_brightens_dark_frame(self):
        rng = np.random.default_rng(1)
        dark = rng.integers(0, 40, size=(128, 128, 3), dtype=np.uint8)
        assert mean_luminance(enhance_low_light(dark)) >= mean_luminance(dark)

    def test_prepare_modes(self):
        dark = np.full((64, 64, 3), 5, dtype=np.uint8)
        assert prepare_frame(dark, "off").tolist() == dark.tolist()
        assert prepare_frame(dark, "on").shape == dark.shape
        assert prepare_frame(dark, "auto").shape == dark.shape
        assert prepare_frame(dark, "off", force_enhance=True).shape == dark.shape


class TestTiling:
    def test_tiles_cover_frame(self):
        tiles = generate_tiles((720, 1280, 3), 640, 0.2)
        assert len(tiles) > 1
        assert min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 1280
        assert min(t[1] for t in tiles) == 0 and max(t[3] for t in tiles) == 720

    def test_small_frame_single_tile(self):
        assert generate_tiles((100, 100, 3), 640, 0.2) == [(0, 0, 100, 100)]

    def test_crop_shape(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert crop(frame, (10, 20, 110, 220)).shape == (200, 100, 3)


class TestNms:
    def _raw(self, x, conf, cls=0):
        return RawDetection(cls, "c", conf, BoundingBox(x, 0, 100, 100))

    def test_overlapping_suppressed(self):
        kept = non_max_suppression([self._raw(0, 0.9), self._raw(5, 0.8)], 0.5)
        assert len(kept) == 1 and kept[0].confidence == 0.9

    def test_different_classes_kept(self):
        kept = non_max_suppression([self._raw(0, 0.9, 0), self._raw(5, 0.8, 1)], 0.5)
        assert len(kept) == 2

    def test_disjoint_kept(self):
        assert len(non_max_suppression([self._raw(0, 0.9), self._raw(500, 0.8)])) == 2


class TestFilters:
    def _det(self, x, y, w, h):
        return Detection(ObjectKind.DRONE, "drone", 0.9, BoundingBox(x, y, w, h))

    def test_min_size(self):
        dets = [self._det(0, 0, 1, 1), self._det(0, 0, 10, 10)]
        assert len(filter_min_size(dets, 2, 2)) == 1

    def test_drone_candidate_area_and_height(self):
        shape = (720, 1280, 3)
        small_high = self._det(100, 50, 40, 30)
        too_big = self._det(0, 0, 900, 600)
        too_low = self._det(100, 650, 40, 30)
        kept = filter_drone_candidates([small_high, too_big, too_low], shape, 0.15, 0.7)
        assert kept == [small_high]

    def test_drone_candidate_without_height_limit(self):
        low = self._det(100, 650, 40, 30)
        assert filter_drone_candidates([low], (720, 1280, 3), 0.15) == [low]
