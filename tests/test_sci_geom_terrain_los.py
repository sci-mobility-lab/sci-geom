"""terrain_blocks_los: the v0.2.0 clearance-ramp change and known-answer case K9."""
import math

import numpy as np
import pytest

from sci_geom.terrain_los import terrain_blocks_los

flat = lambda xy: np.zeros(len(xy))  # noqa: E731


def _v010(observer, target, ground_z_at, *, clearance_m=0.05, sample_step_m=1.0,
          object_height=0.6, start_margin_m=2.0, end_margin_m=1.0):
    """Verbatim v0.1.0 ramp (max(0, h - clearance)), kept to document what changed."""
    observer = np.asarray(observer, float)
    target = np.asarray(target, float)
    delta_xy = target[:2] - observer[:2]
    dist = float(np.linalg.norm(delta_xy))
    step = min(float(sample_step_m), 1.0)
    if dist <= start_margin_m + end_margin_m + step:
        return False
    samples = np.arange(start_margin_m, dist - end_margin_m + 1e-9, step)
    u = samples / dist
    xy = observer[:2] + u[:, None] * delta_xy
    line_z = observer[2] + u * (target[2] - observer[2])
    ground_z = np.asarray(ground_z_at(xy), float)
    clearance = clearance_m + u * max(0.0, min(object_height, 0.6) - clearance_m)
    over = ground_z + clearance >= line_z
    return bool(np.any(over[:-1] & over[1:]))


@pytest.mark.parametrize("eye", [1.08, 1.37, 1.40])
def test_k9_flat_road_object_at_road_level_is_never_blocked(eye):
    """K9: flat road, object height 0, any distance -> the ground never blocks."""
    for d in range(5, 301, 5):
        assert not terrain_blocks_los([0, 0, eye], [d, 0, 0.0], flat, object_height=0.0), d


@pytest.mark.parametrize("eye, first_bad", [(1.08, 43.2), (1.40, 56.0)])
def test_k9_documents_the_v010_defect(eye, first_bad):
    """v0.1.0 reported a flat road as blocked beyond ~2*eye/clearance metres."""
    assert not _v010([0, 0, eye], [first_bad - 3, 0, 0.0], flat, object_height=0.0)
    for d in (2 * first_bad, 150.0):
        assert _v010([0, 0, eye], [d, 0, 0.0], flat, object_height=0.0), d
        assert not terrain_blocks_los([0, 0, eye], [d, 0, 0.0], flat, object_height=0.0), d


def test_identical_to_v010_for_object_heights_at_or_above_clearance():
    rng = np.random.default_rng(11)

    def crest(peak, height, width):
        return lambda xy: height * np.exp(-((np.asarray(xy)[:, 0] - peak) / width) ** 2)

    for _ in range(400):
        g = crest(rng.uniform(20, 80), rng.uniform(0.0, 6.0), rng.uniform(5, 30))
        obs = [0.0, 0.0, rng.uniform(0.9, 1.6)]
        oh = float(rng.uniform(0.05, 2.0))
        tgt = [rng.uniform(10, 150), 0.0, oh]
        assert terrain_blocks_los(obs, tgt, g, object_height=oh) == _v010(obs, tgt, g, object_height=oh)


def test_low_object_still_blocked_by_a_real_crest():
    crest = lambda xy: 1.5 * np.exp(-((np.asarray(xy)[:, 0] - 40.0) / 8.0) ** 2)  # noqa: E731
    assert terrain_blocks_los([0, 0, 1.40], [90, 0, 0.0], crest, object_height=0.0)


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_invalid_object_height_raises(bad):
    with pytest.raises(ValueError):
        terrain_blocks_los([0, 0, 1.08], [50, 0, 0.0], flat, object_height=bad)
