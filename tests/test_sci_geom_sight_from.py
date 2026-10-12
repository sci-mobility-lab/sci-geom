"""available_sight_distance_from: the observer-fixed kernel of the corridor engines (v0.2.0)."""
import warnings

import numpy as np
import pytest

from sci_geom.sight import SightResult, available_sight_distance, available_sight_distance_from
from sci_geom.voxel_los import voxelize

flat = lambda xy: np.zeros(len(xy))  # noqa: E731
crest = lambda xy: 2.0 * np.exp(-((np.asarray(xy)[:, 0] - 80.0) / 10.0) ** 2)  # noqa: E731


def _box(x0, x1, y0, y1, z0, z1, sp=0.05):
    xs = np.arange(x0, x1 + 1e-9, sp)
    ys = np.arange(y0, y1 + 1e-9, sp)
    zs = np.arange(z0, z1 + 1e-9, sp)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def _targets(ground, step=5.0, max_d=150.0):
    d = np.arange(step, max_d + 1e-9, step)
    xyz = np.column_stack([d, np.zeros_like(d), ground(np.column_stack([d, np.zeros_like(d)]))])
    return xyz, d


# Golden values produced on 2026-10-12 by SCI_LiDAR platform/engine/sci_studio/hso_corridor.py
# `_available_sight_distance` (commit 5f0ef90, sci-geom v0.1.0 voxel_los) on these exact fixtures:
# straight trajectory along +x, station 0, target_step 5 m, max_range 150 m, VOXEL 0.15, STEP 0.08,
# clearance 1, min_hit_samples 2, margins 2/1 m. The new kernel must reproduce HSO on them.
GOLDEN = {
    "flat_clear": (flat, None, 1.08, 0.6, 150.0, "range"),
    "flat_wall": (flat, (62.4, 62.6, -3, 3, 0, 3, 0.05), 1.08, 0.6, 60.0, "obstacle"),
    "crest_terrain": (crest, None, 1.08, 0.6, 80.0, "terrain"),
    "low_obstacle_driver": (flat, (39.8, 40.2, -3, 3, 0, 1.30, 0.03), 1.08, 1.08, 40.0, "obstacle"),
    "low_obstacle_cyclist": (flat, (39.8, 40.2, -3, 3, 0, 1.30, 0.03), 1.40, 1.08, 40.0, "obstacle"),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_reproduces_hso_golden_values(name):
    ground, box, eye, h, expected, reason = GOLDEN[name]
    occ = voxelize(_box(*box), 0.15) if box else set()
    targets, d = _targets(ground)
    res = available_sight_distance_from(
        [0.0, 0.0, float(ground(np.array([[0.0, 0.0]]))[0])], targets, d, occ,
        eye_height_m=eye, object_height_m=h, ground_z_at=ground, max_range_m=150.0,
    )
    assert isinstance(res, SightResult)
    assert res.distance_m == expected
    assert res.reason == reason


def test_eye_height_is_applied():
    """The v0.1.0 defect: eye_height_m was accepted and ignored. Here it must matter."""
    occ = voxelize(_box(29.9, 30.1, -3, 3, 0, 1.25, 0.02), 0.05)
    targets, d = _targets(flat, step=2.0, max_d=80.0)
    kw = dict(object_height_m=1.30, voxel=0.05, step=0.02, clearance=0, max_range_m=80.0)
    low = available_sight_distance_from([0, 0, 0], targets, d, occ, eye_height_m=1.08, **kw)
    high = available_sight_distance_from([0, 0, 0], targets, d, occ, eye_height_m=1.40, **kw)
    assert low.reason == "obstacle" and low.distance_m < 80.0
    assert high.distance_m > low.distance_m


def test_bisection_refines_below_target_spacing():
    occ = voxelize(_box(62.4, 62.6, -3, 3, 0, 3), 0.15)
    targets, d = _targets(flat)
    coarse = available_sight_distance_from([0, 0, 0], targets, d, occ, eye_height_m=1.08,
                                           object_height_m=0.6, max_range_m=150.0)
    fine = available_sight_distance_from([0, 0, 0], targets, d, occ, eye_height_m=1.08,
                                         object_height_m=0.6, max_range_m=150.0, refine_tolerance_m=0.1)
    assert coarse.distance_m == 60.0 and not coarse.refined
    assert fine.refined and 60.0 <= fine.distance_m < 65.0
    assert fine.distance_m > coarse.distance_m


def test_not_evaluated_is_not_reported_as_not_visible():
    occ = voxelize(_box(62.4, 62.6, -3, 3, 0, 3), 0.15)
    targets, d = _targets(flat)
    res = available_sight_distance_from([0, 0, 0], targets, d, occ, eye_height_m=1.08, object_height_m=0.6)
    k = res.first_blocked_index
    assert res.visible[k] is False
    assert all(v is True for v in res.visible[:k])
    assert all(v is None for v in res.visible[k + 1:])
    full = available_sight_distance_from([0, 0, 0], targets, d, occ, eye_height_m=1.08, object_height_m=0.6,
                                         evaluate_all=True)
    assert None not in full.visible[: len(d)]
    assert full.distance_m == res.distance_m


def test_end_of_path_before_range_and_no_targets():
    targets, d = _targets(flat, max_d=40.0)
    res = available_sight_distance_from([0, 0, 0], targets, d, set(), eye_height_m=1.08, object_height_m=0.6,
                                        max_range_m=150.0)
    assert (res.distance_m, res.reason) == (40.0, "end")
    none = available_sight_distance_from([0, 0, 0], targets, d + 500.0, set(), eye_height_m=1.08,
                                         object_height_m=0.6, max_range_m=150.0)
    assert (none.distance_m, none.reason) == (0.0, "no_targets")


def test_k9_bicycle_stopping_case_on_flat_road_reaches_the_range_cap():
    targets, d = _targets(flat, max_d=150.0)
    for eye in (1.08, 1.37, 1.40):
        res = available_sight_distance_from([0, 0, 0], targets, d, set(), eye_height_m=eye, object_height_m=0.0,
                                            ground_z_at=flat, max_range_m=150.0)
        assert (res.distance_m, res.reason) == (150.0, "range")


@pytest.mark.parametrize("kwargs", [
    dict(eye_height_m=-1.0, object_height_m=0.6),
    dict(eye_height_m=1.08, object_height_m=float("nan")),
])
def test_invalid_heights_raise(kwargs):
    targets, d = _targets(flat)
    with pytest.raises(ValueError):
        available_sight_distance_from([0, 0, 0], targets, d, set(), **kwargs)


def test_unordered_distances_raise():
    with pytest.raises(ValueError):
        available_sight_distance_from([0, 0, 0], [[5, 0, 0], [3, 0, 0]], [5.0, 3.0], set(),
                                      eye_height_m=1.08, object_height_m=0.6)


def test_walk_back_kernel_warns_that_eye_height_is_ignored():
    stations = np.arange(0.0, 61.0, 10.0)
    positions = np.column_stack((stations, np.zeros_like(stations), np.full_like(stations, 1.4)))
    with pytest.warns(DeprecationWarning, match="eye_height_m is ignored"):
        available_sight_distance(stations, positions, np.array([60.0, 0.0, 1.4]), set(), eye_height_m=1.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert available_sight_distance(stations, positions, np.array([60.0, 0.0, 1.4]), set()) == 60.0
