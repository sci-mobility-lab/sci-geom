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


# Known answers on synthetic corridors (straight path along +x, station 0, targets every 5 m to
# 150 m, VOXEL 0.15, STEP 0.08, clearance 1, min_hit_samples 2, margins 2/1 m). The equivalence
# with SCI_LiDAR's HSO engine is tested in SCI_LiDAR (sci-geom AGENTS.md), not here.
KNOWN = {
    "flat_clear": (flat, None, 1.08, 0.6, (150.0, 150.0), "range"),
    # Last target before the 62.5 m wall is 60 m; the next one (65 m) is behind it.
    "flat_wall": (flat, (62.4, 62.6, -3, 3, 0, 3, 0.05), 1.08, 0.6, (60.0, 60.0), "obstacle"),
    # A 2 m crest centred at 80 m hides a 0.6 m object somewhere past its summit.
    "crest_terrain": (crest, None, 1.08, 0.6, (70.0, 85.0), "terrain"),
}


@pytest.mark.parametrize("name", sorted(KNOWN))
def test_known_answers(name):
    ground, box, eye, h, (low, high), reason = KNOWN[name]
    occ = voxelize(_box(*box), 0.15) if box else set()
    targets, d = _targets(ground)
    res = available_sight_distance_from(
        [0.0, 0.0, float(ground(np.array([[0.0, 0.0]]))[0])], targets, d, occ,
        eye_height_m=eye, object_height_m=h, ground_z_at=ground, max_range_m=150.0,
    )
    assert isinstance(res, SightResult)
    assert low <= res.distance_m <= high
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


def test_path_continuing_beyond_the_cap_is_range_not_end():
    targets, d = _targets(flat, max_d=200.0)
    res = available_sight_distance_from([0, 0, 0], targets, d, set(), eye_height_m=1.08, object_height_m=0.6,
                                        max_range_m=152.0)
    assert (res.distance_m, res.reason) == (150.0, "range")
    assert res.visible[-1] is None  # beyond the cap: not evaluated, not "not visible"


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


def test_walk_back_kernel_refuses_the_ignored_eye_height():
    stations = np.arange(0.0, 61.0, 10.0)
    positions = np.column_stack((stations, np.zeros_like(stations), np.full_like(stations, 1.4)))
    with pytest.raises(TypeError, match="eye_height_m is not used"):
        available_sight_distance(stations, positions, np.array([60.0, 0.0, 1.4]), set(), eye_height_m=1.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert available_sight_distance(stations, positions, np.array([60.0, 0.0, 1.4]), set()) == 60.0


def test_bisection_terminates_below_float_resolution():
    occ = voxelize(_box(62.4, 62.6, -3, 3, 0, 3), 0.15)
    targets, d = _targets(flat)
    res = available_sight_distance_from([0, 0, 0], targets, d, occ, eye_height_m=1.08, object_height_m=0.6,
                                        refine_tolerance_m=1e-16)
    assert res.refined and 60.0 <= res.distance_m < 65.0


def test_refinement_changes_only_the_distance():
    """reason / hit / first_blocked_index keep describing the first blocked SUPPLIED target."""
    occ = voxelize(_box(62.4, 62.6, -3, 3, 0, 3), 0.15)
    targets, d = _targets(flat)
    kw = dict(eye_height_m=1.08, object_height_m=0.6)
    coarse = available_sight_distance_from([0, 0, 0], targets, d, occ, **kw)
    fine = available_sight_distance_from([0, 0, 0], targets, d, occ, refine_tolerance_m=0.05, **kw)
    assert (fine.reason, fine.hit_xyz, fine.first_blocked_index, fine.visible) == (
        coarse.reason, coarse.hit_xyz, coarse.first_blocked_index, coarse.visible)


def test_bisection_stays_on_the_callers_surface():
    """Targets given on a surface 0.3 m above a DEM: refinement must not drop midpoints onto the DEM."""
    dem = lambda xy: np.zeros(len(xy))  # noqa: E731
    occ = voxelize(_box(62.4, 62.6, -3, 3, 0, 3), 0.15)
    targets, d = _targets(lambda xy: np.full(len(xy), 0.3))
    kw = dict(eye_height_m=1.08, object_height_m=0.6, ground_z_at=dem)
    coarse = available_sight_distance_from([0, 0, 0.3], targets, d, occ, **kw)
    fine = available_sight_distance_from([0, 0, 0.3], targets, d, occ, refine_tolerance_m=0.1, **kw)
    assert fine.distance_m >= coarse.distance_m
