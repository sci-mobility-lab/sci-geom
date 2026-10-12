"""Available sight distance along a corridor, at a given eye height.

Two properties are the point of this module, and both are easy to lose:

**1. Available sight distance is the CONTINUOUS run**, walked back from the target until visibility
first breaks -- not the furthest station from which the target happens to be visible. On a corridor
where an object is visible, hidden, then visible again, those two readings differ by the whole
corridor length, and the wrong one silently inflates every downstream result.

**2. Two eye points, binding value governs.** Cyclist 1.40 m and driver 1.08 m at the same conflict
point. A 1.30 m parked vehicle can hide the seated driver while the rider sees over it.

⚠️ **This does NOT reproduce the numbers from the synthetic bikeway prototype, and it should
not.** The prototype uses a pure height gate (`plan_geometry.py:55`):

    blocks = obstruction_height >= EYE[eye]

so a 1.30 m obstruction ALWAYS blocks the 1.08 m driver and NEVER blocks the 1.40 m cyclist,
wherever it sits. Real 3D occlusion depends on the height of the *sight line* at the obstruction,
which depends on both endpoint heights and on where along the line the obstruction sits. Verified:
with a 1.30 m obstruction halfway to a target at 1.40 m, the driver's line is at 1.24 m there
(blocked, 30 m available) and the cyclist's is at 1.40 m (clear, 60 m). Move the same obstruction
close to the target and both are blocked.

The prototype's DECISION logic (gates, scores, treatment classes) is reusable. Its OCCLUSION model
is a 2D simplification, and the error it introduces changes sign with geometry. `DELIVERABLES_SPEC.md`
should not be read as promising that real runs reproduce prototype figures.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from sci_geom.terrain_los import terrain_blocks_los
from sci_geom.voxel_los import blocked, voxelize


def available_sight_distance(
    stations,
    positions,
    target_xyz,
    occupied,
    *,
    eye_height_m=None,
    voxel=0.15,
    step=0.08,
    clearance=1,
    min_hit_samples=2,
    start_margin=2.0,
    end_margin=1.0,
    max_range_m=250.0,
    ground_z_at=None,
    object_height_m=0.6,
) -> float:
    """Return the uninterrupted visible run ending at the last station.

    ⚠️ **Pass `ground_z_at` on any corridor that is not flat.** Two occlusion layers run, in the
    same order as `hso_corridor.py`: the analytic terrain profile first, then the voxel ray march.
    Without the terrain layer the ground itself can never block, because occupancy is built from
    points *above* the ground by construction. Measured on pine_054823 (2537 HSO stations):
    **66 % of stations are blocked by terrain, 32 % by obstacles.** A voxel-only run on that
    corridor returns the range cap almost everywhere, and the number is not conservative -- it is
    structurally blind.

    `ground_z_at` is a callable (N,2) xy -> (N,) ground elevation. None disables the terrain layer.

    ⚠️ **`positions` are EYE-LEVEL observer points; `eye_height_m` is not used.** Up to v0.1.0 the
    parameter was required and silently ignored, so passing 1.08 or 1.40 changed nothing unless the
    caller had already added the height to `positions`. It now only triggers a DeprecationWarning.
    Use `available_sight_distance_from`, which takes ground points and applies the eye height.
    """
    if eye_height_m is not None:
        warnings.warn(
            "available_sight_distance(): eye_height_m is ignored -- positions must already be at "
            "eye level. Use available_sight_distance_from(), which applies the eye height.",
            DeprecationWarning,
            stacklevel=2,
        )
    stations = np.asarray(stations, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if stations.ndim != 1 or positions.shape != (len(stations), 3):
        raise ValueError("stations must be 1-D and positions must have shape (N, 3)")
    if not len(stations):
        return 0.0

    target_station = stations[-1]
    available = 0.0
    for station, observer in zip(stations[::-1], positions[::-1]):
        distance_back = float(target_station - station)
        if distance_back > max_range_m:
            return float(max_range_m)
        if ground_z_at is not None and terrain_blocks_los(
            observer,
            target_xyz,
            ground_z_at,
            sample_step_m=1.0,
            clearance_m=0.05,
            object_height=object_height_m,
            start_margin_m=start_margin,
            end_margin_m=end_margin,
        ):
            break
        is_blocked, _hit, _distance = blocked(
            observer,
            target_xyz,
            occupied,
            voxel,
            step,
            clearance,
            min_hit_samples,
            start_margin,
            end_margin,
        )
        if is_blocked:
            break
        available = min(distance_back, float(max_range_m))
        if available == max_range_m:
            break
    return available


@dataclass(frozen=True)
class SightResult:
    """Outcome of one observer-fixed sight-distance reading.

    `distance_m` is the CONTINUOUS visible run: the distance of the last target seen before the
    first one that is not (refined by bisection when asked). `reason` says why the run ended:

    - ``"range"``     -- every target up to `max_range_m` is visible; `distance_m` is the cap;
    - ``"end"``       -- every target supplied is visible but the path ends before the cap;
    - ``"terrain"``   -- the ground itself breaks the sight line (`terrain_blocks_los`);
    - ``"obstacle"``  -- the voxel occupancy breaks it (`blocked`); `hit_xyz` is the first hit;
    - ``"no_targets"`` -- nothing within range was supplied; `distance_m` is 0.

    `visible` has one entry per supplied target: True/False where evaluated, None where the
    evaluation stopped (after the first block, unless `evaluate_all=True`) or the target lies
    beyond `max_range_m`. "Not evaluated" is never reported as "not visible".
    """

    distance_m: float
    reason: str
    hit_xyz: tuple[float, float, float] | None
    first_blocked_index: int | None
    visible: tuple[bool | None, ...]
    refined: bool


def available_sight_distance_from(
    observer_ground_xyz,
    targets_ground_xyz,
    distances_m,
    occupied,
    *,
    eye_height_m,
    object_height_m,
    voxel=0.15,
    step=0.08,
    clearance=1,
    min_hit_samples=2,
    start_margin=2.0,
    end_margin=1.0,
    max_range_m=250.0,
    ground_z_at=None,
    terrain_sample_step_m=1.0,
    terrain_clearance_m=0.05,
    refine_tolerance_m=None,
    evaluate_all=False,
) -> SightResult:
    """Available sight distance from a FIXED observer to targets moving away along a path.

    This is the geometry of the corridor engines (HSO, stop-line sight distance, ISD): the observer
    stands at a station and the target object is moved outward, `distances_m` apart, until it is
    first lost. `available_sight_distance` above is the opposite geometry (fixed target, observer
    walked back); the two are not interchangeable, because the margins and the terrain clearance
    ramp are asymmetric between eye and target.

    Heights are APPLIED here, unlike `available_sight_distance`:
    eye = `observer_ground_xyz` + `eye_height_m`, each target = its ground point + `object_height_m`.

    Per target, the two occlusion layers run in the HSO order -- terrain first
    (`terrain_blocks_los`), then the voxel march (`blocked`) -- and the first target that fails
    either ends the run. With `refine_tolerance_m`, the transition between the last visible target
    and the first blocked one is bisected (ground interpolated linearly, or read from `ground_z_at`
    when given) until the bracket is narrower than the tolerance, so the reading is no longer
    quantised to the target spacing.

    `targets_ground_xyz` (N,3) must be ordered by strictly increasing `distances_m` (N,).
    """
    observer_ground = np.asarray(observer_ground_xyz, dtype=np.float64).reshape(3)
    targets = np.asarray(targets_ground_xyz, dtype=np.float64).reshape(-1, 3)
    dists = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    if len(targets) != len(dists):
        raise ValueError("targets_ground_xyz and distances_m must have the same length")
    if len(dists) > 1 and not np.all(np.diff(dists) > 0.0):
        raise ValueError("distances_m must be strictly increasing")
    for name, value in (("eye_height_m", eye_height_m), ("object_height_m", object_height_m)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    eye = observer_ground + np.array([0.0, 0.0, float(eye_height_m)])
    lift = np.array([0.0, 0.0, float(object_height_m)])

    def ground_at(xyz):
        if ground_z_at is None:
            return xyz
        z = float(np.asarray(ground_z_at(np.asarray(xyz[:2], dtype=np.float64)[None]))[0])
        return np.array([xyz[0], xyz[1], z])

    def test(ground_xyz):
        """Return (reason, hit) -- reason None when the target is visible."""
        target = ground_xyz + lift
        if ground_z_at is not None and terrain_blocks_los(
            eye,
            target,
            ground_z_at,
            sample_step_m=terrain_sample_step_m,
            clearance_m=terrain_clearance_m,
            object_height=float(object_height_m),
            start_margin_m=start_margin,
            end_margin_m=end_margin,
        ):
            return "terrain", None
        if occupied:
            is_blocked, hit, _ = blocked(
                eye, target, occupied, voxel, step, clearance, min_hit_samples, start_margin, end_margin
            )
            if is_blocked:
                return "obstacle", (None if hit is None else tuple(float(v) for v in hit))
        return None, None

    in_range = dists <= float(max_range_m) + 1e-9
    visible: list[bool | None] = [None] * len(dists)
    if not np.any(in_range):
        return SightResult(0.0, "no_targets", None, None, tuple(visible), False)

    available = 0.0
    reason = None
    hit_xyz = None
    first_blocked = None
    for i in np.flatnonzero(in_range):
        if first_blocked is not None and not evaluate_all:
            break
        why, hit = test(targets[i])
        visible[i] = why is None
        if why is not None and first_blocked is None:
            first_blocked, reason, hit_xyz = int(i), why, hit
        elif why is None and first_blocked is None:
            available = float(dists[i])

    if first_blocked is None:
        last = int(np.flatnonzero(in_range)[-1])
        at_cap = abs(float(dists[last]) - float(max_range_m)) <= 1e-9
        return SightResult(min(available, float(max_range_m)), "range" if at_cap else "end", None,
                           None, tuple(visible), False)

    refined = False
    if refine_tolerance_m is not None and float(refine_tolerance_m) > 0.0:
        if first_blocked == 0:
            lo_xyz, lo_d = observer_ground, 0.0
        else:
            lo_xyz, lo_d = targets[first_blocked - 1], float(dists[first_blocked - 1])
        hi_xyz, hi_d = targets[first_blocked], float(dists[first_blocked])
        while hi_d - lo_d > float(refine_tolerance_m):
            frac = 0.5
            mid_d = lo_d + frac * (hi_d - lo_d)
            mid_xyz = ground_at(lo_xyz + frac * (hi_xyz - lo_xyz))
            why, hit = test(mid_xyz)
            if why is None:
                lo_xyz, lo_d = mid_xyz, mid_d
            else:
                hi_xyz, hi_d, reason, hit_xyz = mid_xyz, mid_d, why, hit
        available = lo_d
        refined = True

    return SightResult(float(available), str(reason), hit_xyz, first_blocked, tuple(visible), refined)


def build_occupancy(points, corridor_xyz, *, voxel=0.20,
                    min_height_m=0.5, max_height_m=10.0,
                    exclude_within_m=0.0, min_neighbors=2):
    """Voxel occupancy of everything standing above the road, for line-of-sight tests.

    ⚠️ **Height is measured above the NEAREST CORRIDOR POSE, never above a global z percentile.**
    That distinction is not pedantic. On a 600 m window of the pine_054823 mountain corridor the
    point cloud spans 71.6 m in z while the trajectory itself spans only 9.4 m, so a single global
    "ground level" is meaningless: it puts the whole cloud inside the band in the low sections and
    excludes every occluder in the high ones. A first run of this engine used a global 2nd-percentile
    z and silently reported "nothing blocks" across an entire corridor.

    No classification is used or needed -- anything standing above the road blocks a sight line
    whatever it is. `min_height_m` exists to drop the road surface itself, not to select a class.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    corridor_xyz = np.asarray(corridor_xyz, dtype=np.float64)
    tree = cKDTree(corridor_xyz[:, :2])
    lateral, idx = tree.query(points[:, :2])
    height = points[:, 2] - corridor_xyz[idx, 2]
    band = (height > min_height_m) & (height < max_height_m)
    if exclude_within_m > 0.0:
        # The road surface and whatever sits on it must not close a sight line ALONG the road.
        # viewshed.py does the same with ROAD_W = 3.8 m against the traced centreline.
        band &= lateral > exclude_within_m
    occ = voxelize(points[band], voxel)
    return filter_isolated_voxels(occ, min_neighbors)


def filter_isolated_voxels(occ, min_neighbors: int):
    """Drop voxels with fewer than `min_neighbors` occupied 26-neighbours.

    ⚠️ **Without this, isolated noise closes sight lines.** Extracted from
    `viewshed.py:_filter_isolated_voxels`, which both viewshed and hso_corridor apply before any
    sweep. Omitting it was a measured error: cross-validating against HSO on pine_054823, station
    10140 reported 35 m against HSO's 115 m -- speckle voxels, not real obstacles.

    A single stray return should never obstruct anything; `min_hit_samples` in `blocked()` guards
    the same failure along the ray, and this guards it in the map.
    """
    if int(min_neighbors) <= 0 or not occ:
        return occ
    required = int(min_neighbors)
    kept = set()
    for vx, vy, vz in occ:
        n = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    if (vx + dx, vy + dy, vz + dz) in occ:
                        n += 1
                        if n >= required:
                            kept.add((vx, vy, vz))
                            break
                if n >= required:
                    break
            if n >= required:
                break
    return kept
