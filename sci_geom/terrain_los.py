"""Analytic terrain line-of-sight: does the GROUND ITSELF break the sight line?

Extracted 2026-08-24 from `platform/engine/sci_studio/viewshed.py:280-311`, where it is the first
of two occlusion layers used by both `viewshed.py` and `hso_corridor.py`.

⚠️ **Why a voxel occupancy grid cannot replace this.** Occupancy is built from points *above* the
ground (a `min_height_m` band exists precisely to drop the road surface), so the ground is excluded
by construction and a crest can never block a sight line in a voxel-only engine. On a hilly
corridor that is not an edge case:

    pine_054823, 2537 HSO stations
      blocked by TERRAIN   1678  (66 %)
      blocked by OBSTACLE   810  (32 %)
      neither                49  ( 2 %)

Two thirds. A voxel-only engine run on that corridor reported "nothing blocks" and returned the
range cap almost everywhere. Terrain is the dominant occluder on rolling terrain; obstacles dominate
in town. Both layers are needed.

Method: sample the straight chord between eye and target every `sample_step_m`, interpolate the
ground height under each sample, and require **two consecutive** samples where the ground (plus a
clearance that ramps from `clearance_m` at the eye to the object height at the target) rises above
the chord. Two consecutive samples, not one, so a single noisy ground cell cannot close a sightline.

This is a FAITHFUL extraction, guarded by an equivalence test. Do not "improve" it without one.
"""
from __future__ import annotations

import math

import numpy as np


def terrain_blocks_los(
    observer,
    target,
    ground_z_at,
    *,
    clearance_m: float = 0.05,
    sample_step_m: float = 1.0,
    object_height: float = 0.6,
    start_margin_m: float = 2.0,
    end_margin_m: float = 1.0,
) -> bool:
    """`ground_z_at` is a callable: (N,2) xy array -> (N,) ground elevations."""
    observer = np.asarray(observer, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta_xy = np.asarray(target[:2] - observer[:2], dtype=np.float64)
    dist = float(np.linalg.norm(delta_xy))
    step = min(float(sample_step_m), 1.0)
    if not math.isfinite(step) or step <= 0.0:
        step = 1.0
    if dist <= start_margin_m + end_margin_m + step:
        return False
    samples = np.arange(start_margin_m, dist - end_margin_m + 1e-9, step, dtype=np.float64)
    if not len(samples):
        return False
    u = samples / dist
    xy = observer[:2] + u[:, None] * delta_xy
    line_z = float(observer[2]) + u * float(target[2] - observer[2])
    ground_z = np.asarray(ground_z_at(xy), dtype=np.float64)
    # Preserve the 0.6 m object SSD clearance while letting taller targets clear crests sooner.
    clearance_object_height = min(float(object_height), 0.6)
    clearance = float(clearance_m) + u * max(0.0, clearance_object_height - float(clearance_m))
    over = ground_z + clearance >= line_z
    return bool(np.any(over[:-1] & over[1:])) if len(over) >= 2 else False
