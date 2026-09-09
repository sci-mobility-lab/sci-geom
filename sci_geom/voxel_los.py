"""Voxel line-of-sight: does a straight segment from an observer to a target hit anything?

Extracted 2026-08-24 from
`src/sci_lidar/03_core/sci_lidar_core/voxel_visibility/isd_visibility_voxels.py:44-100`,
which sat inside an otherwise-dead ROS2 tree while being the shared occlusion kernel of three
live engine modules: `isd_corridor.py`, `hso_corridor.py` and `sight_distance.py`.

⚠️ **This is a FAITHFUL extraction, deliberately not an improvement.** The behaviour must stay
bit-identical to the original, because the RP 26-06 sight-distance engine is validated by
cross-checking against `isd_corridor.py` on the same input — and that check is worthless if the
two have silently diverged. Two known inefficiencies are preserved on purpose:

  - `occupied_neighbor_count` scans the full (2r+1)^3 neighbourhood per sample: 27 dict lookups
    per sample at the default radius 1. A 90 m ray at step 0.08 m is ~30k lookups.
  - sampling can skip voxels when `step > voxel`. Callers avoid it (step 0.08 vs voxel 0.15).

Optimise only behind an equivalence test against this implementation.

Method: sampled-segment ray march. The segment is sampled at `ceil(length/step)` points; each
sample is floored to a voxel key and its neighbourhood tested for occupancy. A block is declared
only after `min_hit_samples` *consecutive* hits, so isolated speckle does not close a sightline.
`start_margin` / `end_margin` suppress hits near either endpoint (the observer's own vehicle, the
target's own body) and reset the consecutive counter.
"""
from __future__ import annotations

import math

import numpy as np


def voxelize(points, voxel: float) -> set[tuple[int, int, int]]:
    """Occupancy as a set of integer voxel keys. No KD-tree, no octree — a plain hash set."""
    idx = np.floor(np.asarray(points, dtype=np.float64) / voxel).astype(np.int64)
    return {tuple(row) for row in idx}


def neighbors(center, radius: int):
    cx, cy, cz = center
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                yield (cx + dx, cy + dy, cz + dz)


def occupied_neighbor_count(center, occupied, radius: int) -> int:
    return sum(1 for nb in neighbors(center, radius) if nb in occupied)


def blocked(
    observer,
    target,
    occupied,
    voxel: float,
    step: float,
    clearance: int,
    min_hit_samples: int = 1,
    start_margin: float = 0.0,
    end_margin: float = 0.0,
    min_occupied_neighbors: int = 1,
):
    """Is the sightline observer -> target obstructed?

    Returns `(is_blocked, first_hit_point_or_None, distance)`. When not blocked, `distance` is
    the full segment length; when blocked, it is the distance to the first hit.
    """
    observer = np.asarray(observer, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    vec = target - observer
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        return False, None, 0.0
    direction = vec / length
    n = max(2, int(math.ceil(length / step)))
    hits = 0
    first_hit = None
    first_hit_distance = 0.0
    for i in range(1, n):
        distance = length * i / n
        if distance < start_margin or distance > length - end_margin:
            hits = 0
            first_hit = None
            continue
        p = observer + direction * distance
        key = tuple(np.floor(p / voxel).astype(np.int64))
        if occupied_neighbor_count(key, occupied, clearance) >= min_occupied_neighbors:
            if hits == 0:
                first_hit = p
                first_hit_distance = distance
            hits += 1
            if hits >= min_hit_samples:
                return True, first_hit, first_hit_distance
        else:
            hits = 0
            first_hit = None
    return False, None, length
