"""Defensive TUM trajectory loading shared by repository tools.

Extracted after comparing ``sci_studio/hso_corridor.py:load_tum``,
``gnss_map/dedrift_trajectory.py:load_tum``, and ``gnss_map/pose_bag_scans.py:parse_tum``.
They disagree: HSO accepts four or more columns and returns only ``t x y z``; dedrift relies on
``loadtxt`` and returns three arrays; pose-bag truncates extra columns, validates ordering and
normalizes quaternions.  This shared loader requires the actual eight-column TUM format, preserves
the file's ``qx qy qz qw`` values and row order, and rejects malformed, non-finite, non-increasing,
empty, or zero-quaternion trajectories.  It never silently sorts or normalizes input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Slerp


def load_tum(path: str | Path) -> np.ndarray:
    """Load ``timestamp tx ty tz qx qy qz qw`` rows as a float64 ``(N, 8)`` array."""
    path = Path(path).expanduser()
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            fields = text.split()
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number}: expected exactly 8 columns "
                    "(timestamp tx ty tz qx qy qz qw)"
                )
            try:
                rows.append([float(value) for value in fields])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric TUM value") from exc

    if not rows:
        raise ValueError(f"{path} contains no TUM poses")
    tum = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(tum).all():
        raise ValueError("trajectory contains non-finite values")
    if np.any(np.diff(tum[:, 0]) <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    if np.any(np.linalg.norm(tum[:, 4:8], axis=1) <= 0):
        raise ValueError("trajectory contains a zero quaternion")
    return tum


def load_tum_xyz(path: str | Path) -> np.ndarray:
    """Load only the translation columns, returning a float64 ``(N, 3)`` array."""
    return load_tum(path)[:, 1:4]


def rewarp_xyz(
    xyz: np.ndarray,
    time: np.ndarray,
    old_t: np.ndarray,
    old_p: np.ndarray,
    old_slerp: Slerp,
    new_t: np.ndarray,
    new_p: np.ndarray,
    new_slerp: Slerp,
) -> np.ndarray:
    """Rigidly attach points to interpolated poses (linear translation, SO(3) Slerp).

    This is a geometric shortcut, not a rerun of mapping or deskewing. The caller
    supplies validated, strictly increasing trajectory knots and matching Slerp
    objects. Query points must be finite and inside both trajectory spans.
    Extracted from SCI_LiDAR rewarp_cloud_trajectory.py without changing its math.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    query = np.asarray(time, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or query.shape != (len(xyz),):
        raise ValueError("expected xyz (N,3) and time (N,)")
    if not np.isfinite(xyz).all() or not np.isfinite(query).all():
        raise ValueError("points and times must be finite")
    if not len(xyz):
        return np.empty((0, 3), dtype=np.float64)
    if query.min() < max(old_t[0], new_t[0]) or query.max() > min(old_t[-1], new_t[-1]):
        raise ValueError("point timestamps are outside trajectory overlap")
    old_pos = np.column_stack([np.interp(query, old_t, old_p[:, axis]) for axis in range(3)])
    new_pos = np.column_stack([np.interp(query, new_t, new_p[:, axis]) for axis in range(3)])
    old_r = old_slerp(query)
    new_r = new_slerp(query)
    local = old_r.inv().apply(np.asarray(xyz, dtype=np.float64) - old_pos)
    return new_r.apply(local) + new_pos
