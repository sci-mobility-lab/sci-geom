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
