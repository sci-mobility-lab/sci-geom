"""WGS-84 geodetic/ECEF/ENU transforms.

Faithfully extracted from
``src/sci_lidar/04_applications/sci_lidar_applications/sight_distance/geodesy.py`` so map tools
can use the repository's one library-quality transform without importing its ROS2 package tree.
The original WGS-84 constants and row-vector inverse convention are preserved exactly.
"""
from __future__ import annotations

import numpy as np

_A = 6378137.0
_E2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)


def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(la) ** 2)
    return np.array([(n + h) * np.cos(la) * np.cos(lo),
                     (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - _E2) + h) * np.sin(la)])


def _enu_rot(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([[-np.sin(lo), np.cos(lo), 0.0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def geodetic_to_enu(lat, lon, h, lat0, lon0, h0):
    """Convert ``(lat, lon, h)`` to ENU centered at ``(lat0, lon0, h0)``."""
    d = ecef(lat, lon, h) - ecef(lat0, lon0, h0)
    return _enu_rot(lat0, lon0) @ d


def enu_to_geodetic(e, n, u, lat0, lon0, h0):
    """Convert datum-frame ENU to ``(lat, lon, h)``; vectorized over ``e, n, u``."""
    e = np.atleast_1d(np.asarray(e, float))
    n = np.atleast_1d(np.asarray(n, float))
    u = np.atleast_1d(np.asarray(u, float))
    rotation = _enu_rot(lat0, lon0)
    base = ecef(lat0, lon0, h0)
    enu = np.column_stack([e, n, u])
    ec = base + enu @ rotation
    try:
        import pyproj
    except ModuleNotFoundError:
        # The root analysis environment intentionally has numpy only.  Iterate the standard
        # ellipsoidal ECEF inverse there; environments with pyproj retain the original path.
        x, y, z = ec[:, 0], ec[:, 1], ec[:, 2]
        radius = np.hypot(x, y)
        lon = np.arctan2(y, x)
        lat = np.arctan2(z, radius * (1 - _E2))
        for _ in range(10):
            prime_vertical = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
            alt = radius / np.cos(lat) - prime_vertical
            lat = np.arctan2(z, radius * (1 - _E2 * prime_vertical / (prime_vertical + alt)))
        prime_vertical = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        alt = radius / np.cos(lat) - prime_vertical
        return np.degrees(lat), np.degrees(lon), alt
    else:
        transformer = pyproj.Transformer.from_crs(4978, 4326, always_xy=True)
        lon, lat, alt = transformer.transform(ec[:, 0], ec[:, 1], ec[:, 2])
        return np.asarray(lat), np.asarray(lon), np.asarray(alt)
