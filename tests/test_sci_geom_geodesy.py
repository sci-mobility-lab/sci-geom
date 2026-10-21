"""Vectorised geodetic_to_enu (v0.2.0).

Before v0.2.0 an array input was not supported, and it did not always fail: N=2 raised a broadcast
error, N=3 silently returned garbage. These tests pin the scalar behaviour and the array contract.
"""
import numpy as np
import pytest

from sci_geom.geodesy import ecef, enu_to_geodetic, geodetic_to_enu, _enu_rot

LAT0, LON0, H0 = 33.7781, -84.3991, 287.6  # Georgia Tech, the lab's corridors


def _scalar_reference(lat, lon, h):
    """The v0.1.0 formula, valid for one point."""
    return _enu_rot(LAT0, LON0) @ (ecef(lat, lon, h) - ecef(LAT0, LON0, H0))


def test_scalar_shape_and_value_unchanged():
    enu = geodetic_to_enu(33.78, -84.40, 290.0, LAT0, LON0, H0)
    assert enu.shape == (3,)
    np.testing.assert_allclose(enu, _scalar_reference(33.78, -84.40, 290.0), rtol=0, atol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 10_000])
def test_array_matches_pointwise_loop(n):
    rng = np.random.default_rng(n)
    lat = LAT0 + rng.uniform(-0.02, 0.02, n)
    lon = LON0 + rng.uniform(-0.02, 0.02, n)
    h = H0 + rng.uniform(-30, 30, n)
    enu = geodetic_to_enu(lat, lon, h, LAT0, LON0, H0)
    assert enu.shape == (3, n)
    idx = np.unique(np.linspace(0, n - 1, min(n, 50)).astype(int))
    expected = np.column_stack([_scalar_reference(lat[i], lon[i], h[i]) for i in idx])
    np.testing.assert_allclose(enu[:, idx], expected, rtol=0, atol=1e-9)


def test_round_trip_below_one_millimetre():
    rng = np.random.default_rng(0)
    lat = LAT0 + rng.uniform(-0.05, 0.05, 2000)
    lon = LON0 + rng.uniform(-0.05, 0.05, 2000)
    h = H0 + rng.uniform(-50, 50, 2000)
    e, n, u = geodetic_to_enu(lat, lon, h, LAT0, LON0, H0)
    lat2, lon2, h2 = enu_to_geodetic(e, n, u, LAT0, LON0, H0)
    e2, n2, u2 = geodetic_to_enu(lat2, lon2, h2, LAT0, LON0, H0)
    assert np.max(np.abs(np.stack([e2 - e, n2 - n, u2 - u]))) < 1e-3


def test_broadcast_scalar_height_and_2d_shape():
    lat = np.full((4, 5), LAT0 + 0.001)
    lon = np.full((4, 5), LON0)
    enu = geodetic_to_enu(lat, lon, H0, LAT0, LON0, H0)
    assert enu.shape == (3, 4, 5)
    # 0.001 deg of latitude at 33.8 N is ~110.9 m north, not the 111.32 m of the flat constant.
    assert np.allclose(enu[1], enu[1, 0, 0])
    assert 110.8 < enu[1, 0, 0] < 111.0
