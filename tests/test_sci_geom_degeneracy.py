"""Analytic ground truth for corridor localizability.

The whole point is the ALONG-TRACK axis. A test that only checks an aggregate number would pass
while the one axis the thesis depends on is unobservable -- which is exactly the failure mode.
"""
import numpy as np
import pytest

from sci_geom.degeneracy import (
    corridor_profile,
    information_matrix,
    localizability,
    normals_from_neighbourhood,
    station_sigma_from_localizability,
)


def straight_corridor_normals(n=2000, seed=0):
    """Two parallel walls plus a flat road: every normal is lateral or vertical, none along."""
    rng = np.random.default_rng(seed)
    lateral = np.tile([1.0, 0.0, 0.0], (n, 1)) * rng.choice([-1, 1], n)[:, None]
    vertical = np.tile([0.0, 0.0, 1.0], (n, 1))
    return np.vstack([lateral, vertical])


def test_a_straight_corridor_is_degenerate_along_its_axis():
    """The documented failure: walls constrain sideways, the road constrains vertically,
    nothing constrains along the corridor."""
    n = straight_corridor_normals()
    along = np.array([0.0, 1.0, 0.0])       # corridor runs along +y; walls face +/-x
    lateral = np.array([1.0, 0.0, 0.0])

    assert localizability(n, along) == pytest.approx(0.0, abs=1e-9)
    assert localizability(n, lateral) == pytest.approx(1.0, abs=1e-9)


def test_a_transverse_surface_restores_the_along_track_constraint():
    """A wall ACROSS the corridor (a building face, a cut, a sign gantry) is what makes the
    along-track direction observable again."""
    n = straight_corridor_normals()
    transverse = np.tile([0.0, 1.0, 0.0], (1500, 1))     # normals pointing along the corridor
    along = np.array([0.0, 1.0, 0.0])

    before = localizability(n, along)
    after = localizability(np.vstack([n, transverse]), along)
    assert before < 1e-6
    assert after > 0.3, "a transverse surface must make the along-track axis observable"


def test_information_matrix_is_the_sum_of_outer_products():
    n = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0]])
    expected = np.array([[2.0, 0, 0], [0, 1.0, 0], [0, 0, 0.0]])
    assert np.allclose(information_matrix(n), expected)


def test_sigma_grows_as_the_inverse_square_root_of_localizability():
    s1 = station_sigma_from_localizability(1.0, sigma_best_m=0.05, sigma_max_m=99.0)
    s4 = station_sigma_from_localizability(0.25, sigma_best_m=0.05, sigma_max_m=99.0)
    assert s1 == pytest.approx(0.05)
    assert s4 == pytest.approx(0.10)          # 4x less information -> 2x the sigma
    # a fully degenerate section reports large-but-finite, never inf or nan
    s0 = station_sigma_from_localizability(0.0, sigma_max_m=5.0)
    assert s0 == 5.0 and np.isfinite(s0)


def test_normals_of_a_plane_all_point_the_same_way():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-5, 5, size=(600, 2))
    plane = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])
    n = normals_from_neighbourhood(plane, k=12)
    assert np.abs(np.abs(n[:, 2]) - 1.0).max() < 1e-6, "a flat plane must give vertical normals"


def test_corridor_profile_flags_the_degenerate_stretch_and_not_the_other():
    """End to end: a corridor whose first half is bare walls and second half has cross-walls.
    The profile must show along_ratio low then high, and station_sigma_m the reverse."""
    rng = np.random.default_rng(1)
    corridor = np.column_stack([np.zeros(201), np.linspace(0, 200, 201), np.zeros(201)])

    def wall_x(x, y0, y1, m=4000):
        return np.column_stack([np.full(m, x) + rng.normal(0, 0.01, m),
                                rng.uniform(y0, y1, m),
                                rng.uniform(0, 4, m)])

    def wall_y(y, m=4000):    # a face ACROSS the corridor
        return np.column_stack([rng.uniform(-6, 6, m),
                                np.full(m, y) + rng.normal(0, 0.01, m),
                                rng.uniform(0, 4, m)])

    pts = np.vstack([
        wall_x(-6, 0, 200), wall_x(6, 0, 200),            # side walls the whole way
        wall_y(120), wall_y(140), wall_y(160), wall_y(180),   # cross faces only in the far half
    ])
    prof = corridor_profile(pts, corridor, window_m=20.0, step_m=20.0, k=12)
    assert prof, "profile must not be empty"

    early = [p["along_ratio"] for p in prof if p["station_m"] < 100]
    late = [p["along_ratio"] for p in prof if p["station_m"] >= 120]
    assert early and late
    assert np.median(early) < np.median(late), "bare-wall stretch must look LESS localizable"

    sig_early = [p["station_sigma_m"] for p in prof if p["station_m"] < 100]
    sig_late = [p["station_sigma_m"] for p in prof if p["station_m"] >= 120]
    assert np.median(sig_early) > np.median(sig_late), "sigma must be larger where it is degenerate"
    assert all(np.isfinite(p["station_sigma_m"]) for p in prof)
