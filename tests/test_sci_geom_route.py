"""Polyline chainage, projection and simplification: V1_EXPORT_SLICE.md §16.3/§16.6.

The fixtures are BENT on purpose. A straight route cannot tell chainage from chord
distance, so a straight-route test would pass even if the implementation measured
the wrong thing (§16.6). Every expected station below is arithmetic done by hand in
the test, never a value read back from the implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from sci_geom.route import (AMBIGUOUS_BRANCH, OFF_ROUTE, chainage, project, simplify)

# An L: 30 m east, then 40 m north. Chord = 50 m, chainage = 70 m -- they differ,
# which is exactly what makes this fixture able to fail.
L_ROUTE = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 40.0]])


def test_chainage_follows_the_bend_not_the_chord():
    assert chainage(L_ROUTE).tolist() == [0.0, 30.0, 70.0]
    chord = float(np.linalg.norm(L_ROUTE[-1] - L_ROUTE[0]))
    assert chord == 50.0 and chainage(L_ROUTE)[-1] == 70.0, "chainage must exceed the chord here"


def test_station_is_measured_along_the_route():
    # 4 m short of the corner and 2 m off to the side: station 26 m, offset 2 m.
    result = project(L_ROUTE, [26.0, 2.0])
    assert result["reason"] is None
    assert result["station_m"] == pytest.approx(26.0)
    assert result["offset_m"] == pytest.approx(2.0)
    assert result["segment"] == 0

    # 10 m up the second leg: 30 + 10 = 40 m.
    second = project(L_ROUTE, [31.5, 10.0])
    assert second["station_m"] == pytest.approx(40.0) and second["segment"] == 1


def test_a_point_past_the_end_is_off_route_never_clamped():
    beyond = project(L_ROUTE, [30.0, 45.0])          # 5 m past the final vertex
    assert beyond["reason"] == OFF_ROUTE and beyond["station_m"] is None, \
        "clamping would fabricate progress beyond the route's coverage"
    before = project(L_ROUTE, [-6.0, 0.0])           # 6 m before the start
    assert before["reason"] == OFF_ROUTE and before["station_m"] is None


def test_a_shared_vertex_is_not_a_competing_branch():
    """Adjacent segments agree on the chainage at their shared vertex."""
    at_corner = project(L_ROUTE, [30.0, 0.0])
    assert at_corner["reason"] is None
    assert at_corner["station_m"] == pytest.approx(30.0)
    assert at_corner["candidates"] >= 2, "both segments should be candidates here"


def test_two_legs_of_a_loop_compete_and_yield_no_station():
    """A hairpin: the outbound and return legs are 4 m apart, so a fix between them
    is within reach of both, but they place it 100 m apart along the route."""
    hairpin = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 4.0], [0.0, 4.0]])
    result = project(hairpin, [50.0, 2.0])
    assert result["reason"] == AMBIGUOUS_BRANCH and result["station_m"] is None
    assert result["candidates"] >= 2


@pytest.mark.parametrize("offset,expected_reason", [
    (15.0, None),          # exactly at the ratified 15 m radius: still on the route
    (15.000001, OFF_ROUTE),  # past it: off-route
])
def test_radius_boundary_is_exact(offset, expected_reason):
    assert project(L_ROUTE, [10.0, offset])["reason"] == expected_reason


def test_a_bend_is_not_an_ambiguity(tmp_path=None):
    """Regression: the first version of this module called every corner ambiguous.

    At a bend, a fix a few metres off the road is in reach of both segments of the
    corner, and their chainages differ by the geometry of the turn. Refusing a
    station there would have made the whole route unusable around every turn --
    which is why consecutive candidates are grouped into one stretch.
    """
    # 10 m before the bend, 3 m off the road. Both segments are candidates: the
    # first at 3 m (station 20), the second at 10 m (station 33). Their chainages
    # differ by 13 m, so the old rule refused a station here; the nearer segment
    # decides, unambiguously.
    corner = project(L_ROUTE, [20.0, 3.0])
    assert corner["reason"] is None, corner
    assert corner["candidates"] >= 2, "both segments of the corner should be candidates"
    assert corner["station_m"] == pytest.approx(20.0)
    assert corner["segment"] == 0 and corner["offset_m"] == pytest.approx(3.0)

    # Closer to the corner the SECOND leg is genuinely the nearer one (2 m against
    # 3 m), and the station follows it round the bend. Not an ambiguity either.
    past = project(L_ROUTE, [28.0, 3.0])
    assert past["reason"] is None and past["segment"] == 1
    assert past["station_m"] == pytest.approx(33.0)


# The 4 m hairpin: at the midpoint the two legs are both within reach but place the
# point 104 m apart along the route, and the connector at the far turn is out of
# radius -- so they are genuinely disjoint stretches. Varying the LIMIT rather than
# the geometry tests the comparison itself, exactly, with nothing contrived.
WIDE_HAIRPIN = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 4.0], [0.0, 4.0]])


@pytest.mark.parametrize("compete_m,expected_reason", [
    (104.0, None),          # exactly at the limit: not competing
    (103.999999, AMBIGUOUS_BRANCH),
])
def test_competition_threshold_is_compared_exactly(compete_m, expected_reason):
    result = project(WIDE_HAIRPIN, [50.0, 2.0], compete_m=compete_m)
    assert result["reason"] == expected_reason, result


def test_disjoint_runs_are_always_far_apart_at_the_ratified_radius():
    """The invariant the ambiguity rule rests on, made executable.

    Two candidate stretches are disjoint only because the route left the 15 m
    neighbourhood and came back, so their chainage gap always exceeds the 15 m
    radius -- and therefore the 5 m competition limit. That is why no variation of
    HOW each run's station interval is summarised (its minimum, its maximum, its
    nearest member) can change any verdict: a sabotage of that detail is
    unobservable by construction, not untested. Pinning the invariant here is the
    honest alternative to a contrived test for a distinction that cannot exist.
    """
    geometries = [
        np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 4.0], [0.0, 4.0]]),                    # hairpin
        np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 30.0], [10.0, 30.0], [10.0, 6.0], [60.0, 6.0]]),
        np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 25.0], [-5.0, 25.0], [-5.0, 2.0], [40.0, 2.0]]),
    ]
    for vertices in geometries:
        cumulative = chainage(vertices)
        starts, ends = vertices[:-1], vertices[1:]
        deltas = ends - starts
        lengths = np.linalg.norm(deltas, axis=1)
        for point in (vertices[:-1] + deltas * 0.5) + np.array([0.0, 1.0]):
            u = np.einsum("ij,ij->i", point - starts, deltas) / lengths ** 2
            feet = starts + u[:, None] * deltas
            distances = np.linalg.norm(point - feet, axis=1)
            candidates = np.flatnonzero((u >= 0) & (u <= 1) & (distances <= 15.0))
            runs, stations = [], cumulative[candidates] + u[candidates] * lengths[candidates]
            for index, station in zip(candidates, stations):
                if runs and index == runs[-1][-1][0] + 1:
                    runs[-1].append((index, station))
                else:
                    runs.append([(index, station)])
            for previous, following in zip(runs, runs[1:]):
                gap = min(s for _i, s in following) - max(s for _i, s in previous)
                assert gap > 15.0, (
                    f"disjoint runs only {gap:.2f} m apart -- the ambiguity rule's premise "
                    f"would no longer hold, and the interval summary WOULD become observable")


def test_projection_refuses_malformed_input():
    with pytest.raises(ValueError, match="at least two vertices"):
        project([[0.0, 0.0]], [0.0, 0.0])
    with pytest.raises(ValueError, match="zero-length segment"):
        project([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], [0.0, 0.0])
    with pytest.raises(ValueError, match="finite XY pair"):
        project(L_ROUTE, [0.0, float("nan")])
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        chainage([0.0, 1.0, 2.0])


def test_simplify_drops_jitter_and_keeps_the_shape():
    # A straight run with 0.5 m of jitter, then a real 40 m turn.
    straight = [[float(x), 0.5 if x % 2 else -0.5] for x in range(0, 31, 2)]
    turn = [[30.0, float(y)] for y in range(5, 41, 5)]
    points = np.array(straight + turn)
    simplified = simplify(points, tolerance_m=2.0)
    assert len(simplified) < len(points)
    assert simplified[0].tolist() == points[0].tolist()
    assert simplified[-1].tolist() == points[-1].tolist()
    # The corner survives: the simplified route still bends by roughly 90 degrees.
    assert chainage(simplified)[-1] > float(np.linalg.norm(points[-1] - points[0]))


def test_simplify_keeps_a_feature_above_the_tolerance():
    points = np.array([[0.0, 0.0], [5.0, 3.0], [10.0, 0.0]])
    assert len(simplify(points, tolerance_m=1.0)) == 3, "a 3 m deviation must survive a 1 m tolerance"
    assert len(simplify(points, tolerance_m=5.0)) == 2, "and be dropped by a 5 m one"


def test_simplify_refuses_a_nonsense_tolerance():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="tolerance_m"):
            simplify(L_ROUTE, bad)
