"""Polyline chainage and point projection. Pure geometry, no session, no I/O.

Used by the V1 phone-GNSS station (`docs/sci_gaze/V1_EXPORT_SLICE.md` §16.3), but
deliberately domain-neutral: it knows nothing about fixes, gaze, time or evidence.

Two rules here are refusals, not conveniences, and they are the reason this is a
module rather than three lines inlined at the call site:

* **No clamping to the route ends.** A point whose perpendicular foot falls beyond
  the first or last vertex is off-route and has no station. Clamping it to the
  terminal vertex would fabricate progress beyond the route's coverage.
* **Competing candidates yield no station.** When two *disjoint* stretches of the
  route would place the same point at materially different chainages -- the two
  legs of a loop, or a branch -- the honest answer is "ambiguous", not the nearest
  guess. **Consecutive segments are one stretch**: at a bend, a point a few metres
  off the road is naturally in reach of both segments of the corner, and their
  chainages differ by the geometry of the turn, not by any ambiguity. Grouping
  candidates into runs of consecutive segments is what keeps every corner of a
  real route usable while still refusing a genuine branch.

All lengths are horizontal (XY) in metres; the caller converts to ENU XY first.
"""
from __future__ import annotations

import numpy as np

OFF_ROUTE = "off_route"
AMBIGUOUS_BRANCH = "ambiguous_branch"
# Distinct from OFF_ROUTE on purpose: the polyline DOES come near this point, but
# its nearest part is an endpoint, and clamping there would fabricate progress.
# "off-route" needs a diagnostic cause, not just a count.
BEYOND_ROUTE_END = "beyond_route_end"

DEFAULT_RADIUS_M = 15.0    # §16.3, ratified: beyond this, a fix is off-route
DEFAULT_COMPETE_M = 5.0    # §16.3, ratified: chainages differing by more than this compete

# A foot that lands past an endpoint by a floating-point crumb is ON the segment.
# Found on real data: the last fix of a traversal IS the route's last vertex, yet
# its parameter came out 1.0000000000000002 and the no-clamping rule called the
# route's own endpoint `off_route`. Hand-made test coordinates land exactly on 1.0,
# which is why no fixture caught it. This tolerance is numerical only -- the
# no-clamping rule is about points genuinely beyond the route, metres away.
_U_EPS = 1e-9
_D_EPS = 1e-9   # metres: a margin for the nearest-endpoint comparison below


def _vertices(vertices) -> np.ndarray:
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError(f"route vertices must be an (n, 2) XY array, got shape {v.shape}")
    if len(v) < 2:
        raise ValueError("a route needs at least two vertices")
    if not np.isfinite(v).all():
        raise ValueError("route vertices must all be finite")
    lengths = np.linalg.norm(np.diff(v, axis=0), axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("route has a zero-length segment; simplify or de-duplicate it first")
    return v


def chainage(vertices) -> np.ndarray:
    """Cumulative horizontal length at each vertex, starting at 0.

    For a bent route this is the distance *along* the polyline, which is longer
    than the straight line between the endpoints -- the whole point of a station.
    """
    v = _vertices(vertices)
    lengths = np.linalg.norm(np.diff(v, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))


def project(vertices, point, *, radius_m=DEFAULT_RADIUS_M, compete_m=DEFAULT_COMPETE_M) -> dict:
    """Project one XY point onto the polyline.

    Returns a dict with `station_m`, `segment`, `offset_m`, `reason` and
    `candidates`. `station_m` is `None` whenever `reason` is set, and `reason` is
    `None` whenever a station is established -- never both, never neither.

    A candidate is a segment whose perpendicular foot lies strictly within the
    segment (parameter in [0, 1]) and no further than `radius_m` from the point.
    With no candidate the point is `off_route`; with candidates whose chainages
    span more than `compete_m` it is `ambiguous_branch`.
    """
    v = _vertices(vertices)
    p = np.asarray(point, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError(f"point must be a finite XY pair, got {point!r}")

    starts, ends = v[:-1], v[1:]
    deltas = ends - starts
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = chainage(v)

    # Unclamped parameter along each segment. Outside [0, 1] the foot is beyond a
    # vertex, so that segment does not carry this point -- see the no-clamping rule.
    u = np.einsum("ij,ij->i", p - starts, deltas) / (lengths ** 2)
    u_clipped = np.clip(u, 0.0, 1.0)   # the station stays inside the route, always
    feet = starts + u_clipped[:, None] * deltas
    distances = np.linalg.norm(p - feet, axis=1)

    inside = (u >= -_U_EPS) & (u <= 1.0 + _U_EPS)
    near = distances <= radius_m
    candidates = np.flatnonzero(inside & near)
    if not len(candidates):
        # No segment carries this point in its interior. Two very different
        # situations hide behind that, and the reviewer asked for them to be told apart:
        # the polyline may still come within the radius -- at a vertex or an end,
        # which the no-clamping rule refuses to use -- or it may simply be far
        # away. A bare "off-route" count cannot be audited.
        clamped = np.linalg.norm(p - (starts + np.clip(u, 0.0, 1.0)[:, None] * deltas), axis=1)
        reason = BEYOND_ROUTE_END if float(clamped.min()) <= radius_m else OFF_ROUTE
        return {"station_m": None, "segment": None, "offset_m": None,
                "reason": reason, "candidates": 0}

    # Group the candidates into runs of CONSECUTIVE segments. One run is one
    # continuous stretch of route -- the two segments of a bend belong together,
    # and the chainage difference across a corner is geometry, not ambiguity.
    runs: list[list[int]] = [[int(candidates[0])]]
    for index in candidates[1:]:
        (runs[-1] if int(index) == runs[-1][-1] + 1 else runs.append([]) or runs[-1]).append(int(index))

    # Compare the station INTERVAL each run spans, not one chosen member of it: a
    # representative would be an arbitrary choice, and an arbitrary choice in a
    # refusal rule is exactly what nobody can audit later. Runs are ordered along
    # the route, so it is enough to look at the gap between consecutive runs.
    #
    # Note for whoever reads this next: with the ratified 15 m radius, two DISJOINT
    # runs are separated by a stretch of route that left the 15 m neighbourhood and
    # came back, so their station gap always exceeds the 5 m competition limit. The
    # comparison below is therefore a guard against a genuine branch or loop, not a
    # knob to tune -- and no variation of it is observable at those thresholds.
    def _interval(run: list[int]) -> tuple[float, float]:
        stations = [float(cumulative[i] + u_clipped[i] * lengths[i]) for i in run]
        return min(stations), max(stations)

    intervals = [_interval(run) for run in runs]
    if any(low - previous_high > compete_m
           for (_previous_low, previous_high), (low, _high) in zip(intervals, intervals[1:])):
        return {"station_m": None, "segment": None, "offset_m": None,
                "reason": AMBIGUOUS_BRANCH, "candidates": int(len(candidates))}

    best = int(candidates[int(np.argmin(distances[candidates]))])

    # If the genuinely nearest point of the polyline is an ENDPOINT -- which the
    # no-clamping rule forbids using -- then falling back to a farther segment
    # would fabricate progress. Found by a test: on (0,0)-(10,0)-(10,7), a point
    # 1 m before the start is 1 m from the first vertex, yet the only usable foot
    # was the corner 11 m away, which handed it station 10 m. Refuse instead.
    # Compare with a margin, not at the bit: for the route's own last vertex the
    # endpoint distance is exactly 0 while its computed foot lands ~1e-15 away, and
    # a bare `<` made the route reject its own endpoint.
    endpoint_distance = float(min(np.linalg.norm(p - v[0]), np.linalg.norm(p - v[-1])))
    if endpoint_distance + _D_EPS < float(distances[best]):
        return {"station_m": None, "segment": None, "offset_m": None,
                "reason": BEYOND_ROUTE_END, "candidates": int(len(candidates))}

    return {
        "station_m": float(cumulative[best] + u_clipped[best] * lengths[best]),
        "segment": best,
        "offset_m": float(distances[best]),
        "reason": None,
        "candidates": int(len(candidates)),
    }


def simplify(points, tolerance_m: float) -> np.ndarray:
    """Douglas-Peucker simplification, keeping the first and last point.

    A route is not "every fix joined up": jitter and retraced passages have to go
    before the polyline means anything. The tolerance is recorded in the route
    snapshot, because it changes the geometry that every later station depends on.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be an (n, 2) XY array, got shape {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("points must all be finite")
    if not (tolerance_m > 0.0) or not np.isfinite(tolerance_m):
        raise ValueError(f"tolerance_m must be finite and positive, got {tolerance_m!r}")
    if len(pts) <= 2:
        return pts.copy()

    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        a, b = pts[first], pts[last]
        span = b - a
        norm = float(np.linalg.norm(span))
        segment = pts[first + 1:last]
        if norm == 0.0:
            deviations = np.linalg.norm(segment - a, axis=1)
        else:
            offsets = segment - a
            deviations = np.abs(span[0] * offsets[:, 1] - span[1] * offsets[:, 0]) / norm
        worst = int(np.argmax(deviations))
        if float(deviations[worst]) > tolerance_m:
            index = first + 1 + worst
            keep[index] = True
            stack.extend([(first, index), (index, last)])
    return pts[keep]
