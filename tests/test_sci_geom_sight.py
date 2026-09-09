import numpy as np
import pandas as pd

from encounters.schema import empty, validate
from sci_geom.sight import build_occupancy, availability_table, available_sight_distance
from sci_geom.voxel_los import voxelize


def _wall(x, y_range=(-1.0, 1.0), z_range=(0.0, 2.0), spacing=0.05):
    ys = np.arange(y_range[0], y_range[1] + spacing, spacing)
    zs = np.arange(z_range[0], z_range[1] + spacing, spacing)
    yy, zz = np.meshgrid(ys, zs)
    return np.column_stack((np.full(yy.size, x), yy.ravel(), zz.ravel()))


def _positions(stations, height):
    return np.column_stack((stations, np.zeros_like(stations), np.full_like(stations, height)))


def _objects(rows):
    template = empty("objects")
    values = [
        [object_id, "object", x, y, z, 1.0, 1.0, station, 0.0, 0.1, 1.0,
         pd.NA, pd.NA, "test"]
        for object_id, station, x, y, z in rows
    ]
    frame = pd.DataFrame(values, columns=template.columns)
    return frame.astype({column: template[column].dtype for column in template})


def test_clear_corridor_returns_exact_span_and_cap():
    stations = np.arange(0.0, 61.0, 10.0)
    positions = _positions(stations, 1.4)
    target = np.array([60.0, 0.0, 1.4])

    assert available_sight_distance(
        stations, positions, target, set(), eye_height_m=1.4
    ) == 60.0
    assert available_sight_distance(
        stations, positions, target, set(), eye_height_m=1.4, max_range_m=25.0
    ) == 25.0


def test_wall_thirty_metres_back_stops_continuous_run():
    stations = np.arange(0.0, 61.0, 10.0)
    occupied = voxelize(_wall(30.0), 0.1)
    result = available_sight_distance(
        stations,
        _positions(stations, 1.4),
        np.array([60.0, 0.0, 1.4]),
        occupied,
        eye_height_m=1.4,
        voxel=0.1,
        clearance=0,
    )
    assert abs(result - 30.0) <= 0.1


def test_low_obstacle_produces_binding_driver_eye_height():
    stations = np.arange(0.0, 61.0, 10.0)
    obstacle_points = np.vstack([
        _wall(x, z_range=(0.0, 1.30), spacing=0.02)
        for x in np.arange(29.8, 30.21, 0.04)
    ])
    obstacle = voxelize(obstacle_points, 0.04)
    target = np.array([60.0, 0.0, 1.4])
    kwargs = dict(voxel=0.04, step=0.02)

    driver = available_sight_distance(
        stations, _positions(stations, 1.08), target, obstacle,
        eye_height_m=1.08, **kwargs,
    )
    cyclist = available_sight_distance(
        stations, _positions(stations, 1.40), target, obstacle,
        eye_height_m=1.40, **kwargs,
    )

    # This eye-height asymmetry is the mechanism the whole contribution rests on.
    assert driver == 30.0
    assert cyclist == 60.0
    assert driver < cyclist


def test_continuity_stops_at_first_break_instead_of_using_maximum():
    stations = np.arange(0.0, 61.0, 10.0)
    occupied = voxelize(_wall(50.0, y_range=(4.8, 5.2)), 0.1)
    result = available_sight_distance(
        stations,
        _positions(stations, 1.4),
        np.array([60.0, 10.0, 1.4]),
        occupied,
        eye_height_m=1.4,
        voxel=0.1,
        clearance=0,
    )
    assert result == 10.0
    assert result != 60.0


def test_availability_table_schema_range_raycast_count_and_order(monkeypatch):
    corridor = pd.DataFrame(
        {
            "station_m": np.array([0.0, 10.0, 20.0]),
            "x": np.array([0.0, 10.0, 20.0]),
            "y": np.zeros(3),
            "z": np.zeros(3),
        }
    )
    objects = _objects([
        ("b", 20.0, 20.0, 2.0, 1.0),
        ("a", 0.0, 0.0, 2.0, 1.0),
    ])
    calls = []

    def clear_ray(*args, **kwargs):
        calls.append(args[0].copy())
        return False, None, 0.0

    monkeypatch.setattr("sci_geom.sight.blocked", clear_ray)
    progress = []
    kwargs = dict(
        eye_heights_m=[1.40, 1.08], max_range_m=10.0,
        progress=lambda completed, total: progress.append((completed, total)),
    )
    first = availability_table(corridor, objects, set(), **kwargs)
    second = availability_table(corridor, objects, set(), **kwargs)

    validate(first, "availability")
    assert set(first["eye_height_m"]) == {1.08, 1.40}
    assert (first["reason"] == "out_of_range").sum() == 4
    assert len(calls) == 16  # 8 in-range rays per run; out-of-range rows were not cast
    assert progress == [(1, 2), (2, 2), (1, 2), (2, 2)]
    assert first.equals(second)
    assert list(first[["object_id", "eye_height_m", "station_m"]].itertuples(index=False, name=None)) == sorted(
        first[["object_id", "eye_height_m", "station_m"]].itertuples(index=False, name=None)
    )


def test_empty_objects_single_station_and_object_at_first_station():
    corridor = pd.DataFrame(
        {"station_m": [0.0], "x": [0.0], "y": [0.0], "z": [0.0]}
    )
    result = availability_table(
        corridor, empty("objects"), set(), eye_heights_m=[1.08, 1.40]
    )
    validate(result, "availability")
    assert result.empty

    objects = _objects([("first", 0.0, 0.0, 0.0, 1.4)])
    result = availability_table(corridor, objects, set(), eye_heights_m=[1.4])
    assert result.to_dict("records") == [
        {"object_id": "first", "station_m": 0.0, "eye_height_m": 1.4,
         "available": True, "reason": "ok"}
    ]
    assert available_sight_distance(
        np.array([0.0]), np.array([[0.0, 0.0, 1.4]]),
        np.array([0.0, 0.0, 1.4]), set(), eye_height_m=1.4,
    ) == 0.0


def test_build_occupancy_uses_a_LOCAL_ground_reference_not_a_global_one():
    """A global z threshold is meaningless on a corridor that climbs.

    Regression guard for a real failure: on 600 m of the pine mountain corridor the cloud spans
    71.6 m in z while the trajectory spans 9.4 m. A global 2nd-percentile ground level reported
    "nothing blocks" across the whole corridor.
    """
    # Corridor climbing 20 m over 100 m of travel.
    corridor = np.column_stack([np.linspace(0, 100, 101),
                                np.zeros(101),
                                np.linspace(0, 20, 101)])
    # One 2 m post above the road at each end -- both must be captured.
    posts = np.vstack([
        np.column_stack([np.full(50, 10.0), np.zeros(50), 2.0 + np.linspace(0, 2, 50)]),
        np.column_stack([np.full(50, 90.0), np.zeros(50), 18.0 + np.linspace(0, 2, 50)]),
    ])
    occ = build_occupancy(posts, corridor, voxel=0.5, min_height_m=0.5, max_height_m=10.0)

    low_post = {k for k in occ if k[0] == int(10.0 // 0.5)}
    high_post = {k for k in occ if k[0] == int(90.0 // 0.5)}
    assert low_post, "post at the bottom of the climb was dropped"
    assert high_post, "post at the top of the climb was dropped -- global-threshold bug"

    # A global percentile would have kept only one of the two.
    global_band = posts[posts[:, 2] > np.percentile(posts[:, 2], 2) + 0.5]
    assert len(global_band) < len(posts), "sanity: the global rule does discard points"


def test_terrain_layer_changes_the_answer_on_a_crest():
    """The regression this whole layer exists for.

    A crest between eye and target blocks the sight line, but the crest is GROUND -- and occupancy
    is built from points above the ground, so a voxel-only engine cannot see it. On pine_054823,
    66% of 2537 HSO stations are terrain-blocked; a voxel-only run returned the range cap almost
    everywhere and the number was structurally blind, not conservative.
    """
    stations = np.arange(0.0, 101.0, 10.0)
    positions = np.column_stack([stations, np.zeros(len(stations)), np.full(len(stations), 1.08)])
    target = np.array([100.0, 0.0, 0.6])
    empty_occ: set = set()   # nothing standing above the ground at all

    def crest(xy):
        return 5.0 * np.exp(-((xy[:, 0] - 50.0) ** 2) / (2 * 20.0 ** 2))

    without = available_sight_distance(stations, positions, target, empty_occ,
                                       eye_height_m=1.08, max_range_m=250.0)
    with_terrain = available_sight_distance(stations, positions, target, empty_occ,
                                            eye_height_m=1.08, max_range_m=250.0,
                                            ground_z_at=crest, object_height_m=0.6)

    assert without == 100.0, "voxel-only sees a fully clear corridor -- it cannot see the ground"
    assert with_terrain < without, "terrain layer must shorten the available distance"
    assert with_terrain <= 50.0, "the crest is at x=50; visibility cannot survive past it"


def test_isolated_voxels_are_dropped_but_clusters_survive():
    """A single stray return must never obstruct anything.

    Regression for a MEASURED error: cross-validating against HSO on pine_054823, station 10140
    reported 35 m against HSO's 115 m. The cause was speckle voxels, not real obstacles. Both
    viewshed.py and hso_corridor.py filter them before any sweep; this engine did not.
    """
    from sci_geom.sight import filter_isolated_voxels

    cluster = {(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)}
    speck = {(50, 50, 50)}
    kept = filter_isolated_voxels(cluster | speck, 2)
    assert cluster <= kept, "a real cluster must survive"
    assert (50, 50, 50) not in kept, "an isolated voxel must be dropped"
    # min_neighbors 0 disables the filter entirely
    assert filter_isolated_voxels(cluster | speck, 0) == cluster | speck


def test_a_lone_speck_no_longer_closes_a_sightline():
    """End to end: the same stray point, with and without the filter."""
    import numpy as np

    from sci_geom.sight import available_sight_distance, filter_isolated_voxels
    from sci_geom.voxel_los import voxelize

    stations = np.arange(0.0, 61.0, 10.0)
    positions = np.column_stack([stations, np.zeros(len(stations)), np.full(len(stations), 1.4)])
    target = np.array([60.0, 0.0, 1.4])
    # one stray return sitting on the line, three voxels wide so min_hit_samples cannot save us
    speck = np.array([[30.0, 0.0, 1.4], [30.05, 0.0, 1.4], [30.1, 0.0, 1.4]])
    raw = voxelize(speck, 0.1)

    unfiltered = available_sight_distance(stations, positions, target, raw,
                                          eye_height_m=1.4, voxel=0.1, clearance=0)
    filtered = available_sight_distance(stations, positions, target,
                                        filter_isolated_voxels(raw, 6),
                                        eye_height_m=1.4, voxel=0.1, clearance=0)
    assert unfiltered < 60.0, "fixture must actually block without the filter"
    assert filtered == 60.0, "speckle must not close the sightline once filtered"


def test_build_occupancy_can_exclude_points_near_the_corridor():
    """The road surface, and whatever sits on it, must not close a sight line ALONG the road.
    viewshed.py does this with ROAD_W = 3.8 m against the traced centreline."""
    import numpy as np

    from sci_geom.sight import build_occupancy

    corridor = np.column_stack([np.linspace(0, 50, 51), np.zeros(51), np.zeros(51)])
    rng = np.random.default_rng(0)
    on_road = np.column_stack([rng.uniform(0, 50, 400), rng.uniform(-1, 1, 400),
                               rng.uniform(1.0, 2.0, 400)])
    off_road = np.column_stack([rng.uniform(0, 50, 400), rng.uniform(8, 10, 400),
                                rng.uniform(1.0, 2.0, 400)])
    pts = np.vstack([on_road, off_road])

    keep_all = build_occupancy(pts, corridor, voxel=0.5, min_neighbors=0)
    excluded = build_occupancy(pts, corridor, voxel=0.5, min_neighbors=0, exclude_within_m=3.8)
    assert len(excluded) < len(keep_all)
    # nothing within 3.8 m of the corridor survives
    assert all(abs(k[1] * 0.5) > 3.0 for k in excluded)
