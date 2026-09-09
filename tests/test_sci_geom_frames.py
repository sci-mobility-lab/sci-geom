"""Tests for the versioned frame / extrinsic / CRS registry.

The behaviours worth guarding here are the refusals, not the algebra. A registry that resolves a
chain is easy; a registry that REFUSES to answer when two extrinsics were in force at once, or
when the vertical datum was never stated, is the thing that prevents a wrong number from being
quoted with confidence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sci_geom import frames


IDENTITY_Q = [0.0, 0.0, 0.0, 1.0]

CRS_BLOCK = {
    "horizontal": "EPSG:6446",
    "vertical": "EPSG:5703",
    "geoid": "GEOID18",
    "epoch": "2010.0",
}


def _registry(transforms, crs=None):
    return frames.loads(
        {
            "schema_version": frames.SCHEMA_VERSION,
            "crs": crs if crs is not None else CRS_BLOCK,
            "transforms": transforms,
        },
        name="<test>",
    )


def _entry(child, parent, translation=(0, 0, 0), quaternion=IDENTITY_Q, **kw):
    d = {
        "child": child,
        "parent": parent,
        "translation": list(translation),
        "quaternion": list(quaternion),
    }
    d.update(kw)
    return d


# --- resolution ------------------------------------------------------------------------------


def test_a_two_hop_chain_composes_in_the_right_order():
    # lidar sits 1 m forward of body; body sits 10 m east of enu.
    reg = _registry(
        [
            _entry("lidar", "body", translation=(1, 0, 0)),
            _entry("body", "enu", translation=(10, 0, 0)),
        ]
    )
    m = reg.resolve("lidar", "enu")
    point_in_enu = frames.apply(m, np.array([[0.0, 0.0, 0.0]]))
    assert np.allclose(point_in_enu, [[11.0, 0.0, 0.0]])


def test_resolving_against_the_chain_direction_uses_the_inverse():
    reg = _registry([_entry("lidar", "body", translation=(1, 2, 3))])
    forward = reg.resolve("lidar", "body")
    backward = reg.resolve("body", "lidar")
    assert np.allclose(forward @ backward, np.eye(4), atol=1e-12)


def test_rotation_is_applied_before_translation():
    # 90 deg about +Z, then a translation expressed in the PARENT frame.
    q = Rotation.from_euler("z", 90, degrees=True).as_quat()
    reg = _registry([_entry("s", "p", translation=(5, 0, 0), quaternion=list(q))])
    got = frames.apply(reg.resolve("s", "p"), np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(got, [[5.0, 1.0, 0.0]], atol=1e-12)


def test_a_frame_resolves_to_itself_as_identity():
    reg = _registry([_entry("lidar", "body")])
    assert np.allclose(reg.resolve("lidar", "lidar"), np.eye(4))


def test_an_unconnected_frame_raises_rather_than_guessing():
    reg = _registry([_entry("lidar", "body"), _entry("gnss", "body")])
    with pytest.raises(frames.RegistryError, match="unknown frame"):
        reg.resolve("lidar", "camera")


def test_a_frame_outside_the_active_window_reports_no_chain():
    reg = _registry(
        [
            _entry(
                "lidar",
                "body",
                valid_from="2026-01-01T00:00:00+00:00",
                valid_to="2026-02-01T00:00:00+00:00",
            ),
            _entry("body", "enu"),
        ]
    )
    after = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with pytest.raises(frames.RegistryError, match="no chain of transforms"):
        reg.resolve("lidar", "enu", at=after)


# --- the point of the whole module: time-varying extrinsics -----------------------------------


def test_the_registry_returns_the_extrinsic_that_was_in_force_that_day():
    """A re-mounted sensor must not silently reinterpret older acquisitions."""
    reg = _registry(
        [
            _entry(
                "lidar",
                "body",
                translation=(1, 0, 0),
                valid_from="2026-01-01T00:00:00+00:00",
                valid_to="2026-06-01T00:00:00+00:00",
                note="original mount",
            ),
            _entry(
                "lidar",
                "body",
                translation=(2, 0, 0),
                valid_from="2026-06-01T00:00:00+00:00",
                note="re-mounted, re-calibrated",
            ),
        ]
    )
    before = reg.resolve("lidar", "body", at=datetime(2026, 3, 1, tzinfo=timezone.utc))
    after = reg.resolve("lidar", "body", at=datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert np.allclose(before[:3, 3], [1, 0, 0])
    assert np.allclose(after[:3, 3], [2, 0, 0])


def test_valid_to_is_exclusive_so_windows_can_abut_without_overlapping():
    reg = _registry(
        [
            _entry("a", "b", translation=(1, 0, 0), valid_to="2026-06-01T00:00:00+00:00"),
            _entry("a", "b", translation=(2, 0, 0), valid_from="2026-06-01T00:00:00+00:00"),
        ]
    )
    boundary = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert np.allclose(reg.resolve("a", "b", at=boundary)[:3, 3], [2, 0, 0])


# --- refusals --------------------------------------------------------------------------------


def test_overlapping_validity_windows_are_refused_at_load():
    with pytest.raises(frames.RegistryError, match="overlapping validity"):
        _registry(
            [
                _entry("a", "b", valid_from="2026-01-01T00:00:00+00:00"),
                _entry("a", "b", valid_from="2026-03-01T00:00:00+00:00"),
            ]
        )


def test_a_frame_with_two_simultaneous_parents_is_refused():
    with pytest.raises(frames.RegistryError, match="must stay a tree"):
        _registry([_entry("lidar", "body"), _entry("lidar", "enu")])


@pytest.mark.parametrize("missing", ["horizontal", "vertical", "geoid", "epoch"])
def test_an_incomplete_crs_is_refused_rather_than_defaulted(missing):
    crs = dict(CRS_BLOCK)
    del crs[missing]
    with pytest.raises(frames.RegistryError, match=missing):
        _registry([_entry("a", "b")], crs=crs)


def test_an_unnormalised_quaternion_is_refused():
    with pytest.raises(frames.RegistryError, match="quaternion norm"):
        _registry([_entry("a", "b", quaternion=[0.0, 0.0, 0.0, 0.9])])


def test_a_naive_timestamp_is_refused_because_bag_time_is_utc():
    with pytest.raises(frames.RegistryError, match="no timezone"):
        _registry([_entry("a", "b", valid_from="2026-01-01T00:00:00")])


def test_valid_to_before_valid_from_is_refused():
    with pytest.raises(frames.RegistryError, match="not after"):
        _registry(
            [
                _entry(
                    "a",
                    "b",
                    valid_from="2026-06-01T00:00:00+00:00",
                    valid_to="2026-01-01T00:00:00+00:00",
                )
            ]
        )


def test_a_self_referential_transform_is_refused():
    with pytest.raises(frames.RegistryError, match="child == parent"):
        _registry([_entry("a", "a")])


def test_an_unknown_schema_version_is_refused():
    with pytest.raises(frames.RegistryError, match="schema_version"):
        frames.loads({"schema_version": 999, "crs": CRS_BLOCK, "transforms": []})


def test_an_empty_registry_is_refused():
    with pytest.raises(frames.RegistryError, match="declares no transforms"):
        _registry([])


# --- io --------------------------------------------------------------------------------------


def test_load_reads_a_file_and_keeps_provenance(tmp_path):
    path = tmp_path / "rig.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": frames.SCHEMA_VERSION,
                "crs": CRS_BLOCK,
                "transforms": [
                    _entry(
                        "lidar",
                        "body",
                        translation=(0.1, 0.2, 0.3),
                        provenance={"method": "checkerboard", "residual_m": 0.0158},
                    )
                ],
            }
        )
    )
    reg = frames.load(path)
    assert reg.crs.vertical == "EPSG:5703"
    assert reg.transforms[0].provenance["residual_m"] == 0.0158


def test_apply_rejects_wrongly_shaped_points():
    reg = _registry([_entry("a", "b")])
    with pytest.raises(frames.RegistryError, match=r"\(N, 3\)"):
        frames.apply(reg.resolve("a", "b"), np.zeros((4, 2)))
