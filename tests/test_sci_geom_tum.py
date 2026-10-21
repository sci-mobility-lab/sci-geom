"""Format and validation tests for the shared TUM loader."""
import numpy as np
import pytest

from sci_geom.tum import load_tum, load_tum_xyz


def test_comments_blank_lines_and_trailing_whitespace_parse_exactly(tmp_path):
    path = tmp_path / "trajectory.tum"
    path.write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        "\n"
        "1.0 10 20 30 0.1 0.2 0.3 0.9   \n"
        "2.0 -1 -2 -3 -0.4 -0.5 -0.6 0.7   # pose two\n"
        "   \n",
        encoding="utf-8",
    )
    expected = np.array([
        [1.0, 10, 20, 30, 0.1, 0.2, 0.3, 0.9],
        [2.0, -1, -2, -3, -0.4, -0.5, -0.6, 0.7],
    ], dtype=np.float64)

    actual = load_tum(path)

    assert actual.shape == (2, 8)
    assert actual.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(actual, expected)
    # TUM quaternion order is explicitly qx, qy, qz, qw -- never qw-first.
    np.testing.assert_array_equal(actual[:, 4:8], expected[:, [4, 5, 6, 7]])
    np.testing.assert_array_equal(load_tum_xyz(path), expected[:, 1:4])


@pytest.mark.parametrize(
    "row",
    [
        "1 2 3 4 0 0 1",
        "1 2 3 4 0 0 0 1 unexpected",
    ],
)
def test_wrong_number_of_columns_raises(tmp_path, row):
    path = tmp_path / "wrong.tum"
    path.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 8 columns"):
        load_tum(path)


def test_non_increasing_timestamps_raise_instead_of_being_silently_sorted(tmp_path):
    path = tmp_path / "unsorted.tum"
    path.write_text("2 0 0 0 0 0 0 1\n1 1 1 1 0 0 0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_tum(path)


# Shared per-point re-posing: analytic cases, independent of the LiDAR adapter.
def test_rewarp_translation_rotation_and_overlap():
    from scipy.spatial.transform import Rotation, Slerp
    from sci_geom.tum import rewarp_xyz
    t = np.array([0., 2.])
    old_p = np.array([[0., 0., 0.], [2., 0., 0.]])
    new_p = old_p + [0., 3., 0.]
    old_r = Slerp(t, Rotation.identity(2))
    new_r = Slerp(t, Rotation.from_rotvec([[0., 0., np.pi / 2], [0., 0., np.pi / 2]]))
    out = rewarp_xyz(np.array([[2., 0., 0.]]), np.array([1.]), t, old_p, old_r, t, new_p, new_r)
    np.testing.assert_allclose(out, [[1., 4., 0.]], atol=1e-12)
    with pytest.raises(ValueError, match="overlap"):
        rewarp_xyz(np.ones((1, 3)), np.array([3.]), t, old_p, old_r, t, new_p, new_r)
    with pytest.raises(ValueError, match="finite"):
        rewarp_xyz(np.ones((1, 3)), np.array([np.nan]), t, old_p, old_r, t, new_p, new_r)
    with pytest.raises(ValueError, match="expected"):
        rewarp_xyz(np.ones((1, 2)), np.array([1.]), t, old_p, old_r, t, new_p, new_r)
    assert rewarp_xyz(np.empty((0, 3)), np.array([]), t, old_p, old_r, t, new_p, new_r).shape == (0, 3)
