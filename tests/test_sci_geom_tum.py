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
