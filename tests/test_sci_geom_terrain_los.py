"""Analytic terrain cases; no field experiment or voxel occupancy involved."""
import numpy as np
import pytest

from sci_geom.terrain_los import terrain_blocks_los


@pytest.mark.parametrize("distance", [20., 40., 60., 100., 200.])
@pytest.mark.parametrize("height", [0., .02, .6, 1.08])
def test_k9_flat_ground_never_blocks_target(distance, height):
    assert not terrain_blocks_los([0, 0, 1.08], [distance, 0, height],
                                 lambda xy: np.zeros(len(xy)), object_height=height)


def test_crest_blocks_ground_target_but_elevated_target_clears():
    def ground(xy):
        return np.where((xy[:, 0] >= 20) & (xy[:, 0] <= 35), 2., 0.)
    assert terrain_blocks_los([0, 0, 1.08], [100, 0, 0], ground, object_height=0.)
    assert terrain_blocks_los([0, 0, 1.08], [100, 0, .6], ground, object_height=.6)
    assert not terrain_blocks_los([0, 0, 1.08], [100, 0, 10.], ground, object_height=10.)


def test_single_spike_cannot_block_but_two_consecutive_samples_do():
    for width, blocked in [(0., False), (1., True)]:
        def ground(xy):
            return np.where((xy[:, 0] >= 20) & (xy[:, 0] <= 20 + width), 2., 0.)
        assert terrain_blocks_los([0, 0, 1.08], [100, 0, 0], ground, object_height=0.) is blocked
