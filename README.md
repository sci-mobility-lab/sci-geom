# sci-geom

Geometry kernels shared by the SCI projects (SCI_LiDAR, SCI_Eye_Tracking). Pure Python + numpy/scipy:
no server, no UI, no ROS2, and no import of any other SCI package.

| Module | What it does |
|---|---|
| `geodesy` | lat/lon ↔ local ENU metres (exact; optional `pyproj`) |
| `frames`, `tum`, `ply_io` | coordinate frames, TUM trajectories, PLY point clouds |
| `route` | stationing along a route |
| `voxel_los`, `terrain_los`, `sight` | line of sight through voxels / terrain, available sight distance |
| `degeneracy` | geometric degeneracy checks |

## Install

As a dependency of another project (pin a tag — never a branch):

```toml
dependencies = ["sci-geom @ git+https://github.com/sci-mobility-lab/sci-geom@v0.1.0"]
```

For development:

```bash
uv sync
uv run pytest
```

## Changing it

A change here reaches a project only when that project bumps its pinned tag. That is deliberate:
a fix made for LiDAR cannot silently change Eye Tracking results. Tag every release (`v0.1.1`, …).
