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
dependencies = ["sci-geom @ git+https://github.com/sci-mobility-lab/sci-geom@v0.2.0"]
```

For development:

```bash
uv sync
uv run pytest
```

## Changing it

A change here reaches a project only when that project bumps its pinned tag. That is deliberate:
a fix made for LiDAR cannot silently change Eye Tracking results. Tag every release (`v0.1.1`, …).

## Releases

| Tag | What changed | Effect on callers |
|---|---|---|
| v0.2.0 | `geodetic_to_enu` vectorised (arrays → `(3, …)`; scalars unchanged). `terrain_blocks_los`: the clearance ramp now descends to object heights below `clearance_m` (known-answer case K9: a flat road no longer blocks a road-level object beyond 2·eye/0.05 m); negative or non-finite heights raise. New `available_sight_distance_from` (observer fixed, targets moved outward, as in the corridor engines; eye and object heights applied; optional bisection; per-target verdicts). `available_sight_distance` raises TypeError when given `eye_height_m`, which it never used. | `terrain_blocks_los` is identical to v0.1.0 whenever object height ≥ `clearance_m`. Callers that passed `eye_height_m` to `available_sight_distance` must drop it (it never changed the result). Array inputs to `geodetic_to_enu` used to fail or silently return garbage. SCI_LiDAR's engines keep their own v0.1.0 ramp copy until they adopt this kernel (thesis EXP-004). |
| v0.1.1 | Shared `tum.rewarp_xyz` and initial K9 patch. | Superseded by v0.2.0: the initial signed ramp changed the historical custom-clearance case above the 0.6 m cap. v0.2.0 restores that case. |
| v0.1.0 | Initial split from SCI_LiDAR. | — |
