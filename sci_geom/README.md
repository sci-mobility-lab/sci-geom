# sci_geom — reusable geometry kernels

This pure library contains degeneracy, geodesy, PLY I/O, sight, terrain line-of-sight, TUM, and voxel line-of-sight modules. It has no CLI and does not require ROS2. `voxel_los.py` was extracted from the legacy ROS tree with an equivalence test.

## Gotchas

- ⚠️ There are no console scripts. Any runnable package module must be invoked with `python -m`.
- ⚠️ `sci_geom` and `encounters` must never import a FastAPI application module. Engines produce versioned artifacts; analysis consumes them, and deliverables must not require a running server.
