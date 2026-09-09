# sci_geom — reusable geometry kernels

This pure library contains degeneracy, geodesy, PLY I/O, sight, terrain line-of-sight, TUM, and voxel line-of-sight modules. It has no CLI and does not require ROS2. `voxel_los.py` was extracted from the legacy ROS tree with an equivalence test.

## Install

The root pyproject packages this directory as `sci-analysis` for Python 3.12+ with NumPy, pandas, and PyArrow. Install it into the engine and gaze environments as needed.

```bash
platform/engine/.venv/bin/pip install -e .
platform/engine/.venv/bin/pip install -e .
```

## Run

There is no command-line entry point; import `sci_geom` from Python.

```bash
platform/engine/.venv/bin/python -c 'import sci_geom'
```

## Test

```bash
./run_tests.sh root
```

This runs the 91 root tests. There is no root venv — `run_tests.sh` uses the engine venv.

## Gotchas

- ⚠️ There are no console scripts. Any runnable package module must be invoked with `python -m`.
- ⚠️ `sci_geom` and `encounters` must never import a FastAPI application module. Engines produce versioned artifacts; analysis consumes them, and deliverables must not require a running server.
