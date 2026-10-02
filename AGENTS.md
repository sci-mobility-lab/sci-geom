# sci-geom — agent rules

- This is a LIBRARY. It must never import another SCI package (`encounters`, `sci_gaze`, `sci_studio`, …),
  a server or a UI. Project-specific tables and schemas belong in the project, not here.
- New code goes into the existing module that owns the concept; add a module only for a new concept.
- Every change ships with a test in `tests/`; run `uv run pytest`.
- Equivalence tests against the original LiDAR implementations live in SCI_LiDAR, not here.
- Releasing: bump `version` in pyproject.toml, tag `vX.Y.Z`, push the tag.
