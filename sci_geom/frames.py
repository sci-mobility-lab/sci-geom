"""Versioned frame, extrinsic and CRS registry.

Why this exists
---------------
Until now every extrinsic in this project lived in an ad-hoc JSON beside the data that produced
it (``~/data/calib/*/calib.json``, ``pine_mountain_calib_output/report.json``) and was re-read by
whichever script needed it -- twelve files reference ``extrinsic`` / ``boresight`` / ``lever``
with no shared source. Two consequences, both of which have already cost time here:

1. **No answer to "which extrinsic was in force when this bag was recorded?"** A rig that is
   re-mounted, re-calibrated or modified silently invalidates every earlier product, and nothing
   records the boundary.
2. **No vertical datum anywhere.** A project that reports cross-slope and grade to a DOT cannot
   state which vertical reference its heights are on.

This module is the one place both are declared. It is deliberately NOT ROS TF: TF is a runtime
pub/sub mechanism with a time-windowed buffer, and offline we want the opposite -- an explicit,
diffable, git-versioned file with validity dates and provenance per transform.

Conventions
-----------
- A transform entry gives the pose of ``child`` expressed in ``parent``: the 4x4 that maps a point
  in child coordinates into parent coordinates (``T_parent_child``).
- Rotations are TUM-order quaternions ``(qx, qy, qz, qw)``, matching every ``.tum`` file in this
  repository.
- ``valid_from`` / ``valid_to`` are ISO-8601 UTC instants. ``valid_to`` is exclusive and may be
  null, meaning "still in force".
- The registry is JSON, not YAML: ``sci_geom`` carries numpy/scipy only and must stay installable
  in the light analysis environment. Human context goes in the ``note`` and ``provenance`` fields.

Loading a registry VALIDATES it. A registry without a vertical CRS, or with two transforms for the
same (child, parent) whose validity windows overlap, is rejected at load rather than silently
resolved -- that is the entire point of the file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA_VERSION = 1


class RegistryError(ValueError):
    """A registry is malformed, ambiguous, or cannot answer the question asked."""


def _parse_instant(value: str | None, what: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RegistryError(f"{what}: not an ISO-8601 instant: {value!r}") from exc
    if parsed.tzinfo is None:
        raise RegistryError(
            f"{what}: {value!r} has no timezone. Extrinsic validity is compared against bag "
            "timestamps, which are UTC; a naive instant is an ambiguity waiting to happen."
        )
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class CRS:
    """The spatial reference a registry's products are expressed in.

    All four fields are required. A horizontal EPSG code alone does not let anyone reproduce a
    height, and an epoch is what makes a NAD83(2011) coordinate mean a specific thing.
    """

    horizontal: str
    vertical: str
    geoid: str
    epoch: str
    note: str = ""

    @staticmethod
    def from_dict(d: dict) -> "CRS":
        missing = [k for k in ("horizontal", "vertical", "geoid", "epoch") if not d.get(k)]
        if missing:
            raise RegistryError(
                "crs is missing required field(s) " + ", ".join(missing) + ". "
                "This is refused rather than defaulted: an unstated vertical datum is exactly the "
                "gap that makes a cross-slope deliverable uncheckable."
            )
        return CRS(
            horizontal=str(d["horizontal"]),
            vertical=str(d["vertical"]),
            geoid=str(d["geoid"]),
            epoch=str(d["epoch"]),
            note=str(d.get("note", "")),
        )


@dataclass(frozen=True)
class Transform:
    """``T_parent_child`` with a validity window and its provenance."""

    child: str
    parent: str
    translation: np.ndarray
    quaternion: np.ndarray
    valid_from: datetime | None
    valid_to: datetime | None
    provenance: dict = field(default_factory=dict)
    note: str = ""

    def matrix(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = Rotation.from_quat(self.quaternion).as_matrix()
        m[:3, 3] = self.translation
        return m

    def covers(self, at: datetime | None) -> bool:
        """Is this entry in force at ``at``? An entry with no window is always in force."""
        if at is None:
            return self.valid_from is None and self.valid_to is None
        if self.valid_from is not None and at < self.valid_from:
            return False
        if self.valid_to is not None and at >= self.valid_to:
            return False
        return True

    @staticmethod
    def from_dict(d: dict) -> "Transform":
        for key in ("child", "parent", "translation", "quaternion"):
            if key not in d:
                raise RegistryError(f"transform entry is missing {key!r}: {d!r}")
        translation = np.asarray(d["translation"], dtype=np.float64)
        quaternion = np.asarray(d["quaternion"], dtype=np.float64)
        if translation.shape != (3,):
            raise RegistryError(f"{d['child']}<-{d['parent']}: translation must be 3 numbers")
        if quaternion.shape != (4,):
            raise RegistryError(
                f"{d['child']}<-{d['parent']}: quaternion must be 4 numbers in TUM order "
                "(qx, qy, qz, qw)"
            )
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise RegistryError(
                f"{d['child']}<-{d['parent']}: quaternion norm is {norm:.9f}, expected 1. "
                "A silently unnormalised quaternion scales every transformed point."
            )
        if d["child"] == d["parent"]:
            raise RegistryError(f"transform has child == parent ({d['child']!r})")
        label = f"{d['child']}<-{d['parent']}"
        valid_from = _parse_instant(d.get("valid_from"), f"{label} valid_from")
        valid_to = _parse_instant(d.get("valid_to"), f"{label} valid_to")
        if valid_from is not None and valid_to is not None and valid_to <= valid_from:
            raise RegistryError(f"{label}: valid_to {valid_to} is not after valid_from {valid_from}")
        return Transform(
            child=str(d["child"]),
            parent=str(d["parent"]),
            translation=translation,
            quaternion=quaternion,
            valid_from=valid_from,
            valid_to=valid_to,
            provenance=dict(d.get("provenance", {})),
            note=str(d.get("note", "")),
        )


def _windows_overlap(a: Transform, b: Transform) -> bool:
    a_start = a.valid_from or datetime.min.replace(tzinfo=timezone.utc)
    b_start = b.valid_from or datetime.min.replace(tzinfo=timezone.utc)
    a_end = a.valid_to or datetime.max.replace(tzinfo=timezone.utc)
    b_end = b.valid_to or datetime.max.replace(tzinfo=timezone.utc)
    return a_start < b_end and b_start < a_end


@dataclass(frozen=True)
class Registry:
    """A validated set of frames and the CRS their root is expressed in."""

    crs: CRS
    transforms: tuple[Transform, ...]
    name: str = ""

    @property
    def frames(self) -> set[str]:
        out: set[str] = set()
        for t in self.transforms:
            out.add(t.child)
            out.add(t.parent)
        return out

    def at(self, when: datetime | None) -> tuple[Transform, ...]:
        return tuple(t for t in self.transforms if t.covers(when))

    def resolve(self, frm: str, to: str, at: datetime | None = None) -> np.ndarray:
        """4x4 mapping points expressed in ``frm`` into ``to``, using entries in force at ``at``."""
        if frm not in self.frames:
            raise RegistryError(f"unknown frame {frm!r}; known frames: {sorted(self.frames)}")
        if to not in self.frames:
            raise RegistryError(f"unknown frame {to!r}; known frames: {sorted(self.frames)}")
        if frm == to:
            return np.eye(4, dtype=np.float64)

        active = self.at(at)
        # adjacency: node -> (neighbour, matrix mapping node coords into neighbour coords)
        adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
        for t in active:
            m = t.matrix()
            adjacency.setdefault(t.child, []).append((t.parent, m))
            adjacency.setdefault(t.parent, []).append((t.child, np.linalg.inv(m)))

        # BFS keeps the path with the fewest compositions, which is also the fewest inverses.
        frontier: list[tuple[str, np.ndarray]] = [(frm, np.eye(4, dtype=np.float64))]
        seen = {frm}
        while frontier:
            node, acc = frontier.pop(0)
            if node == to:
                return acc
            for neighbour, m in adjacency.get(node, []):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                frontier.append((neighbour, m @ acc))

        when = "with no validity filter" if at is None else f"in force at {at.isoformat()}"
        raise RegistryError(
            f"no chain of transforms {when} connects {frm!r} to {to!r}. "
            "Either the extrinsic was never recorded, or its validity window does not cover this "
            "acquisition -- both are real answers, and both are better than a wrong number."
        )


def _validate(crs: CRS, transforms: Iterable[Transform], name: str) -> Registry:
    transforms = tuple(transforms)
    if not transforms:
        raise RegistryError(f"{name}: registry declares no transforms")

    # Two entries for the same directed pair may not be in force at the same time.
    by_pair: dict[tuple[str, str], list[Transform]] = {}
    for t in transforms:
        by_pair.setdefault((t.child, t.parent), []).append(t)
    for (child, parent), entries in by_pair.items():
        for i, a in enumerate(entries):
            for b in entries[i + 1 :]:
                if _windows_overlap(a, b):
                    raise RegistryError(
                        f"{name}: two transforms for {child}<-{parent} have overlapping validity "
                        "windows. Which one applied to a given bag would be a coin flip, so this "
                        "is refused at load."
                    )

    # A frame with two different parents in force at once makes the chain ambiguous.
    parents: dict[str, list[Transform]] = {}
    for t in transforms:
        parents.setdefault(t.child, []).append(t)
    for child, entries in parents.items():
        distinct = {t.parent for t in entries}
        if len(distinct) > 1:
            for i, a in enumerate(entries):
                for b in entries[i + 1 :]:
                    if a.parent != b.parent and _windows_overlap(a, b):
                        raise RegistryError(
                            f"{name}: frame {child!r} has parents {a.parent!r} and {b.parent!r} in "
                            "force at the same time; the frame tree must stay a tree."
                        )
    return Registry(crs=crs, transforms=transforms, name=name)


def load(path: str | Path) -> Registry:
    """Read and validate a registry file."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{path}: not valid JSON: {exc}") from exc
    return loads(raw, name=str(path))


def loads(raw: dict, name: str = "<dict>") -> Registry:
    """Validate an already-parsed registry mapping."""
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RegistryError(
            f"{name}: schema_version is {version!r}, this build understands {SCHEMA_VERSION}"
        )
    if "crs" not in raw:
        raise RegistryError(f"{name}: registry has no 'crs' block")
    crs = CRS.from_dict(raw["crs"])
    transforms = [Transform.from_dict(d) for d in raw.get("transforms", [])]
    return _validate(crs, transforms, name)


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to an (N, 3) array of points."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RegistryError(f"points must be (N, 3), got {points.shape}")
    return points @ matrix[:3, :3].T + matrix[:3, 3]
