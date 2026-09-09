"""Structured PLY I/O extracted from ``platform/engine/sci_studio/ply_io.py``.

That module was the closest canonical implementation among the repository's many small PLY
readers and writers.  This shared copy preserves its supported scalar types and ASCII/binary
behaviour while making truncated or malformed headers fail instead of looping or returning an
accidentally constructed dtype.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "int64": "i8", "uint64": "u8", "long": "i8", "ulong": "u8",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}

NP_PLY_TYPES = {
    ("f", 4): "float",
    ("f", 8): "double",
    ("u", 1): "uchar",
    ("i", 1): "char",
    ("u", 2): "ushort",
    ("i", 2): "short",
    ("u", 4): "uint",
    ("i", 4): "int",
}


def read_ply_struct(path: str | Path):
    path = Path(path)
    with path.open("rb") as f:
        first = f.readline()
        if first.rstrip(b"\r\n") != b"ply":
            raise ValueError("invalid PLY header: missing 'ply' magic")

        fmt = None
        names, types = [], []
        n = None
        in_vertex = False
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError("invalid PLY header: missing end_header")
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("format"):
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError("invalid PLY format declaration")
                fmt = parts[1]
            elif line.startswith("element"):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"invalid PLY element declaration: {line}")
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n = int(parts[2])
                    if n < 0:
                        raise ValueError("PLY vertex count cannot be negative")
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"invalid PLY property declaration: {line}")
                if parts[1] == "list":
                    raise ValueError("PLY vertex list properties are not supported")
                if len(parts) != 3:
                    raise ValueError(f"invalid PLY property declaration: {line}")
                types.append(parts[1])
                names.append(parts[2])
            elif line == "end_header":
                break

        if fmt is None or n is None or not names:
            raise ValueError("invalid PLY header: format, vertex element, and properties are required")
        try:
            dtype = np.dtype([
                (name, "<" + PLY_TYPES[typ]) for name, typ in zip(names, types)
            ])
        except KeyError as exc:
            raise ValueError(f"unsupported PLY property type: {exc.args[0]}") from exc

        if fmt == "binary_little_endian":
            result = np.fromfile(f, dtype=dtype, count=n)
        elif fmt == "ascii":
            result = np.loadtxt(f, dtype=dtype, max_rows=n, ndmin=1)
        else:
            raise ValueError(f"unsupported PLY format: {fmt}")
        if len(result) != n:
            raise ValueError(f"truncated PLY vertex data: expected {n} rows, got {len(result)}")
        return result


def write_ply_struct(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    if arr.dtype.names is None:
        raise ValueError("PLY writer requires a structured array")
    props = []
    for name in arr.dtype.names:
        dt = arr.dtype[name]
        if dt.subdtype is not None:
            raise ValueError(f"unsupported PLY field {name}: subarray dtype {dt}")
        typ = NP_PLY_TYPES.get((dt.kind, dt.itemsize))
        if typ is None:
            raise ValueError(f"unsupported PLY field {name}: dtype {dt}")
        props.append(f"property {typ} {name}\n")
    hdr = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(arr)}\n"
        + "".join(props)
        + "end_header\n"
    )
    with path.open("wb") as f:
        f.write(hdr.encode("utf-8"))
        arr.tofile(f)
