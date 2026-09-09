"""Mesh and solver file formats: INRImage, Medit, Gmsh and getDP node tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["save_inr", "read_medit", "save_msh", "read_pos"]


def save_inr(volume: np.ndarray, path) -> Path:
    """Write a labelled volume in the INRImage format expected by the CGAL
    mesher (port of iso2mesh's ``saveinr``)."""
    volume = np.asarray(volume)
    if volume.dtype == bool:
        volume = volume.astype(np.uint8)
    if volume.dtype == np.uint8:
        btype, bitlen = "unsigned fixed", 8
    elif volume.dtype == np.uint16:
        btype, bitlen = "unsigned fixed", 16
    elif volume.dtype == np.float32:
        btype, bitlen = "float", 32
    elif volume.dtype == np.float64:
        btype, bitlen = "float", 64
    else:
        raise ValueError(f"volume format not supported: {volume.dtype}")

    header = ("#INRIMAGE-4#{{\nXDIM={}\nYDIM={}\nZDIM={}\nVDIM=1\nTYPE={}\n"
              "PIXSIZE={} bits\nCPU=decm\nVX=1\nVY=1\nVZ=1\n").format(
                  volume.shape[0], volume.shape[1], volume.shape[2], btype, bitlen)
    header = header + "\n" * (256 - 4 - len(header)) + "##}\n"

    path = Path(path)
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        # MATLAB writes column-major, which is what the reader expects.
        handle.write(np.asfortranarray(volume).tobytes(order="F"))
    return path


def read_medit(path):
    """Read a Medit ``.mesh`` file (port of iso2mesh's ``readmedit``).

    Returns ``(node, elem, face)``: node coordinates with a trailing label
    column, tetrahedra with a region column and boundary triangles with a
    surface-id column.  Node indices stay 1-based.
    """
    tokens = Path(path).read_text().split()
    node = np.zeros((0, 4))
    elem = np.zeros((0, 5), dtype=np.int64)
    face = np.zeros((0, 4), dtype=np.int64)

    i = 0
    while i < len(tokens):
        key = tokens[i]
        i += 1
        if key == "End":
            break
        if key not in ("Vertices", "Triangles", "Tetrahedra"):
            continue
        count = int(tokens[i])
        i += 1
        width = {"Vertices": 4, "Triangles": 4, "Tetrahedra": 5}[key]
        block = np.asarray(tokens[i:i + width * count], dtype=float)
        i += width * count
        block = block.reshape(count, width)
        if key == "Vertices":
            node = block
        elif key == "Triangles":
            face = block.astype(np.int64)
        else:
            elem = block.astype(np.int64)
    return node, elem, face


def save_msh(node: np.ndarray, elem: np.ndarray, path, region_names=None) -> Path:
    """Write a tetrahedral mesh in Gmsh 2.2 ASCII format (port of ``savemsh``).

    ``elem`` is ``(n, 5)``: four 1-based node indices plus the region id.
    """
    node = np.asarray(node, dtype=float)[:, :3]
    elem = np.asarray(elem, dtype=np.int64)
    if elem.shape[1] < 5:
        elem = np.column_stack([elem, np.ones(len(elem), dtype=np.int64)])

    path = Path(path)
    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write("$Nodes\n%d\n" % len(node))
        for i, (x, y, z) in enumerate(node, start=1):
            handle.write("%d %10.10f %10.10f %10.10f\n" % (i, x, y, z))
        handle.write("$EndNodes\n")
        handle.write("$Elements\n%d\n" % len(elem))
        for i, row in enumerate(elem, start=1):
            region = int(row[4])
            handle.write("%d 4 2 %d %d %d %d %d %d\n"
                         % (i, region, region, row[0], row[1], row[2], row[3]))
        handle.write("$EndElements\n")
    return path


def read_pos(path, ncols: int):
    """Read a getDP ``NodeTable`` result file.

    ``ncols`` is the number of value columns (1 for the voltage, 3 for the
    electric field).  Returns ``(node_ids, values)`` with 1-based node ids.
    """
    ids, values = [], []
    with open(path, "r") as handle:
        handle.readline()                      # skip the '$...' header line
        for line in handle:
            fields = line.split()
            if len(fields) < ncols + 1:
                continue
            try:
                node_id = int(fields[0])
                row = [float(v) for v in fields[1:ncols + 1]]
            except ValueError:
                break                          # textscan stops at the first mismatch
            ids.append(node_id)
            values.append(row)
    if not ids:
        raise ValueError(f"No node values found in {path}")
    return np.asarray(ids, dtype=np.int64), np.asarray(values, dtype=float)
