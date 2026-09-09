"""Python front-end to the CGAL mesher shipped with iso2mesh.

``cgalv2m`` turns a multi-labelled volume into a tetrahedral mesh; here the
volume is written as an INRImage, the bundled ``cgalmesh`` binary is invoked and
the resulting Medit mesh is read back and re-ordered for cache locality.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..config import exe_suffix, lib_dir
from ..io.meshfile import read_medit, save_inr
from ..utils.logging import get_logger
from ..utils.matlab import cart2sph, sortrows

__all__ = ["cgalmesh_path", "sort_mesh", "cgalv2m"]

logger = get_logger()

_RANDSEED = 0x623F9A9E
_DEFAULTS = {"angbound": 30.0, "radbound": 6.0, "distbound": 0.5, "reratio": 3.0}


def cgalmesh_path() -> Path:
    """Location of the ``cgalmesh`` binary for this platform."""
    binary = lib_dir() / "iso2mesh" / "bin" / ("cgalmesh" + exe_suffix())
    if binary.exists():
        binary.chmod(0o755)
    return binary


def sort_mesh(node, elem, face, origin=None):
    """Reorder nodes (and re-index elements) by distance from ``origin``.

    Sorting nodes that are close in space to indices that are close together
    reduces cache misses in the solver.  Port of iso2mesh's ``sortmesh``.
    """
    node = np.asarray(node, dtype=float)
    elem = np.asarray(elem, dtype=np.int64)
    face = np.asarray(face, dtype=np.int64)
    if origin is None:
        origin = node[0, :3]
    origin = np.asarray(origin, dtype=float).ravel()[:3]

    offset = node[:, :3] - origin[None, :]
    theta, phi, radius = cart2sph(offset[:, 0], offset[:, 1], offset[:, 2])
    node_map = sortrows(np.column_stack([radius, phi, theta]))
    sorted_node = node[node_map]

    inverse = np.empty_like(node_map)
    inverse[node_map] = np.arange(len(node_map))      # new index of every old node

    sorted_elem = elem.copy()
    sorted_elem[:, :4] = np.sort(inverse[elem[:, :4] - 1] + 1, axis=1)
    sorted_elem = sorted_elem[sortrows(sorted_elem, range(4))]

    sorted_face = face.copy()
    sorted_face[:, :3] = np.sort(inverse[face[:, :3] - 1] + 1, axis=1)
    sorted_face = sorted_face[sortrows(sorted_face, range(3))]
    return sorted_node, sorted_elem, sorted_face


def cgalv2m(volume, opt, maxvol):
    """Convert a labelled volume into a tetrahedral mesh.

    ``opt`` is a mapping with the CGAL parameters (``radbound``, ``angbound``,
    ``distbound``, ``reratio``); ``maxvol`` is the target maximum tetrahedron
    volume.  Returns ``(node, elem, face)`` with 1-based node indices, a region
    id in the last column of ``elem`` and a boundary id in the last column of
    ``face``.
    """
    volume = np.asarray(volume)
    if volume.dtype == bool:
        volume = volume.astype(np.uint8)
    if volume.dtype != np.uint8:
        raise ValueError("cgalmesher can only handle uint8 volumes; convert the image "
                         "to uint8 first.")
    if not volume.any():
        raise ValueError("no labeled regions found in the input volume.")

    logger.info("creating surface and tetrahedral mesh from a multi-domain volume ...")
    settings = dict(_DEFAULTS)
    if isinstance(opt, dict):
        settings.update({k: v for k, v in opt.items() if k in _DEFAULTS})
    else:
        settings["radbound"] = float(opt)

    binary = cgalmesh_path()
    if not binary.exists():
        raise FileNotFoundError(f"The CGAL mesher {binary} is missing from lib/iso2mesh.")

    with tempfile.TemporaryDirectory(prefix="roast_mesh_") as workdir:
        inr_file = Path(workdir) / "pre_cgalmesh.inr"
        mesh_file = Path(workdir) / "post_cgalmesh.mesh"
        save_inr(volume, inr_file)
        command = [str(binary), str(inr_file), str(mesh_file),
                   str(settings["angbound"]), str(settings["radbound"]),
                   str(settings["distbound"]), str(settings["reratio"]),
                   str(maxvol), str(_RANDSEED)]
        subprocess.run(command)
        if not mesh_file.exists():
            raise RuntimeError("output file was not found, failure was encountered when "
                               "running command:\n" + " ".join(command))
        node, elem, face = read_medit(mesh_file)

    logger.info("node number:\t%d\ntriangles:\t%d\ntetrahedra:\t%d\nregions:\t%d",
                len(node), len(face), len(elem), len(np.unique(elem[:, -1])))
    logger.info("surface and volume meshes complete")

    if len(node):
        node, elem, face = sort_mesh(node, elem, face, node[0, :])
    return node + 0.5, elem, face
