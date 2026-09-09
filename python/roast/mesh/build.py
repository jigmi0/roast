"""Building the volumetric head mesh for the FEM solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import NUM_OF_TISSUE, TISSUE_NAMES
from ..io.matfile import save_mat
from ..io.meshfile import save_msh
from ..utils.logging import get_logger
from .iso2mesh import cgalv2m

__all__ = ["mesh_by_iso2mesh", "region_names"]

logger = get_logger()


def region_names(num_of_gel: int, num_of_elec: int):
    """Region labels of the mesh: the six tissues, then the gels, then the metal."""
    names = list(TISSUE_NAMES)
    names += [f"GEL{i + 1}" for i in range(num_of_gel)]
    names += [f"ELEC{i + 1}" for i in range(num_of_elec)]
    return names


def mesh_by_iso2mesh(subj, mask, elec, gel, opt, geom, uni_tag):
    """Mesh the segmented head together with the electrodes and the gel.

    The mesh coordinates are put into a *pseudo-world* space - voxel indices
    scaled by the header resolution - because solving in true world coordinates
    causes needless complications.  Units are mm; the solver reports mV, so no
    conversion to metres is needed.
    """
    subj = Path(subj)
    directory = subj.parent
    subj_name = subj.stem

    all_mask = np.array(mask.img, dtype=np.uint8, copy=True)
    gel_img = np.asarray(gel.img)
    elec_img = np.asarray(elec.img)
    num_of_gel = int(gel_img.max())
    num_of_elec = int(elec_img.max())

    for i in range(1, num_of_gel + 1):
        all_mask[gel_img == i] = NUM_OF_TISSUE + i
    for i in range(1, num_of_elec + 1):
        all_mask[elec_img == i] = NUM_OF_TISSUE + num_of_gel + i

    node, elem, face = cgalv2m(all_mask, opt, opt["maxvol"])
    node[:, :3] = node[:, :3] + 0.5           # ...and now in voxel space
    for i in range(3):
        node[:, i] = node[:, i] * geom.mat[i, i]

    logger.info("saving mesh...")
    names = region_names(num_of_gel, num_of_elec)
    save_msh(node[:, :3], elem, directory / f"{subj_name}_{uni_tag}.msh", names)
    save_mat(directory / f"{subj_name}_{uni_tag}.mat",
             {"node": node, "elem": elem, "face": face})
    return node, elem, face
