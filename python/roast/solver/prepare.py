"""Preparing the mesh for getDP: electrode surfaces and boundary conditions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import NUM_OF_TISSUE
from ..io.matfile import load_mat, save_mat
from ..utils.logging import get_logger

__all__ = ["free_boundary", "prepare_for_getdp"]

logger = get_logger()

_NOT_MESHED = (
    "{what} {name} was not meshed properly. Reasons may be: 1) electrode size is too "
    "small so the mesher cannot capture it; 2) mesh resolution is not high enough. "
    "Consider using bigger electrodes or increasing the mesh resolution by specifying "
    "the mesh options.")


def free_boundary(tets: np.ndarray) -> np.ndarray:
    """Triangles bounding a set of tetrahedra (those belonging to a single tet).

    ``tets`` holds 1-based node indices; the returned triangles keep that
    numbering.
    """
    tets = np.asarray(tets, dtype=np.int64)
    faces = np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                       tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    keys = np.sort(faces, axis=1)
    _, index, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return faces[index[counts == 1]]


def prepare_for_getdp(subj, node, elem, elec_needed, uni_tag):
    """Add the outer electrode surfaces to the mesh and record their areas.

    The injected current is imposed as a Neumann condition on the *outer*
    surface of every electrode, i.e. the part that does not touch the gel, so
    those triangles are appended to the mesh as their own physical regions.
    """
    subj = Path(subj)
    directory = subj.parent
    subj_name = subj.stem
    ready_file = directory / f"{subj_name}_{uni_tag}_ready.msh"
    if ready_file.exists():
        return ready_file

    node = np.asarray(node, dtype=float)
    elem = np.asarray(elem, dtype=np.int64)
    num_of_elec = len(elec_needed)

    elements_per_elec = []
    areas = np.zeros(num_of_elec)

    for i in range(num_of_elec):
        gel_tets = elem[elem[:, 4] == NUM_OF_TISSUE + i + 1, :4]
        elec_tets = elem[elem[:, 4] == NUM_OF_TISSUE + num_of_elec + i + 1, :4]
        if len(gel_tets) == 0:
            raise ValueError(_NOT_MESHED.format(what="Gel under electrode",
                                                name=elec_needed[i]))
        if len(elec_tets) == 0:
            raise ValueError(_NOT_MESHED.format(what="Electrode", name=elec_needed[i]))

        gel_faces = free_boundary(gel_tets)
        elec_faces = free_boundary(elec_tets)
        interface = np.intersect1d(np.unique(elec_faces), np.unique(gel_faces))
        on_interface = np.isin(elec_faces, interface)
        outer = elec_faces[on_interface.sum(axis=1) != 3]
        elements_per_elec.append(outer)

        vertices = node[:, :3]
        a = vertices[outer[:, 1] - 1] - vertices[outer[:, 0] - 1]
        b = vertices[outer[:, 2] - 1] - vertices[outer[:, 0] - 1]
        areas[i] = np.sum(0.5 * np.linalg.norm(np.cross(a, b), axis=1))

    area_file = directory / f"{subj_name}_{uni_tag}_usedElecArea.mat"
    if not area_file.exists():
        save_mat(area_file, {"area_elecNeeded": areas})

    logger.info("setting up boundary conditions...")
    num_of_part = len(np.unique(elem[:, 4]))
    extra = sum(len(faces) for faces in elements_per_elec)

    source = directory / f"{subj_name}_{uni_tag}.msh"
    with open(source, "r") as fin, open(ready_file, "w") as fout:
        while True:
            line = fin.readline()
            if not line:
                break
            text = line.rstrip("\n")
            if text == "$Elements":
                fout.write(text + "\n")
                num_of_elem = int(fin.readline().strip())
                fout.write("%d\n" % (num_of_elem + extra))
            elif text == "$EndElements":
                index = num_of_elem
                for j, faces in enumerate(elements_per_elec, start=1):
                    tag = num_of_part + j
                    for face in faces:
                        index += 1
                        fout.write("%d 2 2 %d %d %d %d %d \n"
                                   % (index, tag, tag, face[0], face[1], face[2]))
                fout.write(text + "\n")
            else:
                fout.write(text + "\n")
    return ready_file


def load_elec_areas(subj, uni_tag) -> np.ndarray:
    """Read back the electrode surface areas recorded by :func:`prepare_for_getdp`."""
    subj = Path(subj)
    data = load_mat(subj.parent / f"{subj.stem}_{uni_tag}_usedElecArea.mat")
    return np.asarray(data["area_elecNeeded"], dtype=float).ravel()
