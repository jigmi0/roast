"""Readers for the electrode layout files.

``data/capInfo.xlsx`` holds one sheet per EEG system (``10-05``, ``BioSemi``
and ``EGI``); every row is ``name, x, y, z`` with the coordinates given on the
unit sphere of the template head.  ``data/elec72.loc`` lists the candidate
electrodes used for lead-field generation, in the same order as the ``10-05``
sheet.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import cap_info_file, elec_loc_file

__all__ = ["CapInfo", "read_cap_info", "sheet_for_cap_type", "read_elec_loc",
           "read_custom_locations", "custom_locations_path"]


class CapInfo:
    """Electrode names and template coordinates of one EEG system."""

    def __init__(self, names, coords):
        self.names = list(names)
        self.coords = np.asarray(coords, dtype=float)

    def __len__(self):
        return len(self.names)

    def index(self, name: str) -> int:
        """0-based position of ``name``; raises :class:`ValueError` if absent."""
        return self.names.index(name)


def sheet_for_cap_type(cap_type: str) -> str:
    cap_type = str(cap_type).lower()
    if cap_type in ("1020", "1010", "1005"):
        return "10-05"
    if cap_type == "biosemi":
        return "BioSemi"
    if cap_type == "egi":
        return "EGI"
    raise ValueError(f"Unsupported cap type: {cap_type}")


@lru_cache(maxsize=8)
def _read_sheet(path: str, sheet: str):
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    names, coords = [], []
    for row in workbook[sheet].iter_rows(values_only=True):
        if row[0] is None:
            continue
        names.append(str(row[0]))
        coords.append([float(row[1]), float(row[2]), float(row[3])])
    workbook.close()
    return tuple(names), np.asarray(coords, dtype=float)


def read_cap_info(cap_type: str) -> CapInfo:
    """Read the layout of one EEG system.  There is no header row in the file."""
    names, coords = _read_sheet(str(cap_info_file()), sheet_for_cap_type(cap_type))
    return CapInfo(names, coords)


def read_elec_loc(path=None):
    """Read ``elec72.loc``; returns the electrode names with the trailing dots
    stripped, exactly like the MATLAB code does."""
    path = Path(path) if path is not None else elec_loc_file()
    names = []
    with open(path, "r") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 4:
                names.append(fields[3].replace(".", ""))
    return names


def custom_locations_path(subj) -> Path:
    """``<subject folder>/<subject name>_customLocations``."""
    subj = Path(subj)
    directory = subj.parent if str(subj.parent) not in ("", ".") else Path.cwd()
    return directory / (subj.stem + "_customLocations")


def read_custom_locations(subj):
    """Read user-defined electrode locations for a subject.

    Returns ``(names, coords)`` where ``coords`` are 1-based voxel coordinates
    in the space of the *original* MRI.
    """
    path = custom_locations_path(subj)
    if not path.exists():
        raise FileNotFoundError(
            "You specified customized electrode locations but did not provide the "
            "location coordinates. Please put together all the coordinates in a text "
            "file 'subjectName_customLocations' and store it under the subject folder.")
    names, coords = [], []
    with open(path, "r") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 4:
                names.append(fields[0])
                coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return names, np.asarray(coords, dtype=float)
