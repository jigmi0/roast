"""Electrode bookkeeping: which pool each requested electrode comes from.

ROAST accepts three kinds of electrodes at once: those of a standard EEG system
(``capInfo.xlsx``), the four neck electrodes ``nk1``..``nk4``, and user-defined
locations read from ``<subject>_customLocations``.  They are placed in that
order, so :func:`elec_preproc` also returns the permutation that maps the
internal order back to the order the user asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from ..config import NECK_ELECTRODES
from ..io.caps import read_cap_info, read_custom_locations
from ..utils.logging import get_logger

__all__ = ["ElecPara", "elec_preproc"]

logger = get_logger()

_UNKNOWN_ELEC = (
    "Unrecognized electrodes found. It may come from the following mistakes: "
    "1) you specified one cap type (e.g. 1010) but asked the electrode name in the "
    "other system (e.g. BioSemi); 2) you defined some customized electrode location "
    "but forgot to put 'custom' as a prefix in the electrode name; 3) you picked up "
    "one of the electrodes that falls on the ears or eyes (which are removed, see "
    "capLayout.pdf); 4) you asked ROAST to do an electrode that does not belong to "
    "any system (neither 1005, BioSemi, EGI, nor your customized electrodes).")


@dataclass
class ElecPara:
    """Placement parameters of one electrode (or of a whole uniform set)."""

    cap_type: str = "1010"
    elec_type: str = "disc"
    elec_size: np.ndarray = field(default_factory=lambda: np.array([6.0, 2.0]))
    elec_ori: object = None
    # 0-based indices of the requested electrodes within each pool; identical
    # across every entry of a parameter list.
    ind_p: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    ind_n: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    ind_c: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def __post_init__(self):
        self.elec_size = np.atleast_2d(np.asarray(self.elec_size, dtype=float))
        if not isinstance(self.elec_ori, str) and self.elec_ori is not None:
            self.elec_ori = np.atleast_2d(np.asarray(self.elec_ori, dtype=float))

    def row(self, index: int) -> "ElecPara":
        """Pick the parameters of the ``index``-th electrode out of a uniform set."""
        out = replace(self)
        if self.elec_size.shape[0] > 1:
            out.elec_size = self.elec_size[index:index + 1, :]
        if isinstance(self.elec_ori, np.ndarray) and self.elec_ori.shape[0] > 1:
            out.elec_ori = self.elec_ori[index:index + 1, :]
        return out


def _sorted_pool_indices(requested, pool, lower: bool = False):
    """Positions of ``requested`` inside ``pool``, sorted by their pool order.

    Returns ``(ind_in_pool, ind_in_user_input)``, both 0-based.
    """
    lookup = {(name.lower() if lower else name): i for i, name in enumerate(pool)}
    pairs = []
    for user_index, name in enumerate(requested):
        key = name.lower() if lower else name
        if key in lookup:
            pairs.append((lookup[key], user_index))
    pairs.sort(key=lambda pair: pair[0])
    if not pairs:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    ind_pool, ind_user = zip(*pairs)
    return np.asarray(ind_pool, dtype=int), np.asarray(ind_user, dtype=int)


def elec_preproc(subj, elec: Sequence[str], paras: Sequence[ElecPara]):
    """Classify the requested electrodes and record their pool indices.

    Returns ``(paras, ind_to_user_input)``.
    """
    paras = [replace(p) for p in paras]
    cap_type = paras[0].cap_type
    pool_p = read_cap_info(cap_type).names if cap_type.lower() != "none" else []
    pool_c = None

    unknown = 0
    for name in elec:
        if name in pool_p:
            continue
        if name.lower() in NECK_ELECTRODES:
            continue
        if "custom" in name.lower():
            if pool_c is None:
                pool_c = read_custom_locations(subj)[0]
            if name not in pool_c:
                logger.warning("Unrecognized electrode %s", name)
                unknown += 1
            continue
        logger.warning("Unrecognized electrode %s", name)
        unknown += 1
    if unknown:
        raise ValueError(_UNKNOWN_ELEC)

    ind_p, ind_to_user_p = _sorted_pool_indices(elec, pool_p)
    ind_n, ind_to_user_n = _sorted_pool_indices(elec, NECK_ELECTRODES, lower=True)
    ind_c, ind_to_user_c = (_sorted_pool_indices(elec, pool_c)
                            if pool_c is not None
                            else (np.zeros(0, dtype=int), np.zeros(0, dtype=int)))

    for para in paras:
        para.ind_p, para.ind_n, para.ind_c = ind_p, ind_n, ind_c
        if ind_p.size == 0:
            para.cap_type = "none"

    ind_to_user = np.concatenate([ind_to_user_p, ind_to_user_n, ind_to_user_c])
    return paras, ind_to_user.astype(int)
