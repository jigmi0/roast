"""Electrode handling: naming, cap fitting, modelling and rasterising."""

from .preproc import ElecPara, elec_preproc
from .cap import fit_cap_to_individual
from .neck import place_neck_elec
from .scalp import clean_scalp
from .model import place_and_model_electrodes, expand_elec_para
from .mask import generate_elec_mask
from .placement import electrode_placement

__all__ = [
    "ElecPara", "elec_preproc", "fit_cap_to_individual", "place_neck_elec",
    "clean_scalp", "place_and_model_electrodes", "expand_elec_para",
    "generate_elec_mask", "electrode_placement",
]
