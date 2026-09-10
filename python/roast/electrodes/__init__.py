"""From electrode names to electrode and gel masks.

* :mod:`~roast.electrodes.preproc` - classify the requested electrodes into the
  standard cap, the neck electrodes and user-defined locations.
* :mod:`~roast.electrodes.cap` - fit a standard EEG cap onto the individual head
  and optimise its position against the 10-10 spacing.
* :mod:`~roast.electrodes.neck` - project the four optional neck electrodes.
* :mod:`~roast.electrodes.scalp` - close, fill and open the scalp so electrodes
  sit on a smooth surface.
* :mod:`~roast.electrodes.model` - build the electrode and gel point clouds
  along the local surface normal.
* :mod:`~roast.electrodes.mask` - rasterise those point clouds, one label per
  electrode, rejecting overlaps.
* :mod:`~roast.electrodes.placement` - the step that ties all of the above
  together and writes the masks.
"""

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
