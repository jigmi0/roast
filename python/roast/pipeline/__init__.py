"""The three user-facing pipelines and their bookkeeping.

* :mod:`~roast.pipeline.simulate` - ``roast()``: segment, place, mesh, solve, show.
* :mod:`~roast.pipeline.target` - ``roast_target()``: optimise a montage against
  a lead field.
* :mod:`~roast.pipeline.review` - ``review_res()``: re-open the figures of a
  finished run.
* :mod:`~roast.pipeline.options` - validation and normalisation of the options,
  and the comparison that decides whether a run is new.
* :mod:`~roast.pipeline.log` - unique tags, the option records and the readable
  run logs.
"""

from .options import (RoastOptions, TargetOptions, parse_recipe, build_elec_para,
                      validate_mesh_options, validate_conductivities, is_new_options)
from .log import (write_roast_log, write_target_log, write_target_results,
                  save_options, load_options, find_previous_run, options_path)
from .simulate import roast, model_paths, ModelPaths
from .target import roast_target
from .review import review_res, TISSUE_VIEWS

__all__ = [
    "RoastOptions", "TargetOptions", "parse_recipe", "build_elec_para",
    "validate_mesh_options", "validate_conductivities", "is_new_options",
    "write_roast_log", "write_target_log", "write_target_results", "save_options",
    "load_options", "find_previous_run", "options_path",
    "roast", "model_paths", "ModelPaths", "roast_target", "review_res", "TISSUE_VIEWS",
]
