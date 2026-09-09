"""ROAST - Realistic vOlumetric-Approach-based Simulator for Transcranial
electric stimulation.

A Python implementation of the ROAST pipeline.  Three functions do the work:

>>> from roast import roast, roast_target, review_res
>>> roast()                                          # doctest: +SKIP
>>> roast('example/subject1.nii', ['F1', 0.3, 'P2', 0.7, 'C5', -0.6, 'O2', -0.4])
...                                                  # doctest: +SKIP
>>> roast_target('example/subject1.nii', 'leadFieldTag', [-48, -8, 50])
...                                                  # doctest: +SKIP

If you use ROAST in your research, please cite:

  Huang, Y., Datta, A., Bikson, M., Parra, L.C., Realistic vOlumetric-Approach
  to Simulate Transcranial Electric Stimulation - ROAST - a fully automated
  open-source pipeline, Journal of Neural Engineering, 16(5), 2019.

The targeting feature, the New York head and the Multiaxial segmentation carry
their own references; see the documentation.  Released under GPL v3 or later.
"""

from .config import roast_root, data_dir, lib_dir, example_dir
from .pipeline import roast, roast_target, review_res

__version__ = "4.0.0"

__all__ = ["roast", "roast_target", "review_res", "roast_root", "data_dir", "lib_dir",
           "example_dir", "__version__"]
