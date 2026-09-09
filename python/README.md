# ROAST for Python

A full Python implementation of the ROAST pipeline: build a subject-specific
finite-element head model from an MRI, simulate transcranial electric
stimulation, and optimise electrode montages for a brain target.

## Install

```bash
pip install -e python                 # from the repository root
pip install -e "python[targeting]"    # also install cvxpy, needed by roast_target
```

Python 3.9 or newer.  The package finds the bundled `data/`, `lib/` and
`example/` directories relative to the repository; set `ROAST_ROOT` if you move
the package elsewhere.

## Use

```python
from roast import roast, roast_target, review_res

roast()                                                   # demo on the MNI152 head
roast('example/subject1.nii', ['F1', 0.3, 'P2', 0.7, 'C5', -0.6, 'O2', -0.4])
roast('example/subject1.nii', ['Fp1', 1, 'P4', -1], elec_type='pad', elec_ori='ap')

roast(None, 'leadField', simulation_tag='myLeadField')    # hours to a day
roast_target(None, 'myLeadField', [-48, -8, 50], opt_type='max-l1')

review_res('example/subject1.nii', 'myTag', tissue='brain')
```

or from the shell:

```bash
roast simulate example/subject1.nii --recipe F1:0.3 P2:0.7 C5:-0.6 O2:-0.4
roast target --sim-tag myLeadField --target -48 -8 50
roast review example/subject1.nii --sim-tag myTag
```

`python examples/run_examples.py --list` lists every example from the manual;
pass a number to run one of them.

Run everything from the repository root - example MRIs and data files are
addressed relative to the working directory.

## What each step needs

| step | Python | external tool |
|---|---|---|
| RAS conversion, resampling, padding | nibabel, SciPy | - |
| T2 to T1 alignment | SciPy (NMI rigid registration) | - |
| segmentation (default) | drives SPM12 | MATLAB or GNU Octave |
| segmentation (`multiaxial=True`) | drives the bundled CNN | conda (set up on first use) |
| MNI registration (Multiaxial path) | - | `lib/NiftyReg/…/reg_aladin` |
| electrode placement, meshing | NumPy, SciPy, scikit-image | `lib/iso2mesh/bin/cgalmesh` |
| solving | - | `lib/getdp-3.2.0/bin/getdp` |
| targeting | cvxpy | - |
| visualisation | matplotlib | - |

SPM12 is a MATLAB toolbox, so it cannot be rewritten in Python; the port drives
the bundled copy through an interpreter.  Point `ROAST_MATLAB_CMD` at your
`matlab` (or `octave`) executable and `ROAST_SPM_PATH` at SPM12 if it lives
outside `lib/`.  **`multiaxial=True` needs no MATLAB at all** - that segmentation
is a Python CNN - so it is the way to run the whole pipeline in Python.

## Option map

The manual (`docs/roast-manual.md`) is written in MATLAB syntax.  The options map
one to one onto keyword arguments:

| MATLAB | Python |
|---|---|
| `roast(subj, recipe, 'capType', '1010')` | `roast(subj, recipe, cap_type='1010')` |
| `'elecType'`, `'elecSize'`, `'elecOri'` | `elec_type=`, `elec_size=`, `elec_ori=` |
| `'T2'`, `'multiaxial', 'on'`, `'manualGui', 'on'` | `T2=`, `multiaxial=True`, `manual_gui=True` |
| `'meshOptions'`, `'conductivities'` (structs) | `mesh_options=`, `conductivities=` (dicts) |
| `'simulationTag'`, `'resampling', 'on'`, `'zeroPadding', 20` | `simulation_tag=`, `resampling=True`, `zero_pad=20` |
| `{'Fp1', 1, 'P4', -1}` (cell) | `['Fp1', 1, 'P4', -1]` (list) |
| `'coordType'`, `'optType'`, `'targetRadius'`, `'targetingTag'` | `coord_type=`, `opt_type=`, `target_radius=`, `targeting_tag=` |

Add `show=False` to any of the three functions to skip the figures in batch runs.

## Conventions

* **Voxel coordinates are 1-based**, as in MATLAB and in the SPM affines that the
  pipeline consumes: landmarks, electrode locations and mesh nodes all follow
  that convention.  *Array indices* returned by the helpers are 0-based, so they
  can be used directly on NumPy arrays.  `roast.utils.matlab.to_index` converts.
* Volumes are indexed in **column-major (Fortran) order** wherever the original
  used `find`/`ind2sub`, because the ordering of the resulting point clouds
  decides ties in the placement code.
* Results are written under the same names as the MATLAB version
  (`<subj>_<tag>_roastResult.mat`, `..._v.nii`, `..._e.nii`, `..._emag.nii`).
  Option records are JSON (`<subj>_<tag>_roastOptions.json`) rather than `.mat`,
  so a run can be inspected and diffed without MATLAB.

## Differences from the MATLAB implementation

Everything the MATLAB code does is implemented, with these deliberate deviations:

* **Resampling and reslicing** use a 5th-degree B-spline (SciPy's maximum);
  SPM offers 7th degree.
* **T2 alignment** is a normalised-mutual-information rigid registration written
  on top of SciPy rather than a call to `spm_coreg`; the estimate is equivalent
  but not bit-identical.
* **`edge()`** is a re-implementation of MATLAB's Sobel edge detector, including
  the automatic threshold and directional thinning.
* **The manual landmark GUI** presents each step as a projection of the head
  rather than a rotated 3-D rendering.  The click carries the same two
  coordinates and the search along the third axis is identical.
* **Figures** are matplotlib.  The slice viewer, the topography and the surface
  renderings behave like their MATLAB counterparts; matplotlib's 3-D engine
  composites transparent surfaces less convincingly than OpenGL.
* **Surface smoothing** for display (`reviewRes(..., fastRender=0)`) is not
  implemented; the mesh is rendered as it is.

## Tests

```bash
python -m pytest python/tests -q
```

The suite covers the MATLAB compatibility layer, the geometry, the file formats,
the option validation and the targeting programs; it needs no external binaries.
