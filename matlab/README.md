# ROAST - MATLAB implementation

The original MATLAB implementation, grouped by pipeline stage:

| directory | what it does |
|---|---|
| `pipeline/` | `roast`, `roast_target`, `reviewRes`, the option/log bookkeeping and `runExamples` |
| `preprocess/` | RAS re-orientation, resampling, zero padding, T2 alignment, header fixes |
| `segment/` | SPM12 and Multiaxial segmentation, NiftyReg registration, mask clean-up |
| `electrodes/` | electrode naming, cap fitting, electrode/gel modelling, landmark GUIs |
| `geometry/` | point-cloud primitives shared by the electrode code |
| `mesh/` | tetrahedral meshing via iso2mesh |
| `solver/` | boundary conditions, the getDP solve and its post-processing |
| `targeting/` | the optimisation algorithms and their CVX programs |
| `viz/` | slice viewer and 3-D renderings |

## Running it

Start MATLAB **in the root of this repository** (the directory holding
`example/`, `data/` and `lib/`), then:

```matlab
addpath('matlab');
setup_roast;      % puts matlab/ and lib/ on the path
roast
```

The data files live in `data/` (`capInfo.xlsx`, `elec72.loc`, `eTPM.nii`) and are
resolved relative to the working directory, so keep it at the repository root.

See [`../docs/roast-manual.md`](../docs/roast-manual.md) for the full manual, and
[`../python/`](../python/) for the Python port of this code.
