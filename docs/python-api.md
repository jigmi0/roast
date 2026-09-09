# ROAST Python API

The package mirrors the pipeline: one sub-package per stage, each holding the
translation of the corresponding MATLAB functions.

## Module map

| module | MATLAB counterpart |
|---|---|
| `roast.config` | the constants scattered across `roast.m` (tissue labels, TPM landmarks, defaults) |
| `roast.utils.matlab` | `fspecial`, `imfilter`, `imfill`, `bwareaopen`, `edge`, `prctile`, `sortrows`, `ind2sub`, ... |
| `roast.utils.logging` | the `disp` banners |
| `roast.io.nifti` | `load_untouch_nii`, `save_untouch_nii`, `spm_vol` |
| `roast.io.matfile` | reading `_seg8.mat` / `_niftyReg.mat`, writing results |
| `roast.io.meshfile` | `saveinr`, `readmedit`, `savemsh`, reading getDP `.pos` files |
| `roast.io.caps` | `capInfo.xlsx`, `elec72.loc`, `*_customLocations` readers |
| `roast.preprocess.orientation` | `convertToRAS`, `convertToRASpointCloud` |
| `roast.preprocess.resample` | `resampToOneMM`, `spm_reslice`, `spm_get_bbox` |
| `roast.preprocess.padding` | `zeroPadding` |
| `roast.preprocess.realign` | `realignT2`, `spm_coreg` |
| `roast.preprocess.header` | `alignHeader2mni`, `renameSPMres` |
| `roast.segment.spm` | `start_seg` |
| `roast.segment.multiaxial` | `runMultiaxial` |
| `roast.segment.niftyreg` | `runNiftyReg` |
| `roast.segment.touchup` | `segTouchup` |
| `roast.segment.masks` | `binaryMaskGenerate`, `sizeOfObject`, `brainCrop` |
| `roast.geometry.shapes` | `drawLine`, `drawCuboid`, `drawCylinder`, `cylinder2P` |
| `roast.geometry.pointcloud` | `mask2EdgePointCloud`, `map2Points`, `project2ClosestSurfacePoints`, `getDataAroundTar` |
| `roast.geometry.spline` | `ncs2dapprox` |
| `roast.electrodes.preproc` | `elecPreproc` |
| `roast.electrodes.cap` | `fitCap2individual` |
| `roast.electrodes.neck` | `placeNeckElec` |
| `roast.electrodes.scalp` | `cleanScalp` |
| `roast.electrodes.model` | `placeAndModelElectrodes` |
| `roast.electrodes.mask` | `generateElecMask` |
| `roast.electrodes.placement` | `electrodePlacement` |
| `roast.mesh.iso2mesh` | `cgalv2m`, `sortmesh` |
| `roast.mesh.build` | `meshByIso2mesh` |
| `roast.solver.prepare` | `prepareForGetDP`, `freeBoundary` |
| `roast.solver.getdp` | `solveByGetDP` |
| `roast.solver.post` | `postGetDP`, `TriScatteredInterp` |
| `roast.targeting.convex` | `ls_l1`, `ls_l1per`, `max_l1`, `max_l1per`, `lcmv_l1`, `lcmv_l1per` |
| `roast.targeting.currents` | `optimize_currents` |
| `roast.targeting.optimize` | `optimize`, `optimize_prepare`, `optimize_anon` |
| `roast.viz.slices` | `sliceshow` |
| `roast.viz.surfaces` | `viewMRI`, `viewSeg`, `viewElectrodes` |
| `roast.viz.results` | `visualizeRes` |
| `roast.viz.topoplot` | `mytopoplot` |
| `roast.viz.landmarks` | `checkLandmarks`, `getLandmarksManual` |
| `roast.pipeline.options` | the argument checking at the top of `roast.m`/`roast_target.m`, and `isNewOptions` |
| `roast.pipeline.log` | `writeRoastLog` |
| `roast.pipeline.simulate` | `roast.m` |
| `roast.pipeline.target` | `roast_target.m` |
| `roast.pipeline.review` | `reviewRes.m` |
| `roast.cli` | the command line front-end (new) |

## The three entry points

### `roast(subj=None, recipe=None, **options)`

Runs the six pipeline steps and returns
`{'tag', 'options', 'node', 'elem', 'face', 'results'}`, where `results` is
`(vol_all, ef_mag, ef_all)` for a simulation and the lead field for a
`recipe='leadField'` run.  Options: `cap_type`, `elec_type`, `elec_size`,
`elec_ori`, `T2`, `multiaxial`, `manual_gui`, `mesh_options`, `conductivities`,
`simulation_tag`, `resampling`, `zero_pad`, `show`.

### `roast_target(subj, sim_tag, target_coord, **options)`

Optimises a montage against a lead field.  Returns `{'tag', 'options', 'results'}`
with the montage, the resulting field and the intensity/focality at every target.
Options: `coord_type`, `opt_type`, `orient`, `desired_intensity`, `elec_num`,
`target_radius`, `k`, `targeting_tag`, `show`.

### `review_res(subj, sim_tag, tissue='brain', tar_tag=None)`

Re-opens the figures of a finished run and returns the data behind them.

## Working with the pieces

The stages are usable on their own:

```python
from roast.io import NiftiVolume
from roast.segment import brain_crop
from roast.viz import sliceshow, roast_colormap

mask = NiftiVolume.load('example/subject1_T1orT2_SPM_masks.nii')
sliceshow(mask.img, color=roast_colormap(), bbox=brain_crop(mask.img), show=True)
```

```python
from roast.mesh import cgalv2m
node, elem, face = cgalv2m(labelled_volume,
                           {'radbound': 5, 'angbound': 30, 'distbound': 0.3,
                            'reratio': 3}, maxvol=10)
```

```python
from roast.targeting import TargetingProblem, optimize_prepare, optimize
problem = TargetingProblem(target_coord=coords, num_of_targets=1,
                           opt_type='max-l1', target_radius=2.0, n_locs=len(locs))
problem.u = orientations
currents = optimize(optimize_prepare(problem, A, locs), A)
```
