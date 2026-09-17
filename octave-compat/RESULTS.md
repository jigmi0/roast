# master vs the Python port — same example, same recipe

Both implementations were run on ROAST's default example:
`example/MNI152_T1_1mm.nii`, anode **Fp1 +1 mA**, cathode **P4 −1 mA**,
1010 cap, disc electrodes 6×2 mm, default conductivities and mesh options.
Both run logs record identical options.

- `master` (MATLAB source) run under GNU Octave 8.4 — see `README.md`.
- `claude/matlab-python-translation-9kb7su` run with `roast(show=False)`;
  its own test suite passes 89/89.

Both sides drive the *same* bundled SPM12 for segmentation, so that stage is a
shared dependency rather than a reimplementation.

## Pipeline agreement

| Stage | master | Python port | Agreement |
|---|---|---|---|
| RAS-converted input | 182×218×182 | same | bit-identical (max diff 0.0, same affine) |
| SPM12 tissue maps c1–c6 | — | — | bit-identical |
| masks: CSF / bone / skin | 399,154 / 603,576 / 1,271,963 | identical | Dice 1.000000 |
| masks: grey / white | 554,163 / 935,671 | 554,147 / 935,687 | Dice 0.999973 / 0.999984 |
| masks: air | 43,999 | 41,794 | Dice 0.955229 (−5.01%) |
| whole mask volume | — | — | 99.946% of voxels identical |
| electrode / gel masks | 872 / 756 vox | 877 / 759 vox | Dice 0.9754 / 0.9756 |
| mesh | 389,451 nodes / 2,260,476 tets | 388,160 nodes / 2,254,640 tets | 0.33% |
| electrode contact area | 261.12, 259.11 mm² | 259.24, 252.11 mm² | 0.7% / 2.7% |

## Fields in grey + white matter (1,489,834 voxels)

| Quantity | master | Python port | Difference |
|---|---|---|---|
| mean \|E\| | 0.105690 V/m | 0.105598 V/m | **0.086%** |
| median \|E\| | 0.101257 V/m | 0.101157 V/m | 0.099% |
| p99 \|E\| | 0.214110 V/m | 0.214549 V/m | 0.205% |
| correlation of \|E\| | — | — | **r = 0.996548** |
| correlation of voltage | — | — | **r = 0.999998** |

Median per-voxel difference in brain |E| is 0.97% (p95 5.66%).

Over the whole head, |E| correlates at r = 0.9819 with a heavier tail
(median 1.66%, p95 21.9%). Those differences sit at tissue boundaries and
electrode edges, where the field is near-singular and the 0.33% mesh
difference dominates.

## Reading

The port reproduces master. The two are identical through SPM segmentation,
diverge by ~0.05% of voxels in segmentation touch-up (almost entirely the air
mask), which perturbs the mesh by 0.33%, which moves mean brain |E| by 0.09% —
far below the model's own uncertainty from conductivity assumptions.

See the fidelity caveats in `README.md`: master's `edge` and
`TriScatteredInterp` shims implement the same reference algorithms as the port,
so those two steps are not independent.

Reproduce the numbers with `scripts/compare_runs.py` and `scripts/compare2.py`
(edit the two directory paths at the top).

---

# Example suite

`runExamples.m` documents 41 examples. The port mirrors them 1:1. Of those, 17
are runnable in this environment: the rest need conda (the 10 `multiaxial`
examples), a `bikson.hdr`/`.img` that is absent from the repository (2), or a
lead field taking hours to a day, which the 4 `roast_target` and 4 `reviewRes`
examples then depend on.

## Agreement, |E| in grey + white matter

| ex | what it varies | brain r | mean master | mean port | diff | median | p95 |
|---|---|---|---|---|---|---|---|
| 1 | default Fp1/P4 | 0.9965 | 0.10569 | 0.10560 | 0.086% | 0.97% | 5.7% |
| 5 | zero-padding 20 | 0.9919 | 0.10240 | 0.10226 | 0.132% | 1.97% | 8.0% |
| 14 | mixed disc/pad/ring | 0.9943 | 0.16975 | 0.16963 | 0.074% | 1.14% | 6.1% |
| 15 | mixed types, custom sizes | 0.9925 | 0.17024 | 0.16996 | 0.163% | 1.45% | 6.7% |
| 16 | pads oriented a-p | 0.9955 | 0.10347 | 0.10310 | 0.360% | 1.16% | 6.0% |
| 17 | pads, custom orientation | 0.9958 | 0.10342 | 0.10310 | 0.308% | 1.01% | 5.8% |
| 19 | orientations among mixed types | 0.9935 | 0.16724 | 0.16728 | 0.023% | 1.26% | 6.4% |
| 23 | customised conductivities | 0.9958 | 0.23598 | 0.23588 | 0.044% | 0.87% | 4.7% |
| 24 | per-electrode conductivities | 0.9933 | 0.16877 | 0.16861 | 0.097% | 1.26% | 6.4% |

Nine configurations: brain correlation never below **0.9919**, mean |E| never
more than **0.36%** apart, median **0.115%**. Example 21 (custom mesh options)
also agreed, at whole-volume r = 0.9810 / 0.19%, but ran before the
brain-restricted metric existed and is not directly comparable.

Whole-volume correlations are much looser (0.927 to 0.987) and should not be
quoted: two thirds of the volume is outside the head, where |E| is near zero
and tiny absolute differences become large relative ones. Example 5 makes the
point - brain r 0.9919 against whole-volume 0.9275, because zero-padding adds
20 voxels of empty space on every face.

## One genuine defect in the port

**Example 20** - `elecOri` given per electrode, mixing a keyword, an empty and
a direction vector, which the manual documents as "keywords and vectors mixed
as orientations":

```matlab
roast([],{'Fp1',1,'FC4',1,'POz',-2},'electype',{'pad','disc','pad'}, ...
      'elecori',{'ap',[],[0 0.71 0.71]})
```

Master completes this in 564 s. The port raises `ValueError` before the
pipeline starts, in `roast/pipeline/options.py:176`:

```python
if elec_ori is None or isinstance(elec_ori, str) or np.ndim(elec_ori) == 2:
```

`np.ndim` builds an array from its argument, so a ragged per-electrode list
like `['ap', None, [0, 0.71, 0.71]]` raises instead of returning 1. Reproduced
without the pipeline:

```python
np.ndim(["ap", None, [0, 0.71, 0.71]])   # ValueError: inhomogeneous shape
```

The all-vector form (example 19) is unaffected, which is why that example
passes. The guard needs to recognise a per-electrode sequence before asking
numpy for its dimensionality.

## Failures that are environmental, not defects

| ex | master | port |
|---|---|---|
| 7, 9, 11, 18, 22 | getDP did not converge | getDP did not converge / MemoryError |
| 13, 25 | `quantile: called with too many inputs` | getDP did not converge / MemoryError |

Examples 7, 9, 11, 18, 22 and 25 all use `example/subject1.nii`, which is
192x256x256 - 1.75x the voxels of the MNI152 head. getDP factorises with MUMPS
and already peaked at 7.2 GB on the smaller head, so on this 15 GB machine the
larger models exhaust memory and the solution comes back empty, which both
implementations then report as non-convergence. **Both sides fail together, so
these say nothing about the port**; they need a larger machine.

The master-side failures on 13 and 25 are a third Octave shadowing collision,
alongside `vec` and `edge`: `lib/eeglabFunc/sigprocfunc/quantile.m` shadows
Octave's core 3-argument `quantile`. It would be fixed the same way as `vec`,
with a script-local definition in the driver.
