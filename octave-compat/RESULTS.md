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
