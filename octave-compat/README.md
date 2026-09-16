# Running ROAST under GNU Octave

ROAST requires MATLAB. This directory contains the compatibility layer used to
run `roast()` under **GNU Octave 8.4** on Linux, in order to compare `master`
against the Python port on branch `claude/matlab-python-translation-9kb7su`.

Nothing here modifies ROAST itself — every shim lives on the Octave path
outside the ROAST source tree. See `RESULTS.md` for the comparison.

## Prerequisites

```bash
apt-get update && apt-get install -y octave octave-image octave-statistics octave-dev
```

SPM12 ships only MATLAB MEX binaries (`.mexa64`), which Octave cannot load.
Build Octave MEX from the matching SPM12 source (the bundled copy is r7219) and
drop the results into `lib/spm12/`, keeping the bundled `.m` files — the
bundled SPM12 is patched to write the `_rmask.mat` that the segmentation
touch-up needs, so a stock SPM12 will not do:

```bash
git clone --depth 1 --branch r7219 https://github.com/spm/spm12.git /tmp/spm12-src
cd /tmp/spm12-src/src && make PLATFORM=octave all && make PLATFORM=octave install
cd /tmp/spm12-src && find . -name '*.mex' -exec cp --parents {} /path/to/roast/lib/spm12/ \;
```

Octave's `save()` writes its own text format by default, which neither MATLAB
nor scipy can read. Force MATLAB v7 output in `~/.octaverc`:

```matlab
save_default_options("-mat7-binary");
```

## Running

From the ROAST root:

```bash
# the default example
ROAST_PYTHON=/path/to/python \
QT_QPA_PLATFORM=offscreen \
octave-cli --no-gui octave-compat/scripts/run_master.m

# any single example from runExamples.m, by its command text
ROAST_CMD="roast([],[],'meshoptions',struct('radbound',4,'maxvol',8))" \
ROAST_PYTHON=/path/to/python \
octave-cli --no-gui octave-compat/scripts/run_one_master.m

# the whole suite on BOTH implementations, comparing and cleaning up per
# example (each one generates ~1.3 GB across the two trees)
python3 octave-compat/scripts/extract_examples.py > master_cmds.json
python3 octave-compat/scripts/run_example_suite.py 5 7 9 11 13
```

`ROAST_PYTHON` must point at an interpreter with numpy/scipy — the
`TriScatteredInterp` shim uses it (see below).

## What each shim does

| Shim | Why |
|---|---|
| `readtable.m`, `table2cell.m` | Not implemented in Octave. Reads CSV caches of `capInfo.xlsx` (regenerate with `scripts/export_capinfo.py`). Uses `str2double`, not `textscan('%f')`, which loses a ULP. |
| `computer.m` | ROAST switches on MATLAB's `computer('arch')` (`glnxa64`). The no-argument form deliberately keeps Octave's native host string, because iso2mesh's `getexeext()` tests it with `regexp(computer,'86_64')`. |
| `edge.m` | Octave's image-package `edge()` rejects logical input, and its `simple_thinning()` lacks MATLAB's gradient-direction gate. Implements MATLAB's default Sobel detector. |
| `TriRep.m`, `freeBoundary.m` | Not in Octave. Returns the facets referenced by exactly one simplex, re-indexed into the ascending subset of points they use — MATLAB's contract. |
| `TriScatteredInterp.m`, `tsi_helper.py` | Not in Octave, and `griddatan` cannot cope with a 388k-node mesh. Delegates to scipy's Qhull Delaunay + barycentric linear interpolation, which is the algorithm MATLAB uses. Caches the triangulation across the four calls. |
| `save.m` | Octave cannot write MATLAB's `-v7.3` (HDF5) container. Drops the flag; `.octaverc` makes v7 the default, which fits these results. |
| `viewMRI.m`, `viewSeg.m`, `viewElectrodes.m`, `visualizeRes.m` | Headless no-ops. Octave has no `uifigure`. Equivalent to the Python port's `show=False`; all four discard their outputs in `roast.m`, so no numbers change. |

Two further collisions are handled by the driver rather than a shim, so that
**the ROAST checkout is never modified**. Octave resolves script-local
functions ahead of both the current directory and the load path, so
`scripts/run_one_master.m` opens with `1;` and then defines:

- `vec(x, dim)` — `lib/cvx/{functions,sedumi}/vec.m` shadow Octave's core
  2-argument `vec`, which the image package's `fspecial` needs. Defining both
  arities locally restores it without moving CVX aside.
- `viewMRI` / `viewSeg` / `viewElectrodes` / `visualizeRes` as no-ops, which
  is why the copies under `shims/` are only needed if you call `roast`
  interactively rather than through the driver.

One ordering rule still matters: `pkg load image` / `pkg load statistics` must
come **before** the shim directory is added, so the shims take precedence over
the packages.

## Fidelity caveats

`edge` and `TriScatteredInterp` implement the same reference algorithms the
Python port implements, so agreement at those two steps is not independent
evidence. Everything else — segmentation, touch-up, electrode placement,
meshing, and the getDP solve — is produced independently by each side.
