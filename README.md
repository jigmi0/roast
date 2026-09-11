# ROAST: Realistic vOlumetric-Approach-based Simulator for Transcranial electric stimulation

ROAST builds a subject-specific finite-element model of the head from an MRI and
simulates the electric field produced by transcranial electric stimulation (TES).
It also solves the inverse problem: given a brain target, find the electrode
montage that stimulates it best.

This repository contains **two implementations of the same pipeline**:

| | language | entry points | status |
|---|---|---|---|
| [`python/`](python/) | Python 3.9+ | `roast`, `roast_target`, `review_res` | full port, actively developed |
| [`matlab/`](matlab/) | MATLAB | `roast`, `roast_target`, `reviewRes` | original reference implementation |

Both read the same data files, call the same external solvers, and write the
model files (segmentation, electrode masks, mesh, fields, lead field) in the same
layout, so a model built by one can be picked up by the other.  Only the run
bookkeeping (option records, targeting results) is specific to each
implementation.

## Repository layout

```
├── python/          the Python implementation (package, tests, examples)
├── matlab/          the original MATLAB implementation, grouped by pipeline stage
├── data/            capInfo.xlsx, elec72.loc, eTPM.nii, capLayout.pdf
├── example/         example MRIs, including the MNI152 and New York heads
├── lib/             bundled third-party software (SPM12, iso2mesh, getDP, ...)
└── docs/            the full user manual and release notes
```

Run both implementations **from the root of this repository**: the data and
example files are addressed relative to the current directory.

## Quick start (Python)

```bash
pip install -e python            # add [targeting] for the montage optimiser
python -c "from roast import roast; roast()"
```

or from the command line:

```bash
roast simulate                                   # the default demo on the MNI152 head
roast simulate example/subject1.nii --recipe F1:0.3 P2:0.7 C5:-0.6 O2:-0.4
roast simulate --lead-field --tag myLeadField    # prepare for targeting
roast target --sim-tag myLeadField --target -48 -8 50
```

See [`python/README.md`](python/README.md) for installation details, the external
tools each step needs, and how the port differs from the MATLAB original.

## Quick start (MATLAB)

```matlab
addpath('matlab'); setup_roast;
roast
```

`setup_roast` puts `matlab/` and `lib/` on the path.  Keep MATLAB's working
directory at the repository root.

## Documentation

* [`docs/roast-manual.md`](docs/roast-manual.md) - the complete manual: every
  option, 40 worked examples, the outputs and how to review them.  The examples
  are written in MATLAB syntax; [`python/README.md`](python/README.md#option-map)
  maps every option onto its Python keyword.
* [`docs/python-api.md`](docs/python-api.md) - the Python module map and API.
* [`data/capLayout.pdf`](data/capLayout.pdf) - the electrode layouts of the
  10-05, BioSemi-256 and EGI HCGSN-256 systems.
* [`docs/knownIssues.txt`](docs/knownIssues.txt), [`docs/releasingNotes.txt`](docs/releasingNotes.txt)

## Citing ROAST

If you use ROAST in your research, please cite:

> Huang, Y., Datta, A., Bikson, M., Parra, L.C., *Realistic vOlumetric-Approach
> to Simulate Transcranial Electric Stimulation - ROAST - a fully automated
> open-source pipeline*, Journal of Neural Engineering, 16(5), 2019.

The New York head, the targeting algorithms and the Multiaxial segmentation carry
their own references; they are listed at the top of `docs/roast-manual.md`.

## License

General Public License version 3 or later; see [LICENSE.md](LICENSE.md).  ROAST is
an *aggregate*: the license covers the scripts, the documentation and the
individual MRI data under `example/`, but not the third-party programs under
`lib/`, which follow their own licenses.  This software is intended for
non-commercial use.
