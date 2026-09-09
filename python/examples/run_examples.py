#!/usr/bin/env python3
"""Every example from the ROAST documentation, as runnable Python.

Run the whole suite (long!), a single example, or just list them::

    python examples/run_examples.py --list
    python examples/run_examples.py 7
    python examples/run_examples.py --from 26 --to 30

Failures are written to ``errLog.txt`` and do not stop the run, mirroring the
``runExamples.m`` script.  Run this from the root of the repository, since the
example MRIs are addressed relative to it.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roast import roast, roast_target, review_res      # noqa: E402

SUBJECT = "example/subject1.nii"
T2 = "example/subject1_T2.nii"

#: (description, callable) for every documented example, in order.
EXAMPLES = [
    ("default recipe on the MNI152 head",
     lambda: roast()),
    ("default recipe, segmented by Multiaxial",
     lambda: roast(multiaxial=True)),
    ("Analyze .hdr input",
     lambda: roast("example/bikson.hdr")),
    ("Analyze .img input",
     lambda: roast("example/bikson.img")),
    ("10-05 electrode with zero padding",
     lambda: roast(None, ["Exx19", 1, "C4", -1], zero_pad=20)),
    ("10-05 electrode with zero padding, Multiaxial",
     lambda: roast(None, ["Exx19", 1, "C4", -1], zero_pad=20, multiaxial=True)),
    ("four electrodes on a subject",
     lambda: roast(SUBJECT, ["F1", 0.3, "P2", 0.7, "C5", -0.6, "O2", -0.4])),
    ("four electrodes on a subject, Multiaxial",
     lambda: roast(SUBJECT, ["F1", 0.3, "P2", 0.7, "C5", -0.6, "O2", -0.4],
                   multiaxial=True)),
    ("T1 and T2 segmentation",
     lambda: roast(SUBJECT, None, T2=T2)),
    ("T1 and T2 segmentation, Multiaxial",
     lambda: roast(SUBJECT, None, T2=T2, multiaxial=True)),
    ("resampling and zero padding",
     lambda: roast(SUBJECT, ["F1", 0.3, "P2", 0.7, "C5", -0.6, "O2", -0.4],
                   resampling=True, zero_pad=20)),
    ("resampling and zero padding, Multiaxial",
     lambda: roast(SUBJECT, ["F1", 0.3, "P2", 0.7, "C5", -0.6, "O2", -0.4],
                   resampling=True, zero_pad=20, multiaxial=True)),
    ("BioSemi cap, neck and customized electrodes",
     lambda: roast(SUBJECT, ["G12", 0.25, "J7", -0.25, "Nk1", 0.5, "Nk3", -0.5,
                             "custom1", 0.25, "custom3", -0.25], cap_type="biosemi")),
    ("mixed electrode types",
     lambda: roast(None, ["Fp1", 1, "FC4", 1, "POz", -2],
                   elec_type=["disc", "pad", "ring"])),
    ("mixed electrode types with customized sizes",
     lambda: roast(None, ["Fp1", 1, "FC4", 1, "POz", -2],
                   elec_type=["disc", "pad", "ring"],
                   elec_size=[[8, 2], [45, 25, 4], [5, 8, 2]])),
    ("pad electrodes oriented anterior-posterior",
     lambda: roast(None, None, elec_type="pad", elec_ori="ap")),
    ("pad electrodes with a customized orientation",
     lambda: roast(None, None, elec_type="pad", elec_ori=[0.71, 0.71, 0])),
    ("a different orientation for every pad",
     lambda: roast(SUBJECT, ["Fp1", 1, "FC4", 1, "POz", -2], elec_type="pad",
                   elec_ori=[[0.71, 0.71, 0], [-0.71, 0.71, 0], [0, 0.71, 0.71]])),
    ("orientations for the pads among mixed types",
     lambda: roast(None, ["Fp1", 1, "FC4", 1, "POz", -2],
                   elec_type=["pad", "disc", "pad"],
                   elec_ori=[[0.71, 0.71, 0], [0, 0.71, 0.71]])),
    ("keywords and vectors mixed as orientations",
     lambda: roast(None, ["Fp1", 1, "FC4", 1, "POz", -2],
                   elec_type=["pad", "disc", "pad"],
                   elec_ori=["ap", None, [0, 0.71, 0.71]])),
    ("customized mesh options",
     lambda: roast(None, None, mesh_options={"radbound": 4, "maxvol": 8})),
    ("pads on a resampled, padded T1+T2 model",
     lambda: roast(SUBJECT, None, resampling=True, zero_pad=10, T2=T2,
                   elec_type="pad")),
    ("customized conductivities",
     lambda: roast(None, ["Fp1", 1, "FC4", 1, "POz", -2],
                   conductivities={"csf": 0.6, "electrode": 0.1})),
    ("per-electrode conductivities",
     lambda: roast(None, ["Fp1", 1, "FC4", 1, "POz", -2],
                   elec_type=["pad", "disc", "pad"],
                   conductivities={"gel": [1, 0.3, 1],
                                   "electrode": [0.1, 5.9e7, 0.1]})),
    ("everything at once, under a simulation tag",
     lambda: roast(SUBJECT, ["Fp1", 0.3, "F8", 0.2, "POz", -0.4, "Nk1", 0.5,
                             "custom1", -0.6],
                   elec_type=["disc", "pad", "pad", "ring", "disc"],
                   elec_size=[[6, 2], [50, 30, 3], [50, 30, 3], [4, 6, 2], [6, 2]],
                   elec_ori="ap", T2=T2, resampling=True, zero_pad=10,
                   conductivities={"csf": 0.6, "gel": 0.1},
                   simulation_tag="awesomeSimulation")),
    ("the New York head",
     lambda: roast("nyhead")),
    ("the New York head, Multiaxial",
     lambda: roast("nyhead", None, multiaxial=True)),
    ("the New York head, resampled and padded",
     lambda: roast("nyhead", None, resampling=True, zero_pad=25)),
    ("the New York head, resampled and padded, Multiaxial",
     lambda: roast("nyhead", None, resampling=True, zero_pad=25, multiaxial=True)),
    ("ring electrodes on the New York head",
     lambda: roast("nyhead", None, zero_pad=5, elec_type="ring", elec_size=[7, 10, 3],
                   multiaxial=True)),
    ("a different ring size per electrode",
     lambda: roast("nyhead", ["Fp1", 1, "FC4", 1, "POz", -2], elec_type="ring",
                   elec_size=[[7, 10, 3], [6, 8, 3], [4, 6, 2]])),
    ("lead field generation (takes a long time)",
     lambda: roast(None, "leadField", multiaxial=True, zero_pad=10, elec_type="ring",
                   simulation_tag="LFwithMA")),
    ("lead field on the New York head (takes a long time)",
     lambda: roast("nyhead", "leadField", multiaxial=True, zero_pad=10,
                   elec_type="ring", simulation_tag="nyLFwithMA")),
    ("targeting with the defaults",
     lambda: roast_target(None, "LFwithMA", None, targeting_tag="basic")),
    ("three targets, maximum focality",
     lambda: roast_target(None, "LFwithMA",
                          [[52, 184, 72], [25, 80, 72], [139, 171, 72]],
                          coord_type="voxel", opt_type="wls-l1", k=0.002,
                          target_radius=4, targeting_tag="3targets_radialIn")),
    ("three targets, mixed orientations",
     lambda: roast_target(None, "LFwithMA",
                          [[52, 184, 72], [25, 80, 72], [139, 171, 72]],
                          coord_type="voxel", opt_type="wls-l1", k=0.002,
                          orient=["radial-in", [-1, 1, 1], "posterior"],
                          targeting_tag="mixed_orient")),
    ("two targets with the optimal orientation",
     lambda: roast_target("nyhead", "nyLFwithMA", [[-48, -8, 50], [48, -8, 50]],
                          opt_type="lcmv-l1", orient="optimal",
                          targeting_tag="nyOptOri")),
    ("review a simulation",
     lambda: review_res(SUBJECT, "awesomeSimulation", "all")),
    ("review a targeting run, bone surface",
     lambda: review_res(None, "LFwithMA", "bone", tar_tag="3targets_radialIn")),
    ("review a targeting run, white matter surface",
     lambda: review_res(None, "LFwithMA", "white", tar_tag="mixed_orient")),
    ("review a targeting run on the New York head",
     lambda: review_res("nyhead", "nyLFwithMA", tar_tag="nyOptOri")),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", nargs="*", type=int, help="example numbers to run")
    parser.add_argument("--list", action="store_true", help="list the examples and exit")
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--to", dest="stop", type=int, default=len(EXAMPLES))
    args = parser.parse_args(argv)

    if args.list:
        for number, (description, _) in enumerate(EXAMPLES, start=1):
            print(f"{number:3d}. {description}")
        return 0

    numbers = args.example or list(range(args.start, args.stop + 1))
    for number in numbers:
        description, run = EXAMPLES[number - 1]
        print(f"\n=== Example {number}: {description} ===")
        try:
            run()
        except Exception:                       # keep going, like runExamples.m
            with open("errLog.txt", "a") as handle:
                handle.write(f"error at running example {number}:\n")
                handle.write(traceback.format_exc() + "\n")
            print(f"Example {number} FAILED; see errLog.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
