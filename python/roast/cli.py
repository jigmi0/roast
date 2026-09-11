"""Command-line interface.

    roast simulate example/subject1.nii --recipe F1:0.3 P2:0.7 C5:-0.6 O2:-0.4
    roast simulate example/subject1.nii --lead-field --tag subj1LF
    roast target   example/subject1.nii --sim-tag subj1LF --target -48 -8 50
    roast review   example/subject1.nii --sim-tag 20190101T120000
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .config import CAP_TYPES, ELEC_TYPES, OPT_TYPES


def _parse_recipe(pairs, parser: argparse.ArgumentParser):
    """``F1:0.3 P2:-0.3`` -> ``['F1', 0.3, 'P2', -0.3]``."""
    recipe = []
    for pair in pairs:
        name, _, current = pair.rpartition(":")
        try:
            recipe += [name, float(current)]
        except ValueError:
            current = None
        if not name or current is None:
            parser.error(f"'{pair}' is not an electrode:current pair, e.g. Fp1:1")
    return recipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roast", description="Realistic vOlumetric-Approach-based Simulator for "
                                  "Transcranial electric stimulation")
    parser.add_argument("--version", action="store_true", help="print the version")
    sub = parser.add_subparsers(dest="command")

    simulate = sub.add_parser("simulate", help="run a TES simulation")
    simulate.add_argument("subject", nargs="?", default=None,
                          help="path of the subject MRI, or 'nyhead'")
    simulate.add_argument("--recipe", nargs="+", metavar="NAME:CURRENT",
                          help="electrode/current pairs in mA, summing to zero")
    simulate.add_argument("--lead-field", action="store_true",
                          help="generate a lead field for targeting instead")
    simulate.add_argument("--cap-type", default="1010", choices=CAP_TYPES)
    simulate.add_argument("--elec-type", default="disc", choices=ELEC_TYPES)
    simulate.add_argument("--elec-size", nargs="+", type=float, default=None,
                          help="electrode size in mm")
    simulate.add_argument("--elec-ori", default=None, help="pad orientation (lr/ap/si)")
    simulate.add_argument("--t2", default=None, help="optional T2 image")
    simulate.add_argument("--multiaxial", action="store_true",
                          help="segment with the deep-learning model instead of SPM12")
    simulate.add_argument("--manual-gui", action="store_true",
                          help="inspect and correct the landmarks by hand")
    simulate.add_argument("--resample", action="store_true",
                          help="resample the MRI to 1 mm isotropic first")
    simulate.add_argument("--zero-pad", type=int, default=0,
                          help="add this many empty slices around the MRI")
    simulate.add_argument("--tag", default=None, help="name of this run")
    simulate.add_argument("--no-show", action="store_true", help="do not open figures")

    target = sub.add_parser("target", help="optimise a montage for a brain target")
    target.add_argument("subject", nargs="?", default=None)
    target.add_argument("--sim-tag", required=True,
                        help="tag of the lead-field run to use")
    target.add_argument("--target", nargs="+", type=float, required=True,
                        metavar="X Y Z", help="target coordinates, three per target")
    target.add_argument("--coord-type", default="mni", choices=("mni", "voxel"))
    target.add_argument("--opt-type", default="max-l1", choices=OPT_TYPES)
    target.add_argument("--orient", default=None, help="desired field orientation")
    target.add_argument("--desired-intensity", type=float, default=1.0)
    target.add_argument("--elec-num", type=int, default=None)
    target.add_argument("--target-radius", type=int, default=2)
    target.add_argument("-k", type=float, default=None)
    target.add_argument("--tag", default=None, help="name of this targeting run")
    target.add_argument("--no-show", action="store_true")

    review = sub.add_parser("review", help="re-open the figures of a finished run")
    review.add_argument("subject", nargs="?", default=None)
    review.add_argument("--sim-tag", required=True)
    review.add_argument("--tar-tag", default=None)
    review.add_argument("--tissue", default="brain")
    review.add_argument("--no-show", action="store_true", help="do not open figures")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "simulate":
        from .pipeline import roast

        recipe = "leadField" if args.lead_field else (
            _parse_recipe(args.recipe, parser) if args.recipe else None)
        roast(args.subject, recipe, cap_type=args.cap_type, elec_type=args.elec_type,
              elec_size=args.elec_size, elec_ori=args.elec_ori, T2=args.t2,
              multiaxial=args.multiaxial, manual_gui=args.manual_gui,
              simulation_tag=args.tag, resampling=args.resample,
              zero_pad=args.zero_pad, show=not args.no_show)
    elif args.command == "target":
        from .pipeline import roast_target

        coords = np.asarray(args.target, dtype=float)
        if coords.size % 3:
            parser.error("--target takes three numbers per target")
        roast_target(args.subject, args.sim_tag, coords.reshape(-1, 3),
                     coord_type=args.coord_type, opt_type=args.opt_type,
                     orient=args.orient, desired_intensity=args.desired_intensity,
                     elec_num=args.elec_num, target_radius=args.target_radius,
                     k=args.k, targeting_tag=args.tag, show=not args.no_show)
    else:
        from .pipeline import review_res

        review_res(args.subject, args.sim_tag, tissue=args.tissue, tar_tag=args.tar_tag,
                   show=not args.no_show)

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()                      # keep the figures open until they are closed
    return 0


if __name__ == "__main__":            # pragma: no cover
    sys.exit(main())
