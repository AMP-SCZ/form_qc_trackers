"""Entry point.

Input/output folders come from -i/-o, or the FORMQC_ANOMALY_INPUT /
FORMQC_ANOMALY_OUTPUT environment variables; with none set the constructor
raises a clear error (no machine-specific path is baked in).

    python -m simple_anomaly_detection -i PATH -o PATH # explicit paths
    FORMQC_ANOMALY_INPUT=PATH python -m simple_anomaly_detection
    python -m simple_anomaly_detection --demo          # run on synthetic data
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from .runner import SimpleAnomalyDetector, DEFAULT_COMBINED_MAX_PER_COMBO


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="simple_anomaly_detection",
                                description="Multi-detector anomaly report.")
    p.add_argument("-i", "--input",  default=None, help="input folder (combined CSVs)")
    p.add_argument("-o", "--output", default=None, help="output folder")
    p.add_argument("--cap", type=int, default=5000, help="max rows per anomaly type")
    p.add_argument("--combined-max-per-combo", type=int,
                   default=DEFAULT_COMBINED_MAX_PER_COMBO,
                   help="max rows per (network, variable-combo) for the cluster "
                        "detectors in combined_ranked, so the headline isn't "
                        "dominated by one repeating combo (0 disables; "
                        f"default {DEFAULT_COMBINED_MAX_PER_COMBO})")
    p.add_argument("--demo", action="store_true",
                   help="generate synthetic CSVs and run against them")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.demo:
        from .demo import build_demo_dataset
        demo_root = tempfile.mkdtemp(prefix="simple_anomaly_demo_")
        in_dir = os.path.join(demo_root, "in")
        out_dir = os.path.join(demo_root, "out")
        build_demo_dataset(in_dir)
        print(f"[demo] synthetic CSVs in {in_dir}")
        det = SimpleAnomalyDetector(input_path=in_dir, output_path=out_dir,
                                    cap_per_type=args.cap,
                                    combined_max_per_combo=args.combined_max_per_combo)
        det.run()
        print(f"[demo] report in {out_dir}")
        return 0

    kwargs = {"cap_per_type": args.cap,
              "combined_max_per_combo": args.combined_max_per_combo}
    if args.input:
        kwargs["input_path"] = args.input
    if args.output:
        kwargs["output_path"] = args.output
    det = SimpleAnomalyDetector(**kwargs)
    det.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
