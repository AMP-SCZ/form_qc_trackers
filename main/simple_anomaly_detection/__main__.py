"""Entry point.

Input/output folders come from -i/-o, or the FORMQC_ANOMALY_INPUT /
FORMQC_ANOMALY_OUTPUT environment variables; with none set the constructor
raises a clear error (no machine-specific path is baked in).  The metadata
contracts come from -d/--dependencies or FORMQC_ANOMALY_DEPENDENCIES, which
unlike the I/O paths falls back to the package's own dependencies folder.

    python -m simple_anomaly_detection -i PATH -o PATH # clinical-only default
    python -m simple_anomaly_detection -i PATH -d PATH/dependencies
    FORMQC_ANOMALY_INPUT=PATH python -m simple_anomaly_detection
    python -m simple_anomaly_detection --demo          # run on synthetic data

The default scope is the intersection of the study's Clinical-measures form
mapping and the REDCap data dictionary's own field metadata: a variable is
reportable only if both call it a measurement.  The dictionary is required
whenever clinical scoping is on; a run fails loudly rather than falling back to
the wider name-only allowlist, which contained PHI-marked fields.

Use ``--all-variables`` only when deliberately debugging outside the
production Clinical-measures scope.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from .runner import (
    SimpleAnomalyDetector, DEFAULT_COMBINED_MAX_PER_COMBO,
    DEFAULT_MAX_ROWS_PER_VARIABLE, DEFAULT_MAX_ROWS_PER_SUBJECT,
)


def _positive_int(v):
    """argparse type for --cap: a cap of 0 (or negative) would make every
    per-type frame .head(0) and combined_cap 0, silently writing an empty
    report with exit 0. Reject it so the misread (--combined-max-per-combo
    documents '0 disables', which does NOT apply to --cap) fails loudly."""
    iv = int(v)
    if iv <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive integer (a cap of 0 would write an empty "
            "report); omit --cap for the default of 5000")
    return iv


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="simple_anomaly_detection",
                                description="Multi-detector anomaly report.")
    p.add_argument("-i", "--input",  default=None, help="input folder (combined CSVs)")
    p.add_argument("-o", "--output", default=None, help="output folder")
    p.add_argument("-d", "--dependencies", default=None,
                   help="folder holding the metadata contracts "
                        "(missingness_domain_forms.json, "
                        "grouped_variables.json, important_form_vars.json, "
                        "data_dictionary/); default: the package's own "
                        "dependencies folder")
    p.add_argument("--cap", type=_positive_int, default=5000, help="max rows per anomaly type")
    p.add_argument("--combined-max-per-combo", type=int,
                   default=DEFAULT_COMBINED_MAX_PER_COMBO,
                   help="backward-compatible name for the max rows per "
                        "underlying variable in combined_ranked (applies to "
                        "single variables and every pair/triple leg; 0 disables; "
                        f"default {DEFAULT_COMBINED_MAX_PER_COMBO})")
    p.add_argument("--max-per-variable", type=int,
                   default=DEFAULT_MAX_ROWS_PER_VARIABLE,
                   help="max review rows per underlying variable across an "
                        "entire detector tab (0 disables)")
    p.add_argument("--max-per-subject", type=int,
                   default=DEFAULT_MAX_ROWS_PER_SUBJECT,
                   help="max review rows per (network, subject) in each tab "
                        "(0 disables)")
    p.add_argument("--all-variables", action="store_true",
                   help="debug-only opt-out of the default Clinical measures "
                        "variable scope")
    p.add_argument("--data-dictionary", default=None,
                   help="REDCap data dictionary CSV used to confirm which "
                        "clinical-form fields hold measurement data "
                        "(default: discovered beside the repository; ignored "
                        "with --all-variables)")
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
                                    combined_max_per_combo=args.combined_max_per_combo,
                                    max_rows_per_variable=args.max_per_variable,
                                    max_rows_per_subject=args.max_per_subject,
                                    # Synthetic demo names do not exist in the
                                    # production REDCap variable map.
                                    clinical_only=False)
        det.run()
        print(f"[demo] report in {out_dir}")
        return 0

    kwargs = {"cap_per_type": args.cap,
              "combined_max_per_combo": args.combined_max_per_combo,
              "max_rows_per_variable": args.max_per_variable,
              "max_rows_per_subject": args.max_per_subject,
              "clinical_only": not args.all_variables,
              "data_dictionary_path": args.data_dictionary}
    if args.input:
        kwargs["input_path"] = args.input
    if args.output:
        kwargs["output_path"] = args.output
    if args.dependencies:
        kwargs["dependencies_path"] = args.dependencies
    det = SimpleAnomalyDetector(**kwargs)
    det.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
