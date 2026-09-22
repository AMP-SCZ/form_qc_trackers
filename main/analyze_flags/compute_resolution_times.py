"""Per-flag time-to-resolve by proposed QC category from tracker histories.

    python compute_resolution_times.py --run-dir HISTORY --mapping CATEGORIES.xlsx

Input reports and category mappings are supplied by the caller.

Mirrors the pipeline's own semantics (analyze_flags/flag_analytics.py and
graph_recovered_flags.py):
  * operator jump decisions applied (excluded ADDITION jumps -> entry dropped;
    excluded REMOVAL jumps -> disappearance not treated as a resolution);
  * resolution_eligible respected (PRONET V2-only testing episodes excluded);
  * resolved = NOT is_currently_open;
  * resolution date = Resolution_observed when the column exists. The 8/19
    history CSVs predate that column entirely, so the OPERATIVE definition for
    this run is graph_recovered_flags.py's older-artifact fallback: the whole
    column is taken from Latest_seen, making every duration a LOWER BOUND
    (the true fix happened after the last revision the flag was present).
    Individual nulls are never filled from Latest_seen;
  * clinical-form filter identical to FlagAnalytics (the same scope that
    produced flag_template_counts_*.csv, whose messages were categorized).

Time to resolve = (resolution date - Earliest_seen) in days.

Writes a NON-PHI per-entry CSV (network, category, canonical_template,
days_to_resolve) and prints aggregate stats only.
"""
import main  # Make deployed sibling packages available to this CLI.

import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "analyze_flags_output"
OUT_CSV = ROOT / "resolution_times_by_category.csv"

from analyze_flags.manual_review import (  # noqa: E402
    entry_key_from_row,
    load_jump_decisions,
)
from analyze_flags.canonicalize import normalize_form_name  # noqa: E402

try:
    from analyze_flags.flag_analytics import FlagAnalytics
    CLINICAL_FORMS = FlagAnalytics.CLINICAL_FORMS
except Exception as error:  # Utils import chain can break on Windows
    print(f"NOTE: FlagAnalytics import failed ({type(error).__name__}: "
          f"{error}); using a local copy of CLINICAL_FORMS.")
    CLINICAL_FORMS = frozenset([
        "lifetime_ap_exposure_screen", "past_pharmaceutical_treatment",
        "health_conditions_medical_historypsychiatric_histo",
        "current_pharmaceutical_treatment_floating_med_125",
        "current_pharmaceutical_treatment_floating_med_2650",
        "adverse_events", "conversion_form",
        "scid5_schizotypal_personality_sciddpq",
        "family_interview_for_genetic_studies_figs",
        "sofas_screening", "sofas_followup",
        "psychs_p1p8", "psychs_p9ac32", "psychs_p1p8_fu", "psychs_p9ac32_fu",
        "psychs_p1p8_fu_hc", "psychs_p9ac32_fu_hc",
        "psychs_av_recording_run_sheet", "premorbid_adjustment_scale",
        "nsipr", "cdss", "assist", "cssrs_baseline", "cssrs_followup",
        "global_functioning_social_scale",
        "global_functioning_social_scale_followup",
        "global_functioning_role_scale",
        "global_functioning_role_scale_followup",
        "perceived_discrimination_scale", "pubertal_developmental_scale",
        "oasis", "item_promis_for_sleep", "perceived_stress_scale", "pgis",
        "psychosis_polyrisk_score", "bprs", "psychosocial_treatment_form",
        "scid5_psychosis_mood_substance_abuse",
    ])

BOOL_TRUE = {"true", "1", "yes"}
BOOL_ALL = {"true", "false", "1", "0", "yes", "no"}

MAPPING = ROOT / "template_mapping_recategorized_updated.xlsx"
if not MAPPING.exists():
    MAPPING = ROOT / "template_mapping_recategorized.xlsx"


def parse_dates(series):
    parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce",
                            utc=True)
    return parsed.dt.tz_localize(None).dt.normalize()


def load_network(network, template_to_cat, apply_exclusions=True, run_dir=RUN_DIR):
    run_dir = Path(run_dir).expanduser().resolve()
    csv_path = run_dir / f"open_tracker_row_history_{network}.csv"
    df = pd.read_csv(csv_path, keep_default_na=False)
    df.columns = df.columns.str.replace(" ", "_")
    n0 = len(df)
    view = "conservative" if apply_exclusions else "raw"
    print(f"\n=== {network} [{view}]: {n0} history entries ===")

    # --- operator jump decisions (same as FlagAnalytics._apply_jump_decisions)
    df["resolution_excluded"] = False
    if apply_exclusions:
        decisions = load_jump_decisions(str(run_dir), network)
        print(f"jumps: {decisions['n_jumps']} total, "
              f"{decisions['n_undecided']} undecided (undecided => excluded)")
        keys = [entry_key_from_row(r) for r in df.itertuples(index=False)]
        if decisions["excluded_added"]:
            keep = pd.Series(
                [k not in decisions["excluded_added"] for k in keys],
                index=df.index)
            print(f"dropped {int((~keep).sum())} entries from excluded "
                  "addition jumps")
            keys = [k for k, kept in zip(keys, keep) if kept]
            df = df[keep].copy()
        if decisions["excluded_removed"]:
            res_excl = pd.Series(
                [k in decisions["excluded_removed"] for k in keys],
                index=df.index)
            df.loc[res_excl, "resolution_excluded"] = True
            print(f"marked {int(res_excl.sum())} entries resolution_excluded "
                  "(excluded removal jumps)")

    # --- dates and booleans
    for col in ("Earliest_seen", "Latest_seen"):
        df[col] = parse_dates(df[col])
    if "Resolution_observed" in df.columns:
        res_raw = df["Resolution_observed"].astype(str).str.strip()
        df["Resolution_observed"] = parse_dates(df["Resolution_observed"])
        malformed = (res_raw.ne("") & df["Resolution_observed"].isna()).sum()
        if malformed:
            raise SystemExit(f"{malformed} malformed Resolution_observed values")
    else:
        # Older artifact format. Same fallback graph_recovered_flags.py uses:
        # bucket resolutions by Latest_seen (last revision the flag was
        # present) — durations are then a LOWER BOUND on time-to-resolve.
        print("NOTE: no Resolution_observed column (older artifact format); "
              "using Latest_seen — durations are lower-bound estimates.")
        df["Resolution_observed"] = df["Latest_seen"]

    open_raw = df["is_currently_open"].astype(str).str.strip().str.casefold()
    assert open_raw.isin(BOOL_ALL).all(), "invalid is_currently_open"
    df["is_currently_open"] = open_raw.isin(BOOL_TRUE)

    if "resolution_eligible" in df.columns:
        eligible_raw = (df["resolution_eligible"].astype(str).str.strip()
                        .str.casefold())
        assert eligible_raw.isin(BOOL_ALL).all(), "invalid resolution_eligible"
        df["resolution_eligible"] = eligible_raw.isin(BOOL_TRUE)
    elif network == "PRONET" and "source" in df.columns:
        # Backward compatibility, same as the pipeline: PRONET V2 is
        # testing-only, so only entries seen in V1 prove a live resolution.
        df["resolution_eligible"] = df["source"].map(
            lambda value: "V1" in str(value).split("+"))
    else:
        df["resolution_eligible"] = True

    # --- clinical-form filter (same scope as flag_template_counts)
    lookup = {normalize_form_name(f): f for f in CLINICAL_FORMS}
    clinical = df["General_Flag"].map(normalize_form_name).isin(lookup)
    print(f"clinical filter: kept {int(clinical.sum())}, "
          f"dropped {int((~clinical).sum())} non-clinical entries")
    df = df[clinical].copy()

    # --- resolved selection
    n_all = len(df)
    n_open = int(df["is_currently_open"].sum())
    resolved = df[~df["is_currently_open"]
                  & ~df["resolution_excluded"]
                  & df["resolution_eligible"]].copy()
    print(f"resolved-eligible entries: {len(resolved)} "
          f"(of {n_all}; {n_open} still open; "
          f"{int((~df['is_currently_open'] & df['resolution_excluded']).sum())} "
          "closed-but-excluded-removal; "
          f"{int((~df['is_currently_open'] & ~df['resolution_excluded'] & ~df['resolution_eligible']).sum())} "
          "ineligible)")

    both_bounds = (resolved["Resolution_observed"].notna()
                   & resolved["Earliest_seen"].notna())
    no_obs = int((~both_bounds).sum())
    resolved = resolved[both_bounds].copy()
    print(f"dropped {no_obs} resolved entries missing a date bound; "
          f"{len(resolved)} remain")

    resolved["days_to_resolve"] = (
        resolved["Resolution_observed"] - resolved["Earliest_seen"]
    ).dt.days
    bad = ~(resolved["days_to_resolve"] >= 0)  # catches negatives AND NaN
    if bad.any():
        print(f"WARNING: {int(bad.sum())} negative/NaN durations — dropping")
        resolved = resolved[~bad]

    resolved["category"] = resolved["canonical_template"].map(template_to_cat)
    unmapped = resolved["category"].isna()
    if unmapped.any():
        print(f"WARNING: {int(unmapped.sum())} resolved entries with unmapped "
              "templates (showing templates only):")
        for tpl in resolved.loc[unmapped, "canonical_template"].unique()[:10]:
            print(f"  - {tpl[:160]}")
        resolved = resolved[~unmapped]

    resolved["network"] = network
    resolved["view"] = "conservative" if apply_exclusions else "raw"
    return resolved[["view", "network", "category", "canonical_template",
                     "days_to_resolve"]]


def main(run_dir=RUN_DIR, mapping_path=MAPPING, output_path=OUT_CSV,
         networks=("PRESCIENT", "PRONET")):
    mapping_path = Path(mapping_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    tmap = pd.read_excel(mapping_path, sheet_name="templates")
    template_to_cat = dict(zip(tmap["canonical_template"].astype(str),
                               tmap["proposed_category"].astype(str)))
    try:
        extra = pd.read_excel(mapping_path, sheet_name="raw_view_extra_templates")
        for tpl, cat in zip(extra["canonical_template"].astype(str),
                            extra["proposed_category"].astype(str)):
            template_to_cat.setdefault(tpl, cat)
        print(f"raw-view extras: {len(extra)} templates")
    except ValueError:
        print("no raw_view_extra_templates sheet")
    print(f"mapping: {mapping_path.name}, {len(template_to_cat)} templates")

    frames = [load_network(net, template_to_cat, apply_exclusions=flag,
                           run_dir=run_dir)
              for flag in (True, False)
              for net in networks]
    out = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"\nWrote {output_path} ({len(out)} rows)")

    for view in ("conservative", "raw"):
        sub = out[out["view"] == view]
        print(f"\n=== [{view}] days-to-resolve by category (pooled) ===")
        stats = (sub.groupby("category")["days_to_resolve"]
                 .agg(n="size", median="median", mean="mean",
                      q1=lambda s: s.quantile(0.25),
                      q3=lambda s: s.quantile(0.75),
                      min="min", max="max")
                 .sort_values("median", ascending=False))
        print(stats.round(1).to_string())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="directory containing open_tracker_row_history_<NETWORK>.csv and jump reviews")
    parser.add_argument("--mapping", type=Path, required=True,
                        help="workbook containing canonical templates and proposed categories")
    parser.add_argument("--output", type=Path,
                        default=Path("resolution_times_by_category.csv"),
                        help="output CSV (default: %(default)s)")
    parser.add_argument("--networks", nargs="+", choices=("PRESCIENT", "PRONET"),
                        default=["PRESCIENT", "PRONET"],
                        help="networks to analyze (default: both)")
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    main(args.run_dir, args.mapping, args.output,
         networks=tuple(dict.fromkeys(args.networks)))


if __name__ == "__main__":
    cli()
