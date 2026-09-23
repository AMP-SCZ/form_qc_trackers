"""Diagnose why the MED-QC-01 pharm flag (modification date before latest
assessment date) is absent from PRONET output.

Standalone on purpose: imports nothing from the project (so it runs even where
`utils.Utils` path-splitting misbehaves) and prints AGGREGATE COUNTS ONLY —
no subject IDs, no dates, no row values — so the output is safe to share.

Run from the project root of the environment that produced the QC output:

    python diagnose_pronet_pharm_flag.py

It walks the funnel a MED-QC-01 flag must survive, per network:

  stage 1  combined_qc_flags parquets (old/new/current): did the flag mint,
           and if so were its `reports` blanked before the tracker stage?
  stage 2  which combined CSVs even contain chrpharm_date_mod/_2 columns
           (header-only reads)
  stage 3  re-derive expected mints from the CSVs + dependency JSONs and
           report where the funnel collapses to zero.
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

MSG_SNIPPET = "is before the latest assessment date"
PHARM_FORM_PREFIX = "current_pharmaceutical_treatment_floating_med"
MOD_VARS = ["chrpharm_date_mod", "chrpharm_date_mod_2"]

# Mirrors PharmChecks.missing_code_list for the date path: after the
# YYYY-MM-DD parse only date-shaped sentinels can slip through.
MISSING_DATE_STRINGS = {"1909-09-09", "1903-03-03"}
MISSING_STRINGS = {"", "nan", "nat", "none", "null"}

TP_LIST = (
    ["screening", "baseline"]
    + [f"month{x}" for x in range(1, 13)]
    + ["month18", "month24", "floating", "conversion"]
)


def csv_path(comb_path, tp, network):
    tp_name = tp.replace("month", "month_").replace("floating", "floating_forms")
    net_name = network.replace("PRONET", "ProNET")
    return f"{comb_path}AMPSCZ-combined-redcap_{tp_name}_{net_name}-day1to1.csv"


def parse_valid_date(val):
    """Replicates PharmChecks._normalize_date_str closely enough for mod dates."""
    s = str(val).strip()
    if not s or s.lower() in MISSING_STRINGS:
        return None
    s = s.split(" ")[0].strip()
    if s in MISSING_DATE_STRINGS:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    comb_path = cfg["paths"]["combined_csv_path"]
    dep_path = cfg["paths"]["dependencies_path"]
    out_path = cfg["paths"]["output_path"]
    if cfg.get("testing_enabled") == "True":
        out_path += "testing/"

    section("CONFIG GATES (report-blanking behavior in create_row_output)")
    print(f"  recruited_only    = {cfg.get('recruited_only')}"
          "   (True => rows with recruitment_status_v2 != 'recruited' get reports blanked)")
    print(f"  withdrawn_enabled = {cfg.get('withdrawn_enabled')}"
          "   (False => rows with visit_status_string == 'removed' get reports blanked)")
    print(f"  testing_enabled   = {cfg.get('testing_enabled')}")

    # ------------------------------------------------------------------ #
    # STAGE 1: did the flag mint into the parquet handoff at all?
    # ------------------------------------------------------------------ #
    section("STAGE 1: MED-QC-01 rows in combined_qc_flags parquets")
    for stage in ["new", "current", "old"]:
        pq = f"{out_path}combined_outputs/{stage}_output/combined_qc_flags.parquet"
        if not os.path.exists(pq):
            print(f"[{stage}] not found: {pq}")
            continue
        df = pd.read_parquet(pq)
        df["error_message"] = df["error_message"].astype(str)
        hits = df[df["error_message"].str.contains(MSG_SNIPPET, regex=False)]
        print(f"[{stage}] total rows={len(df)}  MED-QC-01 rows by network: "
              f"{hits['network'].value_counts().to_dict() or '{} (NONE)'}")
        for network in ["PRONET", "PRESCIENT"]:
            sub = hits[hits["network"] == network]
            if sub.empty:
                continue
            blank = (sub["reports"].astype(str).str.strip() == "").sum()
            print(f"    {network}: n={len(sub)}  reports_blank={blank}  "
                  f"reports_routed={len(sub) - blank}  "
                  f"displayed_timepoints={sub['displayed_timepoint'].value_counts().to_dict()}  "
                  f"currently_resolved={sub['currently_resolved'].astype(str).value_counts().to_dict()}")
        # Control group: ALL flags on the current-pharm floating forms.
        # If PRONET has none of these either, the whole floating-row family is
        # being suppressed (blanking gates), not MED-QC-01 specifically.
        ctrl = df[df["displayed_form"].astype(str).str.startswith(PHARM_FORM_PREFIX)]
        for network in ["PRONET", "PRESCIENT"]:
            sub = ctrl[ctrl["network"] == network]
            blank = (sub["reports"].astype(str).str.strip() == "").sum()
            print(f"    control (any current-pharm-form flag) {network}: "
                  f"n={len(sub)}  reports_blank={blank}  reports_routed={len(sub) - blank}")

    # ------------------------------------------------------------------ #
    # STAGE 2: which combined CSVs carry the mod-date columns?
    # ------------------------------------------------------------------ #
    section("STAGE 2: which combined CSVs contain chrpharm_date_mod / _2 (header-only)")
    mod_csvs = {"PRONET": [], "PRESCIENT": []}
    for network in ["PRONET", "PRESCIENT"]:
        for tp in TP_LIST:
            path = csv_path(comb_path, tp, network)
            if not os.path.exists(path):
                print(f"  {network} {tp}: CSV MISSING ({os.path.basename(path)})")
                continue
            cols = pd.read_csv(path, nrows=0).columns
            present = [v for v in MOD_VARS if v in cols]
            if present:
                mod_csvs[network].append((tp, path, present))
                print(f"  {network} {tp}: has {present}")
        if not mod_csvs[network]:
            print(f"  {network}: NO combined CSV contains the mod-date columns "
                  "=> the check can never fire for this network")

    # ------------------------------------------------------------------ #
    # STAGE 3: re-derive expected mints and find the collapsing gate
    # ------------------------------------------------------------------ #
    section("STAGE 3: expected-mint funnel per network (aggregate counts)")
    tp_ranges = json.load(open(f"{dep_path}earliest_latest_dates_per_tp.json"))
    subject_info = json.load(open(f"{dep_path}subject_info.json"))
    print(f"  earliest_latest_dates_per_tp.json: {len(tp_ranges)} subjects; "
          f"subject_info.json: {len(subject_info)} subjects")

    for network in ["PRONET", "PRESCIENT"]:
        print(f"\n  --- {network} ---")
        rows_with_mod = 0
        funnel = {
            "rows_with_valid_mod_date": 0,
            "  and subject in subject_info (row-loop guard)": 0,
            "  and subject in tp_date_ranges": 0,
            "  and >=1 tp entry has a valid top-level 'latest'": 0,
            "  and latest assessment > latest mod date (WOULD MINT)": 0,
            "    minted but blanked: inclusion_status != 'included'": 0,
            "    minted but blanked: recruitment == 'negative_screen'": 0,
            "    minted but blanked: visit_status_string == 'removed' (active when withdrawn_enabled false)": 0,
            "    would-blank IF recruited_only: recruitment != 'recruited'": 0,
            "  minted AND reports survive current config": 0,
        }
        subjects_minting = set()
        for tp, path, present in mod_csvs[network]:
            usecols = ["subjectid"] + present
            for extra in ["visit_status_string", "recruitment_status_v2"]:
                if extra in pd.read_csv(path, nrows=0).columns:
                    usecols.append(extra)
            df = pd.read_csv(path, usecols=usecols, keep_default_na=False,
                             low_memory=False)
            for row in df.itertuples(index=False):
                mods = [parse_valid_date(getattr(row, v))
                        for v in present if hasattr(row, v)]
                mods = [m for m in mods if m is not None]
                if not mods:
                    continue
                rows_with_mod += 1
                funnel["rows_with_valid_mod_date"] += 1
                sid = getattr(row, "subjectid")
                if sid not in subject_info:
                    continue
                funnel["  and subject in subject_info (row-loop guard)"] += 1
                sub_dates = tp_ranges.get(sid, {})
                if not sub_dates:
                    continue
                funnel["  and subject in tp_date_ranges"] += 1
                latests = [parse_valid_date(info.get("latest", ""))
                           for info in sub_dates.values()
                           if isinstance(info, dict)]
                latests = [d for d in latests if d is not None]
                if not latests:
                    continue
                funnel["  and >=1 tp entry has a valid top-level 'latest'"] += 1
                if max(latests) <= max(mods):
                    continue
                funnel["  and latest assessment > latest mod date (WOULD MINT)"] += 1
                subjects_minting.add(sid)

                incl = str(subject_info[sid].get("inclusion_status", "")).lower()
                recruit = str(getattr(row, "recruitment_status_v2", "")).strip().lower()
                removed = str(getattr(row, "visit_status_string", "")) == "removed"
                blanked = False
                if incl != "included":
                    funnel["    minted but blanked: inclusion_status != 'included'"] += 1
                    blanked = True
                if recruit == "negative_screen":
                    funnel["    minted but blanked: recruitment == 'negative_screen'"] += 1
                    blanked = True
                if removed and cfg.get("withdrawn_enabled") is False:
                    funnel["    minted but blanked: visit_status_string == 'removed' (active when withdrawn_enabled false)"] += 1
                    blanked = True
                if recruit != "recruited":
                    funnel["    would-blank IF recruited_only: recruitment != 'recruited'"] += 1
                    if cfg.get("recruited_only") is True:
                        blanked = True
                if not blanked:
                    funnel["  minted AND reports survive current config"] += 1

        for label, count in funnel.items():
            print(f"    {label}: {count}")
        print(f"    unique subjects that would mint: {len(subjects_minting)}")
        if rows_with_mod and funnel["  and latest assessment > latest mod date (WOULD MINT)"] == 0:
            print("    >>> funnel collapses BEFORE minting - see the first zero above")
        elif funnel["  minted AND reports survive current config"] == 0 and rows_with_mod:
            print("    >>> flags mint but every one is report-blanked - see blanking lines")

    print("\nDone. Paste this output back - it contains aggregate counts only.")


if __name__ == "__main__":
    sys.exit(main())
