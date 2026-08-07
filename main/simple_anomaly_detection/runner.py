"""SimpleAnomalyDetector - load AMPSCZ combined CSVs, run the configured
detectors, write one ranked Excel workbook plus per-type CSVs.

Paths live in ``__init__`` so editing this class is the only configuration step.
No config.json read; no source-data writes; per-detector try/except isolation.
"""

from __future__ import annotations

import glob
import os
import re
import time
import traceback
import uuid
from typing import Dict, List, Tuple

import pandas as pd

from . import (
    standard_outlier,
    time_series,
    correlative,
    site_network,
    format_string,
    whole_subject_site,
    date_anomaly,
    cluster_3d,
    cluster_3d_longitudinal,
    rater_effect,
)
from .common import (
    FINDING_COLUMNS, classify_columns, coerce_to_finding_frame,
)


_DETECTORS = [
    ("standard_outlier", standard_outlier),
    ("time_series", time_series),
    ("correlative", correlative),
    ("site_network", site_network),
    ("format_string", format_string),
    ("whole_subject_site", whole_subject_site),
    ("date_anomaly", date_anomaly),
    ("cluster_3d", cluster_3d),
    ("cluster_3d_longitudinal", cluster_3d_longitudinal),
    ("rater_effect", rater_effect),
]

# Detectors whose findings name a variable COMBO ("a | b | c"); the runner
# writes an additive per-combo summary index for each (see _build_combo_frames /
# cluster_3d.summarize_combos) so a combo that flags many subjects becomes one
# row instead of recurring across many per-subject rows.
_COMBO_SUMMARY_TYPES = ("cluster_3d", "cluster_3d_longitudinal")

# The anomaly_type values whose combined-headline rows are capped per
# (network, variable) so a few variables can't dominate the top of the report.
#   - The cluster detectors: a correlated block yields C(M,3) triples over the
#     same few variables, all sharing one "a | b | c" label.
#   - correlative: after _dedup_correlative, one bad value re-flagged for many
#     subjects collapses to rows that share a single dominant-variable (or pair)
#     label, so the same variable can otherwise flood the headline. (Its
#     binary-binary joint-cell rows are exempted in _diversify_combined -- they
#     are distinct subjects, not a flooding variable.)
# Capping defers the lower-severity tail to the per-type sheet; that sheet is
# itself bounded by cap_per_type, so it is a full-detail fallback only while a
# detector's raw output fits that cap. The cluster detectors additionally
# surface the variable combo in their per-combo summary index; correlative has
# no such index. Detectors NOT listed here are left alone -- their rows that
# share a variable label are distinct subjects, so capping would HIDE findings.
_COMBO_ANOMALY_TYPES = (cluster_3d.ANOMALY_TYPE, cluster_3d_longitudinal.ANOMALY_TYPE,
                        correlative.ANOMALY_TYPE)
# Per (network, combo) cap on combo-detector rows in combined_ranked (0 disables).
# Deferred per-subject detail stays in the per-type sheet (+ per-combo index for
# the cluster detectors); see _diversify_combined for the exact guarantee.
DEFAULT_COMBINED_MAX_PER_COMBO = 5

# Per-detector overrides for the per-type row cap. Detectors not listed here use
# cap_per_type. date_anomaly scans date columns all-pairs (within- and cross-
# timepoint), so it can legitimately emit far more rows than the other
# detectors; give it a higher ceiling so its per-type sheet isn't truncated.
PER_TYPE_CAP_OVERRIDES = {"date_anomaly": 100_000}

# Global severity floor: rows whose severity_score is below this value are
# dropped before any sheet is built (per-type, combined_ranked, and the combo
# summary indexes all inherit it). Severity is the shared 0..100 scale where a
# detector's own flag threshold lands at ~50 (see common.py); 0.0 keeps every
# flagged row. Raise this to suppress lower-severity findings from the output.
MIN_SEVERITY = 0.0

NETWORKS = ("PRONET", "PRESCIENT")

TIMEPOINTS = (
    "screening", "baseline",
    "month1", "month2", "month3", "month4", "month5", "month6",
    "month7", "month8", "month9", "month10", "month11", "month12",
    "month18", "month24",
    "conversion", "floating",
)


def _tp_file_name(tp: str) -> str:
    if tp.startswith("month") and tp != "monthly":
        n = tp.replace("month", "")
        if n.isdigit():
            return f"month_{n}"
    if tp == "floating":
        return "floating_forms"
    return tp


def _basename_tp_ok(basename: str, tp_file: str) -> bool:
    """True if ``tp_file`` appears in ``basename`` as an exact timepoint token:
    not preceded by an alphanumeric character and not followed by a digit.

    Prevents the unanchored glob fallback from matching the wrong month --
    e.g. "month_1" must not match "month_10"/"month_18", and "month_2" must
    not match "month_24". Tolerant of the underscore/hyphen separator variant.
    """
    base = basename.lower()
    for tf in {tp_file.lower(), tp_file.lower().replace("_", "-")}:
        if re.search(r"(?<![a-z0-9])" + re.escape(tf) + r"(?![0-9])", base):
            return True
    return False


class SimpleAnomalyDetector:
    """Run the configured anomaly detectors and write a ranked Excel report.

    Parameters (all configurable here in ``__init__``):
        input_path  : folder containing AMPSCZ-combined-redcap CSVs
        output_path : folder for the report + per-type CSVs (created if missing)
        cap_per_type: max rows per anomaly type in the output (default 5000)
        networks    : networks to scan (default both)
        timepoints  : timepoints to scan (default all)
        max_rows_per_csv : optional sample cap when smoke-testing on huge files
    """

    def __init__(
        self,
        input_path: str = "/data/predict1/data_from_nda/formsdb/generated_outputs/combined/PROTECTED/",
        output_path: str = "/home/ob001/output",
        cap_per_type: int = 5000,
        networks: Tuple[str, ...] = NETWORKS,
        timepoints: Tuple[str, ...] = TIMEPOINTS,
        max_rows_per_csv: int = 0,
        combined_cap: int = 0,
        combined_max_per_combo: int = DEFAULT_COMBINED_MAX_PER_COMBO,
        per_type_caps: Dict[str, int] = None,
    ):
        # Resolve paths: explicit arg > env var > error. No machine-specific
        # absolute default is baked in -- the old hardcoded C:/Users/owenb/...
        # default only worked on one box and silently produced a NO_DATA run
        # everywhere else. Use --demo / build_demo_dataset for synthetic data.
        input_path = input_path or os.environ.get("FORMQC_ANOMALY_INPUT")
        if not input_path:
            raise ValueError(
                "input_path is required: pass input_path=..., or set the "
                "FORMQC_ANOMALY_INPUT environment variable to the folder of "
                "AMPSCZ-combined CSVs.")
        self.input_path = os.path.abspath(input_path)
        output_path = output_path or os.environ.get("FORMQC_ANOMALY_OUTPUT")
        if not output_path:
            # Default to a SIBLING of input (never a child -- _guard_paths
            # refuses to run when output is inside input, so output handling
            # can never touch the source CSVs).
            output_path = os.path.join(os.path.dirname(self.input_path),
                                       "simple_anomaly_output")
        self.output_path = os.path.abspath(output_path)
        self.cap_per_type = int(cap_per_type)
        self.networks = tuple(networks)
        self.timepoints = tuple(timepoints)
        self.max_rows_per_csv = int(max_rows_per_csv)
        # Global cap on the combined ranked sheet (0 -> cap_per_type x n_detectors).
        self.combined_cap = int(combined_cap) if combined_cap else self.cap_per_type * len(_DETECTORS)
        # Per (network, combo) cap applied to the combo detectors when building
        # the combined headline, so it isn't dominated by one repeating combo.
        self.combined_max_per_combo = int(combined_max_per_combo)
        # Per-detector per-type cap overrides (detector name -> cap). Detectors
        # absent from this map fall back to cap_per_type.
        self.per_type_caps = dict(PER_TYPE_CAP_OVERRIDES if per_type_caps is None
                                  else per_type_caps)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _candidate_paths(self, tp: str, network: str) -> List[str]:
        tp_file = _tp_file_name(tp)
        templates = [
            f"AMPSCZ-combined-redcap_{tp_file}_{network}-day1to1.csv",
            f"AMPSCZ-combined-redcap_{tp_file}_{network}.csv",
            f"AMPSCZ-combined-redcap_{tp_file}_{network}-day1to1_combined.csv",
        ]
        out = [os.path.join(self.input_path, name) for name in templates]
        # Case variants. Add ProNET explicitly since .title() gives "Pronet".
        net_variants = {network, network.lower(), network.upper(), network.title()}
        if network.upper() == "PRONET":
            net_variants.update({"ProNET", "pronet", "Pronet"})
        tp_variants = {tp_file, tp_file.lower(), tp_file.upper(), tp_file.title(),
                       tp_file.replace("_", "-"), tp_file.replace("-", "_")}
        # Glob fallback for fuzzily-named files. The patterns use unanchored
        # tokens, so "*month_1*" also matches month_10/11/12/18 (and
        # "*month_2*" matches month_24). Validate each match's basename so the
        # timepoint token is exact -- the token must not be preceded by an
        # alphanumeric nor followed by a digit -- before accepting it. The
        # exact-named templates above are trusted and not re-validated.
        for nv in net_variants:
            for tv in tp_variants:
                for hit in glob.glob(os.path.join(self.input_path, f"*{tv}*{nv}*.csv")):
                    if _basename_tp_ok(os.path.basename(hit), tp_file):
                        out.append(hit)
        seen, ordered = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return ordered

    def _load_slices(self) -> Dict[Tuple[str, str], pd.DataFrame]:
        slices: Dict[Tuple[str, str], pd.DataFrame] = {}
        load_log: List[dict] = []
        for network in self.networks:
            for tp in self.timepoints:
                path = None
                for cand in self._candidate_paths(tp, network):
                    if os.path.isfile(cand):
                        path = cand
                        break
                if not path:
                    load_log.append({"network": network, "timepoint": tp,
                                     "path": "", "rows": 0, "status": "missing"})
                    continue
                # on_bad_lines="warn" surfaces malformed rows instead of
                # silently dropping them. Pipe to stderr so the run-status
                # output reflects what actually happened.
                try:
                    df = pd.read_csv(path, dtype=str, keep_default_na=False,
                                     on_bad_lines="warn", low_memory=False)
                except Exception as e:
                    load_log.append({"network": network, "timepoint": tp,
                                     "path": path, "rows": 0,
                                     "status": f"read_error:{type(e).__name__}: {e}"})
                    print(f"  [load] WARN {os.path.basename(path)}: {e}")
                    continue
                if self.max_rows_per_csv and len(df) > self.max_rows_per_csv:
                    df = df.sample(self.max_rows_per_csv, random_state=0).reset_index(drop=True)
                if "subjectid" not in df.columns:
                    for alt in ("subject_id", "src_subject_id", "study_id"):
                        if alt in df.columns:
                            df = df.rename(columns={alt: "subjectid"})
                            break
                if "subjectid" not in df.columns:
                    load_log.append({"network": network, "timepoint": tp,
                                     "path": path, "rows": int(len(df)),
                                     "status": "no_subjectid_column"})
                    print(f"  [load] SKIP {os.path.basename(path)}: no subjectid column")
                    continue
                slices[(network, tp)] = df
                load_log.append({"network": network, "timepoint": tp,
                                 "path": path, "rows": int(len(df)),
                                 "status": "loaded"})
                print(f"  [load] {network}/{tp}: {len(df):>5} rows from "
                      f"{os.path.basename(path)}")
        self._load_log = load_log
        missing = [r for r in load_log if r["status"] == "missing"]
        if missing and len(missing) < 60:
            print(f"  [load] {len(missing)} (network, timepoint) slices missing "
                  f"(this is fine if you only have a subset of the data)")
        return slices

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, pd.DataFrame]:
        t0 = time.time()
        self._guard_paths()
        os.makedirs(self.output_path, exist_ok=True)
        print(f"[simple_anomaly_detection] input : {self.input_path}")
        print(f"[simple_anomaly_detection] output: {self.output_path}")

        # Write a marker right at the start so a half-completed run is
        # always distinguishable from a stale prior run.
        self._write_run_marker("started")

        per_slice = self._load_slices()
        if not per_slice:
            print("[simple_anomaly_detection] no slices loaded; nothing to do.")
            self._write_no_data_marker()
            self._write_run_marker("no_data")
            return {}

        # Classify every slice once and share with detectors. Earlier
        # each detector re-classified per slice (or per stack) and the
        # cumulative cost was 100+ classify_columns calls on big runs.
        print("[simple_anomaly_detection] classifying columns once per slice ...")
        t_classify = time.time()
        classify_per_slice: Dict[Tuple[str, str], dict] = {}
        for key, df in per_slice.items():
            try:
                classify_per_slice[key] = classify_columns(df)
            except Exception as e:
                print(f"  [classify] WARN slice {key}: {e}")
                classify_per_slice[key] = None
        print(f"[simple_anomaly_detection] classify done in "
              f"{time.time() - t_classify:.1f}s")

        # Collect UNCAPPED detector output first; per-type caps and the
        # combined ranking are derived afterwards. Capping per type BEFORE
        # combining (the old flow) made combined_ranked unfaithful: a
        # flooding type's dropped mid-severity tail left holes that a quiet
        # type's lower-severity rows ranked above. We now cap per type for
        # the per-type sheets but build combined from the PRE-cap rows + one
        # global cap.
        raw: Dict[str, pd.DataFrame] = {}
        detector_log: List[dict] = []
        for name, mod in _DETECTORS:
            print(f"[simple_anomaly_detection] running {name} ...")
            t1 = time.time()
            status = "ok"
            err = ""
            try:
                rows = mod.detect_all(per_slice, classify=classify_per_slice)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                status = "crashed"
                print(f"[simple_anomaly_detection] WARNING: {name} crashed: {err}")
                traceback.print_exc()
                rows = []
            frame = coerce_to_finding_frame(rows)
            # Apply the global severity floor here so every downstream sheet
            # (per-type caps, combined_ranked, combo summaries) inherits it.
            if MIN_SEVERITY > 0 and not frame.empty:
                sev = pd.to_numeric(frame["severity_score"], errors="coerce")
                frame = frame[sev >= MIN_SEVERITY].reset_index(drop=True)
            raw[name] = frame
            detector_log.append({"detector": name, "status": status,
                                 "duration_s": round(time.time() - t1, 2),
                                 "error": err})

        # Cross-detector suppression on the UNCAPPED frames: when site_network
        # explains a (variable, site) shift, drop individual standard_outlier
        # rows it subsumes (but keep individuals more extreme than the shift;
        # see _suppress_redundant_outliers).
        raw["standard_outlier"] = self._suppress_redundant_outliers(
            raw.get("standard_outlier"),
            raw.get("site_network"),
        )

        def _cap_by_severity(df: pd.DataFrame, n: int) -> pd.DataFrame:
            if df is None or df.empty:
                return df if df is not None else pd.DataFrame(columns=list(FINDING_COLUMNS))
            return (df.sort_values("severity_score", ascending=False, na_position="last")
                      .head(n).reset_index(drop=True))

        results: Dict[str, pd.DataFrame] = {}
        for entry in detector_log:
            name = entry["detector"]
            cap = self.per_type_caps.get(name, self.cap_per_type)
            capped = _cap_by_severity(raw.get(name), cap)
            results[name] = capped
            entry["n_flags"] = int(len(capped))
            print(f"[simple_anomaly_detection] {name}: {len(capped)} flags "
                  f"({entry['duration_s']:.1f}s)")

        # Build the combined headline. The cluster detectors contribute ONE
        # combo-summary row per (network, variable-combo) -- built from their
        # PRE-cap per-subject rows -- instead of their per-subject rows, so the
        # headline shows a spread of distinct 3D combinations rather than the
        # same few correlated variable-blocks repeated across many subjects. The
        # per-subject detail stays in the per-type sheets + the *_combos index;
        # every other detector contributes its rows unchanged.
        combined_parts: List[pd.DataFrame] = []
        for name, df in raw.items():
            if df is None or df.empty:
                continue
            if name in _COMBO_SUMMARY_TYPES:
                atype = (str(df["anomaly_type"].iloc[0])
                         if "anomaly_type" in df.columns else name)
                combo_rows = cluster_3d.combos_for_combined(
                    df.to_dict("records"), atype)
                combined_parts.append(coerce_to_finding_frame(combo_rows))
            else:
                combined_parts.append(df)
        if combined_parts:
            combined = pd.concat(combined_parts, ignore_index=True, sort=False)
            # Cap the remaining combo detector (correlative) per (network,
            # variable) BEFORE the severity sort/cap, so a flooding dominant
            # variable can't crowd out other signal. The cluster combos are
            # already one row per combo, so this is a no-op for them.
            n_before = len(combined)
            combined = self._diversify_combined(
                combined, self.combined_max_per_combo, _COMBO_ANOMALY_TYPES)
            if len(combined) < n_before:
                print(f"[simple_anomaly_detection] combined: capped combo rows "
                      f"to <= {self.combined_max_per_combo} per (network, combo) "
                      f"({n_before - len(combined)} rows deferred to per-type + "
                      f"combo-index sheets)")
            combined = (combined.sort_values("severity_score", ascending=False,
                                             na_position="last")
                        .head(self.combined_cap).reset_index(drop=True))
        else:
            combined = pd.DataFrame(columns=list(FINDING_COLUMNS))

        results["combined_ranked"] = combined
        # Stash the PRE-cap frames so the *_combos index counts every flagged
        # subject per combo, not just those surviving the per-type severity cap.
        self._raw_frames = raw
        self._detector_log = detector_log
        self._write_report(results)

        crashed = [r["detector"] for r in detector_log if r["status"] == "crashed"]
        if crashed:
            self._write_run_marker(f"completed_with_detector_crash:{','.join(crashed)}")
        else:
            self._write_run_marker("completed")
        print(f"[simple_anomaly_detection] done in {time.time() - t0:.1f}s; "
              f"{len(combined)} total flags.")
        return results

    @staticmethod
    def _suppress_redundant_outliers(std_df: pd.DataFrame,
                                     site_df: pd.DataFrame) -> pd.DataFrame:
        """Drop standard_outlier rows that a site_network row already explains
        AND that are no more extreme than that site row.

        site_network flags a per-SITE distribution shift; standard_outlier
        flags an INDIVIDUAL subject's extreme value. The two are not the same
        phenomenon, so we suppress an individual row only when its severity is
        <= the explaining site row's severity (both on the shared 0..100
        scale). A subject far beyond the site-level shift is kept -- a wide
        site IQR practically guarantees individuals worth keeping.

        Keyed on (network, variable, site_id) WITHOUT timepoint: site_network
        pools all timepoints into one network stack and emits timepoint=''
        rows, so suppression is intentionally cross-timepoint. (Re-adding
        timepoint to the key would never match the site rows' empty tp and
        would silently disable suppression entirely.) Pure-variable site rows
        only -- correlation-drift rows whose variable looks like 'A | B' are
        ignored.

        site_network now collapses multi-site rows to one row whose ``site_id``
        is "(N sites)" and whose ``sites`` column lists the concrete sites
        (see site_network._collapse_multisite). Expand that list back to the
        individual (network, variable, site) keys here so suppression still
        matches the standard_outlier rows' concrete sites; each site is capped
        by its OWN drift severity (the parallel ``site_severities`` column),
        NOT the collapsed row's group max -- otherwise an individual at a site
        that drifted only weakly would be over-suppressed. Fall back to the
        single ``site_id`` / ``severity_score`` for any uncollapsed row.
        """
        if std_df is None or std_df.empty or site_df is None or site_df.empty:
            return std_df if std_df is not None else pd.DataFrame()
        single_var = site_df[~site_df["variable"].astype(str).str.contains(r"\|", regex=True)]
        if single_var.empty:
            return std_df
        has_sites_col = "sites" in single_var.columns
        has_sev_col = "site_severities" in single_var.columns
        # Max explaining site-row severity per (network, variable, site_id).
        site_sev: Dict[Tuple[str, str, str], float] = {}
        sev_site = pd.to_numeric(single_var["severity_score"], errors="coerce").fillna(0.0)
        sites_list = (single_var["sites"].astype(str) if has_sites_col
                      else single_var["site_id"].astype(str))
        persite_sev_list = (single_var["site_severities"].astype(str) if has_sev_col
                            else [None] * len(single_var))

        def _clean(tok: str) -> str:
            t = tok.strip()
            return "" if t.lower() in ("nan", "none") else t

        for net, var, site_id, sites_str, persite_str, sev in zip(
            single_var["network"].astype(str),
            single_var["variable"].astype(str),
            single_var["site_id"].astype(str),
            sites_list,
            persite_sev_list,
            sev_site,
        ):
            # Zip the concrete sites with their own severities BEFORE filtering
            # so the two stay index-aligned. Rows that never collapsed (e.g.
            # cross-network) carry NaN here -> fall back to (site_id, sev).
            site_toks = [_clean(s) for s in str(sites_str).split(",")]
            sev_toks = ([_clean(s) for s in str(persite_str).split(",")]
                        if persite_str is not None else [])
            pairs = []
            for k, s in enumerate(site_toks):
                if not s:
                    continue
                site_s = float(sev)
                if k < len(sev_toks) and sev_toks[k]:
                    try:
                        site_s = float(sev_toks[k])
                    except ValueError:
                        site_s = float(sev)
                pairs.append((s, site_s))
            if not pairs:
                pairs = [(site_id, float(sev))]
            for site, site_s in pairs:
                key = (net, var, site)
                if site_s > site_sev.get(key, -1.0):
                    site_sev[key] = site_s
        if not site_sev:
            return std_df

        def _keep(r) -> bool:
            cap = site_sev.get((str(r["network"]), str(r["variable"]), str(r["site_id"])))
            if cap is None:
                return True  # no site row explains this triple
            try:
                row_sev = float(r["severity_score"])
            except (TypeError, ValueError):
                row_sev = 0.0
            return row_sev > cap  # keep only individuals MORE extreme than the site shift

        keep_mask = std_df.apply(_keep, axis=1)
        return std_df.loc[keep_mask].reset_index(drop=True)

    @staticmethod
    def _diversify_combined(combined: pd.DataFrame, cap_per_combo: int,
                            combo_types) -> pd.DataFrame:
        """Keep the combined headline from being dominated by a few variables.

        For the anomaly types in ``combo_types`` (the cluster detectors and
        correlative), keep at most ``cap_per_combo`` highest-severity rows per
        (network, variable). Deferred rows remain in the per-type sheet (itself
        bounded by cap_per_type, so this is full detail only while a detector's
        raw output fits that cap); the cluster detectors additionally surface
        the variable combo in their per-combo summary index. Rows from every
        OTHER detector pass through untouched -- their rows that share a variable
        label are distinct subjects with no fallback view, so capping them would
        hide real findings rather than de-duplicate.

        Correlative binary-binary joint-cell rows (variable "a=x & b=y") are
        EXEMPTED from the cap: they are distinct subjects sharing one rare cell
        (correlative._dedup_correlative leaves them un-collapsed for exactly this
        reason), not one variable flooding the list, and all carry the cell's
        single severity -- so capping them would drop arbitrary distinct subjects.
        ``cap_per_combo <= 0`` disables (returns ``combined`` unchanged).
        """
        if combined is None or combined.empty or cap_per_combo <= 0:
            return combined
        if not {"anomaly_type", "network", "variable", "severity_score"} <= set(combined.columns):
            return combined
        is_combo = combined["anomaly_type"].astype(str).isin(set(combo_types))
        # Exempt binary-binary joint-cell rows (the " & " marker is unique to
        # correlative._binary_binary_pairs; cluster combos use " | ").
        bb_mask = combined["variable"].astype(str).str.contains(" & ", regex=False)
        is_combo = is_combo & ~bb_mask
        if not is_combo.any():
            return combined
        combo_part = combined.loc[is_combo].copy()
        combo_part["_sev"] = pd.to_numeric(
            combo_part["severity_score"], errors="coerce").fillna(float("-inf"))
        combo_part["_net"] = combo_part["network"].astype(str)
        combo_part["_var"] = combo_part["variable"].astype(str)
        keep_idx = (combo_part.sort_values("_sev", ascending=False, kind="stable")
                    .groupby(["_net", "_var"], sort=False)
                    .head(cap_per_combo).index)
        # Keep every non-combo row + the capped combo rows, in the original order.
        mask = (~is_combo) | combined.index.isin(keep_idx)
        return combined.loc[mask]

    def _guard_paths(self) -> None:
        """Refuse to run if the output folder is inside the input folder.

        Without this, a future cleanup step (or a directory-tree tool the
        user runs against `output_path`) could touch the source CSVs.
        """
        in_norm = os.path.normcase(os.path.realpath(self.input_path))
        out_norm = os.path.normcase(os.path.realpath(self.output_path))
        if out_norm == in_norm or out_norm.startswith(in_norm + os.sep):
            raise RuntimeError(
                f"output_path ({self.output_path}) is inside input_path "
                f"({self.input_path}). Refusing to run -- pick an output "
                f"folder outside the input tree so the source CSVs cannot "
                f"be touched by output handling."
            )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _atomic_to_csv(self, df: pd.DataFrame, final_path: str) -> None:
        """Write to a unique tmp file then os.replace into place so a partial
        write never replaces a previous good output."""
        d, name = os.path.split(final_path)
        token = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
        tmp_path = os.path.join(d, f".{name}.{token}.tmp")
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, final_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def _build_run_status_df(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        load_log = getattr(self, "_load_log", [])
        detector_log = getattr(self, "_detector_log", [])
        rows: List[dict] = []
        for r in load_log:
            rows.append({"section": "load", **r})
        for r in detector_log:
            rows.append({"section": "detector", **r})
        rows.append({"section": "summary",
                     "loaded_slices": sum(1 for r in load_log if r["status"] == "loaded"),
                     "missing_slices": sum(1 for r in load_log if r["status"] == "missing"),
                     "crashed_detectors": sum(1 for r in detector_log if r["status"] == "crashed"),
                     "total_flags": sum(int(len(v)) for k, v in results.items() if k != "combined_ranked")})
        return pd.DataFrame(rows)

    def _build_combo_frames(self, results: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Per-combo summary index for the cluster detectors: one row per
        variable combo, ranked by how many subjects it flags at high severity
        (the ``n_subjects_sev_ge_*`` column), with the high-severity subjects
        listed (``cluster_3d.summarize_combos``).

        Additive -- kept SEPARATE from ``results`` so it never inflates the
        per-detector flag counts in summary.csv / run_status. Built from the
        PRE-cap per-subject frames (``self._raw_frames``) so ``n_subjects`` and
        the severities count EVERY flagged subject per combo, not just those
        surviving the per-type severity cap (which would undercount once a
        cluster detector's raw output exceeds cap_per_type); falls back to the
        capped ``results`` frames if the pre-cap stash is unavailable."""
        raw_frames = getattr(self, "_raw_frames", None)
        frames: Dict[str, pd.DataFrame] = {}
        for name in _COMBO_SUMMARY_TYPES:
            df = (raw_frames or {}).get(name)
            if df is None:
                df = results.get(name)
            if df is None or df.empty or "variable" not in df.columns:
                continue
            summary = cluster_3d.summarize_combos(df.to_dict("records"))
            if summary:
                frames[f"{name}_combos"] = pd.DataFrame(summary)
        return frames

    def _write_report(self, results: Dict[str, pd.DataFrame]) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        csv_dir = os.path.join(self.output_path, "by_type")
        os.makedirs(csv_dir, exist_ok=True)
        write_failures: List[str] = []

        for name, df in results.items():
            path = os.path.join(csv_dir, f"{name}.csv")
            try:
                self._atomic_to_csv(df, path)
            except Exception as e:
                print(f"  [write] WARN {name}.csv: {e}")
                write_failures.append(f"{name}.csv: {e}")

        # Additive per-combo summaries (cluster detectors). Written as their own
        # by_type CSVs + xlsx sheets; intentionally excluded from summary.csv so
        # the per-detector flag counts stay meaningful. The build is isolated
        # like every other write step here so an additive summary can never
        # abort the rest of the report (or leave the run marker stuck at
        # "started").
        try:
            combo_frames = self._build_combo_frames(results)
        except Exception as e:
            print(f"  [write] WARN combo summary: {e}")
            write_failures.append(f"combo_frames: {e}")
            combo_frames = {}
        for name, df in combo_frames.items():
            path = os.path.join(csv_dir, f"{name}.csv")
            try:
                self._atomic_to_csv(df, path)
            except Exception as e:
                print(f"  [write] WARN {name}.csv: {e}")
                write_failures.append(f"{name}.csv: {e}")

        summary = pd.DataFrame(
            [(k, len(v)) for k, v in results.items() if k != "combined_ranked"],
            columns=["anomaly_type", "n_flags"],
        ).sort_values("n_flags", ascending=False)
        try:
            self._atomic_to_csv(summary, os.path.join(self.output_path, "summary.csv"))
        except Exception as e:
            print(f"  [write] WARN summary.csv: {e}")
            write_failures.append(f"summary.csv: {e}")

        status_df = self._build_run_status_df(results)
        try:
            self._atomic_to_csv(status_df, os.path.join(self.output_path, "run_status.csv"))
        except Exception as e:
            print(f"  [write] WARN run_status.csv: {e}")

        # PID + uuid in temp filename so concurrent runs don't collide.
        xlsx_path = os.path.join(self.output_path, "anomaly_report.xlsx")
        tmp_path = os.path.join(
            self.output_path,
            f".anomaly_report.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.xlsx",
        )
        try:
            with pd.ExcelWriter(tmp_path, engine="openpyxl") as xw:
                results["combined_ranked"].to_excel(xw, sheet_name="combined_ranked", index=False)
                for name, df in results.items():
                    if name == "combined_ranked":
                        continue
                    sheet = name[:31]
                    try:
                        df.to_excel(xw, sheet_name=sheet, index=False)
                    except Exception as e:
                        print(f"  [write] WARN sheet {sheet}: {e}")
                        write_failures.append(f"sheet {sheet}: {e}")
                for name, df in combo_frames.items():
                    sheet = name[:31]
                    try:
                        df.to_excel(xw, sheet_name=sheet, index=False)
                    except Exception as e:
                        print(f"  [write] WARN sheet {sheet}: {e}")
                        write_failures.append(f"sheet {sheet}: {e}")
                summary.to_excel(xw, sheet_name="summary", index=False)
                status_df.to_excel(xw, sheet_name="run_status", index=False)
            os.replace(tmp_path, xlsx_path)
            print(f"  [write] {xlsx_path}")
        except ImportError:
            print("  [write] openpyxl not installed; only CSVs written. "
                  "`pip install openpyxl` to get the xlsx workbook.")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        except Exception as e:
            print(f"  [write] WARN xlsx: {e}")
            write_failures.append(f"xlsx: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        if write_failures:
            self._write_run_marker(f"completed_with_write_warnings:{len(write_failures)}")

    def _write_run_marker(self, status: str) -> None:
        """A tiny file that records the most recent run's outcome. Lets a
        reviewer tell at a glance whether the report next to it is fresh,
        partial, or a no-data placeholder."""
        path = os.path.join(self.output_path, "RUN_STATUS.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    f"status={status}\n"
                    f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                    f"input_path={self.input_path}\n"
                    f"output_path={self.output_path}\n"
                    f"cap_per_type={self.cap_per_type}\n"
                )
        except Exception:
            pass

    def _write_no_data_marker(self) -> None:
        """Wipe stale report artifacts and replace with an explicit
        no-data marker so a reviewer doesn't mistake an old anomaly_report
        for the current run."""
        for fname in ("anomaly_report.xlsx", "summary.csv", "run_status.csv"):
            fp = os.path.join(self.output_path, fname)
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
        by_type = os.path.join(self.output_path, "by_type")
        if os.path.isdir(by_type):
            for fname in os.listdir(by_type):
                if fname.endswith(".csv"):
                    try:
                        os.remove(os.path.join(by_type, fname))
                    except OSError:
                        pass
        marker = os.path.join(self.output_path, "NO_DATA.txt")
        try:
            with open(marker, "w", encoding="utf-8") as f:
                f.write(
                    "No CSV slices matched any (network, timepoint) under "
                    f"input_path={self.input_path}\n"
                    f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                )
        except Exception:
            pass
