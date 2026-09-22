"""SimpleAnomalyDetector - load AMPSCZ combined CSVs, run the configured
detectors on Clinical-measures variables, and write one ranked Excel workbook
plus per-type CSVs.

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
    timeline_anomaly,
    cluster_3d,
    cluster_3d_longitudinal,
    rater_effect,
)
from .common import (
    FINDING_COLUMNS, classify_columns, coerce_to_finding_frame,
)
from .clinical_variables import (
    ClinicalVariableContractError,
    filter_findings_to_clinical,
    load_clinical_variable_catalog,
    scope_slices_to_clinical,
)
from .selection import select_diverse_findings


_DETECTORS = [
    ("standard_outlier", standard_outlier),
    ("time_series", time_series),
    ("correlative", correlative),
    ("site_network", site_network),
    ("format_string", format_string),
    ("whole_subject_site", whole_subject_site),
    ("date_anomaly", date_anomaly),
    ("timeline_anomaly", timeline_anomaly),
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
DEFAULT_COMBINED_MAX_PER_COMBO = 10

# Reviewer-facing diversity limits. These apply across each entire tab and to
# every underlying variable leg, so ``a`` cannot evade a cap by recurring as
# ``a | b``, ``a | c``, ... or by repeating once per network.
# The combined headline is intentionally tighter than detail tabs.
DEFAULT_MAX_ROWS_PER_VARIABLE = 10
DEFAULT_MAX_ROWS_PER_SUBJECT = 10
DEFAULT_COMBINED_MAX_PER_VARIABLE = 10

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
        max_rows_per_variable : per-tab quota for each underlying variable
            leg in a detector tab (pair/triple labels count against every leg)
        max_rows_per_subject : per-network subject detail quota per tab
        combined_max_per_variable : tighter variable-leg quota for the combined
            headline; defaults to combined_max_per_combo for compatibility
        dependencies_path : folder holding the metadata contracts
            (missingness_domain_forms.json, grouped_variables.json,
            important_form_vars.json, and a data_dictionary/ subfolder).
            Explicit arg > FORMQC_ANOMALY_DEPENDENCIES > the installed
            package's own dependencies folder. The per-file arguments below
            still win individually where supplied.
        clinical_only : restrict production detection and every report output
            to variables mapped to the study's Clinical measures domain and
            confirmed by the REDCap data dictionary to hold measurement data
        data_dictionary_path : REDCap data dictionary CSV; discovered beside
            the repository when omitted, and required whenever clinical_only
            is set
    """

    def __init__(
        self,
        input_path: str = None,
        output_path: str = None,
        cap_per_type: int = 5000,
        networks: Tuple[str, ...] = NETWORKS,
        timepoints: Tuple[str, ...] = TIMEPOINTS,
        max_rows_per_csv: int = 0,
        combined_cap: int = 0,
        combined_max_per_combo: int = DEFAULT_COMBINED_MAX_PER_COMBO,
        per_type_caps: Dict[str, int] = None,
        max_rows_per_variable: int = DEFAULT_MAX_ROWS_PER_VARIABLE,
        max_rows_per_subject: int = DEFAULT_MAX_ROWS_PER_SUBJECT,
        combined_max_per_variable: int = None,
        dependencies_path: str = None,
        clinical_only: bool = True,
        clinical_domain_map_path: str = None,
        grouped_variables_path: str = None,
        important_form_vars_path: str = None,
        data_dictionary_path: str = None,
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
        # Same arg > env var precedence as the I/O paths, but this one has a
        # sane default (the installed package's own dependencies folder), so an
        # unset value is not an error.
        dependencies_path = (dependencies_path
                             or os.environ.get("FORMQC_ANOMALY_DEPENDENCIES"))
        if dependencies_path:
            dependencies_path = os.path.abspath(dependencies_path)
            if not os.path.isdir(dependencies_path):
                raise ValueError(
                    "dependencies_path is not a directory: "
                    f"{dependencies_path}")
        self.dependencies_path = dependencies_path
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
        self.per_type_caps = {**PER_TYPE_CAP_OVERRIDES, **(per_type_caps or {})}
        self.max_rows_per_variable = int(max_rows_per_variable)
        self.max_rows_per_subject = int(max_rows_per_subject)
        self.combined_max_per_variable = int(
            self.combined_max_per_combo if combined_max_per_variable is None
            else combined_max_per_variable)
        self.clinical_only = bool(clinical_only)
        self.clinical_domain_map_path = clinical_domain_map_path
        self.grouped_variables_path = grouped_variables_path
        self.important_form_vars_path = important_form_vars_path
        self.data_dictionary_path = data_dictionary_path

    def _form_contract_path(self):
        """important_form_vars.json, honouring dependencies_path.

        Returns None when neither is set so the detector keeps its own default.
        """
        if self.important_form_vars_path:
            return self.important_form_vars_path
        if self.dependencies_path:
            return os.path.join(self.dependencies_path,
                                "important_form_vars.json")
        return None

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

                # Every downstream detector assumes one physical row per
                # participant within a timepoint. Blank IDs can join unrelated
                # records longitudinally; duplicate IDs can repeat one anomaly
                # or let an arbitrary first row define a trajectory. Fail
                # closed for those ambiguous rows and persist the counts.
                rows_read = int(len(df))
                before_exact = len(df)
                df = df.drop_duplicates(keep="first").copy()
                exact_duplicate_rows = int(before_exact - len(df))
                sid = df["subjectid"].astype(str).str.strip()
                # Store the canonical value used by the identity checks.  A
                # leading/trailing space at one visit must not prevent that
                # participant from joining their other timeline records or
                # corrupt the site prefix shown in findings.
                df["subjectid"] = sid
                blank = sid.eq("") | sid.str.lower().isin(
                    {"nan", "none", "na", "n/a"})
                duplicate_ids = set(sid[~blank & sid.duplicated(keep=False)])
                ambiguous = blank | sid.isin(duplicate_ids)
                blank_rows = int(blank.sum())
                duplicate_subjectids = int(len(duplicate_ids))
                duplicate_rows = int((~blank & sid.isin(duplicate_ids)).sum())
                if ambiguous.any():
                    df = df.loc[~ambiguous].copy().reset_index(drop=True)
                    print(f"  [load] WARN {network}/{tp}: excluded {blank_rows} "
                          f"blank-ID row(s) and {duplicate_rows} row(s) across "
                          f"{duplicate_subjectids} duplicate subject id(s)")
                rows_after_identity = int(len(df))
                rows_sampled_out = 0
                if self.max_rows_per_csv and len(df) > self.max_rows_per_csv:
                    # Debug-only sample after identity validation so sampling
                    # cannot accidentally retain one side of a conflicting ID.
                    df = df.sample(self.max_rows_per_csv,
                                   random_state=0).reset_index(drop=True)
                    rows_sampled_out = rows_after_identity - int(len(df))
                slices[(network, tp)] = df
                load_log.append({"network": network, "timepoint": tp,
                                 "path": path, "rows_read": rows_read,
                                 "rows_after_identity": rows_after_identity,
                                 "rows": int(len(df)),
                                 "rows_sampled_out": rows_sampled_out,
                                 "exact_duplicate_rows_collapsed": exact_duplicate_rows,
                                 "blank_id_rows_excluded": blank_rows,
                                 "duplicate_subjectids_excluded": duplicate_subjectids,
                                 "duplicate_rows_excluded": duplicate_rows,
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

        # Production reports are fail-closed to the study's explicit
        # ``Clinical measures`` domain.  Scope the DATA before classification
        # so nonclinical fields cannot influence correlations, clusters, site
        # statistics, or whole-subject/site composites.  Interview dates and
        # rater ids survive only as detector context and are blocked again at
        # the finding-row boundary below.
        if self.clinical_only:
            try:
                catalog = load_clinical_variable_catalog(
                    domain_map_path=self.clinical_domain_map_path,
                    grouped_variables_path=self.grouped_variables_path,
                    important_form_vars_path=self.important_form_vars_path,
                    data_dictionary_path=self.data_dictionary_path,
                    dependencies_path=self.dependencies_path,
                )
                per_slice, scope_audit = scope_slices_to_clinical(
                    per_slice, catalog)
            except Exception:
                self._clear_report_artifacts()
                self._write_run_marker("failed_clinical_variable_contract")
                raise
            self._clinical_catalog = catalog
            self._clinical_scope_audit = scope_audit
            observed_clinical = int((
                scope_audit.get("disposition", pd.Series(dtype=str))
                == "retained_clinical").sum())
            observed_context = int((
                scope_audit.get("disposition", pd.Series(dtype=str))
                == "retained_context").sum())
            observed_excluded = int((
                scope_audit.get("disposition", pd.Series(dtype=str))
                == "excluded").sum())
            if observed_clinical == 0:
                self._clear_report_artifacts()
                self._write_run_marker("failed_no_observed_clinical_variables")
                raise ClinicalVariableContractError(
                    "Clinical-only anomaly detection found zero observed "
                    "Clinical measures variables in the loaded CSV slices. "
                    "The export schema may be wrong or newer than the variable "
                    "mapping; see RUN_STATUS.txt and update the dependencies "
                    "instead of falling back to all variables.")
            print(
                "[simple_anomaly_detection] clinical scope: "
                f"{observed_clinical} observed clinical variable(s), "
                f"{observed_context} context field(s), "
                f"{observed_excluded} nonclinical/unmapped field(s) excluded "
                f"across {len(per_slice)} slice(s)"
            )
        else:
            self._clinical_catalog = None
            self._clinical_scope_audit = pd.DataFrame()

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
            if self.clinical_only and name in {
                    "date_anomaly", "timeline_anomaly"}:
                # Dates are retained solely as elapsed-time context for the
                # longitudinal detector.  The QC pipeline has a dedicated Date
                # Report; operational date fields are not clinical measures.
                # Keep timeline_anomaly's policy explicit as well: running it
                # and then silently filtering every auxiliary date leg would
                # waste work and misleadingly report status="ok" with zero.
                rows = []
                status = "skipped_clinical_scope"
            else:
                try:
                    detector_kwargs = {"classify": classify_per_slice}
                    if name == "timeline_anomaly":
                        detector_kwargs["important_form_vars_path"] = (
                            self._form_contract_path())
                    rows = mod.detect_all(per_slice, **detector_kwargs)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    status = "crashed"
                    print(f"[simple_anomaly_detection] WARNING: {name} crashed: {err}")
                    traceback.print_exc()
                    rows = []
            frame = coerce_to_finding_frame(rows)
            n_filtered_nonclinical = 0
            if self.clinical_only:
                frame, n_filtered_nonclinical = filter_findings_to_clinical(
                    frame, self._clinical_catalog)
            # Apply the global severity floor here so every downstream sheet
            # (per-type caps, combined_ranked, combo summaries) inherits it.
            if MIN_SEVERITY > 0 and not frame.empty:
                sev = pd.to_numeric(frame["severity_score"], errors="coerce")
                frame = frame[sev >= MIN_SEVERITY].reset_index(drop=True)
            raw[name] = frame
            detector_log.append({"detector": name, "status": status,
                                 "duration_s": round(time.time() - t1, 2),
                                 "error": err,
                                 "n_filtered_nonclinical": n_filtered_nonclinical})

        # Cross-detector suppression on the UNCAPPED frames: when site_network
        # explains a (variable, site) shift, drop individual standard_outlier
        # rows it subsumes (but keep individuals more extreme than the shift;
        # see _suppress_redundant_outliers).
        raw["standard_outlier"] = self._suppress_redundant_outliers(
            raw.get("standard_outlier"),
            raw.get("site_network"),
        )

        results: Dict[str, pd.DataFrame] = {}
        for entry in detector_log:
            name = entry["detector"]
            cap = self.per_type_caps.get(name, self.cap_per_type)
            candidates = raw.get(name)
            selected = select_diverse_findings(
                candidates,
                row_cap=cap,
                max_per_variable=self.max_rows_per_variable,
                max_per_subject=self.max_rows_per_subject,
            )
            results[name] = selected
            n_candidates = int(len(candidates)) if candidates is not None else 0
            entry["n_candidates"] = n_candidates
            entry["n_flags"] = int(len(selected))
            entry["n_omitted"] = n_candidates - int(len(selected))
            entry["row_cap"] = int(cap)
            entry["max_rows_per_variable"] = self.max_rows_per_variable
            entry["max_rows_per_subject"] = self.max_rows_per_subject
            print(f"[simple_anomaly_detection] {name}: {len(selected)} review rows "
                  f"from {n_candidates} candidates ({entry['n_omitted']} "
                  f"duplicate/invalid/quota-deferred; "
                  f"{entry['duration_s']:.1f}s)")

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
            n_before = len(combined)
            combined = select_diverse_findings(
                combined,
                row_cap=self.combined_cap,
                max_per_variable=self.combined_max_per_variable,
                max_per_subject=self.max_rows_per_subject,
                cap_pseudo_variables=True,
            )
            print(f"[simple_anomaly_detection] combined: {len(combined)} review "
                  f"rows from {n_before} candidates after universal variable, "
                  f"subject, duplicate, and issue-round limits")
        else:
            combined = pd.DataFrame(columns=list(FINDING_COLUMNS))

        results["combined_ranked"] = combined
        # Stash the PRE-cap frames so the *_combos index counts every flagged
        # subject per combo, not just those surviving the per-type severity cap.
        self._raw_frames = raw
        self._detector_log = detector_log
        write_failures = self._write_report(results)

        crashed = [r["detector"] for r in detector_log if r["status"] == "crashed"]
        if write_failures:
            self._write_run_marker(f"completed_with_write_warnings:{len(write_failures)}")
        elif crashed:
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

        Keyed on (network, timepoint, variable, site_id). site_network now
        estimates distributions within each timepoint; allowing a baseline
        site shift to suppress an unrelated month-12 cell would hide evidence.
        Pure-variable site rows only -- correlation-drift rows whose variable
        looks like 'A | B' are ignored.

        site_network now collapses multi-site rows to one row whose ``site_id``
        is "(N sites)" and whose ``sites`` column lists the concrete sites
        (see site_network._collapse_multisite). Expand that list back to the
        individual (network, variable, site) keys here so suppression still
        matches the standard_outlier rows' concrete sites; each site is capped
        by its OWN drift severity (the parallel ``site_severities`` column),
        NOT the collapsed row's group max -- otherwise an individual at a site
        that drifted only weakly would be over-suppressed. Fall back to the
        single ``site_id`` / ``severity_score`` for any uncollapsed row.

        Only the MEDIAN-shift sub-test may cap a value outlier. site_network
        emits three single-variable drift kinds -- median-shift, spread-shift
        and missingness-shift -- and all three land here (the ``|`` filter only
        removes correlation rows). But a site's wide IQR or high missingness
        says nothing about whether an individual's PRESENT value is a typo, and
        both reach severity ~95 trivially; letting them build the cap would
        delete genuine value outliers at any under-collecting / wide-spread
        site. Restrict the cap to median-shift rows (the sub-test whose signal
        -- the site's central value is displaced -- is the only one an
        individual extreme value can be "explained by").
        """
        if std_df is None or std_df.empty or site_df is None or site_df.empty:
            return std_df if std_df is not None else pd.DataFrame()
        single_var = site_df[~site_df["variable"].astype(str).str.contains(r"\|", regex=True)]
        if "method" in single_var.columns:
            single_var = single_var[
                single_var["method"].astype(str) == "cross-site MAD on site medians"
            ]
        if single_var.empty:
            return std_df
        has_sites_col = "sites" in single_var.columns
        has_sev_col = "site_severities" in single_var.columns
        # Max explaining site-row severity per
        # (network, timepoint, variable, site_id).
        site_sev: Dict[Tuple[str, str, str, str], float] = {}
        sev_site = pd.to_numeric(single_var["severity_score"], errors="coerce").fillna(0.0)
        sites_list = (single_var["sites"].astype(str) if has_sites_col
                      else single_var["site_id"].astype(str))
        persite_sev_list = (single_var["site_severities"].astype(str) if has_sev_col
                            else [None] * len(single_var))

        def _clean(tok: str) -> str:
            t = tok.strip()
            return "" if t.lower() in ("nan", "none") else t

        tp_values = (single_var["timepoint"].astype(str)
                     if "timepoint" in single_var.columns
                     else [""] * len(single_var))
        for net, tp, var, site_id, sites_str, persite_str, sev in zip(
            single_var["network"].astype(str),
            tp_values,
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
                key = (net, tp, var, site)
                if site_s > site_sev.get(key, -1.0):
                    site_sev[key] = site_s
        if not site_sev:
            return std_df

        def _keep(r) -> bool:
            cap = site_sev.get((str(r["network"]), str(r.get("timepoint", "")),
                                str(r["variable"]), str(r["site_id"])))
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
        scope_audit = getattr(self, "_clinical_scope_audit", pd.DataFrame())
        rows: List[dict] = []
        for r in load_log:
            rows.append({"section": "load", **r})
        for r in detector_log:
            rows.append({"section": "detector", **r})
        catalog = getattr(self, "_clinical_catalog", None)
        rows.append({"section": "summary",
                     "clinical_only": self.clinical_only,
                     "clinical_domain": ("Clinical measures"
                                         if self.clinical_only else "unrestricted"),
                     # Printed rather than asserted: the allowlist size moves
                     # whenever the data dictionary is refreshed, and this is
                     # how that drift becomes visible.
                     "allowlist_variables": (
                         len(catalog.clinical_variables) if catalog else 0),
                     "dictionary_excluded_variables": (
                         catalog.dictionary_excluded_count if catalog else 0),
                     "data_dictionary": (
                         catalog.data_dictionary_path if catalog else ""),
                     "observed_clinical_variables": int((
                         scope_audit.get("disposition", pd.Series(dtype=str))
                         == "retained_clinical").sum()),
                     "observed_context_variables": int((
                         scope_audit.get("disposition", pd.Series(dtype=str))
                         == "retained_context").sum()),
                     "observed_excluded_variables": int((
                         scope_audit.get("disposition", pd.Series(dtype=str))
                         == "excluded").sum()),
                     "loaded_slices": sum(1 for r in load_log if r["status"] == "loaded"),
                     "missing_slices": sum(1 for r in load_log if r["status"] == "missing"),
                     "crashed_detectors": sum(1 for r in detector_log if r["status"] == "crashed"),
                     "total_candidates": sum(int(r.get("n_candidates", 0))
                                             for r in detector_log),
                     "total_filtered_nonclinical_findings": sum(int(
                         r.get("n_filtered_nonclinical", 0))
                         for r in detector_log),
                     "total_written": sum(int(len(v)) for k, v in results.items()
                                          if k != "combined_ranked"),
                     "total_omitted": sum(int(r.get("n_omitted", 0))
                                          for r in detector_log),
                     "combined_written": int(len(results.get(
                         "combined_ranked", pd.DataFrame())))})
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
                combo_df = pd.DataFrame(summary)
                combo_df["anomaly_type"] = f"{name}_combo_summary"
                combo_df["variable"] = combo_df["variables"]
                combo_df["severity_score"] = combo_df["max_severity"]
                combo_df["raw_score"] = combo_df["n_subjects"]
                combo_df["subjectid"] = "(combo summary)"
                selected = select_diverse_findings(
                    combo_df,
                    row_cap=self.per_type_caps.get(name, self.cap_per_type),
                    max_per_variable=self.max_rows_per_variable,
                    max_per_subject=0,
                    cap_pseudo_variables=True,
                )
                frames[f"{name}_combos"] = selected.drop(
                    columns=["anomaly_type", "variable", "severity_score",
                             "raw_score", "subjectid"], errors="ignore")
        return frames

    def _write_report(self, results: Dict[str, pd.DataFrame]) -> List[str]:
        os.makedirs(self.output_path, exist_ok=True)
        no_data = os.path.join(self.output_path, "NO_DATA.txt")
        if os.path.exists(no_data):
            try:
                os.remove(no_data)
            except OSError:
                pass
        csv_dir = os.path.join(self.output_path, "by_type")
        os.makedirs(csv_dir, exist_ok=True)
        audit_dir = os.path.join(self.output_path, "audit_candidates")
        os.makedirs(audit_dir, exist_ok=True)
        write_failures: List[str] = []

        # One row per observed source column.  This is deliberately outside
        # anomaly_report.xlsx: the workbook contains clinical finding rows
        # only, while this sidecar explains exactly what was retained, used as
        # context, or excluded.
        if self.clinical_only:
            scope_path = os.path.join(
                self.output_path, "clinical_variable_scope.csv")
            try:
                self._atomic_to_csv(
                    getattr(self, "_clinical_scope_audit", pd.DataFrame()),
                    scope_path,
                )
            except Exception as e:
                print(f"  [write] WARN clinical_variable_scope.csv: {e}")
                write_failures.append(f"clinical_variable_scope.csv: {e}")
        else:
            stale_scope = os.path.join(
                self.output_path, "clinical_variable_scope.csv")
            if os.path.exists(stale_scope):
                try:
                    os.remove(stale_scope)
                except OSError as e:
                    write_failures.append(
                        f"stale clinical_variable_scope.csv: {e}")

        for name, df in results.items():
            path = os.path.join(csv_dir, f"{name}.csv")
            try:
                self._atomic_to_csv(df, path)
            except Exception as e:
                print(f"  [write] WARN {name}.csv: {e}")
                write_failures.append(f"{name}.csv: {e}")

        # Candidate evidence before reviewer-facing duplicate/entity/diversity
        # selection. Kept outside the workbook so tabs stay concise while every
        # deferred row remains auditable.
        for name, df in getattr(self, "_raw_frames", {}).items():
            path = os.path.join(audit_dir, f"{name}_candidates.csv")
            try:
                self._atomic_to_csv(df, path)
            except Exception as e:
                print(f"  [write] WARN audit candidate {name}.csv: {e}")
                write_failures.append(f"audit/{name}.csv: {e}")

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

        expected_by_type = {f"{name}.csv" for name in results} | {
            f"{name}.csv" for name in combo_frames}
        for fname in os.listdir(csv_dir):
            if fname.endswith(".csv") and fname not in expected_by_type:
                try:
                    os.remove(os.path.join(csv_dir, fname))
                except OSError as e:
                    write_failures.append(f"stale by_type/{fname}: {e}")
        expected_audit = {f"{name}_candidates.csv" for name in getattr(
            self, "_raw_frames", {})}
        for fname in os.listdir(audit_dir):
            if fname.endswith(".csv") and fname not in expected_audit:
                try:
                    os.remove(os.path.join(audit_dir, fname))
                except OSError as e:
                    write_failures.append(f"stale audit/{fname}: {e}")
        for name, df in combo_frames.items():
            path = os.path.join(csv_dir, f"{name}.csv")
            try:
                self._atomic_to_csv(df, path)
            except Exception as e:
                print(f"  [write] WARN {name}.csv: {e}")
                write_failures.append(f"{name}.csv: {e}")

        log_by_name = {r.get("detector"): r for r in getattr(
            self, "_detector_log", [])}
        summary_rows = []
        for name, df in results.items():
            if name == "combined_ranked":
                continue
            log = log_by_name.get(name, {})
            n_candidates = int(log.get("n_candidates", len(df)))
            n_written = int(len(df))
            summary_rows.append({
                "anomaly_type": name,
                "n_candidates": n_candidates,
                "n_filtered_nonclinical": int(log.get(
                    "n_filtered_nonclinical", 0)),
                "n_written": n_written,
                "n_omitted": n_candidates - n_written,
                "row_cap": log.get("row_cap", self.cap_per_type),
                "max_rows_per_variable": self.max_rows_per_variable,
                "max_rows_per_subject": self.max_rows_per_subject,
            })
        summary = pd.DataFrame(summary_rows).sort_values(
            "n_candidates", ascending=False)
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
            write_failures.append(f"run_status.csv: {e}")

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

        return write_failures

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
                    f"max_rows_per_variable={self.max_rows_per_variable}\n"
                    f"max_rows_per_subject={self.max_rows_per_subject}\n"
                    f"combined_max_per_variable={self.combined_max_per_variable}\n"
                    f"clinical_only={self.clinical_only}\n"
                    f"clinical_domain={'Clinical measures' if self.clinical_only else 'unrestricted'}\n"
                )
                catalog = getattr(self, "_clinical_catalog", None)
                if catalog is not None:
                    f.write(
                        f"allowlist_variables={len(catalog.clinical_variables)}\n"
                        f"dictionary_excluded_variables={catalog.dictionary_excluded_count}\n"
                        f"data_dictionary={catalog.data_dictionary_path}\n"
                    )
        except Exception:
            pass

    def _clear_report_artifacts(self) -> None:
        """Remove only generated report artifacts from an unsuccessful run.

        This prevents an older unrestricted workbook or candidate CSV from
        remaining beside a failure marker and being mistaken for the current
        clinical-only result.
        """
        for fname in ("anomaly_report.xlsx", "summary.csv", "run_status.csv",
                      "clinical_variable_scope.csv", "NO_DATA.txt"):
            fp = os.path.join(self.output_path, fname)
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
        for subdir in ("by_type", "audit_candidates"):
            folder = os.path.join(self.output_path, subdir)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.endswith(".csv"):
                    try:
                        os.remove(os.path.join(folder, fname))
                    except OSError:
                        pass

    def _write_no_data_marker(self) -> None:
        """Wipe stale report artifacts and replace with an explicit
        no-data marker so a reviewer doesn't mistake an old anomaly_report
        for the current run."""
        self._clear_report_artifacts()
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
