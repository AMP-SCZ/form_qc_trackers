import os
import sys
import json
import pandas as pd
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)

from utils.utils import Utils

"""
Produce a long-format, numeric-only parquet companion to the existing
wide multi-TP CSV, suitable as input to LongitudinalAnomalyChecks.

INPUT (does NOT modify):
  combined_csv_path/AMPSCZ-combined-redcap_<tp>_<NETWORK>-day1to1.csv
    — the same per-timepoint combined REDCap CSVs MultiTPDataCollector
    reads. Reading these directly (rather than the narrow wide
    dependencies/multi_tp_{network}_combined.csv) is what makes the
    long parquet broad enough for anomaly detection. The wide CSV is
    preserved untouched.
  dependencies/variable_type_distributions.json
  dependencies/grouped_variables.json   (uses 'var_forms', 'blood_vars')
  dependencies/subject_info.json        (cohort lookup)

OUTPUT (atomic, tmp + os.replace):
  dependencies/multi_tp_{network}_numeric_long.parquet

ROW SEMANTICS:
  The long parquet contains BOTH valid-numeric rows AND missing-code /
  sentinel rows, for full auditability of every (subject, tp, var)
  observation. Downstream consumers (LongitudinalAnomalyChecks) MUST
  filter to `value_numeric.notna() & ~is_missing_code` before any
  scoring — failing to do so would corrupt residual statistics.

  `value` preserves the cell text as parsed from the CSV with
  dtype=str and keep_default_na=False. dtype=str disables pandas'
  numeric inference, so leading zeros / trailing zeros / scientific
  notation in the parsed text are kept (without dtype=str, '042'
  would become int 42 and round-trip as '42'). This is preservation
  of the parsed cell text — not a literal byte-equivalence guarantee
  against the raw file, since the CSV reader may still normalize
  line endings, BOMs, and quoted-field escapes per its standard
  rules.

  `is_missing_code` is True iff the stripped string equals one of the
  project missing codes (-3, -9, -99, 999, 1909-09-09 family, plus
  blank/'nan'/'NA' sentinel variants). The set is derived from
  Utils.missing_code_set + a local extension for the blank/sentinel
  forms, since Utils does not include those.

  `keep_non_numeric_audit_rows` (config; default True) controls
  whether observations whose raw value is neither a project missing
  code nor numerically coercible (e.g., a free-text entry that
  slipped past the population-wide ratio gate) are kept in the long
  parquet. When True, such rows appear with is_missing_code=False
  and value_numeric=NaN, and the detector still filters them out via
  `value_numeric.notna() & ~is_missing_code`. When False, they are
  dropped entirely. Default is True for full auditability.

FAILURE MODES (no silent fallbacks):
  - any required per-tp combined CSV missing → FileNotFoundError
  - any required dependency JSON empty/unreadable → RuntimeError
  - subjectid column missing from any per-tp CSV → RuntimeError
  - zero variables pass metadata-only pruning → RuntimeError
  - zero variables pass distinct-values gate → RuntimeError

INTENTIONAL TWO-PASS CSV SCAN:
  Pass 1 (`_refine_eligibility`) reads each per-tp CSV with usecols
  restricted to candidate variables to compute distinct-values per
  variable. Pass 2 (`_melt_one_tp`) re-reads with the final eligible
  set and melts to long. Two passes (versus buffering everything in
  memory between classification and melt) was chosen for memory
  safety at AMP-SCZ scale (~thousands of subjects × ~16 timepoints
  × hundreds of numeric variables) and clearer eligibility
  determination — eligibility is finalized once before any long row
  is emitted.

DESIGN HOOKS RESERVED FOR LATER:
  - config['longitudinal_anomaly']['data_dictionary_path']
    When non-empty, the categorical-coded heuristic in
    `_refine_eligibility` could be replaced with a field_type lookup.
    v1 ignores this and uses conservative numeric discovery only.
  - `event_date` column is left blank in v1. Populating it requires
    either re-reading per-tp CSVs to look up interview_date_var per
    form (extra I/O) or extending MultiTPDataCollector (forbidden in
    N1). Schema column reserved so a v2 fill-in is non-breaking.
  - `source_form` is best-effort: filled from
    grouped_variables['var_forms'] when present, blank otherwise.
"""


class MultiTPLongFormatProducer:
    """
    Read per-tp combined REDCap CSVs, apply numeric-only inclusion
    filters, melt to long format, write atomic per-network parquet.
    """

    def __init__(self, config_dict=None):
        self.utils = Utils()
        if config_dict is None:
            with open(f'{self.utils.absolute_path}/config.json',
                      'r') as f:
                config_dict = json.load(f)
        self.config_info = config_dict
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depen_path = self.config_info['paths']['dependencies_path']

        lon_cfg = self.config_info.get('longitudinal_anomaly', {})
        self.networks = list(
            lon_cfg.get('networks', ['PRONET', 'PRESCIENT']))
        self.min_obs_per_variable = int(
            lon_cfg.get('min_observations_per_variable', 30))
        self.min_distinct_values = int(
            lon_cfg.get('min_distinct_values', 5))
        self.exclude_integer_low_cardinality = bool(
            lon_cfg.get('exclude_integer_low_cardinality', True))
        self.max_integer_distinct_for_categorical = int(
            lon_cfg.get('max_integer_distinct_for_categorical', 6))
        self.keep_non_numeric_audit_rows = bool(
            lon_cfg.get('keep_non_numeric_audit_rows', True))
        self.excluded_name_patterns = [
            p.lower() for p in lon_cfg.get(
                'excluded_name_patterns',
                ['comment', 'note', 'describe',
                 '_specify', '_other'])
        ]
        # Reserved hook for future data-dictionary-driven field_type
        # filtering; v1 does not consult any data dictionary.
        self.data_dictionary_path = lon_cfg.get(
            'data_dictionary_path', '')

        # Required dependency JSONs. Utils.load_dependency_json returns
        # {} on JSONDecodeError; treat empty as fatal so a corrupt or
        # missing dep cannot silently produce a detector-corrupting
        # empty long parquet.
        self.var_type_dists = self.utils.load_dependency_json(
            'variable_type_distributions.json')
        if not self.var_type_dists:
            raise RuntimeError(
                "FATAL: variable_type_distributions.json is empty "
                "or unreadable. Run "
                "process_variables.MultiTPDataCollector first."
            )
        self.grouped_vars = self.utils.load_dependency_json(
            'grouped_variables.json')
        if not self.grouped_vars:
            raise RuntimeError(
                "FATAL: grouped_variables.json is empty or "
                "unreadable. Run process_variables first."
            )
        self.subject_info = self.utils.load_dependency_json(
            'subject_info.json')
        if not self.subject_info:
            raise RuntimeError(
                "FATAL: subject_info.json is empty or unreadable. "
                "Run process_variables.CollectSubjectInfo first."
            )
        # Best-effort variable→form lookup. Always present per
        # grouped_variables.json schema (verified against active
        # call sites in general_checks, define_ranges, qc_branching_
        # logic), but defended with .get('var_forms', {}) so the
        # producer doesn't crash if the structure changes.
        self.forms_per_var = self.grouped_vars.get('var_forms', {})

        # Excluded-by-design: identifiers, barcodes, and physical
        # storage locations are unique-per-sample and would saturate
        # any anomaly metric with noise.
        blood = self.grouped_vars.get('blood_vars', {})
        self.excluded_id_vars = frozenset(
            blood.get('id_variables', [])
            + blood.get('barcode_variables', [])
            + blood.get('position_variables', []))

        # Local string-only missing-code set. Utils.missing_code_set
        # contains both numeric and string forms of REDCap codes
        # (-3, -9, -99, 999, 1909-09-09 family) but does NOT include
        # blank/sentinel strings. Cast every member to its string
        # form and union with the blank/sentinel variants so that a
        # single Series.isin() call handles all forms uniformly.
        self._missing_str_set = frozenset(
            str(v) for v in self.utils.missing_code_set
        ) | frozenset({
            '', 'nan', 'NaN', 'NAN',
            'NA', 'na', 'n/a', 'N/A', 'N/a',
            'null', 'Null', 'NULL', 'None', 'none',
        })

    def __call__(self):
        for network in self.networks:
            self.produce_for_network(network)

    def produce_for_network(self, network: str):
        tp_list = list(self.utils.create_timepoint_list())
        tp_list.extend(['floating', 'conversion'])
        csv_paths = [(tp, self._csv_path(network, tp)) for tp in tp_list]

        # Validate all expected per-tp CSVs exist BEFORE doing any
        # work. Failing partway through after writing a partial long
        # parquet would silently corrupt downstream comparisons.
        missing = [p for _, p in csv_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"FATAL: per-tp combined CSV(s) missing for "
                f"network {network}: {missing}. Refusing to produce "
                f"a partial long-format parquet."
            )

        candidates = self._prune_by_metadata()
        if not candidates:
            raise RuntimeError(
                f"FATAL: no variable survived metadata-only pruning "
                f"for network {network} (excluded ID/barcode/position "
                f"vars + name patterns + variable_type_distributions "
                f"ratio gate). Inspect grouped_variables.json and "
                f"variable_type_distributions.json."
            )
        eligible = self._refine_eligibility(candidates, csv_paths)
        if not eligible:
            raise RuntimeError(
                f"FATAL: no variable passed the distinct-values + "
                f"categorical heuristic for network {network}. "
                f"Refusing to write an empty long-format parquet — "
                f"downstream anomaly detection would silently emit "
                f"zero flags. Lower min_distinct_values or set "
                f"exclude_integer_low_cardinality=false in config."
            )

        long_dfs = []
        for tp, path in csv_paths:
            piece = self._melt_one_tp(path, tp, network, eligible)
            if piece is not None and len(piece) > 0:
                long_dfs.append(piece)
        if not long_dfs:
            raise RuntimeError(
                f"FATAL: eligibility produced {len(eligible)} "
                f"variables but no observations survived to long "
                f"format for network {network}. Indicates a "
                f"data-shape mismatch between classification and "
                f"melt passes — investigate before continuing."
            )
        full_long = pd.concat(long_dfs, ignore_index=True)

        out_path = (f"{self.depen_path}"
                    f"multi_tp_{network}_numeric_long.parquet")
        self._atomic_write_parquet(full_long, out_path)

        print(
            f"[produce_multi_tp_long] {network}: "
            f"{len(full_long)} observations across "
            f"{len(eligible)} variables → {out_path}"
        )

    def _csv_path(self, network: str, tp: str) -> str:
        # Match the path-mangling in qc_forms_main.iterate_combined_dfs
        # exactly — same source files, same naming convention.
        return (
            f"{self.comb_csv_path}AMPSCZ-combined-redcap_"
            f"{tp.replace('month', 'month_').replace('floating', 'floating_forms')}"
            f"_{network.replace('PRONET', 'ProNET')}-day1to1.csv"
        )

    def _prune_by_metadata(self) -> set:
        """
        Drop candidates by name patterns, identifier/barcode/position
        membership, and variable_type_distributions ratio. Returns
        the surviving set of base variable names. Reads no CSVs.
        """
        survivors = set()
        for var, dist in self.var_type_dists.items():
            if not isinstance(dist, dict):
                continue
            if var in self.excluded_id_vars:
                continue
            lower = var.lower()
            if any(pat in lower for pat in self.excluded_name_patterns):
                continue
            num_count = int(dist.get('num', 0))
            non_blank_typed = (
                num_count
                + int(dist.get('string', 0))
                + int(dist.get('date', 0))
                + int(dist.get('other', 0))
            )
            if non_blank_typed == 0:
                continue
            if num_count / non_blank_typed < 0.5:
                continue
            if num_count < self.min_obs_per_variable:
                continue
            survivors.add(var)
        return survivors

    def _refine_eligibility(
        self, candidates: set, csv_paths: list
    ) -> set:
        """
        Pass 1 of two-pass scan. Read each per-tp CSV restricted to
        {subjectid} ∪ candidates and accumulate distinct (numeric,
        non-missing-code) values per candidate. Apply distinct-count
        + categorical heuristic gates. Returns final eligible set.

        Behavior:
          - If subjectid is missing from any per-tp CSV → fail loud.
          - If a per-tp CSV contains subjectid but NO candidate cols
            (e.g. early timepoint with a different form set) → skip
            that CSV silently for the eligibility scan; candidates
            may still be populated from other timepoints.
        """
        candidate_set = set(candidates)
        cols_filter = candidate_set | {'subjectid'}
        distinct_vals = {v: set() for v in candidates}

        for _tp, path in csv_paths:
            df = pd.read_csv(
                path,
                keep_default_na=False,
                encoding='utf-8',
                on_bad_lines='error',
                dtype=str,
                usecols=lambda c: c in cols_filter,
            )
            if 'subjectid' not in df.columns:
                raise RuntimeError(
                    f"FATAL: subjectid column missing from "
                    f"{path}. Refusing to proceed — combined CSV "
                    f"with no subjectid is structurally broken."
                )
            present_cands = [c for c in df.columns
                             if c in candidate_set]
            if not present_cands:
                continue
            for col in present_cands:
                s = df[col].str.strip()
                mask = ~s.isin(self._missing_str_set)
                if not mask.any():
                    continue
                vals = pd.to_numeric(
                    s[mask], errors='coerce').dropna()
                if len(vals) > 0:
                    distinct_vals[col].update(vals.tolist())

        eligible = set()
        for var, vals in distinct_vals.items():
            if len(vals) < self.min_distinct_values:
                continue
            if self.exclude_integer_low_cardinality:
                if (len(vals) <= self.max_integer_distinct_for_categorical
                        and all(float(x).is_integer() for x in vals)):
                    continue
            eligible.add(var)
        return eligible

    def _melt_one_tp(
        self, path: str, tp: str, network: str, eligible: set
    ) -> pd.DataFrame:
        """
        Pass 2 of two-pass scan. Read this tp's CSV restricted to
        {subjectid} ∪ eligible, melt to long, attach metadata
        columns. Returns a DataFrame with the long schema.

        Behavior:
          - subjectid missing → fail loud (matches Pass 1).
          - tp's CSV has no eligible cols → return None (skipped by
            caller). Common when an early-cohort timepoint doesn't
            collect a measurement that later cohorts do.
        """
        cols_filter = eligible | {'subjectid'}
        df = pd.read_csv(
            path,
            keep_default_na=False,
            encoding='utf-8',
            on_bad_lines='error',
            dtype=str,
            usecols=lambda c: c in cols_filter,
        )
        if 'subjectid' not in df.columns:
            raise RuntimeError(
                f"FATAL: subjectid column missing from {path}. "
                f"Refusing to proceed — combined CSV with no "
                f"subjectid is structurally broken."
            )
        present_eligible = [c for c in df.columns if c in eligible]
        if not present_eligible:
            return None

        long = df.melt(
            id_vars='subjectid',
            value_vars=present_eligible,
            var_name='variable',
            value_name='value',
        )
        # `value` is already string (dtype=str applied at read time);
        # no further mutation. Preserves the cell text as parsed from
        # the CSV — leading zeros, trailing zeros, scientific
        # notation are all retained.

        # Missing-code mask BEFORE coercion so that '999' / '-3' /
        # 1909-09-09 are never interpreted as real measurements.
        long['is_missing_code'] = (
            long['value'].str.strip().isin(self._missing_str_set)
        )
        coerced = pd.to_numeric(long['value'], errors='coerce')
        long['value_numeric'] = coerced.where(~long['is_missing_code'])
        long.loc[long['is_missing_code'], 'value_numeric'] = float('nan')

        # Optionally drop non-numeric non-missing-code rows
        # (`keep_non_numeric_audit_rows` config; default True keeps
        # them for audit completeness). Such rows are rare but
        # possible for object-dtype columns containing the
        # occasional free-text entry. Detector filters via
        # `value_numeric.notna() & ~is_missing_code` either way; the
        # config controls only whether they're persisted at all.
        if not self.keep_non_numeric_audit_rows:
            drop = (long['value_numeric'].isna()
                    & ~long['is_missing_code'])
            long = long.loc[~drop].reset_index(drop=True)

        cohort_map = {
            sid: (info or {}).get('cohort', '')
            for sid, info in self.subject_info.items()
        }
        long['cohort'] = long['subjectid'].map(cohort_map).fillna('')
        long['network'] = network
        long['timepoint'] = tp
        # event_date intentionally blank in v1; see module docstring.
        long['event_date'] = ''
        long['source_form'] = (
            long['variable'].map(self.forms_per_var).fillna('')
        )

        return long[[
            'subjectid', 'network', 'timepoint', 'cohort',
            'event_date', 'source_form', 'variable', 'value',
            'value_numeric', 'is_missing_code',
        ]]

    def _atomic_write_parquet(
        self, df: pd.DataFrame, final_path: str
    ):
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        tmp_path = f"{final_path}.{os.getpid()}.tmp"
        try:
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, final_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise


if __name__ == '__main__':
    MultiTPLongFormatProducer()()
