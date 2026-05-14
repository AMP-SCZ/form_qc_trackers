import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_types.discovery._common import atomic_write_parquet

"""
Cross-detector subject-severity aggregator
(Layer 2 / discovery, default-off).

Each discovery detector produces its own candidates parquet. A subject
with a real data-quality issue often fires on several detectors at
once — e.g., a copy-paste pattern between baseline and m_2 will hit
copy_forward, duplicate_record (within-subject), and possibly
longitudinal_delta (because the post-copy m_4 visit jumps from the
"fake stable" baseline).

This aggregator reads every detector's parquet, groups by
(network, subjectid), and emits a single ranked subject summary so
triage starts with the highest-signal subjects across all detectors
instead of paging through 8 separate sheets.

REQUIRES: each source detector publishes a `severity_normalized`
column in [0, 100]. T2.2 added this to every discovery detector.

INPUT (read-only):
  combined_outputs/discovery/{detector}_candidates.parquet for each
  enabled detector. Missing files are silently skipped (a detector
  may not have been run this cycle).

OUTPUT (atomic, tmp + os.replace):
  combined_outputs/discovery/subject_anomaly_summary.parquet

OUTPUT SCHEMA:
  subject, network, total_flags, total_severity_normalized,
  mean_severity_normalized, detectors_firing,
  flags_by_detector (JSON), top_severity_normalized,
  top_detector, top_error_message, dates_detected

CONSTRAINTS HONORED:
  - Reads only discovery output parquets — no source data.
  - Writes only the single summary parquet.
  - Does NOT append to combined_qc_flags.parquet.
  - Atomic write via PID-stamped tmp + os.replace.
"""


# Map detector key -> output filename. Keep this in sync with
# run_all_discovery.py.
_DETECTOR_FILES = (
    ('missingness_summary', 'missingness_anomaly_summary.parquet'),
    ('copy_forward', 'copy_forward_candidates.parquet'),
    ('longitudinal_delta', 'longitudinal_delta_candidates.parquet'),
    ('point_spike', 'point_spike_candidates.parquet'),
    ('internal_consistency',
     'internal_consistency_candidates.parquet'),
    ('pairwise_relationship',
     'pairwise_relationship_candidates.parquet'),
    ('mahalanobis', 'mahalanobis_candidates.parquet'),
    ('pca_reconstruction',
     'pca_reconstruction_candidates.parquet'),
    ('local_outlier', 'local_outlier_candidates.parquet'),
    ('trajectory_shape', 'trajectory_shape_candidates.parquet'),
    ('isolation_forest',
     'isolation_forest_candidates.parquet'),
    ('date_anomaly', 'date_anomaly_candidates.parquet'),
    ('duplicate_record', 'duplicate_record_candidates.parquet'),
)


class SubjectSummary:
    """
    Cross-detector subject-severity aggregator.
    """

    OUTPUT_COLUMNS = [
        'subject', 'network',
        'total_flags', 'total_severity_normalized',
        'mean_severity_normalized', 'detectors_firing',
        'flags_by_detector', 'top_severity_normalized',
        'top_detector', 'top_error_message', 'dates_detected',
    ]

    def __init__(self, config_dict=None):
        self.utils = Utils()
        if config_dict is None:
            with open(f'{self.utils.absolute_path}/config.json',
                      'r') as f:
                config_dict = json.load(f)
        self.config_info = config_dict
        self.output_path = self.config_info['paths']['output_path']
        if str(self.config_info.get('testing_enabled', '')) == "True":
            self.output_path = self.output_path + "testing/"

        ss = self.config_info.get(
            'discovery', {}).get('subject_summary', {})
        cap = ss.get('max_subjects', 500)
        self.max_subjects = (
            int(cap) if cap is not None else None)
        self._counters = self._fresh_counters()

    def _fresh_counters(self) -> dict:
        return {
            'detectors_attempted': 0,
            'detectors_loaded': 0,
            'detectors_missing': 0,
            'total_flags_seen': 0,
            'subjects_in_summary': 0,
        }

    def __call__(self):
        return self.run()

    def run(self):
        self._counters = self._fresh_counters()
        all_flags = self._load_detector_outputs()
        summary = self._aggregate(all_flags)
        self._counters['subjects_in_summary'] = len(summary)
        self._write_output(summary)
        self._print_summary()

    def _load_detector_outputs(self) -> pd.DataFrame:
        """
        Read every detector's candidates parquet (if present), tag
        each row with its detector key, and concatenate. Rows
        without a usable subjectid (e.g. missingness site-aggregate
        rows where subject is blank) are dropped — they don't roll up
        to a subject summary.
        """
        disc_dir = os.path.join(
            self.output_path, 'combined_outputs', 'discovery')
        pieces = []
        for detector_key, fname in _DETECTOR_FILES:
            self._counters['detectors_attempted'] += 1
            path = os.path.join(disc_dir, fname)
            if not os.path.isfile(path):
                self._counters['detectors_missing'] += 1
                continue
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                # A corrupt parquet from one detector shouldn't
                # break the aggregator for the rest.
                print(
                    f"[subject_summary] WARNING: failed to read "
                    f"{path}: {e}; skipping detector "
                    f"{detector_key}."
                )
                continue
            if len(df) == 0:
                self._counters['detectors_loaded'] += 1
                continue
            df = df.copy()
            df['_detector'] = detector_key
            pieces.append(df)
            self._counters['detectors_loaded'] += 1
        if not pieces:
            return pd.DataFrame()
        out = pd.concat(pieces, ignore_index=True, sort=False)
        # Use 'subject' if present (tracker alias); fall back to
        # subjectid. Drop blank / site-aggregate rows.
        if 'subject' in out.columns:
            out['_subject'] = (
                out['subject'].fillna('').astype(str).str.strip())
        else:
            out['_subject'] = ''
        if 'subjectid' in out.columns:
            sid = out['subjectid'].fillna('').astype(str).str.strip()
            out.loc[out['_subject'] == '', '_subject'] = sid
        out = out[out['_subject'] != ''].copy()
        out = out[~out['_subject'].str.contains(
            'aggregate', case=False, na=False)]
        self._counters['total_flags_seen'] = len(out)
        return out

    def _aggregate(self, all_flags: pd.DataFrame) -> pd.DataFrame:
        if len(all_flags) == 0:
            return self._empty_output_df()
        today = str(datetime.today().date())
        # Coerce expected columns where they exist.
        sev_col = (
            'severity_normalized'
            if 'severity_normalized' in all_flags.columns
            else None)
        if sev_col is None:
            # Older detector output without severity_normalized —
            # treat each flag as severity 0 (count-only ranking).
            all_flags = all_flags.copy()
            all_flags['severity_normalized'] = 0.0
            sev_col = 'severity_normalized'
        all_flags[sev_col] = (
            pd.to_numeric(
                all_flags[sev_col], errors='coerce').fillna(0.0))
        if 'network' not in all_flags.columns:
            all_flags['network'] = ''

        # Find each subject's top flag (the row that contributed the
        # largest single severity) — its detector and message are the
        # most useful starting point for triage.
        sorted_flags = all_flags.sort_values(
            sev_col, ascending=False)
        top_per_subject = sorted_flags.drop_duplicates(
            subset=['_subject', 'network'], keep='first')

        records = []
        for (subj, network), grp in all_flags.groupby(
                ['_subject', 'network'], sort=False):
            total_flags = int(len(grp))
            total_sev = float(grp[sev_col].sum())
            mean_sev = float(grp[sev_col].mean())
            detector_counts = (
                grp['_detector'].value_counts().to_dict())
            top = top_per_subject[
                (top_per_subject['_subject'] == subj)
                & (top_per_subject['network'] == network)
            ]
            top_sev = (
                float(top.iloc[0][sev_col]) if len(top) > 0 else 0.0)
            top_det = (
                str(top.iloc[0]['_detector'])
                if len(top) > 0 else '')
            top_msg = ''
            if len(top) > 0 and 'error_message' in top.columns:
                top_msg = str(top.iloc[0]['error_message'])
            records.append({
                'subject': str(subj),
                'network': str(network),
                'total_flags': total_flags,
                'total_severity_normalized': total_sev,
                'mean_severity_normalized': mean_sev,
                'detectors_firing': len(detector_counts),
                'flags_by_detector': json.dumps(
                    {k: int(v) for k, v in
                     sorted(detector_counts.items())}),
                'top_severity_normalized': top_sev,
                'top_detector': top_det,
                'top_error_message': top_msg,
                'dates_detected': today,
            })
        df = pd.DataFrame(records)
        df = df.sort_values(
            ['total_severity_normalized', 'total_flags'],
            ascending=[False, False]).reset_index(drop=True)
        if (self.max_subjects is not None
                and len(df) > self.max_subjects):
            df = df.head(self.max_subjects).copy()
        return self._format_output(df)

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in (
            'subject', 'network', 'flags_by_detector',
            'top_detector', 'top_error_message', 'dates_detected',
        ):
            out[col] = df[col].astype(str)
        for col in (
            'total_severity_normalized',
            'mean_severity_normalized',
            'top_severity_normalized',
        ):
            out[col] = df[col].astype(float)
        out['total_flags'] = df['total_flags'].astype('int64')
        out['detectors_firing'] = (
            df['detectors_firing'].astype('int64'))
        return out[self.OUTPUT_COLUMNS]

    def _empty_output_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            'subject': pd.Series(dtype='object'),
            'network': pd.Series(dtype='object'),
            'total_flags': pd.Series(dtype='int64'),
            'total_severity_normalized': pd.Series(dtype='float64'),
            'mean_severity_normalized': pd.Series(dtype='float64'),
            'detectors_firing': pd.Series(dtype='int64'),
            'flags_by_detector': pd.Series(dtype='object'),
            'top_severity_normalized': pd.Series(dtype='float64'),
            'top_detector': pd.Series(dtype='object'),
            'top_error_message': pd.Series(dtype='object'),
            'dates_detected': pd.Series(dtype='object'),
        })

    def _write_output(self, df: pd.DataFrame):
        out_dir = os.path.join(
            self.output_path, 'combined_outputs', 'discovery')
        os.makedirs(out_dir, exist_ok=True)
        final_path = os.path.join(
            out_dir, 'subject_anomaly_summary.parquet')
        atomic_write_parquet(df, final_path)
        if len(df) == 0:
            print(
                f"[subject_summary] no per-subject flags rolled up "
                f"→ {final_path}"
            )
        else:
            top = df.head(10)[
                ['subject', 'network', 'total_flags',
                 'detectors_firing',
                 'total_severity_normalized', 'top_detector']]
            print(
                f"[subject_summary] {len(df)} subjects with flags "
                f"→ {final_path}"
            )
            print(f"Top 10:\n{top.to_string(index=False)}")

    def _print_summary(self):
        c = self._counters
        rows = [
            ("detectors attempted", c['detectors_attempted']),
            ("detectors loaded", c['detectors_loaded']),
            ("detectors missing", c['detectors_missing']),
            ("total flags seen", c['total_flags_seen']),
            ("subjects in summary (post-cap)",
             c['subjects_in_summary']),
        ]
        width = max(len(label) for label, _ in rows)
        print("[subject_summary] summary:")
        for label, count in rows:
            print(f"  {label:<{width}}  {count}")


if __name__ == '__main__':
    SubjectSummary()()
