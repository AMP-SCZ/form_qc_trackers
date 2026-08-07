"""Export the TYPES of flags raised on SCID5 forms to an Excel
workbook — one row per canonical template per network, with counts.
No per-instance / per-subject rows are included.

Reads the entry-level merged tracker history written by
``analyze_flags/estimate_resolved.py`` for both networks, filters to
the SCID5 forms, and emits a single-sheet workbook with one row per
(network, canonical_template):

  * ``network`` — PRESCIENT or PRONET
  * ``canonical_template`` — the templated message (variables / values
    replaced with placeholders so different instances of the same
    check group together)
  * ``ever_raised`` — total instances of this template observed
    across the merged history for that network
  * ``currently_outstanding`` — instances of this template still open
    in the live tracker
  * ``example_form`` — which SCID5 form this template fires on
    (scid5_psychosis_mood_substance_abuse or
    scid5_schizotypal_personality_sciddpq)
  * ``example_variable`` — one variable name that triggers this
    template (helps identify the check at a glance)

The output is templates-only, so it doesn't carry Subject IDs or
other per-instance PHI. Still lives under ``analyze_flags_dir`` so
it sits alongside the rest of the analyze_flags artifacts.

To export types for a different form (or set of forms), edit
``SCID5_FORMS`` and the ``OUTPUT_BASENAME``, or duplicate this script.
"""

import json
import os
import sys

import pandas as pd

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from analyze_flags.canonicalize import normalize_form_name
from analyze_flags.paths import (
    ensure_analyze_flags_artifact_dir,
    open_tracker_row_history_csv_basename,
    pipeline_output_path_from_config,
)


class ScidFlagExporter():
    NETWORKS = ['PRESCIENT', 'PRONET']

    # The SCID5 forms tracked by clinical_checks. Both belong to the
    # clinical-forms set in flag_analytics.CLINICAL_FORMS — the filter
    # below matches on normalize_form_name keys (lowercase, separators
    # stripped) so V1 capitalization / underscore / space variants of
    # the same form still match these canonical V2 spellings.
    SCID5_FORMS = frozenset([
        'scid5_psychosis_mood_substance_abuse',
        'scid5_schizotypal_personality_sciddpq',
    ])

    OUTPUT_BASENAME = 'scid5_flag_types.xlsx'

    SUMMARY_COLUMNS = [
        'network',
        'canonical_template',
        'ever_raised',
        'currently_outstanding',
        'example_form',
        'example_variable',
    ]

    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json', 'r') as file:
            self.config_info = json.load(file)
        self.output_path = pipeline_output_path_from_config(self.config_info)
        self.analyze_flags_dir = ensure_analyze_flags_artifact_dir(self.output_path)

    def run_script(self):
        network_dfs = {}
        for network in self.NETWORKS:
            csv_path = os.path.join(
                self.analyze_flags_dir,
                open_tracker_row_history_csv_basename(network),
            )
            if not os.path.exists(csv_path):
                print(
                    f"{network}: no merged history CSV at {csv_path}; "
                    f"run estimate_resolved.py first. Skipping."
                )
                continue
            df = pd.read_csv(csv_path, keep_default_na=False)
            if 'General_Flag' not in df.columns:
                print(
                    f"{network}: General_Flag column missing in {csv_path}; "
                    f"file may be in an older row-level format. Skipping."
                )
                continue
            # Compare on stripped keys; rewrite matches to the
            # canonical V2 spelling so example_form is consistent.
            lookup = {normalize_form_name(f): f for f in self.SCID5_FORMS}
            form_keys = df['General_Flag'].map(normalize_form_name)
            mask = form_keys.isin(lookup)
            sub = df[mask].copy()
            if not sub.empty:
                sub['General_Flag'] = form_keys[mask].map(lookup)
            if sub.empty:
                print(f"{network}: no SCID5 entries found.")
            else:
                # Count unique templates after grouping for the log
                # so the operator knows whether the workbook will be
                # informative.
                templates = (
                    sub['canonical_template'].nunique()
                    if 'canonical_template' in sub.columns else 0
                )
                print(
                    f"{network}: {len(sub)} SCID5 entries across "
                    f"{templates} distinct canonical templates."
                )
            network_dfs[network] = sub

        summary_df = self._build_summary(network_dfs)
        out_path = os.path.join(self.analyze_flags_dir, self.OUTPUT_BASENAME)
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Templates', index=False)
        print(
            f"Wrote {out_path}: {len(summary_df)} canonical templates "
            f"across {summary_df['network'].nunique() if not summary_df.empty else 0} networks."
        )

    def _build_summary(self, network_dfs):
        """One row per (network, canonical_template). Sorted by
        network then ever_raised desc so the most common templates
        land at the top of each network section."""
        rows = []
        for network, df in network_dfs.items():
            if df.empty or 'canonical_template' not in df.columns:
                continue
            # is_currently_open round-trips through CSV as 'True'/'False'.
            open_mask = (
                df['is_currently_open'].astype(str).str.strip().eq('True')
            )
            grouped = df.groupby('canonical_template', dropna=False)
            for canon, group in grouped:
                ever = len(group)
                outstanding = int(open_mask.loc[group.index].sum())
                example_var = (
                    group['variable'].iloc[0]
                    if 'variable' in group.columns else ''
                )
                example_form = (
                    group['General_Flag'].iloc[0]
                    if 'General_Flag' in group.columns else ''
                )
                rows.append({
                    'network': network,
                    'canonical_template': canon,
                    'ever_raised': ever,
                    'currently_outstanding': outstanding,
                    'example_form': example_form,
                    'example_variable': example_var,
                })
        if not rows:
            return pd.DataFrame(columns=self.SUMMARY_COLUMNS)
        return (
            pd.DataFrame(rows, columns=self.SUMMARY_COLUMNS)
            .sort_values(['network', 'ever_raised'], ascending=[True, False])
            .reset_index(drop=True)
        )


if __name__ == '__main__':
    ScidFlagExporter().run_script()
