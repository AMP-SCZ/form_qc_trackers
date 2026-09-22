"""Multi-dimensional analytics for merged tracker-flag history.

Consumes the entry-level history CSV produced by estimate_resolved.py: one row
per unique (Subject, Timepoint, General_Flag, variable, canonical_template)
episode, with source-aware date bounds, raw message evidence, and an explicit
is_currently_open state.

Per network, produces:
  * outstanding_stacked_by_form_{network}.png and its backing CSV;
  * outstanding_stacked_by_timepoint_{network}.png and its backing CSV;
  * flag_template_counts_{network}.csv;
  * resolved_variable_values_{network}.csv, a status summary by template;
  * resolved_values_per_entry_{network}.csv, detail for every eligible
    resolved entry, including a conservative prior-value inference; and
  * resolved_before_after_values_{network}.csv, limited to rows where both the
    last observed bad value and the latest current value are available.

Resolved-value outputs cover every report form rather than the graph allowlist.
PRONET V2-only testing episodes are excluded; PRESCIENT V1 history remains
eligible because V1 was production before promotion. Current values come from
the latest combined data pull (or the multi-timepoint dependency), not a
resolution-time snapshot.

Operator decisions in jumps_{network}.xlsx are applied before outputs: entries
born in excluded addition jumps are removed, and excluded removal jumps are not
treated as resolutions. Pass apply_jump_exclusions=False for the raw view.

Graph date ranges use tracker metadata rather than today, so reruns against the
same history are reproducible. Pass as_of_date to override the right edge.
"""
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)
from utils.utils import Utils
from analyze_flags.manual_review import (
    entry_key_from_row,
    load_jump_decisions,
)
from analyze_flags.paths import (
    ensure_analyze_flags_artifact_dir,
    open_tracker_row_history_csv_basename,
    pipeline_output_path_from_config,
    tracker_metadata_json_basename,
)
from analyze_flags.canonicalize import normalize_form_name, normalize_timepoint
from analyze_flags.value_extraction import infer_before_value


class FlagAnalytics():
    NETWORKS = ['PRESCIENT', 'PRONET']

    STATUS_FILLED = 'filled'
    STATUS_STILL_BLANK = 'still_blank'
    STATUS_MISSING_CODE = 'missing_code'
    STATUS_SUBJECT_MISSING = 'subject_missing'
    STATUS_SUBJECT_DUPLICATE = 'subject_duplicate'
    STATUS_VARIABLE_MISSING = 'variable_missing'
    STATUS_CSV_MISSING = 'csv_missing'
    STATUS_COLUMNS = [
        STATUS_FILLED,
        STATUS_STILL_BLANK,
        STATUS_MISSING_CODE,
        STATUS_SUBJECT_MISSING,
        STATUS_SUBJECT_DUPLICATE,
        STATUS_VARIABLE_MISSING,
        STATUS_CSV_MISSING,
    ]
    RESOLVED_VALUE_DETAIL_COLUMNS = [
        'Subject', 'Timepoint', 'General_Flag', 'variable',
        'message_variable', 'canonical_template', 'before_value_available',
        'before_value', 'before_value_source', 'Latest_seen', 'status',
        'current_value',
    ]

    # Sentinel used in stacked outputs when a row's form / timepoint
    # is blank. Better than dropping the row (which understates the
    # total) but still visually distinct from a real category.
    BLANK_LABEL = '<blank>'

    # Clinical forms only — set per user direction. Followup variants
    # (`_fu`, `_fu_hc`, `_followup`) are included here because the
    # filter checks the (pre-followup-merge) General_Flag against this
    # set BEFORE the followup-merge step below collapses them.
    # Matching is done on normalize_form_name keys (lowercase, all
    # separators stripped), so V1 spellings like 'SOFAS Screening'
    # still hit 'sofas_screening'; keep the canonical V2 spellings
    # here — they double as the display labels. Edit this list to
    # change which forms feed the analytics outputs.
    CLINICAL_FORMS = frozenset([
        "lifetime_ap_exposure_screen",
        "past_pharmaceutical_treatment",
        "health_conditions_medical_historypsychiatric_histo",
        "current_pharmaceutical_treatment_floating_med_125",
        "current_pharmaceutical_treatment_floating_med_2650",
        "adverse_events",
        "conversion_form",
        "scid5_schizotypal_personality_sciddpq",
        "family_interview_for_genetic_studies_figs",
        "sofas_screening",
        "sofas_followup",
        "psychs_p1p8",
        "psychs_p9ac32",
        "psychs_p1p8_fu",
        "psychs_p9ac32_fu",
        "psychs_p1p8_fu_hc",
        "psychs_p9ac32_fu_hc",
        "psychs_av_recording_run_sheet",
        "premorbid_adjustment_scale",
        "nsipr",
        "cdss",
        "assist",
        "cssrs_baseline",
        "cssrs_followup",
        "global_functioning_social_scale",
        "global_functioning_social_scale_followup",
        "global_functioning_role_scale",
        "global_functioning_role_scale_followup",
        "perceived_discrimination_scale",
        "pubertal_developmental_scale",
        "oasis",
        "item_promis_for_sleep",
        "perceived_stress_scale",
        "pgis",
        "psychosis_polyrisk_score",
        "bprs",
        "psychosocial_treatment_form",
        "scid5_psychosis_mood_substance_abuse",
    ])

    # Followup-suffix stripper. Listed longest-first so `_fu_hc` matches
    # before `_fu` (alternation is left-to-right).
    _FOLLOWUP_SUFFIX_RE = re.compile(r'_(?:fu_hc|fu|followup)$')

    # After stripping a followup suffix, some forms still need a
    # remap because the screening form uses a non-`_screening`
    # naming convention. E.g. stripping `_followup` from
    # `cssrs_followup` yields `cssrs`, but the screening form is
    # `cssrs_baseline`. This map covers those cases; anything not
    # listed falls through unchanged.
    _FORM_REMAP_POST_STRIP = {
        'sofas': 'sofas_screening',
        'cssrs': 'cssrs_baseline',
    }

    def __init__(self, as_of_date=None, apply_jump_exclusions=True):
        """``as_of_date``: optional override for the rightmost date
        plotted. Defaults to the metadata's ``newest_revision_overall``
        so reruns against the same CSV are reproducible. Pass a
        ``pd.Timestamp`` / ``datetime`` / ISO string to override.

        ``apply_jump_exclusions``: when True (default), the operator's
        decisions in ``jumps_{network}.xlsx`` are applied on load (see
        module docstring). False = raw, unfiltered analytics."""
        self.utils = Utils()
        self.apply_jump_exclusions = apply_jump_exclusions
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json', 'r') as file:
            self.config_info = json.load(file)
        self.output_path = pipeline_output_path_from_config(self.config_info)
        self.analyze_flags_dir = ensure_analyze_flags_artifact_dir(self.output_path)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depend_path = self.config_info['paths']['dependencies_path']
        if as_of_date is None:
            self.as_of_date = None
        else:
            ts = pd.to_datetime(as_of_date)
            # tz_localize raises on already-tz-aware values; tz_convert
            # raises on tz-naive. Branch on the actual tz state so
            # either input shape works.
            if getattr(ts, 'tzinfo', None) is not None:
                ts = ts.tz_convert(None) if hasattr(ts, 'tz_convert') else ts.tz_localize(None)
            self.as_of_date = ts.normalize()

    def run_script(self):
        for network in self.NETWORKS:
            self._run_for_network(network)

    def _run_for_network(self, network):
        csv_path = os.path.join(
            self.analyze_flags_dir,
            open_tracker_row_history_csv_basename(network),
        )
        if not os.path.exists(csv_path):
            print(
                f"No merged history CSV for {network} at {csv_path}; "
                f"run estimate_resolved.py first. Skipping."
            )
            return

        try:
            all_df = self._load_history(csv_path, network, clinical_only=False)
        except (
                pd.errors.EmptyDataError, pd.errors.ParserError,
                UnicodeDecodeError, OSError) as error:
            print(
                f"WARNING: cannot read merged history CSV for {network}: "
                f"{type(error).__name__}: {str(error)[:200]}; preserving prior "
                "value outputs."
            )
            return
        if all_df is None:
            print(
                f"{network}: invalid history schema; preserving prior value "
                "outputs."
            )
            return
        # Value recovery is an entry-level operation and is not limited to the
        # clinical-form graph allowlist. Calling this before the empty-history
        # return ensures a clean run replaces stale value artifacts with
        # header-only files.
        self._write_resolved_value_stats(all_df, network)
        if all_df.empty:
            self._clear_non_value_outputs(network)
            print(f"{network} history CSV is empty; analytics outputs were cleared.")
            return
        df = self._load_history(csv_path, network)
        if df.empty:
            self._clear_non_value_outputs(network)
            print(
                f"{network}: no clinical-form entries for graphs/counts; "
                "non-value outputs were cleared and resolved value outputs "
                "were written."
            )
            return

        # Counts do not depend on a usable date range.  Write them before the
        # graph cutoff checks so a successful run cannot retain stale counts.
        self._write_canonical_counts(df, network)

        metadata = self._load_metadata(network)
        # Effective right-edge for the outstanding curves. Priority:
        # explicit as_of_date > metadata newest_revision_overall >
        # max(Latest_seen) from the CSV (fallback).
        if self.as_of_date is not None:
            cutoff = self.as_of_date
        elif metadata and metadata.get('newest_revision_overall'):
            ts = pd.to_datetime(metadata['newest_revision_overall'])
            if getattr(ts, 'tzinfo', None) is not None:
                ts = ts.tz_convert(None) if hasattr(ts, 'tz_convert') else ts.tz_localize(None)
            cutoff = ts.normalize()
        else:
            cutoff = df['Latest_seen'].dropna().max()
        if pd.isna(cutoff):
            self._clear_stacked_output(network, 'form')
            self._clear_stacked_output(network, 'timepoint')
            print(f"{network}: cannot determine cutoff date; skipping.")
            return

        # Stacked outstanding curves — one PNG + one CSV per dim.
        self._produce_stacked_outputs(
            df, network, cutoff,
            group_col='General_Flag', group_label='form',
        )
        self._produce_stacked_outputs(
            df, network, cutoff,
            group_col='Timepoint', group_label='timepoint',
        )
    # ------------------------------------------------------------------
    # History load
    # ------------------------------------------------------------------

    def _load_history(self, csv_path, network, clinical_only=True):
        """Read the entry-level merged history CSV and normalize the
        date columns to tz-naive midnight Timestamps so per-day
        searchsorted bounds align without tz comparison errors.

        Jump exclusions are applied here, BEFORE the followup-merge
        below, because the operator workbook's entry keys carry the
        raw (pre-merge) General_Flag values."""
        df = pd.read_csv(csv_path, keep_default_na=False)
        df.columns = df.columns.str.replace(' ', '_')
        required = ('Subject', 'Earliest_seen', 'Latest_seen', 'General_Flag',
                    'Timepoint', 'variable', 'canonical_template',
                    'is_currently_open')
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(
                f"WARNING: {csv_path} missing columns {missing} "
                f"(found {list(df.columns)}); skipping this network."
            )
            return None
        df = self._apply_jump_decisions(df, network)
        for col in ('Earliest_seen', 'Latest_seen'):
            parsed = pd.to_datetime(df[col], errors='coerce', utc=True)
            df[col] = parsed.dt.tz_localize(None).dt.normalize()
        # Boolean audit fields round-trip through CSV as strings.
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
        open_raw = (
            df['is_currently_open'].astype(str).str.strip().str.casefold())
        invalid_open = ~open_raw.isin(boolean_values)
        if invalid_open.any():
            print(
                f"WARNING: {csv_path} contains {int(invalid_open.sum())} "
                "invalid is_currently_open value(s); skipping this network."
            )
            return None
        df['is_currently_open'] = open_raw.isin({'true', '1', 'yes'})
        if 'message_variable' not in df.columns:
            df['message_variable'] = df['variable']
        else:
            message_variable = (
                df['message_variable'].fillna('').astype(str).str.strip())
            df['message_variable'] = message_variable.where(
                message_variable.ne(''), df['variable'])
        if 'message_value_conflict' in df.columns:
            conflict_raw = (
                df['message_value_conflict'].astype(str).str.strip()
                .str.casefold())
            invalid_conflict = ~conflict_raw.isin(boolean_values)
            if invalid_conflict.any():
                print(
                    f"WARNING: {csv_path} contains "
                    f"{int(invalid_conflict.sum())} invalid "
                    "message_value_conflict value(s); skipping this network."
                )
                return None
            df['message_value_conflict'] = conflict_raw.isin(
                {'true', '1', 'yes'})
        else:
            df['message_value_conflict'] = False
        if 'resolution_eligible' in df.columns:
            eligible_raw = (
                df['resolution_eligible'].astype(str).str.strip()
                .str.casefold())
            invalid_eligible = ~eligible_raw.isin(boolean_values)
            if invalid_eligible.any():
                print(
                    f"WARNING: {csv_path} contains "
                    f"{int(invalid_eligible.sum())} invalid "
                    "resolution_eligible value(s); skipping this network."
                )
                return None
            df['resolution_eligible'] = eligible_raw.isin(
                {'true', '1', 'yes'})
        elif network == 'PRONET' and 'source' in df.columns:
            # Backward compatibility for histories written before the explicit
            # eligibility column: PRONET V2 is testing-only.
            df['resolution_eligible'] = df['source'].map(
                lambda value: 'V1' in str(value).split('+'))
        else:
            # PRESCIENT V1 was production before the V2 promotion.
            df['resolution_eligible'] = True
        if not clinical_only:
            return df

        # Restrict only the established graph/count outputs to clinical forms.
        # The resolved before/current exports returned above use all forms.
        # Comparison happens on normalize_form_name keys (lowercase,
        # separators stripped) so V1's capitalization / underscore /
        # space variants still match; matched rows are rewritten to
        # the canonical V2 spelling so the followup merge and graph
        # labels see one vocabulary.
        lookup = self._clinical_form_lookup()
        form_keys = df['General_Flag'].map(normalize_form_name)
        clinical_mask = form_keys.isin(lookup)
        kept = int(clinical_mask.sum())
        dropped = len(df) - kept
        if dropped:
            print(
                f"{csv_path}: kept {kept} clinical-form entries, "
                f"dropped {dropped} non-clinical entries."
            )
        df = df[clinical_mask].copy()
        if df.empty:
            return df
        df['General_Flag'] = form_keys[clinical_mask].map(lookup)
        # Merge followup variants into their screening/baseline forms
        # so e.g. psychs_p1p8 / psychs_p1p8_fu / psychs_p1p8_fu_hc all
        # canonicalize to psychs_p1p8 and cssrs_followup merges with
        # cssrs_baseline. Applied AFTER the clinical filter (and the
        # rewrite to canonical V2 spellings) so the suffix regex sees
        # the underscored V2 names.
        df['General_Flag'] = df['General_Flag'].map(self._normalize_clinical_form)
        return df

    @classmethod
    def _clinical_form_lookup(cls):
        """{normalize_form_name(form): canonical V2 spelling} for every
        clinical form — comparison key in, display label out."""
        return {normalize_form_name(f): f for f in cls.CLINICAL_FORMS}

    def _apply_jump_decisions(self, df, network):
        """Apply the operator's include/exclude decisions from
        ``jumps_{network}.xlsx``:
          * excluded ADDITION jump -> affected entries dropped
            entirely (artifact flags);
          * excluded REMOVAL jump -> affected entries kept but marked
            ``resolution_excluded`` so the resolved-value outputs skip
            them (their disappearance wasn't a real resolution).
        Always adds the ``resolution_excluded`` column so downstream
        code can rely on it."""
        df['resolution_excluded'] = False
        if not self.apply_jump_exclusions or df.empty:
            return df
        decisions = load_jump_decisions(self.analyze_flags_dir, network)
        if decisions['n_jumps'] == 0:
            return df
        if decisions['n_undecided']:
            print(
                f"{network}: {decisions['n_undecided']} of "
                f"{decisions['n_jumps']} jumps have no include decision "
                f"yet — they are EXCLUDED by default. Review "
                f"jumps_{network}.xlsx to change that."
            )
        excluded_added = decisions['excluded_added']
        excluded_removed = decisions['excluded_removed']
        if not excluded_added and not excluded_removed:
            return df
        keys = [
            entry_key_from_row(r) for r in df.itertuples(index=False)
        ]
        if excluded_added:
            keep = pd.Series(
                [k not in excluded_added for k in keys], index=df.index,
            )
            dropped = int((~keep).sum())
            if dropped:
                print(
                    f"{network}: dropped {dropped} entries from excluded "
                    f"addition jumps."
                )
            keys = [k for k, kept in zip(keys, keep) if kept]
            df = df[keep].copy()
        if excluded_removed:
            res_excl = pd.Series(
                [k in excluded_removed for k in keys], index=df.index,
            )
            n_excl = int(res_excl.sum())
            if n_excl:
                print(
                    f"{network}: {n_excl} entries marked "
                    f"resolution_excluded via removal jumps."
                )
            df['resolution_excluded'] = res_excl
        return df

    def _normalize_clinical_form(self, form):
        """Strip followup suffixes (`_fu_hc`, `_fu`, `_followup`) and
        apply the post-strip remap for forms whose screening variant
        uses a non-`_screening` suffix (cssrs_baseline, sofas_screening).
        """
        if form is None:
            return ''
        base = self._FOLLOWUP_SUFFIX_RE.sub('', str(form))
        return self._FORM_REMAP_POST_STRIP.get(base, base)

    def _load_metadata(self, network):
        path = os.path.join(
            self.analyze_flags_dir,
            tracker_metadata_json_basename(network),
        )
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: {path} unreadable ({e}); falling back to "
                  f"inferred cutoff.")
            return None

    # ------------------------------------------------------------------
    # Stacked outstanding curves
    # ------------------------------------------------------------------

    def _produce_stacked_outputs(self, df, network, cutoff,
                                 group_col, group_label):
        if group_col not in df.columns:
            self._clear_stacked_output(network, group_label)
            print(
                f"{network}: missing {group_col} column; skipping "
                f"{group_label} stack."
            )
            return
        # Cursor HIGH: dropna() independently on Earliest / Latest
        # produces unequal sort arrays where searchsorted can give
        # negative counts. Filter rows where BOTH bounds are valid
        # together before sorting.
        valid_mask = df['Earliest_seen'].notna() & df['Latest_seen'].notna()
        valid = df[valid_mask]
        if valid.empty:
            self._clear_stacked_output(network, group_label)
            print(
                f"{network}: no rows with both Earliest_seen and "
                f"Latest_seen parseable; skipping {group_label} stack."
            )
            return

        start_date = valid['Earliest_seen'].min()
        end_date = cutoff
        if pd.isna(end_date):
            end_date = valid['Latest_seen'].max()
        if start_date > end_date:
            self._clear_stacked_output(network, group_label)
            print(
                f"{network}: earliest ({start_date}) is after cutoff "
                f"({end_date}); skipping {group_label} stack."
            )
            return
        date_index = pd.date_range(start_date, end_date, freq='D')

        curves = self._outstanding_curves_per_group(
            valid, group_col, date_index, cutoff,
        )
        if not curves:
            self._clear_stacked_output(network, group_label)
            print(f"{network}: no curves to plot for {group_label}.")
            return

        wide_df = (
            pd.DataFrame(curves, index=date_index)
            .fillna(0)
            .astype('int64')
        )
        wide_df.index.name = 'date'
        wide_df['total'] = wide_df.sum(axis=1)

        csv_path = os.path.join(
            self.analyze_flags_dir,
            f"outstanding_by_{group_label}_{network}.csv",
        )
        wide_df.to_csv(csv_path)
        print(f"Wrote {csv_path}")

        png_path = os.path.join(
            self.analyze_flags_dir,
            f"outstanding_stacked_by_{group_label}_{network}.png",
        )
        self._render_stacked_plot(
            wide_df.drop(columns=['total']), network, group_label, png_path,
        )
        print(f"Wrote {png_path}")

    def _clear_stacked_output(self, network, group_label):
        """Replace one successful no-data stack with an explicit empty CSV."""

        csv_path = os.path.join(
            self.analyze_flags_dir,
            f"outstanding_by_{group_label}_{network}.csv",
        )
        pd.DataFrame(columns=['date', 'total']).to_csv(csv_path, index=False)
        png_path = os.path.join(
            self.analyze_flags_dir,
            f"outstanding_stacked_by_{group_label}_{network}.png",
        )
        try:
            os.remove(png_path)
        except FileNotFoundError:
            pass
        print(f"Wrote {csv_path}: no rows; removed stale plot if present.")

    def _clear_non_value_outputs(self, network):
        """Clear graph/count artifacts after a successful empty result."""

        self._clear_stacked_output(network, 'form')
        self._clear_stacked_output(network, 'timepoint')
        self._write_empty_canonical_counts(network)

    def _outstanding_curves_per_group(self, df, group_col, date_index, cutoff):
        """One outstanding-per-day series per unique group_col value.
        A flag is outstanding on day d iff Earliest_seen <= d <=
        effective_Latest. effective_Latest extends still-open flags
        past the plot window so they remain counted on the rightmost
        day. Mirrors visualize_data/graph_errors.py:_outstanding_curve.

        Uses the explicit ``is_currently_open`` column rather than
        inferring "still open" from ``Latest_seen == newest_day`` so
        a flag whose source tracker stopped being updated isn't
        incorrectly marked open.
        """
        still_open = df['is_currently_open'].astype(bool)
        # Open-sentinel pushes still-open flags past end_date so they
        # remain counted on the rightmost day.
        open_sentinel = date_index[-1] + pd.Timedelta(days=1)
        effective_latest = df['Latest_seen'].where(~still_open, open_sentinel)
        earliest = df['Earliest_seen']

        idx_np = date_index.to_numpy()
        curves = {}
        for group_val, group_idx in df.groupby(group_col, dropna=False).groups.items():
            # Coerce label rather than skip non-string group values
            # (Cursor MEDIUM: blank/null labels currently dropped =
            # totals understated). Real blanks become '<blank>' so
            # they appear in the stack but are visually distinct.
            if group_val is None or (isinstance(group_val, float)
                                     and pd.isna(group_val)):
                label = self.BLANK_LABEL
            else:
                label = str(group_val).strip() or self.BLANK_LABEL

            sub_e = earliest.loc[group_idx]
            sub_l = effective_latest.loc[group_idx]
            # We already filtered to rows with BOTH bounds valid in
            # _produce_stacked_outputs, so the lengths here match.
            if sub_e.empty:
                continue
            e_sorted = np.sort(sub_e.to_numpy())
            l_sorted = np.sort(sub_l.to_numpy())
            # side='right' on Earliest, side='left' on Latest is the
            # inclusive-on-both-ends convention from graph_errors.py.
            started_le_d = np.searchsorted(e_sorted, idx_np, side='right')
            ended_lt_d = np.searchsorted(l_sorted, idx_np, side='left')
            # Same-label groups (e.g. multiple stale-data buckets all
            # labeled '<blank>') accumulate into the same curve.
            curve = (started_le_d - ended_lt_d).astype('int64')
            if label in curves:
                curves[label] = curves[label] + curve
            else:
                curves[label] = curve
        return curves

    def _render_stacked_plot(self, wide_df, network, group_label, png_path):
        if wide_df.empty or wide_df.shape[1] == 0:
            print(f"{network}: nothing to plot for {group_label}.")
            return
        col_order = wide_df.max().sort_values(ascending=False).index.tolist()
        df_ordered = wide_df[col_order]
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(col_order), 1)))

        # Legend goes BELOW the plot in a multi-column grid. ncol is
        # chosen to keep the legend roughly 6 rows tall regardless of
        # entry count, so the plot area itself isn't dwarfed.
        import math
        n = len(col_order)
        target_legend_rows = 6
        ncol = max(1, math.ceil(n / target_legend_rows))

        fig, ax = plt.subplots(figsize=(18, 10))
        ax.stackplot(
            df_ordered.index,
            [df_ordered[c].values for c in col_order],
            labels=col_order,
            colors=colors,
        )
        ax.set_title(
            f"{network} Outstanding Queries Over Time by {group_label.title()}"
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Outstanding Query Count')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            fontsize='small',
            ncol=ncol,
            frameon=False,
            borderaxespad=0,
        )
        # tight_layout + bbox_inches='tight' together expand the saved
        # PNG canvas downward to include the off-axes legend; the plot
        # area itself keeps its full figure height.
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Canonical-template counts
    # ------------------------------------------------------------------

    def _write_canonical_counts(self, df, network):
        if df.empty or 'canonical_template' not in df.columns:
            print(
                f"{network}: no canonical_template column; skipping "
                f"canonical-template counts."
            )
            self._write_empty_canonical_counts(network)
            return

        ever = (
            df.groupby('canonical_template').size().rename('ever_raised')
        )
        outstanding = (
            df[df['is_currently_open']]
            .groupby('canonical_template').size()
            .rename('currently_outstanding')
        )
        example = (
            df.groupby('canonical_template')
            .first()[['General_Flag', 'variable']]
            .rename(columns={
                'General_Flag': 'example_form',
                'variable': 'example_variable',
            })
        )
        out = pd.concat([ever, outstanding, example], axis=1)
        out['currently_outstanding'] = (
            out['currently_outstanding'].fillna(0).astype('int64')
        )
        out = (
            out.sort_values(by='ever_raised', ascending=False)
            .reset_index()
        )

        csv_path = os.path.join(
            self.analyze_flags_dir,
            f"flag_template_counts_{network}.csv",
        )
        out.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}: {len(out)} unique templates.")

    def _write_empty_canonical_counts(self, network):
        columns = [
            'canonical_template', 'ever_raised', 'currently_outstanding',
            'example_form', 'example_variable',
        ]
        csv_path = os.path.join(
            self.analyze_flags_dir,
            f"flag_template_counts_{network}.csv",
        )
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}: 0 unique templates.")

    # ------------------------------------------------------------------
    # Resolved-flag variable-value status
    # ------------------------------------------------------------------

    def _write_resolved_value_stats(self, df, network):
        if df.empty:
            print(f"{network}: no entries; clearing resolved value stats.")
            self._write_empty_resolved_value_outputs(network)
            return
        # resolution_excluded = the entry's disappearance was an
        # operator-excluded removal jump, not a real resolution. PRONET V2-only
        # episodes are testing observations, not live resolutions.
        if 'resolution_eligible' in df.columns:
            resolution_eligible = (
                df['resolution_eligible'].astype(str).str.strip()
                .str.casefold().isin({'true', '1', 'yes'}))
        elif network == 'PRONET' and 'source' in df.columns:
            resolution_eligible = df['source'].map(
                lambda value: 'V1' in str(value).split('+'))
        else:
            resolution_eligible = pd.Series(True, index=df.index)
        resolved = df[
            ~df['is_currently_open']
            & ~df['resolution_excluded']
            & resolution_eligible
        ]
        if resolved.empty:
            print(
                f"{network}: no eligible resolved entries; clearing resolved "
                "value stats."
            )
            self._write_empty_resolved_value_outputs(network)
            return

        # Bucket by Timepoint so we read each combined CSV once.
        rows_by_tp = defaultdict(list)
        for r in resolved.itertuples(index=False):
            rows_by_tp[r.Timepoint].append(r)

        status_counts = defaultdict(lambda: {s: 0 for s in self.STATUS_COLUMNS})
        per_entry_rows = []
        missing_code_set = self._missing_code_set()

        def record(r, status, value=''):
            """One resolved entry's outcome feeds both the per-template
            summary and the per-entry detail CSV."""
            status_counts[r.canonical_template][status] += 1
            message_variable = getattr(r, 'message_variable', r.variable)
            value_conflict = (
                str(getattr(r, 'message_value_conflict', False))
                .strip().casefold() in {'true', '1', 'yes'})
            before = (
                None if value_conflict else infer_before_value(
                    message_variable, getattr(r, 'message', '')))
            per_entry_rows.append({
                'Subject': r.Subject,
                'Timepoint': r.Timepoint,
                'General_Flag': r.General_Flag,
                'variable': r.variable,
                'message_variable': message_variable,
                'canonical_template': r.canonical_template,
                'before_value_available': before is not None,
                'before_value': before.value if before is not None else '',
                'before_value_source': (
                    before.source if before is not None else ''),
                'Latest_seen': r.Latest_seen,
                'status': status,
                'current_value': str(value),
            })

        for timepoint, rows in rows_by_tp.items():
            # An empty/blank timepoint produces a malformed combined-CSV
            # path (e.g. '..._\_ProNET-day1to1.csv'). The file lookup
            # would fail with file-not-found, eventually marking these
            # entries as csv_missing — short-circuit so the diagnostic
            # is accurate ("blank timepoint" vs. "file missing").
            tp_norm = str(timepoint).strip() if timepoint is not None else ''
            if not tp_norm:
                print(
                    f"{network}: {len(rows)} resolved entries have blank "
                    f"Timepoint; marking as {self.STATUS_CSV_MISSING}."
                )
                for r in rows:
                    record(r, self.STATUS_CSV_MISSING)
                continue
            csv_path = self._current_values_csv_path(network, timepoint)
            cdf, csv_error = self._load_combined_csv(csv_path)
            if cdf is None:
                print(
                    f"{network}/{timepoint}: combined CSV unreadable "
                    f"({csv_error}); marking {len(rows)} resolved entries "
                    f"as {self.STATUS_CSV_MISSING}."
                )
                for r in rows:
                    record(r, self.STATUS_CSV_MISSING)
                continue
            if 'subjectid' not in cdf.columns:
                print(
                    f"{network}/{timepoint}: combined CSV at {csv_path} "
                    f"has no 'subjectid' column; marking {len(rows)} "
                    f"resolved entries as {self.STATUS_CSV_MISSING}."
                )
                for r in rows:
                    record(r, self.STATUS_CSV_MISSING)
                continue
            subject_positions = defaultdict(list)
            for i, sid in enumerate(cdf['subjectid'].astype(str)):
                subject_positions[sid].append(i)

            for r in rows:
                subj = str(r.Subject)
                positions = subject_positions.get(subj, [])
                if not positions:
                    record(r, self.STATUS_SUBJECT_MISSING)
                    continue
                if len(positions) != 1:
                    record(r, self.STATUS_SUBJECT_DUPLICATE)
                    continue
                if r.variable not in cdf.columns:
                    record(r, self.STATUS_VARIABLE_MISSING)
                    continue
                raw = cdf.iloc[positions[0]][r.variable]
                current_value = '' if pd.isna(raw) else str(raw)
                normalized_value = current_value.strip()
                if normalized_value == '':
                    record(r, self.STATUS_STILL_BLANK, current_value)
                elif normalized_value in missing_code_set:
                    record(r, self.STATUS_MISSING_CODE, current_value)
                else:
                    record(r, self.STATUS_FILLED, current_value)

        if not status_counts:
            print(f"{network}: nothing to write for resolved value stats.")
            self._write_empty_resolved_value_outputs(network)
            return

        self._write_per_entry_values(per_entry_rows, network)

        rows_out = []
        for tmpl, counts in status_counts.items():
            total = sum(counts.values())
            row = {'canonical_template': tmpl}
            for s in self.STATUS_COLUMNS:
                row[f'count_{s}'] = counts[s]
            row['total_resolved'] = total
            rows_out.append(row)
        out = (
            pd.DataFrame(rows_out)
            .sort_values('total_resolved', ascending=False)
            .reset_index(drop=True)
        )

        csv_path = os.path.join(
            self.analyze_flags_dir,
            f"resolved_variable_values_{network}.csv",
        )
        out.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}: {len(out)} unique templates.")

    def _write_empty_resolved_value_outputs(self, network):
        """Replace successful no-data results with explicit empty schemas."""

        self._write_per_entry_values([], network)
        summary_columns = [
            'canonical_template',
            *(f'count_{status}' for status in self.STATUS_COLUMNS),
            'total_resolved',
        ]
        csv_path = os.path.join(
            self.analyze_flags_dir,
            f"resolved_variable_values_{network}.csv",
        )
        pd.DataFrame(columns=summary_columns).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}: 0 unique templates.")

    def _write_per_entry_values(self, per_entry_rows, network):
        """Write complete resolved detail plus its trustworthy pair subset.

        current_value is the latest combined-CSV value, not a historical
        resolution-time snapshot. The paired output includes only entries
        where the raw message proves the before value and the current lookup
        succeeded."""
        out = pd.DataFrame(
            per_entry_rows, columns=self.RESOLVED_VALUE_DETAIL_COLUMNS)
        if not out.empty:
            out = out.sort_values(
                ['General_Flag', 'canonical_template', 'Subject']
            ).reset_index(drop=True)
        csv_path = os.path.join(
            self.analyze_flags_dir,
            f"resolved_values_per_entry_{network}.csv",
        )
        out.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}: {len(out)} resolved entries.")
        value_available_statuses = {
            self.STATUS_FILLED,
            self.STATUS_STILL_BLANK,
            self.STATUS_MISSING_CODE,
        }
        paired = out[
            out['before_value_available'].eq(True)
            & out['status'].isin(value_available_statuses)
        ].copy()
        paired_path = os.path.join(
            self.analyze_flags_dir,
            f"resolved_before_after_values_{network}.csv",
        )
        paired.to_csv(paired_path, index=False)
        print(
            f"Wrote {paired_path}: {len(paired)} entries with known before "
            "and current values."
        )

    def _missing_code_set(self):
        """Union of Utils.missing_code_set and str-coerced
        Utils.missing_code_list, matching extract_blank_flags.py
        comparison style."""
        base = set(self.utils.missing_code_set)
        base.update(str(c) for c in self.utils.missing_code_list)
        return base

    def _current_values_csv_path(self, network, timepoint):
        """Return the current-data source that originally fed the flag."""

        if normalize_timepoint(timepoint) == 'multiple_timepoints':
            return os.path.join(
                self.depend_path, f'multi_tp_{network}_combined.csv')
        return self._combined_csv_path(network, timepoint)

    def _combined_csv_path(self, network, timepoint):
        """Match qc_forms_main.py:102 / extract_blank_flags.py:
        monthN -> month_N, floating -> floating_forms,
        PRONET -> ProNET. Timepoint is normalized first as defense
        in depth (estimate_resolved.py already normalizes upstream)."""
        tp_munged = normalize_timepoint(timepoint).replace(
            'month', 'month_'
        ).replace('floating', 'floating_forms')
        net_munged = network.replace('PRONET', 'ProNET')
        return (
            f"{self.comb_csv_path}AMPSCZ-combined-redcap_"
            f"{tp_munged}_{net_munged}-day1to1.csv"
        )

    def _load_combined_csv(self, csv_path):
        if not os.path.isfile(csv_path):
            return None, f"file not found: {csv_path}"
        try:
            df = pd.read_csv(
                csv_path,
                keep_default_na=False,
                encoding='utf-8',
                on_bad_lines='error',
                dtype=str,
            )
            return df, None
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            return None, f"{type(e).__name__}: {str(e)[:200]}"


if __name__ == '__main__':
    FlagAnalytics().run_script()
