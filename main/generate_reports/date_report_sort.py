import re
import pandas as pd

"""
Pure helper for ordering the tracker's "Date Report" tab (the backward
visit-date flags from the `date` qc_type) greatest-to-least by days
apart.

WHY PARSE THE MESSAGE INSTEAD OF A NUMERIC COLUMN
  The day-count is not a standalone column in combined_qc_flags, and a custom
  numeric column would still be dropped before this tracker reads it — just at
  a different, later step than it once was. (Historically it was dropped in
  reconciliation: append_all_cols only copied columns that survived the
  old/new outer-merge with an `_old`/`_new` suffix, so a new-only column,
  merging UN-suffixed, was skipped and never reached current_output. That gap
  is now closed by an `elif col in present_cols` fallback in append_all_cols,
  so new-only columns DO survive into current_output.) The remaining reason
  still holds: create_trackers.convert_to_shared_format selects ONLY the
  columns listed in formatted_column_names via
  `merged_df[list(columns_names.values())]`, so any column not registered
  there is dropped at the Excel step regardless. The flag message IS a core,
  schema-stable, always-registered column, so the magnitude (already written
  as "... is N day(s) before ...") rides along reliably and is parsed here.

Kept import-light (re + pandas) so it is unit-testable without
constructing CreateTrackers (which imports dropbox + Utils).
"""

# Matches the day-count DateChecks writes into each flag message:
# "... is 17 day(s) before the visit date at ...".
_DAYS_PATTERN = re.compile(r'(\d+)\s*day\(s\)\s*before')


def _days_from_flag(text):
    """Largest day-count in a (possibly '|'-joined) flag message, or -1
    when none is present so unparseable rows sort last."""
    nums = _DAYS_PATTERN.findall(str(text))
    return max((int(n) for n in nums), default=-1)


def sort_date_report_rows(df, flags_col='Flags',
                          resolved_cols=('Date Resolved', 'Manually Resolved'),
                          days_col='Days Apart'):
    """
    Order the Date Report tab greatest-to-least by days apart and add a
    visible `days_col` so the ordering is legible.

    Unresolved rows sort above resolved / manually-resolved ones (matching
    the resolved-to-bottom arrangement convert_to_shared_format already
    applies), each block ordered by days apart descending. Rows with no
    parseable count sort last within their block and show a blank
    `days_col`. Returns `df` unchanged when it is empty or has no
    `flags_col`.
    """
    if df is None or df.empty or flags_col not in df.columns:
        return df

    days = df[flags_col].map(_days_from_flag)
    resolved = pd.Series(False, index=df.index)
    for col in resolved_cols:
        if col in df.columns:
            resolved = resolved | (df[col].astype(str).str.strip() != '')

    out = df.copy()
    # Blank rather than -1 for unparseable rows in the displayed column.
    out[days_col] = days.where(days >= 0, '')
    out = (
        out.assign(_resolved=resolved.astype(int), _days=days)
           .sort_values(['_resolved', '_days'],
                        ascending=[True, False], kind='mergesort')
           .drop(columns=['_resolved', '_days'])
           .reset_index(drop=True)
    )
    return out
