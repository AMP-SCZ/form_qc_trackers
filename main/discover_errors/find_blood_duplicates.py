"""
One-off blood ID / barcode duplicate scan over
dependencies/Prescient_bloods_combined.xlsx.

Each row is a (subject, vial-slot) record. Two distinct checks:

1. ID variables (`chrblood_wbid` / `seid` / `plid` / `bcid`) — globally
   unique. Any value appearing on >1 row is flagged regardless of
   whether the rows belong to the same subject or different subjects.

2. Box barcodes (`chrblood_wbbox` / `sebox` / `plbox` / `bcbox`) —
   intentionally shared across vials (multiple vials per box), so only
   the (barcode, matching-position) tuple needs to be unique. A duplicate
   barcode is flagged only if the paired position variable
   (`chrblood_wbpos` / `sepos` / `plpos` / `bcpos`) is also equal —
   i.e. two vials claiming the same slot in the same box.

Missing-value handling mirrors `utils.Utils.missing_code_list` and the
`extra_skip_strings` set used by `cross_subject_blood_duplicate_check` in
`qc_types/fluid_checks.py` so behavior is consistent with the production
pipeline.
"""

import os
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.realpath(__file__))
INPUT_PATH = os.path.join(HERE, 'dependencies', 'PrescientStudy_Prescient_BloodSpecimenCombined_05.08.2026.csv')

ID_VARS = [
    'chrblood_wbid',
    'chrblood_seid',
    'chrblood_plid',
    'chrblood_bcid',
]

# (barcode_var, position_var) — duplicate flagged only when both values match.
BARCODE_POSITION_PAIRS = [
    ('chrblood_wbbox', 'chrblood_wbpos'),
    ('chrblood_sebox', 'chrblood_sepos'),
    ('chrblood_plbox', 'chrblood_plpos'),
    ('chrblood_bcbox', 'chrblood_bcpos'),
]

# Mirrors `Utils.missing_code_list` (utils/utils.py:20) — every numeric
# and string form of every sentinel. Any value here, raw or stringified,
# is treated as "no data" and skipped before duplicate detection.
_MISSING_RAW = {
    -3, -9, -99, 999,
    -3.0, -9.0, -99.0, 999.0,
    '-3', '-9', '-99', '999',
    '-3.0', '-9.0', '-99.0', '999.0',
    '1909-09-09', '1903-03-03', '1901-01-01',
}

# String sentinels (compared after `.strip().lower()`). Adds case
# variants of pandas-ish missing markers (`'nan'`, `'nat'`, etc.) on top
# of the production list. These are the things `pd.read_excel` or upstream
# data-cleanup steps occasionally leak through as literal strings.
_MISSING_STRINGS_LOWER = {
    '', 'na', 'n/a', 'nan', 'nat', 'none', 'null',
    '-3', '-9', '-99', '999',
    '-3.0', '-9.0', '-99.0', '999.0',
    '1909-09-09', '1903-03-03', '1901-01-01',
}


def _is_missing_string(s_lower):
    return s_lower in _MISSING_STRINGS_LOWER


def normalize(val):
    """Return a canonical string for `val`, or None if it should be skipped."""
    # 1. Pandas-native missing markers: np.nan, pd.NA, pd.NaT, None.
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        # `pd.isna` can choke on exotic types; fall through and let the
        # string path catch it.
        pass

    # 2. Raw-value membership in the production missing-code list (catches
    #    numeric -99, 999, -3.0, etc. before any string conversion).
    if val in _MISSING_RAW:
        return None

    # 3. String-normalize. Strip both ASCII whitespace and the non-breaking
    #    space (\xa0), which Excel exports occasionally include.
    s = str(val).strip().strip('\xa0').strip()
    if not s:
        return None
    if _is_missing_string(s.lower()):
        return None

    # 4. Collapse int-shaped floats so "12345" / "12345.0" / 12345 / 12345.0
    #    all hash to the same bucket. Without this, the same ID exported
    #    as int in one row and float in another would not be detected as
    #    a duplicate. (Mirrors `fluid_checks.py:272-279`.)
    try:
        f = float(s)
        if f != f:  # NaN; defensive for the path where pd.isna missed it.
            return None
        if f.is_integer():
            s = str(int(f))
    except (ValueError, OverflowError):
        pass

    # 5. Re-check the missing list AFTER collapse. Without this, the float
    #    `999.0` slips through step 3 (`'999.0'` not in the lower set on
    #    its own — wait, it is — but `'-3.0'`/`'-99.0'` likewise need
    #    coverage at both ends because the collapse rewrites the string).
    if _is_missing_string(s.lower()):
        return None

    # 6. Blood team treats PRONET-prefixed values as non-IDs (see
    #    `barcode_format_check` and `cross_subject_blood_duplicate_check`
    #    in fluid_checks.py). PRESCIENT data shouldn't have these but
    #    matching production is cheap.
    if 'pronet' in s.lower():
        return None

    # 7. Final defensive guard: never return the literal string "nan" /
    #    "nat" / etc. as a real value, regardless of how we got here.
    if _is_missing_string(s.lower()):
        return None

    return s


def _safe_str(v):
    """Stringify metadata for output — empty string for NaN/None."""
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    s = str(v)
    # Guard against numpy/pandas NaN that stringifies to 'nan'.
    if s.strip().lower() in {'nan', 'nat', 'none'}:
        return ''
    return s


def _build_location_row(df, idx):
    row = df.iloc[idx]
    return (
        f"{_safe_str(row.get('subjectkey', ''))} "
        f"(Row#={_safe_str(row.get('Row#', ''))}, "
        f"visit={_safe_str(row.get('visit', ''))}, "
        f"drawdate={_safe_str(row.get('chrblood_drawdate', ''))})"
    )


def _emit_dup_row(df, var, value, idxs, *, position_variable='', position_value=''):
    # Filter NaN subjectids out of the distinct-subject set so they don't
    # masquerade as a "different subject" and falsely flip cross_subject.
    subj_strs = [_safe_str(df.at[i, 'subjectkey']) for i in idxs]
    distinct_subjects = sorted({s for s in subj_strs if s})
    n_missing_subj = sum(1 for s in subj_strs if not s)
    return {
        'variable': var,
        'position_variable': position_variable,
        'value': value,
        'position_value': position_value,
        'occurrences': len(idxs),
        'distinct_subjects': len(distinct_subjects),
        'cross_subject': len(distinct_subjects) > 1,
        'subjects': '; '.join(distinct_subjects),
        'locations': ' | '.join(_build_location_row(df, i) for i in idxs),
    }


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    out_rows = []
    summary = []

    # ---- Check 1: ID vars must be globally unique ----
    for var in ID_VARS:
        if var not in df.columns:
            print(f"[skip] column not in file: {var}")
            continue

        groups = {}
        skipped_missing = 0
        for idx, raw in df[var].items():
            key = normalize(raw)
            if key is None:
                skipped_missing += 1
                continue
            groups.setdefault(key, []).append(idx)

        dup_values = {v: idxs for v, idxs in groups.items() if len(idxs) > 1}
        summary.append((
            var, '',
            len(dup_values),
            sum(len(i) for i in dup_values.values()),
            skipped_missing,
        ))

        for value, idxs in dup_values.items():
            out_rows.append(_emit_dup_row(df, var, value, idxs))

    # ---- Check 2: barcode duplicates only flagged when paired position matches ----
    for barcode_var, pos_var in BARCODE_POSITION_PAIRS:
        if barcode_var not in df.columns:
            print(f"[skip] column not in file: {barcode_var}")
            continue
        if pos_var not in df.columns:
            print(f"[skip] position column not in file: {pos_var}")
            continue

        groups = {}
        skipped_missing = 0
        for idx in df.index:
            bc_key = normalize(df.at[idx, barcode_var])
            pos_key = normalize(df.at[idx, pos_var])
            # Need both values to evaluate "same barcode AND same position".
            if bc_key is None or pos_key is None:
                skipped_missing += 1
                continue
            groups.setdefault((bc_key, pos_key), []).append(idx)

        dup_values = {k: idxs for k, idxs in groups.items() if len(idxs) > 1}
        summary.append((
            barcode_var, pos_var,
            len(dup_values),
            sum(len(i) for i in dup_values.values()),
            skipped_missing,
        ))

        for (bc_value, pos_value), idxs in dup_values.items():
            out_rows.append(_emit_dup_row(
                df, barcode_var, bc_value, idxs,
                position_variable=pos_var, position_value=pos_value,
            ))

    out_df = pd.DataFrame(out_rows)
    if not out_df.empty:
        out_df = out_df.sort_values(
            ['cross_subject', 'variable', 'value'],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(HERE, f'blood_duplicates_{ts}.csv')
    out_df.to_csv(out_path, index=False)

    print()
    print("Per-variable counts:")
    for var, pos_var, n_values, n_rows, n_missing in summary:
        label = f"{var}+{pos_var}" if pos_var else var
        print(f"  {label:<35s}  {n_values:>5d} duplicated values  "
              f"({n_rows} affected rows, {n_missing} missing/skipped)")
    print()
    print(f"Wrote {len(out_df)} duplicate-value rows -> {out_path}")


if __name__ == '__main__':
    main()
