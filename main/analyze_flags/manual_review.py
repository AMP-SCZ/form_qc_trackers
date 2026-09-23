"""IO helpers for the two manual-review artifacts produced/consumed by
the analyze_flags scripts:

* ``template_mapping.xlsx`` — one cross-network workbook listing every
  canonical template (sheet 'templates') and every flag variable
  (sheet 'variables') ever observed, each with a ``maps_to`` column.
  The operator fills ``maps_to`` to declare "treat A as B" (renamed
  flags / renamed variables). ``estimate_resolved.py`` loads the
  mapping at the start of a run and applies it BEFORE entry keying,
  so a mapped rename no longer produces a fake resolve + fake new
  flag pair; it then re-seeds the workbook afterwards, preserving
  every ``maps_to`` the operator already entered.

* ``jumps_{network}.xlsx`` — one workbook per network listing every
  mass addition/removal detected between consecutive Dropbox
  revisions (sheet 'jumps', with an ``include`` column) plus the
  exact entry keys each jump touched (sheet 'affected_entries').
  ``flag_analytics.py`` and ``graph_recovered_flags.py`` call
  ``load_jump_decisions`` to apply the operator's decisions:
  a jump is INCLUDED only when its ``include`` cell is affirmative
  ('yes' / 'y' / 'true' / '1' / 'include' / 'keep', any case).
  Blank or anything else = excluded, so the first run (before any
  manual review) behaves like an automatic spike filter.
"""

import os
import tempfile

import pandas as pd

from analyze_flags.paths import (
    jumps_workbook_basename,
    template_mapping_basename,
)

# Order matters: this is the dedup key used by estimate_resolved.py
# and must match the history CSV's column names.
ENTRY_KEY_COLUMNS = [
    'Subject', 'Timepoint', 'General_Flag', 'variable',
    'canonical_template',
]

_AFFIRMATIVE = frozenset(['yes', 'y', 'true', '1', 'include', 'keep'])

_TEMPLATE_SHEET = 'templates'
_VARIABLE_SHEET = 'variables'
_JUMPS_SHEET = 'jumps'
_AFFECTED_SHEET = 'affected_entries'


class JumpDecisionIntegrityError(RuntimeError):
    """A jump workbook cannot safely support the requested exclusions."""


def is_affirmative(value) -> bool:
    return str(value).strip().lower() in _AFFIRMATIVE


def _read_sheet(path, sheet_name):
    """Read one sheet as all-string cells ('' for blanks). Returns an
    empty DataFrame when the file or sheet doesn't exist."""
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(
            path, sheet_name=sheet_name, keep_default_na=False, dtype=str,
        )
    except ValueError:
        # Sheet missing (e.g. hand-trimmed workbook).
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _atomic_write_workbook(path, sheets):
    """Write a workbook to a sibling temporary file, then replace it.

    Keeping the temporary file on the same filesystem makes ``os.replace``
    atomic: a crash or failed Excel serialization therefore cannot truncate a
    previously valid manual-review workbook.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".xlsx", dir=directory,
    )
    os.close(fd)
    try:
        with pd.ExcelWriter(temporary_path, engine='openpyxl') as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _write_workbook(path, sheets):
    """Atomically write {sheet_name: DataFrame}.

    If the destination is locked (commonly because it is open in Excel on
    Windows), atomically write ``<path>.new.xlsx`` instead so both the old
    workbook and the newly generated result remain intact.
    """
    try:
        _atomic_write_workbook(path, sheets)
        return path
    except (PermissionError, OSError) as e:
        fallback = f"{path}.new.xlsx"
        print(
            f"WARNING: could not write {path} ({type(e).__name__}: {e}); "
            f"writing to {fallback} instead. Close the file in Excel and "
            f"merge manually."
        )
        _atomic_write_workbook(fallback, sheets)
        return fallback


# ----------------------------------------------------------------------
# Template / variable mapping
# ----------------------------------------------------------------------

def template_mapping_path(analyze_flags_dir):
    return os.path.join(analyze_flags_dir, template_mapping_basename())


def _flatten_mapping(raw_map):
    """Resolve chains (A->B, B->C becomes A->C) with a small hop cap
    so an accidental cycle can't hang the run."""
    flat = {}
    for src in raw_map:
        target = raw_map[src]
        seen = {src}
        hops = 0
        while target in raw_map and target not in seen and hops < 10:
            seen.add(target)
            target = raw_map[target]
            hops += 1
        if target != src:
            flat[src] = target
    return flat


def _sheet_to_mapping(df, key_col):
    if df.empty or key_col not in df.columns or 'maps_to' not in df.columns:
        return {}
    raw = {}
    for key, target in zip(df[key_col], df['maps_to']):
        key_s = str(key).strip()
        target_s = str(target).strip()
        if key_s and target_s and key_s != target_s:
            raw[key_s] = target_s
    return _flatten_mapping(raw)


def load_template_mapping(analyze_flags_dir):
    """Returns (template_map, variable_map) — both possibly empty
    dicts when the workbook doesn't exist yet or has no maps_to
    entries filled in."""
    path = template_mapping_path(analyze_flags_dir)
    template_map = _sheet_to_mapping(
        _read_sheet(path, _TEMPLATE_SHEET), 'canonical_template'
    )
    variable_map = _sheet_to_mapping(
        _read_sheet(path, _VARIABLE_SHEET), 'variable'
    )
    if template_map or variable_map:
        print(
            f"Loaded manual mapping from {path}: "
            f"{len(template_map)} template merges, "
            f"{len(variable_map)} variable merges."
        )
    return template_map, variable_map


def update_template_mapping(analyze_flags_dir, seen_templates,
                            seen_variables):
    """Re-seed the mapping workbook with everything observed this run,
    PRESERVING the operator's existing maps_to entries — including
    rows for templates/variables not seen this run (older history the
    operator may still care about stays mapped).

    ``seen_templates``: {canonical_template: {'n_entries', 'networks'
    (set), 'example_form', 'example_variable'}}.
    ``seen_variables``: {variable: {'n_entries', 'networks' (set),
    'example_form'}}.
    """
    path = template_mapping_path(analyze_flags_dir)

    def build_rows(seen, key_col, existing_df, extra_cols):
        existing = {}
        if not existing_df.empty and key_col in existing_df.columns:
            for _, row in existing_df.iterrows():
                k = str(row.get(key_col, '')).strip()
                if k:
                    existing[k] = row
        rows = []
        for key, info in seen.items():
            old = existing.pop(key, None)
            row = {
                key_col: key,
                'maps_to': (
                    str(old.get('maps_to', '')).strip()
                    if old is not None else ''
                ),
                'n_entries': info.get('n_entries', 0),
                'networks': '+'.join(sorted(info.get('networks', set()))),
            }
            for c in extra_cols:
                row[c] = info.get(c, '')
            rows.append(row)
        # Rows the operator mapped previously but that weren't seen
        # this run — keep them so the mapping survives partial runs.
        for key, old in existing.items():
            row = {
                key_col: key,
                'maps_to': str(old.get('maps_to', '')).strip(),
                'n_entries': old.get('n_entries', ''),
                'networks': old.get('networks', ''),
            }
            for c in extra_cols:
                row[c] = old.get(c, '')
            rows.append(row)
        cols = [key_col, 'maps_to', 'n_entries', 'networks'] + extra_cols
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df
        # Most frequent first — those are the ones worth reviewing.
        df['_sort'] = pd.to_numeric(df['n_entries'], errors='coerce').fillna(0)
        df = (
            df.sort_values('_sort', ascending=False)
            .drop(columns=['_sort'])
            .reset_index(drop=True)
        )
        return df

    templates_df = build_rows(
        seen_templates, 'canonical_template',
        _read_sheet(path, _TEMPLATE_SHEET),
        ['example_form', 'example_variable'],
    )
    variables_df = build_rows(
        seen_variables, 'variable',
        _read_sheet(path, _VARIABLE_SHEET),
        ['example_form'],
    )
    written = _write_workbook(path, {
        _TEMPLATE_SHEET: templates_df,
        _VARIABLE_SHEET: variables_df,
    })
    print(
        f"Wrote {written}: {len(templates_df)} templates, "
        f"{len(variables_df)} variables (fill in maps_to to merge "
        f"renamed flags, then re-run estimate_resolved.py)."
    )


# ----------------------------------------------------------------------
# Jump workbooks
# ----------------------------------------------------------------------

def jumps_workbook_path(analyze_flags_dir, network):
    return os.path.join(analyze_flags_dir, jumps_workbook_basename(network))


JUMP_SHEET_COLUMNS = [
    'jump_id', 'include', 'source', 'direction', 'observed_at',
    'previous_revision', 'n_entries', 'open_entries_after',
    'sample_entries',
]


def write_jumps_workbook(analyze_flags_dir, network, jump_rows,
                         affected_rows):
    """``jump_rows``: list of dicts with JUMP_SHEET_COLUMNS keys
    (except ``include``, which this function fills). ``affected_rows``:
    list of dicts with 'jump_id' + ENTRY_KEY_COLUMNS keys.

    The operator's existing ``include`` decisions are carried over by
    jump_id, which is stable across runs (built from source +
    revision timestamp + direction)."""
    path = jumps_workbook_path(analyze_flags_dir, network)
    existing = _read_sheet(path, _JUMPS_SHEET)
    prior_include = {}
    if not existing.empty and 'jump_id' in existing.columns \
            and 'include' in existing.columns:
        for jid, inc in zip(existing['jump_id'], existing['include']):
            jid_s = str(jid).strip()
            inc_s = str(inc).strip()
            if jid_s and inc_s:
                prior_include[jid_s] = inc_s

    rows = []
    for jump in jump_rows:
        row = dict(jump)
        row['include'] = prior_include.get(row['jump_id'], '')
        rows.append(row)
    jumps_df = pd.DataFrame(rows, columns=JUMP_SHEET_COLUMNS)
    if not jumps_df.empty:
        jumps_df = jumps_df.sort_values('observed_at').reset_index(drop=True)

    affected_df = pd.DataFrame(
        affected_rows, columns=['jump_id'] + ENTRY_KEY_COLUMNS,
    )

    written = _write_workbook(path, {
        _JUMPS_SHEET: jumps_df,
        _AFFECTED_SHEET: affected_df,
    })
    undecided = int((jumps_df['include'] == '').sum()) if not jumps_df.empty else 0
    print(
        f"Wrote {written}: {len(jumps_df)} jumps "
        f"({undecided} awaiting an include decision), "
        f"{len(affected_df)} affected entries."
    )


def load_jump_decisions(analyze_flags_dir, network):
    """Returns {'excluded_added': set, 'excluded_removed': set,
    'n_jumps': int, 'n_undecided': int}.

    The sets contain ENTRY_KEY_COLUMNS tuples for entries touched by
    jumps the operator has NOT marked include-affirmative. Missing
    workbook -> empty sets (nothing excluded) so consumers degrade
    gracefully when estimate_resolved.py hasn't been re-run yet."""
    result = {
        'excluded_added': set(),
        'excluded_removed': set(),
        'n_jumps': 0,
        'n_undecided': 0,
    }
    path = jumps_workbook_path(analyze_flags_dir, network)
    if not os.path.isfile(path):
        return result
    jumps_df = _read_sheet(path, _JUMPS_SHEET)
    required_jump_columns = ['jump_id', 'include', 'direction', 'n_entries']
    missing_jump_columns = [
        c for c in required_jump_columns if c not in jumps_df.columns
    ]
    if missing_jump_columns:
        raise JumpDecisionIntegrityError(
            f"{path} '{_JUMPS_SHEET}' sheet is missing columns "
            f"{missing_jump_columns}. Refusing to interpret a malformed "
            "decision workbook as having no exclusions."
        )
    if jumps_df.empty:
        return result
    result['n_jumps'] = len(jumps_df)

    excluded_direction = {}
    seen_jump_ids = set()
    for _, row in jumps_df.iterrows():
        include_val = str(row.get('include', '')).strip()
        if not include_val:
            result['n_undecided'] += 1
        jid = str(row.get('jump_id', '')).strip()
        direction = str(row.get('direction', '')).strip().lower()
        if not jid or direction not in ('added', 'removed'):
            raise JumpDecisionIntegrityError(
                f"{path} contains a jump row with an empty jump_id or invalid "
                f"direction {direction!r}."
            )
        if jid in seen_jump_ids:
            raise JumpDecisionIntegrityError(
                f"{path} contains duplicate jump_id {jid!r}."
            )
        seen_jump_ids.add(jid)
        if not is_affirmative(include_val):
            excluded_direction[jid] = direction

    if not excluded_direction:
        return result
    affected_df = _read_sheet(path, _AFFECTED_SHEET)
    if affected_df.empty:
        raise JumpDecisionIntegrityError(
            f"{path} excludes {len(excluded_direction)} jump(s), but its "
            f"'{_AFFECTED_SHEET}' sheet is missing or empty. Refusing to "
            "apply a partial decision set."
        )
    missing = [c for c in ['jump_id'] + ENTRY_KEY_COLUMNS
               if c not in affected_df.columns]
    if missing:
        raise JumpDecisionIntegrityError(
            f"{path} '{_AFFECTED_SHEET}' sheet is missing columns {missing}. "
            "Refusing to apply a partial decision set."
        )

    affected_df = affected_df.copy()
    affected_df['jump_id'] = affected_df['jump_id'].astype(str).str.strip()
    affected_ids = set(affected_df['jump_id'])
    missing_ids = sorted(set(excluded_direction) - affected_ids)
    if missing_ids:
        sample = ', '.join(missing_ids[:5])
        raise JumpDecisionIntegrityError(
            f"{path} has excluded jump(s) with no affected entries: {sample}. "
            "Refusing to apply a partial decision set."
        )

    # ``n_entries`` is written by estimate_resolved and gives us an
    # independent completeness check.  Validate both row and unique-key counts
    # when that count is present; duplicated or truncated affected rows would
    # otherwise silently change which history entries are filtered.
    for _, jump_row in jumps_df.iterrows():
        jid = str(jump_row.get('jump_id', '')).strip()
        if jid not in excluded_direction:
            continue
        expected_raw = pd.to_numeric(
            jump_row.get('n_entries', ''), errors='coerce'
        )
        if (pd.isna(expected_raw) or float(expected_raw) < 0
                or not float(expected_raw).is_integer()):
            raise JumpDecisionIntegrityError(
                f"{path} jump {jid!r} has invalid n_entries "
                f"{jump_row.get('n_entries', '')!r}."
            )
        expected = int(expected_raw)
        matched = affected_df[affected_df['jump_id'] == jid]
        normalized_keys = matched[ENTRY_KEY_COLUMNS].apply(
            lambda column: column.astype(str).str.strip()
        )
        if normalized_keys.eq('').any(axis=None):
            raise JumpDecisionIntegrityError(
                f"{path} jump {jid!r} has a blank affected-entry key."
            )
        unique_keys = normalized_keys.drop_duplicates()
        if len(matched) != expected or len(unique_keys) != expected:
            raise JumpDecisionIntegrityError(
                f"{path} jump {jid!r} declares {expected} affected entries, "
                f"but the sheet contains {len(matched)} rows / "
                f"{len(unique_keys)} unique keys. Refusing to apply a "
                "partial decision set."
            )

    for row in affected_df.itertuples(index=False):
        direction = excluded_direction.get(str(row.jump_id).strip())
        if direction is None:
            continue
        key = tuple(
            str(getattr(row, c)).strip() for c in ENTRY_KEY_COLUMNS
        )
        result[f'excluded_{direction}'].add(key)
    return result


def entry_key_from_row(row):
    """Build the ENTRY_KEY_COLUMNS tuple from any object exposing the
    key columns as attributes (itertuples row) — same normalization
    as load_jump_decisions so set membership matches."""
    return tuple(str(getattr(row, c)).strip() for c in ENTRY_KEY_COLUMNS)
