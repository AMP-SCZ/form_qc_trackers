"""Shared helpers for parsing and canonicalizing Specific_Flags
entries. Used by ``estimate_resolved.py`` when building the entry-
level merged history and by downstream consumers
(``flag_analytics.py``, ``graph_recovered_flags.py``) when re-doing
the same parsing on already-canonicalized rows.

Centralized here so a regex tweak affects both sides of the producer/
consumer boundary at once.
"""

import re

# Each Specific_Flags entry is "{variable} : {message}" joined by
# " | " (per form_check.py:353 / create_trackers.py:342). The producer
# always uses the space-colon-space form, but historical revisions
# / hand-edits sometimes drift to "var:msg" or "var :msg". The split
# below accepts any of those variants — first colon wins, surrounding
# whitespace is optional. Returns ('', '') if no colon is present so
# the caller can drop the entry.
_ENTRY_SPLIT_RE = re.compile(r'^\s*([^:]+?)\s*:\s*(.*?)\s*$', re.DOTALL)

# Used by _canonicalize_message — pre-compiled so repeated calls don't
# rebuild them per entry.
_DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')
_PAREN_RE = re.compile(r'\([^)]*\)')
_BRACKET_RE = re.compile(r'\[[^\]]*\]')
_BARE_NUMBER_RE = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])')

# Subject IDs / GUIDs / barcodes — uppercase token containing at least
# one digit (e.g. "SUB001", "NDAR_INV12345678", "PRESCIENT001"). Without
# this pass, free-form interpolations like
#   "subject(s): SUB001 at baseline, SUB002 at month1"
# (the {conflict_str} in duplicate-barcode templates) fragment into
# thousands of distinct canonical templates because the IDs vary per row.
_SUBJECT_ID_RE = re.compile(r'\b[A-Z][A-Z0-9_]*\d[A-Z0-9_]*\b')

# Recognized timepoint vocabulary. Matches both V2 ('screening',
# 'baseline', 'monthN', 'floating', 'conversion') and pre-V2 ('screen',
# 'baseln', 'month_N') forms. Used after _SUBJECT_ID_RE so that
# "<id> at baseline" / "<id> at month1" / "<id> at conversion" all
# collapse to "<id> at <tp>".
_TIMEPOINT_RE = re.compile(
    r'\b(?:screening|screen|baseline|baseln|month[\s_]?\d+|floating|conversion)\b',
    re.IGNORECASE,
)

# Final pass: collapse any sequence of "<id>" / "<id> at <tp>" entries
# (singular or comma-separated list) to a single "<ids>" token so the
# number of conflicts in a {conflict_str} interpolation doesn't
# fragment the canonical. After this:
#   "<id> at <tp>"                     -> "<ids>"
#   "<id> at <tp>, <id> at <tp>"       -> "<ids>"
#   "<id>, <id>, <id>"                 -> "<ids>"
# Standalone "<id>" outside a list still collapses to "<ids>" — the
# placeholder name is slightly off for the singular case but it
# preserves the grouping invariant.
_ID_LIST_RE = re.compile(
    r'<id>(?:\s+at\s+<tp>)?(?:\s*,\s*<id>(?:\s+at\s+<tp>)?)*'
)

# Pre-V2 timepoint vocabulary that needs collapsing to V2 so the same
# logical flag from V1 and V2 dedupes at the entry-level key. Mirrors
# extract_blank_flags.py:_normalize_timepoint.
_MONTH_TP_RE = re.compile(r'month[\s_]*(\d+)')

# Separators stripped from form names before cross-version comparison.
# V1 named forms with different capitalization / underscores / spaces
# than V2 (e.g. 'SOFAS Screening' vs 'sofas_screening'), so any
# separator-and-case-only difference must compare equal.
_FORM_SEPARATOR_RE = re.compile(r'[\s_\-]+')


def split_entry(entry):
    """Split a single Specific_Flags entry into (variable, message).

    Accepts any of these whitespace variants around the colon:
      ``var : msg``  (canonical)
      ``var:msg``    (no whitespace)
      ``var :msg``   (left-only)
      ``var: msg``   (right-only)

    Returns ('', '') if the entry has no colon at all so the caller
    can skip it without crashing.
    """
    if not entry:
        return '', ''
    m = _ENTRY_SPLIT_RE.match(entry)
    if not m:
        return '', ''
    var_part = m.group(1).strip()
    msg_part = m.group(2).strip()
    return var_part, msg_part


def normalize_timepoint(tp):
    """Collapse pre-V2 timepoint forms to the V2 vocabulary used by
    qc_forms_main.py:102 ('screening', 'baseline', 'monthN', 'floating',
    'conversion'). Without this, the same logical (Subject, Timepoint,
    Form, variable) flag from V1's 'screen' and V2's 'screening' would
    dedupe as two separate entries.
    """
    if tp is None:
        return ''
    s = str(tp).strip()
    if not s:
        return ''
    s_lower = s.lower()
    if s_lower == 'baseln':
        return 'baseline'
    if s_lower == 'screen':
        return 'screening'
    m = _MONTH_TP_RE.fullmatch(s_lower)
    if m:
        return f'month{m.group(1)}'
    return s_lower


def normalize_form_name(val):
    """Collapse a form name to its cross-version comparison key:
    lowercase with all spaces / underscores / hyphens removed, so
    V1's ``'SOFAS Screening'`` / ``'Sofas_Screening'`` and V2's
    ``'sofas_screening'`` all become ``'sofasscreening'``. Consumers
    that want pretty labels map these keys back to the canonical V2
    spelling (see flag_analytics._clinical_form_lookup)."""
    if val is None:
        return ''
    return _FORM_SEPARATOR_RE.sub('', str(val).strip().lower())


def normalize_general_flag(val):
    """Strip any ``' : status'`` suffix V1 sometimes appended after the
    form name (the regex tolerates zero whitespace before the colon so
    both ``'form : ok'`` and ``'form:ok'`` collapse to ``'form'``),
    then collapse to the cross-version comparison key via
    ``normalize_form_name`` so V1/V2 capitalization, underscore, and
    space differences dedupe to one entry key.
    """
    if val is None:
        return ''
    base = re.sub(r'\s*:.*$', '', str(val)).strip()
    return normalize_form_name(base)


def canonicalize_message(variable, message):
    """Reduce a (variable, message) pair to a canonical template.

    Steps (order matters — see comments inline):
      1. Replace the prefix variable name anywhere in the message with
         ``<var>``. Many templates embed the variable name in the
         message itself (e.g. ``chrsofas_lowscore (-9) is outside...``);
         without this, the same check against different variables
         canonicalizes differently.
      2. Dates ``YYYY-MM-DD`` -> ``<date>``. Must run BEFORE the
         bare-number pass, else ``2026-04-15`` collapses to
         ``<num>-<num>-<num>``.
      3. Paren contents -> ``(<val>)``. Catches interpolated state
         like ``(chrscid_a25 = 3, chrscid_a51 = , chrscid_d26 = -3)``
         and ``(25)`` without per-template knowledge.
      4. Bracket contents -> ``[<range>]``. Catches valid-range
         expressions like ``[0, 100]``.
      5. Subject-style IDs (UPPER+digits) -> ``<id>``. Catches free-form
         interpolations like ``subject(s): SUB001`` and GUID values
         like ``NDAR_INV12345678`` that fragment templates otherwise.
      6. Timepoint vocabulary -> ``<tp>``. Catches the "at baseline" /
         "at month1" / "at conversion" tails attached to subject IDs
         in conflict lists.
      7. Bare signed numbers -> ``<num>``. The lookbehind / lookahead
         ensure embedded numbers inside identifiers
         (``chrnsipr_item1_rating``, ``chrpsychs_fu_4d30``) are NOT
         touched.

    Returns ``'<var> : <canonicalized message>'`` so the full entry
    template is recoverable from one string.
    """
    if not message:
        return '<var> : '
    msg = message
    if variable:
        msg = re.sub(rf'\b{re.escape(variable)}\b', '<var>', msg)
    msg = _DATE_RE.sub('<date>', msg)
    msg = _PAREN_RE.sub('(<val>)', msg)
    msg = _BRACKET_RE.sub('[<range>]', msg)
    msg = _SUBJECT_ID_RE.sub('<id>', msg)
    msg = _TIMEPOINT_RE.sub('<tp>', msg)
    msg = _ID_LIST_RE.sub('<ids>', msg)
    msg = _BARE_NUMBER_RE.sub('<num>', msg)
    return f'<var> : {msg}'


def explode_specific_flags(specific_flags_str):
    """Yield (variable, message, canonical_template) tuples for every
    well-formed entry in a Specific_Flags string. Empty entries and
    entries missing a colon are skipped silently — counting drops is
    the caller's responsibility (see ``estimate_resolved.py``'s
    rejection counters).
    """
    if not specific_flags_str:
        return
    for entry in str(specific_flags_str).split('|'):
        entry = entry.strip()
        if not entry:
            continue
        variable, message = split_entry(entry)
        if not variable:
            continue
        canonical = canonicalize_message(variable, message)
        yield variable, message, canonical
