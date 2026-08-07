"""Detector 10 - Rater scoring effects.

Finds raters who record a *specific value* of a scored variable far more
often than other raters -- i.e. scoring that looks driven by who is filling
out the form rather than by the subject in front of them.

How a rater is identified
-------------------------
Most AMPSCZ forms carry a rater-id field whose name contains ``ampscz_id``
or ``redcap_id`` (the AMPSCZ / REDCap id of the person who administered the
form). Those are the per-form "who scored this" columns. The authoritative
list lives in the REDCap data dictionary; we look it up via ``utils`` (the
data dictionary's ``Form Name`` also tells us which items each rater field
governs). The data dictionary is not always present (it is not part of a
local slice, and ``Utils()`` cannot even construct on Windows -- its
``absolute_path`` splits ``realpath`` on '/'), so the operational detection
is a column-name scan on each slice: any column whose name contains one of
``RATER_ID_SUBSTRINGS`` is a candidate. A cardinality guard
(``_valid_rater_col``) then drops columns that are really a per-SUBJECT id
(e.g. ``chric_ampscz_id`` is the participant "Record ID") by requiring at
least ``MIN_RATERS_FOR_VAR`` distinct values that each cover >=
``MIN_SUBJECTS_PER_RATER`` DISTINCT subjects -- a per-subject id has one
subject per value and fails this regardless of how many timepoints are
pooled.

What "scored by the rater" means (form scoping)
-----------------------------------------------
A form's rater field only tells us who scored *that form's* items, never
another form's. So each rater field is paired only with the items on its own
form: by ``Form Name`` when the data dictionary / ``grouped_variables.json``
``var_forms`` map is available, else by the rater field's name prefix
(``chrpanss_interview_ampscz_id`` -> ``chrpanss_*``). Cross-form attribution
would be wrong because a different person may have rated the other form.

The unit of analysis is a SUBJECT, not a form
---------------------------------------------
Forms are pooled across timepoints (a rater is the same person across visits)
and then collapsed to one row per (rater, subject) carrying that subject's
MODAL value for the item (ties -> smallest value, deterministically). This is
deliberate: the same subject rated by the same rater at every visit is NOT
independent evidence, so counting forms would both inflate the test and let a
SINGLE subject a rater always scores ``v`` masquerade as a scoring tendency.
Counting subjects removes both problems -- "a" below is a count of distinct
subjects.

The contrast (two tiers)
------------------------
For each (rater field, scored item, value v) we compare the fraction of a
rater's SUBJECTS recorded as ``item == v`` against a comparison group, with a
two-proportion z-test plus effect-size gates (the rater must use v on >=
``MIN_RATER_RATE`` of their subjects, at least ``MIN_RATE_DIFF`` above the
comparison rate, and on >= ``MIN_VALUE_SUBJECTS`` distinct subjects, so a
trivially-significant difference at large n -- or a single subject -- does not
flag).

  within-site  (preferred): for a site with >= 2 high-volume raters whose
               HOME (modal) site is that site, compare each such rater to the
               OTHER home raters at the same site. Same site == same subject
               population, so a difference is a rater fingerprint, not a
               subject difference. This is the part the site-level detector
               cannot see, and what makes a finding high-confidence. Each
               rater is screened at its home site only, against the other home
               raters -- so a stray cross-site form neither double-emits nor
               contaminates the comparison.

  cohort       (fallback): when a rater is the ONLY high-volume rater at its
               home site (the common single-rater-per-site case), OR its
               within-site contrast is degenerate because the co-rater is
               under-sampled at home, compare it to all other raters in the
               network. This restores coverage but CANNOT be separated from a
               site / subject-population effect, so such findings are labelled
               "site-confounded" with a lower severity. Review them alongside
               the site-level checks.

A rater is judged in exactly one tier, chosen by its home site, so a whole
site scoring high together does not light up each of its raters, and one
rater is never reported twice for the same (item, value).

Caveats (stated, not hidden)
----------------------------
* Raters are not randomly assigned to subjects; one may take the early or
  more severe cases. Even the within-site contrast shows the rater "differs
  from co-located raters", not proven bias.
* A rater is assigned to its modal site; a rater split evenly across sites
  resolves by site-code order (rare, and each rater is still judged once).
* The per-subject collapse uses each subject's MODAL value, so this detector
  targets raters for whom v is the TYPICAL score of many subjects. A rater who
  merely inflates v (e.g. scores it on ~40% of every subject's visits without
  it ever being the modal value) is by design NOT caught here -- that subtler,
  partial-leaning pattern is left to a per-visit-rate analysis to avoid
  re-introducing the repeated-visit dependence this collapse removes.
* Modal collapse is also not visit-count neutral: a subject's modal value
  concentrates on the per-VISIT plurality value as visit count grows, so a
  value that is merely the most common per-visit value gets a HIGHER modal rate
  for raters whose subjects have more visits. When visits/subject correlates
  with the rater (caseload maturity, attrition), a high-volume rater can look
  like they over-use that plurality value with NO real scoring difference.
  The within-site tier largely cancels this (co-located raters share a visit
  schedule); the cohort tier does not, which is a further reason its findings
  are flagged site-confounded. (A per-visit-rate statistic would be neutral to
  visit count and is the principled successor to modal collapse.)

Output is one row per (rater, item, value) -- a rater-level finding, not a
per-subject one. ``subjectid`` carries ``(rater <id>)`` and ``site_id`` the
rater's site, mirroring how the site detectors emit ``(site-level)`` rows.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .common import (
    Finding, ID_COLUMNS, calibrate, classify_columns, clean_missing_string,
    short, stack_by_network,
)

ANOMALY_TYPE = "rater_effect"

# A column is a rater-id field if its name contains one of these.
#
# AMPSCZ combined exports identify the person who entered each form via a
# per-form REDCap-user column (``chr<form>_redcap_user``) -- that is the real
# "who scored this" field in production data. The ``ampscz_id`` / ``redcap_id``
# forms are the demo / legacy convention and are kept so the synthetic demo and
# its regression tests still resolve. ``redcap_user`` is a substring of the
# plural ``redcap_users`` variant (chrsofas_redcap_users[_fu]), so that is
# covered automatically; ``redacp_user`` and ``redcao_user`` are real
# misspellings present in the AMPSCZ data dictionary, and ``username`` covers
# ``chrpred_username``. The structural ``redcap_event_name`` /
# ``redcap_repeat_*`` columns contain none of these substrings, so they are
# never mistaken for rater fields (and the cardinality guard would drop them
# regardless). (``chroasis_redcap`` -- a bare ``_redcap`` suffix -- is the one
# listed field this does not catch; bare ``redcap`` is intentionally excluded
# because it WOULD collide with the structural redcap_* columns.)
RATER_ID_SUBSTRINGS: Tuple[str, ...] = (
    "ampscz_id", "redcap_id",
    "redcap_user", "redacp_user", "redcao_user", "username",
)

# Known per-SUBJECT id fields that also contain "ampscz_id" but are NOT
# raters -- grouped_variables.json labels both "Record ID" (the participant
# id). Excluded by name in addition to the cardinality guard, which catches
# any other near-1:1 id column generically (it counts distinct SUBJECTS per
# value, so a subject id fails even when timepoints are pooled).
SUBJECT_ID_VAR_NAMES: Tuple[str, ...] = (
    "chric_ampscz_id", "chric_reconsent_ampscz_id",
)

# Minimum name-prefix length before we'll prefix-scope a rater field to items
# (so a stray "redcap_id" with prefix "redcap" doesn't grab unrelated cols).
PREFIX_MIN_LEN = 4

MIN_SUBJECTS_PER_RATER = 20      # a rater needs this many distinct subjects before
                                # we'll test them (a stable per-rater proportion;
                                # tune to your real per-rater subject volumes)
MIN_RATERS_FOR_VAR = 2          # need >= 2 qualifying raters (else nothing to
                                # contrast / the column is a per-subject id)
MIN_SITE_RATERS_FOR_WITHIN = 2  # >= 2 home raters at a site -> within-site tier
MIN_VALUE_TOTAL = 10            # a value must cover this many subjects in the
                                # comparison universe to be tested
MIN_VALUE_SUBJECTS = 5          # the rater must record the value for this many
                                # of their DISTINCT subjects (kills single-subject
                                # / repeated-visit artefacts)
MIN_RATER_RATE = 0.30           # the rater uses the value for at least this fraction
MIN_RATE_DIFF = 0.20            # at least this far above the comparison rate
# z gates are intentionally high. The search space (forms x items x values x
# sites x raters) is large, and the two-proportion normal approximation is
# anti-conservative at small per-rater n, so a modest gate lets chance spikes
# through. These thresholds sit in the empirical gap between pure-noise cells
# (|z| ~ 3-4 at a few dozen subjects) and genuine systematic effects (|z| well
# into the high single digits / double digits). Tune for your data volume.
Z_GATE_WITHIN = 4.0             # within-site: confound-controlled but smaller n
Z_GATE_COHORT = 5.0             # cohort: larger n but site-confounded -> stricter
# Severity calibration (shared family): z at the within-site gate maps to ~50.
Z_SEV_THRESHOLD = 4.0
Z_SEV_EXTREME = 12.0
# Site-confounded (single-rater-site) findings are dialled down vs the
# confound-controlled within-site findings.
SITE_CONFOUNDED_SEVERITY_FACTOR = 0.85
# A low_card column with more distinct values than this is treated as
# near-continuous (mis-classified) and skipped for per-value testing.
MAX_DISTINCT_VALUES = 25


# ---------------------------------------------------------------------------
# Rater-variable resolution (data dictionary -> grouped_variables -> prefix).
# ---------------------------------------------------------------------------

def _load_var_to_form() -> Tuple[Dict[str, str], str]:
    """Best-effort variable -> form map used to scope each rater field to the
    items it governs. Tries, in order:

      1. the REDCap data dictionary via ``utils`` (the requested path):
         field "Variable / Field Name" -> "Form Name";
      2. ``grouped_variables.json`` ``var_forms`` via ``utils``;
      3. nothing -> ``({}, "name_prefix_only")`` and the detector scopes by
         the rater field's name prefix instead.

    Lazy + fully guarded: ``utils.utils`` imports ``dropbox`` at module load
    and ``Utils()`` cannot construct on Windows, so an eager import would
    break this package's numpy/pandas-only self-containment and its demo.
    Read-only; never writes, never raises.
    """
    try:
        from utils.utils import Utils  # noqa: PLC0415 (lazy on purpose)
        u = Utils()
    except Exception:
        return {}, "name_prefix_only"

    try:
        dd = u.read_data_dictionary()
        if ("Variable / Field Name" in dd.columns
                and "Form Name" in dd.columns):
            m = {str(k): str(v) for k, v in
                 zip(dd["Variable / Field Name"], dd["Form Name"])
                 if str(k) != ""}
            if m:
                return m, "data_dictionary"
    except Exception:
        pass

    try:
        gv = u.load_dependency_json("grouped_variables.json")
        var_forms = gv.get("var_forms") if isinstance(gv, dict) else None
        if isinstance(var_forms, dict) and var_forms:
            return {str(k): str(v) for k, v in var_forms.items()}, \
                "grouped_variables.var_forms"
    except Exception:
        pass

    return {}, "name_prefix_only"


def is_rater_variable(name: str) -> bool:
    """True if ``name`` is a per-form rater-id field (not a subject id)."""
    if not isinstance(name, str):
        return False
    s = name.lower()
    if s in {c.lower() for c in ID_COLUMNS}:
        return False
    if s in {n.lower() for n in SUBJECT_ID_VAR_NAMES}:
        return False
    return any(sub in s for sub in RATER_ID_SUBSTRINGS)


def _valid_rater_col(rater_ser: pd.Series, subj_ser: pd.Series) -> bool:
    """Cardinality guard, subject-aware so it survives timepoint pooling.

    A usable rater field has >= MIN_RATERS_FOR_VAR distinct values that each
    cover >= MIN_SUBJECTS_PER_RATER DISTINCT subjects. A per-subject id column
    has exactly one subject per value -> zero qualifying values -> rejected,
    no matter how many timepoints are stacked (form counts would not catch
    this, since a subject id repeats once per visit)."""
    cleaned = clean_missing_string(rater_ser.astype(object))
    mask = cleaned.notna()
    if int(mask.sum()) == 0:
        return False
    # Canonicalise the rater id too: a numeric id can drift '1234' / '1234.0'
    # across pooled timepoints (same float-upcast as the value axis), which
    # would split one rater in two and undercount their subjects. No-op for
    # alphanumeric ids.
    df = pd.DataFrame({"r": _canonicalize_values(cleaned[mask]).to_numpy(),
                       "s": subj_ser[mask].astype(str).to_numpy()})
    subj_per_rater = df.groupby("r")["s"].nunique()
    return int((subj_per_rater >= MIN_SUBJECTS_PER_RATER).sum()) >= MIN_RATERS_FOR_VAR


def _form_prefix(rater_col: str) -> str:
    """The form-name prefix of a rater field: everything before the rater-id
    substring, trailing '_' stripped.

    ``chrpsychs_scr_redcap_user`` -> ``chrpsychs_scr`` (precise: its items are
    ``chrpsychs_scr_*``, NOT the broader ``chrpsychs_*`` that also covers the
    fu/av forms a DIFFERENT user rated). ``panss_rater_ampscz_id`` ->
    ``panss_rater`` (whose items are ``panss_*``, not ``panss_rater_*`` -- so
    the caller falls back to the first name token for this shape)."""
    s = rater_col.lower()
    cut = len(s)
    for sub in RATER_ID_SUBSTRINGS:
        i = s.find(sub)
        if i != -1:
            cut = min(cut, i)
    return s[:cut].rstrip("_")


def _scored_vars_for_rater(rater_col: str, scored_candidates: List[str],
                           var_to_form: Dict[str, str]) -> List[str]:
    """Items that ``rater_col`` governs: same Form Name when known, else the
    rater field's name prefix. Never includes another rater field."""
    scoped: List[str] = []
    form = var_to_form.get(rater_col)
    if form:
        scoped = [v for v in scored_candidates
                  if v != rater_col and var_to_form.get(v) == form]
    if not scoped:
        # Try the precise suffix-stripped form prefix first
        # (``chrpsychs_scr_redcap_user`` -> ``chrpsychs_scr_*``), then fall
        # back to the bare first name token (``panss_rater_ampscz_id`` ->
        # ``panss_*``, and any real form whose items share only the first
        # token). The precise prefix avoids attributing another form's items
        # (rated by another user) to this one; the first-token fallback keeps
        # the demo / forms whose rater field carries an extra name segment.
        first_tok = rater_col.split("_")[0].lower()
        for pref_base in (_form_prefix(rater_col), first_tok):
            if len(pref_base) < PREFIX_MIN_LEN:
                continue
            pref = pref_base + "_"
            # Prefix fallback rescues items the dictionary never mapped (partial
            # dictionaries / local slices), but must NOT override an explicit
            # different-form assignment: keep a prefix match only if the
            # dictionary is silent on it (None) or agrees it's on this form.
            scoped = [v for v in scored_candidates
                      if v != rater_col and v.lower().startswith(pref)
                      and var_to_form.get(v) in (None, form)]
            if scoped:
                break
    return [v for v in scoped if not is_rater_variable(v)]


# ---------------------------------------------------------------------------
# Value canonicalisation + per-subject collapse.
# ---------------------------------------------------------------------------

def _canonicalize_values(ser: pd.Series) -> pd.Series:
    """Collapse numeric-spelling variants so the same number is one value.

    The combined CSVs are all-string and a column can be int-typed at one
    timepoint ('4') and float-typed at another ('4.0') after a pandas upcast;
    keying the crosstab on the raw string would split one value across columns
    (diluting the rate and inflating the distinct-value count). Numbers are
    rendered canonically (integers without a trailing '.0'); genuinely
    non-numeric values pass through unchanged."""
    s = ser.astype(str)
    parsed = pd.to_numeric(s, errors="coerce")
    num_mask = parsed.notna()
    if not num_mask.any():
        return s

    def _fmt(x: float) -> str:
        xf = float(x)
        return str(int(xf)) if xf.is_integer() else f"{xf:g}"

    out = s.copy()
    out.loc[num_mask] = parsed[num_mask].map(_fmt)
    return out


def _collapse_modal(rater: pd.Series, subj: pd.Series,
                    val: pd.Series) -> pd.DataFrame:
    """One row per (rater, subject) with the subject's MODAL value for that
    rater (ties broken to the smallest value, deterministically). Removes the
    repeated-visit pseudoreplication so the test unit is a subject."""
    df = pd.DataFrame({"rater": rater.to_numpy(),
                       "subj": subj.to_numpy(),
                       "val": val.to_numpy()})
    cnt = df.groupby(["rater", "subj", "val"]).size().reset_index(name="n")
    # Most frequent value wins; ties -> smallest value (stable, rater-neutral).
    cnt = cnt.sort_values(["rater", "subj", "n", "val"],
                          ascending=[True, True, False, True])
    return cnt.drop_duplicates(["rater", "subj"], keep="first")[
        ["rater", "subj", "val"]]


# ---------------------------------------------------------------------------
# Core statistic: per-value over-use screen (vectorised, NaN-safe).
# ---------------------------------------------------------------------------

def _screen(ct: pd.DataFrame, z_gate: float) -> List[dict]:
    """Two-proportion over-use screen over a rater x value count table.

    ``ct`` is a contingency table (rows = raters, cols = values) of SUBJECT
    counts within ONE comparison universe (home raters of one site for the
    within-site tier, the whole network for the cohort tier). For every
    (rater R, value v):

        a      = R's subjects with value v          = ct[R, v]
        n_R    = R's subjects                        = row sum
        b      = others' subjects with value v       = col sum - a
        n_rest = others' subjects                     = grand total - n_R
        z      = (a/n_R - b/n_rest) / pooled SE   (one-sided: high)

    A cell is returned only if it clears the z gate AND every effect-size /
    sample-size gate (so big-n significance alone never flags). Returns a
    list of dicts; pure and unit-tested.
    """
    if ct is None or ct.empty:
        return []
    raters = list(ct.index)
    values = list(ct.columns)
    A = ct.to_numpy(dtype=float)              # [R, V]
    n_R = A.sum(axis=1)                       # [R]
    colsum = A.sum(axis=0)                    # [V]
    total = float(A.sum())
    B = colsum[None, :] - A                   # [R, V] others' count of value
    n_rest = total - n_R                      # [R]
    nR = n_R[:, None]
    nrest = n_rest[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        pR = A / nR
        pRest = B / nrest
        pooled = (A + B) / (nR + nrest)
        se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / nR + 1.0 / nrest))
        Z = (pR - pRest) / se

    mask = (
        (Z >= z_gate)
        & ((pR - pRest) >= MIN_RATE_DIFF)
        & (pR >= MIN_RATER_RATE)
        & (A >= MIN_VALUE_SUBJECTS)
        & (nR >= MIN_SUBJECTS_PER_RATER)
        & (nrest >= MIN_SUBJECTS_PER_RATER)
        & (colsum[None, :] >= MIN_VALUE_TOTAL)
        & np.isfinite(Z)
        & (se > 0)
    )
    out: List[dict] = []
    for ri, vi in zip(*np.where(mask)):
        out.append({
            "rater": str(raters[ri]),
            "value": str(values[vi]),
            "a": int(A[ri, vi]),
            "n_rater": int(n_R[ri]),
            "b": int(B[ri, vi]),
            "n_rest": int(n_rest[ri]),
            "p_rater": float(pR[ri, vi]),
            "p_rest": float(pRest[ri, vi]),
            "z": float(Z[ri, vi]),
        })
    return out


def _cohort_rate_excl(cohort_ct: pd.DataFrame, rater: str, value: str) -> float:
    """Network-wide subject-rate of ``value`` among raters OTHER than
    ``rater`` (for displaying cohort context on a within-site finding).
    -1.0 if not computable."""
    if value not in cohort_ct.columns or rater not in cohort_ct.index:
        return -1.0
    a = float(cohort_ct.at[rater, value])
    n_R = float(cohort_ct.loc[rater].sum())
    col = float(cohort_ct[value].sum())
    total = float(cohort_ct.to_numpy().sum())
    n_rest = total - n_R
    if n_rest <= 0:
        return -1.0
    return (col - a) / n_rest


# ---------------------------------------------------------------------------
# Detection.
# ---------------------------------------------------------------------------

def detect_all(per_slice: Dict, classify: Dict = None, **_) -> List[dict]:
    var_to_form, source = _load_var_to_form()
    by_network: Dict[str, List] = defaultdict(list)
    for (network, tp), df in per_slice.items():
        if df is None or df.empty or "subjectid" not in df.columns:
            continue
        by_network[network].append((tp, df))

    rows: List[dict] = []
    for network, items in by_network.items():
        try:
            rows.extend(_detect_network(network, items, var_to_form, source))
        except Exception as e:
            print(f"  [rater_effect] WARN network {network}: {e}")
    return rows


def _detect_network(network: str, items: List, var_to_form: Dict[str, str],
                    source: str) -> List[dict]:
    # Pool timepoints (a rater is the same person across visits); the
    # per-subject collapse downstream removes the repeated-visit dependence.
    stacked = stack_by_network(items)
    cats = classify_columns(stacked)
    scored_candidates = [c for c in (cats["binary"] + cats["low_card"])
                         if c != "_tp" and not is_rater_variable(c)]

    subj_all = stacked["subjectid"]
    name_matched = [c for c in stacked.columns if is_rater_variable(c)]
    rater_cols = [c for c in name_matched
                  if _valid_rater_col(stacked[c], subj_all)]

    # ---- Funnel diagnostics (column NAMES only -- schema, never cell values,
    # so this is PHI-safe). Explains an empty result at each stage. ----
    if not name_matched:
        raw = [c for c in stacked.columns
               if any(s in str(c).lower() for s in RATER_ID_SUBSTRINGS)]
        print(f"  [rater_effect] {network}: 0 rater fields -- no column name "
              f"contains {list(RATER_ID_SUBSTRINGS)} (after excluding subject "
              f"record ids). {len(raw)} column(s) matched the substring but "
              f"were treated as subject ids: {raw[:8]}. NOTE: if these CSVs are "
              f"DPACC exports, per-form rater identifier fields are stripped on "
              f"export (data dictionary Identifier?=='y') -- they exist in the "
              f"dictionary but not in this data. Run on a source that retains "
              f"identifiers.")
        return []
    dropped = [c for c in name_matched if c not in rater_cols]
    if dropped:
        print(f"  [rater_effect] {network}: {len(dropped)} name-matched rater "
              f"column(s) failed the cardinality guard (need >= "
              f"{MIN_RATERS_FOR_VAR} raters each covering >= "
              f"{MIN_SUBJECTS_PER_RATER} subjects): {dropped[:8]}")

    out: List[dict] = []
    n_pairs = 0
    n_unscoped = 0
    for rcol in rater_cols:
        scoped = _scored_vars_for_rater(rcol, scored_candidates, var_to_form)
        if not scoped:
            n_unscoped += 1
            print(f"  [rater_effect] {network}: rater field {rcol!r} scoped to "
                  f"0 testable items (form={var_to_form.get(rcol, '?')!r}, "
                  f"prefix={rcol.split('_')[0]!r}); no binary/low_card item "
                  f"matched its form/prefix.")
            continue
        for scol in scoped:
            n_pairs += 1
            out.extend(_rater_item_findings(network, stacked, rcol, scol,
                                            var_to_form))
    print(f"  [rater_effect] {network}: {len(rater_cols)}/{len(name_matched)} "
          f"rater field(s) passed the guard, {n_pairs} (rater, item) pairs "
          f"scanned ({n_unscoped} rater field(s) had no scoped items) -> "
          f"{len(out)} flags (scoping source: {source})")
    return out


def _rater_item_findings(network: str, stacked: pd.DataFrame, rater_col: str,
                         scored_col: str, var_to_form: Dict[str, str]
                         ) -> List[dict]:
    rater_ser = clean_missing_string(stacked[rater_col].astype(object))
    val_ser = clean_missing_string(stacked[scored_col].astype(object))
    subj = stacked["subjectid"].astype(str)
    keep = rater_ser.notna() & val_ser.notna()
    if int(keep.sum()) < MIN_SUBJECTS_PER_RATER * MIN_RATERS_FOR_VAR:
        return []
    rater_k = _canonicalize_values(rater_ser[keep])  # numeric id float-drift safe
    val_k = _canonicalize_values(val_ser[keep])
    subj_k = subj[keep]
    if val_k.nunique() < 2 or val_k.nunique() > MAX_DISTINCT_VALUES:
        return []

    # Collapse to one row per (rater, subject): the subject's modal value.
    collapsed = _collapse_modal(rater_k, subj_k, val_k)
    if collapsed.empty:
        return []
    collapsed["site"] = collapsed["subj"].str[:2]

    cohort_ct = pd.crosstab(collapsed["rater"], collapsed["val"])
    qual = [r for r in cohort_ct.index
            if int(cohort_ct.loc[r].sum()) >= MIN_SUBJECTS_PER_RATER]
    if len(qual) < MIN_RATERS_FOR_VAR:
        return []

    # Each rater's HOME (modal) site, and the home raters per site.
    rater_site_ct = pd.crosstab(collapsed["rater"], collapsed["site"])
    site_of_rater = rater_site_ct.idxmax(axis=1).to_dict()
    raters_by_site: Dict[str, List[str]] = defaultdict(list)
    for r in qual:
        raters_by_site[str(site_of_rater.get(r, ""))].append(r)
    multi_sites = {s for s, rl in raters_by_site.items()
                   if len(rl) >= MIN_SITE_RATERS_FOR_WITHIN}

    form = var_to_form.get(rater_col, "")
    out: List[dict] = []
    # Raters that received a VALID within-site contrast (their own AND the other
    # home raters' home-at-site subject counts both clear MIN_SUBJECTS_PER_RATER).
    # Only these are genuinely arbitrated by the within tier; any other rater --
    # incl. a well-sampled one whose sole co-rater is under-sampled at home --
    # must fall through to the cohort tier rather than being dropped from both.
    within_tested: set = set()

    # --- within-site tier: home raters of a site contrasted against each
    # other, on that site's SUBJECTS only. We restrict to BOTH the home raters
    # AND collapsed["site"] == site: a home rater who also rates a few subjects
    # at other sites must not bring those off-site subjects into the same-site
    # comparison (that would break the "same subject population" guarantee and
    # can fabricate a confident false positive). A rater whose within-site
    # contrast is degenerate (it, or its sole co-rater, has
    # < MIN_SUBJECTS_PER_RATER subjects AT home) is recorded as NOT within-tested
    # and falls through to the cohort (site-confounded) tier below -- so a
    # well-sampled rater is never silently dropped from BOTH tiers just because
    # a co-rater is thin at home.
    for site in multi_sites:
        home = collapsed[collapsed["rater"].isin(raters_by_site[site])
                         & (collapsed["site"] == site)]
        site_ct = pd.crosstab(home["rater"], home["val"])
        # Record which home raters got a usable same-site contrast: both their
        # own and the other home raters' home-at-site subject counts clear the
        # gate (this mirrors _screen's per-rater nR / nrest requirement).
        site_total = int(site_ct.to_numpy().sum())
        for r in site_ct.index:
            nR = int(site_ct.loc[r].sum())
            if nR >= MIN_SUBJECTS_PER_RATER and (site_total - nR) >= MIN_SUBJECTS_PER_RATER:
                within_tested.add(str(r))
        for d in _screen(site_ct, Z_GATE_WITHIN):
            out.append(_make_finding(
                network, site, rater_col, scored_col, form, d,
                tier="within_site",
                cohort_rate=_cohort_rate_excl(cohort_ct, d["rater"], d["value"]),
            ))

    # --- cohort tier: raters NOT arbitrated by a valid within-site contrast --
    # the sole high-volume rater at their home site, PLUS any rater whose
    # within-site test was degenerate (co-rater thin at home). Both are labelled
    # site-confounded, the appropriate lower-confidence tier.
    for d in _screen(cohort_ct, Z_GATE_COHORT):
        if d["rater"] in within_tested:
            continue
        site = str(site_of_rater.get(d["rater"], ""))
        out.append(_make_finding(
            network, site, rater_col, scored_col, form, d,
            tier="cohort", cohort_rate=d["p_rest"],
        ))
    return out


def _make_finding(network: str, site: str, rater_col: str, scored_col: str,
                  form: str, d: dict, tier: str, cohort_rate: float) -> dict:
    rid = d["rater"]
    z = d["z"]
    p_rater = d["p_rater"]
    p_rest = d["p_rest"]
    value = d["value"]
    base_sev = calibrate(z, Z_SEV_THRESHOLD, Z_SEV_EXTREME)
    net_str = (f"{cohort_rate:.0%}" if cohort_rate >= 0 else "n/a")

    if tier == "within_site":
        severity = base_sev
        method = "rater value over-use vs co-located raters (within-site)"
        confidence = "within-site"
        observed = (f"{scored_col}={value} for {p_rater:.0%} of rater {short(rid)}'s "
                    f"subjects ({d['a']}/{d['n_rater']}) vs {p_rest:.0%} for other "
                    f"raters at site {site}")
        expected = (f"co-located raters at {site}: {p_rest:.0%}; "
                    f"network: {net_str}")
        explanation = (
            f"Rater {short(rid)} at site {site} records {scored_col}={value} for far "
            f"more of their subjects than the other rater(s) at the SAME site "
            f"({p_rater:.0%} vs {p_rest:.0%}, two-proportion z={z:.1f} on subjects). "
            f"Same-site comparison holds the subject population fixed, so this reads "
            f"as a rater scoring tendency rather than a subject difference (raters "
            f"are not randomly assigned to subjects)."
        )
    else:  # cohort / site-confounded
        severity = SITE_CONFOUNDED_SEVERITY_FACTOR * base_sev
        method = ("rater value over-use vs network cohort "
                  "(site-confounded; single-rater site)")
        confidence = "site-confounded"
        observed = (f"{scored_col}={value} for {p_rater:.0%} of rater {short(rid)}'s "
                    f"subjects ({d['a']}/{d['n_rater']}) vs {p_rest:.0%} across the "
                    f"rest of the network")
        expected = f"rest of network: {p_rest:.0%}"
        explanation = (
            f"Rater {short(rid)} records {scored_col}={value} for far more of their "
            f"subjects than the rest of the network ({p_rater:.0%} vs {p_rest:.0%}, "
            f"two-proportion z={z:.1f} on subjects). This rater is the only "
            f"high-volume rater for this form at site {site}, so the difference "
            f"CANNOT be separated from a site / subject-population effect -- review "
            f"alongside the site-level checks."
        )

    return Finding(
        anomaly_type=ANOMALY_TYPE,
        severity_score=severity,
        raw_score=z,
        network=network,
        timepoint="(all timepoints)",
        site_id=site,
        subjectid=f"(rater {short(rid)})",
        variable=scored_col,
        variables_involved=f"{rater_col} -> {scored_col}",
        observed_value=observed,
        expected_value=expected,
        explanation=explanation,
        method=method,
        extra={
            "rater_id": rid,
            "rater_variable": rater_col,
            "form": form,
            "scored_value": value,
            "confidence": confidence,
            "n_rater_subjects": d["n_rater"],
            "n_compare_subjects": d["n_rest"],
            "n_subjects_value": d["a"],
            "rate_rater": round(p_rater, 4),
            "rate_compare": round(p_rest, 4),
            "rate_network_other": (round(cohort_rate, 4) if cohort_rate >= 0 else ""),
            "z_score": round(z, 2),
        },
    ).to_row()
