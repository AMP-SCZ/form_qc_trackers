"""Conservative recovery of a flag's last observed field value.

Tracker history stores the raw message that was last seen before a flag
disappeared.  Some message families include the value of the displayed
variable; many only describe a relationship or an expected value.  This
module extracts only the former and returns ``None`` for everything else.

An empty string is a real, known value for blank-field flags, so callers must
distinguish it from ``None`` rather than relying on truthiness.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class BeforeValue:
    """One exact pre-resolution value and how it was recovered."""

    value: str
    source: str


_PROPOSED_EVIDENCE_RE = re.compile(
    r"(?:^|\n)\s*Variables\s*&\s*values(?P<version>\s+v(?:2|3))?:\s*"
    r"(?P<values>.*)\Z",
    re.IGNORECASE | re.DOTALL,
)
# Structured evidence is added only to Proposed Checks rows, whose producer
# prefixes every base message with a check ID.  Legacy checks interpolate raw
# values into ordinary prose, so a value containing a newline and the evidence
# label must not be allowed to masquerade as producer-owned evidence.  Cross
# Checks also use bracketed IDs but never emit this payload.
_TRUSTED_PROPOSED_BODY_RE = re.compile(
    r"^\[(?!CROSS-QC-)[A-Z0-9][A-Z0-9_.:-]*\]\s+",
    re.IGNORECASE,
)
# "Variable is blank." is the current producer wording
# (qc_types/general_checks.py); "Value is empty" is the dominant pre-V2
# tracker spelling of the same assertion.  Both state the field held no
# value at observation time, so both prove an exact empty-string before
# value.  The missing-code state used its own wording ("Variable is a
# missing code."), which stays unrecoverable because the exact code is
# not in the message.
_BLANK_FLAG_RE = re.compile(
    r"^(?:\[[^\]]+\]\s*)?(?:Variable\s+is\s+blank|Value\s+is\s+empty)\.?$",
    re.IGNORECASE,
)
_MISSING_REASON_BLANK_RE = re.compile(
    r"^Missing data button clicked, but reason not specified\.?$",
    re.IGNORECASE,
)
_CROSS_CHECK_RE = re.compile(
    r"^\[(?P<check_id>CROSS-QC-\d{3})\]", re.IGNORECASE)

# Cross Checks deliberately show relationship-only labels. These enabled
# rules nevertheless constrain the first non-date (tracker-prefix) operand to
# one exact value. Pair the ID with that exact raw variable as a guard against
# catalog reordering or a stale/manual tracker prefix.
_CROSS_CHECK_FIXED_BEFORE = {
    "CROSS-QC-005": ("chrchs_mar", "0"),
    "CROSS-QC-006": ("chrchs_tob", "0"),
    "CROSS-QC-009": ("chrcdss_calg8", "0"),
    "CROSS-QC-030": ("chrchs_mens", "1"),
    "CROSS-QC-031": ("chrchs_mens", "1"),
    "CROSS-QC-039": ("chrcrit_excl9", "1"),
    "CROSS-QC-086": ("chrsaliva_tob", "1"),
    "CROSS-QC-088": ("chrassist_whoassist_often1", "0"),
    "CROSS-QC-094": ("chrassist_whoassist_use3", "1"),
    "CROSS-QC-096": ("chrassist_whoassist_use7", "1"),
    "CROSS-QC-098": ("chrassist_whoassist_use4", "1"),
    "CROSS-QC-100": ("chrassist_whoassist_use6", "1"),
    "CROSS-QC-101": ("chrassist_whoassist_use6", "0"),
    "CROSS-QC-102": ("chrassist_whoassist_use9", "1"),
    "CROSS-QC-104": ("chrassist_whoassist_use8", "1"),
    "CROSS-QC-122": ("chrchs_tob", "1"),
    "CROSS-QC-141": ("chrchs_mar", "1"),
    "CROSS-QC-144": ("chrsaliva_mar", "1"),
}

# Message families whose interpolated values are provably NOT the raw stored
# cell, or whose interpolation provenance cannot be proven, audited 2026-08-22:
#   * pharmaceutical date chronology — every date passes through
#     pharm_checks._date_to_str / strftime (parse-and-reformat), so the
#     rendered date need not equal the raw stored string;
#   * cross-subject duplicate fluids — fluid_checks normalizes the value
#     (strip + integer collapse, "12345.0" -> "12345") before display;
#   * "Incorrect units used" — reviewer-refuted recoveries; the numeric
#     lab value is exposed to dtype-inference re-rendering upstream;
#   * demographics-age comparison — reviewer-refuted recoveries;
#   * V1-only wordings with no surviving emitter (the "later assessment
#     date in", "is before last visit/questionnaire", "not done
#     properly", and future-date families) — nothing in the tree proves
#     what those interpolations contained.
# Any equality or parenthetical inside these messages must fail closed
# before it can reach the generic parsers, so the list is checked ahead of
# every message-pattern rule.  Blank-flag and structured-evidence recovery
# are unaffected: blanks are proven by the message semantics, not by
# trusting an interpolation, and evidence is producer-owned.
_UNPROVEN_INTERPOLATION_RES = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"^(?:\[[^\]]+\]\s*)?Incorrect units used\b",
        r"^(?:\[[^\]]+\]\s*)?Duplicate (?:blood )?(?:ID/barcode|ID|barcode)"
        r" value\b",
        r"^(?:\[[^\]]+\]\s*)?Duplicate positions? found\b",
        r"^(?:\[[^\]]+\]\s*)?Difference between demographics age\b",
        r"\bcannot occur after the form modification date\b",
        r"\bPlease update the pharmaceutical treatment modification date\b",
        r"\brather than the ongoing code\b",
        r"\binstead of the ongoing code\b",
        r"\bcannot be ongoing\b",
        r"\blater assessment date in\b",
        r"\bis before last visit/questionnaire\b",
        r"\bnot done properly\b",
        r"^Date \([^()]*\) is in the future\.?$",
    )
)

# Every date the pharmaceutical chronology checks interpolate is
# parse-and-reformatted, so no message can prove the raw stored string of
# a pharm date/onset/offset/first-dose variable.  Blank-flag proof is
# unaffected (checked before this rule).
_PHARM_DATE_VARIABLE_RE = re.compile(
    r"chrpharm_.*(?:date|onset|offset|firstdose)", re.IGNORECASE)


def _is_unproven_interpolation(variable: str, body: str) -> bool:
    """Whether any value in ``body`` is barred from message-pattern proof."""

    if _PHARM_DATE_VARIABLE_RE.match(variable):
        return True
    return any(
        pattern.search(body) for pattern in _UNPROVEN_INTERPOLATION_RES)


# A quoted Python scalar or a compact unquoted REDCap scalar. The latter is
# deliberately whitespace-free; free-text RHS values need structured evidence
# rather than a guess about where the value ends. The unquoted branch is lazy
# so the explicit boundary can leave one sentence terminator outside the value.
_SCALAR_TOKEN = (
    r"(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|[^\s,;/()]+?)")
_OBSERVED_VALUE_BOUNDARY = (
    r"(?=\s*(?:[,;/)]|\b(?:and|but)\b|\.(?:\s|$)|$))")


def infer_before_value(variable: object, message: object) -> BeforeValue | None:
    """Return the exact last bad value for ``variable`` when provable.

    ``message`` is the raw (non-canonicalized) tracker message retained by
    :mod:`analyze_flags.estimate_resolved`.  Pattern matching is intentionally
    allowlisted.  Thresholds, recommendations, values belonging only to other
    variables, redacted evidence, and conflicting observations all fail
    closed as ``None``.
    """

    var = "" if variable is None else str(variable).strip()
    msg = "" if message is None else str(message).strip()
    if not var or not msg:
        return None

    marker_match = _PROPOSED_EVIDENCE_RE.search(msg)
    evidence_match = _trusted_proposed_evidence_match(msg)
    if marker_match is not None and evidence_match is None:
        # Do not reinterpret the marker-shaped suffix through the generic
        # named-equality parsers below.  In an ordinary legacy message it is
        # untrusted raw data, and any apparent variable=value pair is tainted.
        return None
    body = msg
    if evidence_match:
        body = msg[:evidence_match.start()].rstrip()
        rendered_version = evidence_match.group("version") or ""
        evidence_version = (
            int(rendered_version.strip().lower().removeprefix("v"))
            if rendered_version else 1)
        evidence_result, target_was_present = _from_proposed_evidence(
            var,
            evidence_match.group("values"),
            version=evidence_version,
            blank_is_proven=(
                evidence_version >= 3
                or bool(_BLANK_FLAG_RE.fullmatch(body))),
        )
        if target_was_present:
            return evidence_result

    if _BLANK_FLAG_RE.fullmatch(body):
        return BeforeValue("", "blank_flag")
    if _MISSING_REASON_BLANK_RE.fullmatch(body):
        return BeforeValue("", "message_pattern")

    cross_match = _CROSS_CHECK_RE.match(body)
    if cross_match:
        contract = _CROSS_CHECK_FIXED_BEFORE.get(
            cross_match.group("check_id").upper())
        if contract is not None and var == contract[0]:
            return BeforeValue(contract[1], "check_contract")
        # The remaining Cross Check families are ranges or otherwise do not
        # prove one exact value for their displayed operand.
        return None

    # Families whose interpolations are reformatted, normalized, or of
    # unprovable provenance never reach the message-pattern parsers.
    if _is_unproven_interpolation(var, body):
        return None

    fixed_value = _from_fixed_message(body)
    if fixed_value is not None:
        return BeforeValue(fixed_value, "check_contract")

    family_value = _from_family_pattern(var, body)
    if family_value is not None:
        return BeforeValue(family_value, "message_pattern")

    observed = []
    observed.extend(_named_equality_values(var, body))
    observed.extend(_named_is_equal_values(var, body))
    observed = list(dict.fromkeys(observed))
    if len(observed) == 1:
        return BeforeValue(observed[0], "message_pattern")
    return None


def strip_proposed_evidence(message: object) -> str:
    """Remove display-only Proposed Checks evidence from a flag message.

    Evidence values change as data changes and therefore must not become part
    of the canonical episode key used by Analyze Flags.
    """

    msg = "" if message is None else str(message)
    match = _trusted_proposed_evidence_match(msg)
    return msg[:match.start()].rstrip() if match else msg


def _trusted_proposed_evidence_match(message: str) -> re.Match | None:
    """Return a marker only when its base message has the producer contract."""

    match = _PROPOSED_EVIDENCE_RE.search(message)
    if match is None:
        return None
    body = message[:match.start()].rstrip()
    return match if _TRUSTED_PROPOSED_BODY_RE.match(body) else None


def _from_proposed_evidence(
        variable: str, rendered_values: str, *, version: int = 1,
        blank_is_proven: bool = False) -> tuple[BeforeValue | None, bool]:
    matches = []
    for pair in rendered_values.split(";"):
        name, separator, rendered = pair.partition("=")
        if not separator or name.strip() != variable:
            continue
        matches.append(rendered.strip())

    if not matches:
        return None, False
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        return None, True

    rendered = unique[0]
    if rendered == "<blank>":
        # V3 reserves this token for a true empty/null value. Earlier evidence
        # used the same spelling for textual missing sentinels and literal data,
        # so it remains unknown unless the base flag independently proves blank.
        if blank_is_proven:
            return BeforeValue("", "proposed_evidence"), True
        return None, True
    if (rendered.casefold() in {"[redacted]", "<unavailable>"}
            or rendered.endswith("...")):
        return None, True

    if version >= 3 and rendered.startswith("@json:"):
        try:
            decoded = json.loads(rendered[len("@json:"):])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, True
        if not isinstance(decoded, str):
            return None, True
        return BeforeValue(decoded, "proposed_evidence"), True
    if version >= 3 and rendered.startswith("@json"):
        return None, True

    # V1/V2 replaced these delimiters without a reversible marker. Their raw
    # value is unknowable, even though the rendered preview looks complete.
    if version < 3 and (
            any(char in rendered for char in {"¦", "；"})
            or bool(re.search(r"\s", rendered))):
        return None, True

    # The pre-v3 producer quoted only reserved literal tokens. Decode exactly
    # those legacy tagged cases; ordinary quote characters are data and must be
    # preserved (for example, a raw value of "'7'").
    legacy_reserved = {
        "'<blank>'", '"<blank>"',
        "'<unavailable>'", '"<unavailable>"',
        "'[REDACTED]'", '"[REDACTED]"',
    }
    if version < 3 and rendered in legacy_reserved:
        decoded = _decode_scalar(rendered)
        return (
            BeforeValue(decoded, "proposed_evidence")
            if decoded is not None else None,
            True,
        )
    return BeforeValue(rendered, "proposed_evidence"), True


def _from_fixed_message(message: str) -> str | None:
    """Values proved by a complete, fixed check-message contract."""

    normalized = message.strip().rstrip(".").casefold()
    if normalized in {"value is 0", "speech sample not uploaded to box"}:
        return "0"
    if normalized.startswith("m6 clicked on baseline psychs,"):
        return "M6"
    return None


def _from_family_pattern(variable: str, message: str) -> str | None:
    escaped = re.escape(variable)

    # The active CSSRS contradiction has a fixed coded-state contract: recent
    # is 2 (yes) and lifetime is 1 (no).  Both variable names are present, so
    # recover either operand only when the complete producer sentence matches.
    match = re.fullmatch(
        r"Recent variable \((?P<recent>[A-Za-z_][A-Za-z0-9_]*)\) was "
        r"answered as yes, but lifetime variable "
        r"\((?P<lifetime>[A-Za-z_][A-Za-z0-9_]*)\) was answered as no\.",
        message,
        re.IGNORECASE,
    )
    if match:
        if variable == match.group("recent"):
            return "2"
        if variable == match.group("lifetime"):
            return "1"

    # BPRS emits the exact raw values for two BPRS fields.  Restrict this to
    # that family rather than treating arbitrary "x is y" prose as evidence.
    if variable.startswith("chrbprs_"):
        match = re.fullmatch(
            r"(?P<first>chrbprs_[A-Za-z0-9_]+) is "
            r"(?P<first_value>[^,\r\n]+), but "
            r"(?P<second>chrbprs_[A-Za-z0-9_]+) is "
            r"(?P<second_value>[^\r\n]+)\.",
            message,
            re.IGNORECASE,
        )
        if match:
            if variable == match.group("first"):
                return match.group("first_value").strip()
            if variable == match.group("second"):
                return match.group("second_value").strip()

    # This OASIS message is emitted only when the first item is exactly 0.
    match = re.fullmatch(
        r"Marked as having no anxiety in chroasis_oasis_1, but "
        r"chroasis_oasis_[2-5] is equal to .+\.",
        message,
        re.IGNORECASE,
    )
    if match and variable == "chroasis_oasis_1":
        return "0"

    # Forward conversion criteria flags name their displayed criterion and its
    # exact observed value. Restrict to the complete producer sentence so an
    # arbitrary prose fragment using "is" is never treated as evidence.
    match = re.fullmatch(
        rf"{escaped}\s+is\s+(?P<value>{_SCALAR_TOKEN}),\s+"
        r"but participant is not marked as converted\.",
        message,
        re.IGNORECASE,
    )
    if match:
        return _decode_scalar(match.group("value"))

    # Direction A of the missingness reconciliation is deliberately NOT
    # recoverable (audited 2026-08-22): utils.check_if_missing also reports a
    # form as missing from the PRESCIENT completion status (3/4) and from the
    # sparse-form fill heuristic, so the per-form missing button need not
    # hold 1 — or anything at all — when this flag fires.

    if variable == "chrblood_drawdate":
        match = re.fullmatch(
            r"Blood draw date \((?P<draw>[^()]*)\) is later than date sent "
            r"to lab \((?P<lab>[^()]*)\)\.",
            message,
            re.IGNORECASE,
        )
        if match:
            return match.group("draw").strip()

    # The cross-timepoint blood-date flag is attached to the baseline row, so
    # its displayed chrblood_interview_date value is the baseline operand.
    if variable == "chrblood_interview_date":
        match = re.fullmatch(
            r"Month 2 blood interview date \((?P<month2>[^()]*)\) is before "
            r"the baseline blood interview date \((?P<baseline>[^()]*)\)\.",
            message,
            re.IGNORECASE,
        )
        if match:
            return match.group("baseline").strip()
        match = re.fullmatch(
            r"Baseline \((?P<baseline>[^()]*)\) and month 2 "
            r"\((?P<month2>[^()]*)\) blood interview dates are -?\d+ days "
            r"apart \(should be within 90 days\)\.",
            message,
            re.IGNORECASE,
        )
        if match:
            return match.group("baseline").strip()

    # Current visit date appears before the earlier comparison date.  Match
    # this family before any generic parenthetical rule.
    match = re.search(
        r"^Visit date at\s+.+?\s+\((?P<value>[^()]*)\)\s+is\s+"
        r"\d+\s+day\(s\)\s+before\b",
        message,
        re.IGNORECASE,
    )
    if match:
        return match.group("value").strip()

    if variable in {"chrguid_guid", "chrguid_pseudoguid"}:
        match = re.search(
            r"^GUID in incorrect format\.\s*GUID was reported to be\s+"
            r"(?P<value>.+)\.$",
            message,
            re.IGNORECASE,
        )
        if match:
            # The message template appends exactly one sentence terminator;
            # greedy capture preserves a real trailing period in the value.
            return match.group("value").strip()

    if variable.startswith("chrblood_"):
        match = re.search(
            r"^Barcode\s*\((?P<value>[^()]*)\)\s+"
            r"(?:length is not|contains non-numeric characters)",
            message,
            re.IGNORECASE,
        )
        if match:
            return match.group("value").strip()

    if variable == "chriq_fsiq":
        match = re.search(
            rf"^FSIQ Miscalculated\s*\(Recorded as\s+(?P<value>{_SCALAR_TOKEN})\s*,",
            message,
            re.IGNORECASE,
        )
        if match:
            return _decode_scalar(match.group("value"))

    # Legacy general and clinical range messages explicitly name the target.
    for pattern in (
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s+value\s*"
        rf"\((?P<value>[^()]*)\)\s+is\s+out of range\b",
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s*"
        rf"\((?P<value>[^()]*)\)\s+is\s+"
        rf"(?:outside\b|less than\b|greater than\b)",
    ):
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group("value").strip()

    # Proposed Checks before the structured evidence column was introduced.
    proposed_patterns = (
        rf"^(?:\[[^\]]+\]\s*)?Value\s+(?P<value>{_SCALAR_TOKEN})\s+"
        rf"(?:is outside|violates|is not a real canonical)\b",
        rf"^(?:\[[^\]]+\]\s*)?Yes/No value\s+(?P<value>{_SCALAR_TOKEN})\s+is\b",
        rf"^(?:\[[^\]]+\]\s*)?Expanded checkbox value\s+"
        rf"(?P<value>{_SCALAR_TOKEN})\s+is\b",
        rf"^(?:\[[^\]]+\]\s*)?Numeric field contains\s+.*?value\s+"
        rf"(?P<value>{_SCALAR_TOKEN})\s*\.?$",
        rf"^(?:\[[^\]]+\]\s*)?Integer field contains\s+.*?value\s+"
        rf"(?P<value>{_SCALAR_TOKEN})\s*\.?$",
        rf"^(?:\[[^\]]+\]\s*)?Stored calculated value\s+"
        rf"(?P<value>{_SCALAR_TOKEN})\s+differs\b",
        rf"^(?:\[[^\]]+\]\s*)?Stored (?:SCID endpoint|schizotypal calculated value)\s+"
        rf"(?P<value>{_SCALAR_TOKEN})\s+differs\b",
        rf"^(?:\[[^\]]+\]\s*)?Hidden choice\s+"
        rf"(?P<value>{_SCALAR_TOKEN})\s+is present\b",
    )
    for pattern in proposed_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return _decode_scalar(match.group("value"))

    # Subject-level discovery flags include an explicit raw observation.
    match = re.search(
        rf"^(?:{escaped})(?:\s+at\s+[^:]+)?\s*:\s*value="
        rf"(?P<value>{_SCALAR_TOKEN})(?:,|\s|$)",
        message,
        re.IGNORECASE,
    )
    if match:
        return _decode_scalar(match.group("value"))

    match = re.search(
        rf"^(?:{escaped})\s*:\s*.*?\bobserved\s+"
        rf"(?P<value>{_SCALAR_TOKEN})(?:;|,|\s|$)",
        message,
        re.IGNORECASE,
    )
    if match:
        return _decode_scalar(match.group("value"))
    return None


def _named_equality_values(variable: str, message: str) -> list[str]:
    escaped = re.escape(variable)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s*=\s*"
        rf"(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)*"
        rf"(?P<value>{_SCALAR_TOKEN}){_OBSERVED_VALUE_BOUNDARY}",
        re.IGNORECASE,
    )
    return _unambiguous_matches(pattern, message)


def _named_is_equal_values(variable: str, message: str) -> list[str]:
    escaped = re.escape(variable)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s+"
        rf"is\s+equal\s+to\s+(?P<value>{_SCALAR_TOKEN})"
        rf"{_OBSERVED_VALUE_BOUNDARY}",
        re.IGNORECASE,
    )
    return _unambiguous_matches(pattern, message)


def _unambiguous_matches(pattern: re.Pattern, message: str) -> list[str]:
    values = []
    for match in pattern.finditer(message):
        # Do not reinterpret expected-state prose as an observation.
        prefix = message[max(0, match.start() - 40):match.start()]
        if re.search(r"(?:should|must|has to|expected)\s+(?:be\s+)?$", prefix,
                     re.IGNORECASE):
            continue
        value = _decode_scalar(match.group("value"))
        if value is not None:
            values.append(value)
    return values


def _decode_scalar(token: str) -> str | None:
    token = token.strip()
    if not token or token.endswith("..."):
        return None
    if token[0:1] in {"'", '"'}:
        try:
            decoded = ast.literal_eval(token)
        except (SyntaxError, ValueError):
            return None
        return str(decoded)
    return token
