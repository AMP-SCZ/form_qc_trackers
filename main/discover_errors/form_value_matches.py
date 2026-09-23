"""Summarize value overlap between matching source-form instances.

This is a standalone exploratory program; it is not wired into ``run_qc.py``
and never modifies a combined REDCap export.  A form instance is identified by
both its REDCap form name and its timepoint, so, for example, baseline blood
and month-2 blood are different instances.

For each participant (keyed by network as well as ``subjectid``), the program
represents a form as a multiset of its substantive cell values plus a count of
blank/missing cells. Per-form control fields (missing marker/reason,
completion status, interview/entry dates, and REDCap user) come from
``important_form_vars.json`` and never establish form presence. Missing
controls, dates, and user fields never participate in matching; completion
fields are also excluded unless explicitly requested. Explicitly missing form
instances are excluded. A form key is retained when at least one participant
has a substantive nonmissing value; unmarked all-missing observations from
other participants are retained so one-sided missingness can lower their
pairwise score.

For two forms A and B, a nonmissing value can be matched only once.  Missing
occurrences cancel one-for-one only when both forms have them; excess missing
occurrences remain unmatched denominator values::

    matches = sum(min(count_A[value], count_B[value]) for value in values)
    cancelled_missing = min(missing_A, missing_B)
    effective_A = nonmissing_A + missing_A - cancelled_missing
    effective_B = nonmissing_B + missing_B - cancelled_missing
    percentage = 100 * 2 * matches / (effective_A + effective_B)

That percentage is the symmetric Dice overlap. Only instances with the same
source-form name are compared; their timepoints may be the same or different.
For two different timepoints, records are compared within the same participant.
For a form/timepoint paired with itself, every unordered pair of distinct
participant records is compared; a record is never compared with itself. The
output reports both the median and arithmetic mean of those record-pair
percentages.

The project treats ``999`` as a missing sentinel globally even though some
pharmaceutical fields use it as a meaningful code.  Use
``--include-missing-codes`` to compare nonblank sentinel codes as ordinary
encoded values; this never makes a sentinel-only form key present.  Values are
otherwise compared as exact, whitespace-trimmed text.  Use
``--numeric-equivalence`` to make spellings such as ``1``, ``1.0``, and
``01.000`` equivalent.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import glob
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Hashable, Iterable, Mapping, Sequence

import pandas as pd


DEFAULT_CHUNK_SIZE = 250

TIMEPOINTS = (
    "screening",
    "baseline",
    *(f"month{number}" for number in range(1, 13)),
    "month18",
    "month24",
    "floating",
    "conversion",
)
TIMEPOINT_ORDER = {timepoint: index for index, timepoint in enumerate(TIMEPOINTS)}

CANONICAL_FILE_RE = re.compile(
    r"^AMPSCZ-combined-redcap_"
    r"(?P<timepoint>screening|baseline|month_(?:[1-9]|1[0-2]|18|24)|"
    r"floating_forms|conversion)_"
    r"(?P<network>ProNET|PRONET|PRESCIENT)-day1to1\.csv$",
    re.IGNORECASE,
)

OUTPUT_COLUMNS = (
    "form_1",
    "timepoint_1",
    "form_2",
    "timepoint_2",
    "median_match_percentage",
    "mean_match_percentage",
    "record_pairs_compared",
    "unique_participants_compared",
    "participant_records_form_1",
    "participant_records_form_2",
    "median_matching_values",
    "median_usable_values_form_1",
    "median_usable_values_form_2",
    "networks_represented",
)

# String sentinels are checked case-insensitively after trimming. Numeric
# sentinel matching separately catches equivalent spellings such as -9.00.
_MISSING_TEXT = frozenset({
    "na", "n/a", "nan", "nat", "null", "none", "<na>",
    "1909-09-09", "1903-03-03", "1901-01-01",
})
_MISSING_NUMBERS = frozenset({
    Decimal("-3"), Decimal("-9"), Decimal("-99"), Decimal("999"),
})

MISSING_VALUE_TOKEN: tuple[str, str] = ("missing", "")
_ADMIN_FIELD_KEYS = (
    "missing_var", "missing_spec_var", "completion_var",
    "interview_date_var", "entry_date_var", "redcap_user_var",
)

ParticipantKey = tuple[str, str]
FormKey = tuple[str, str]  # (timepoint, source form)
ValueCounter = Counter[Hashable]


@dataclass(frozen=True)
class _PreparedValueProfile:
    """Cached invariants for repeated overlap checks against one profile."""

    raw_size: int
    missing_count: int
    nonmissing_size: int
    real_values: Mapping[Hashable, int]
    real_items: tuple[tuple[Hashable, int], ...]


ObservationIndex = dict[FormKey, dict[ParticipantKey, ValueCounter]]


class ScannerInputError(ValueError):
    """An input/configuration error that should produce exit status 2."""


@dataclass(frozen=True)
class InputSpec:
    path: Path
    network: str
    timepoint: str


def _normalize_timepoint(filename_value: str) -> str:
    value = filename_value.casefold()
    if value == "floating_forms":
        return "floating"
    if value.startswith("month_"):
        return f"month{value.removeprefix('month_')}"
    return value


def parse_input_spec(path: Path | str) -> InputSpec:
    """Parse network and timepoint from one canonical combined filename."""
    resolved = Path(path).expanduser().resolve()
    match = CANONICAL_FILE_RE.fullmatch(resolved.name)
    if match is None:
        raise ScannerInputError(
            "Combined input filename is not canonical: "
            f"{resolved.name}. Expected AMPSCZ-combined-redcap_"
            "<timepoint>_<ProNET|PRESCIENT>-day1to1.csv."
        )
    raw_network = match.group("network").upper()
    network = "PRONET" if raw_network == "PRONET" else raw_network
    return InputSpec(
        path=resolved,
        network=network,
        timepoint=_normalize_timepoint(match.group("timepoint")),
    )


def _expand_explicit_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_pattern in patterns:
        expanded = os.path.expanduser(raw_pattern)
        if glob.has_magic(expanded):
            matches = sorted(Path(item).resolve() for item in glob.glob(expanded))
            if not matches:
                raise ScannerInputError(
                    f"Explicit input pattern matched no files: {raw_pattern}")
            paths.extend(matches)
        else:
            path = Path(expanded).resolve()
            if not path.is_file():
                raise ScannerInputError(
                    f"Explicit combined input does not exist: {path}")
            paths.append(path)
    return paths


def discover_inputs(
        explicit_inputs: Sequence[str], input_dir: Path | None, *,
        allow_partial_input_dir: bool = False) -> list[InputSpec]:
    """Return unique, canonically sorted combined input specifications."""
    if explicit_inputs:
        paths = _expand_explicit_inputs(explicit_inputs)
    else:
        if input_dir is None:
            raise ScannerInputError(
                "No combined inputs were supplied and no input directory is configured")
        if not input_dir.is_dir():
            raise ScannerInputError(
                f"Combined input directory does not exist: {input_dir}")
        paths = sorted(
            path.resolve() for path in input_dir.iterdir()
            if path.is_file() and CANONICAL_FILE_RE.fullmatch(path.name)
        )
        if not paths:
            raise ScannerInputError(
                f"No canonical combined CSV files found in {input_dir}")

    specs: list[InputSpec] = []
    seen_paths: set[Path] = set()
    seen_slots: dict[tuple[str, str], Path] = {}
    for path in paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        spec = parse_input_spec(path)
        slot = (spec.network, spec.timepoint)
        prior = seen_slots.get(slot)
        if prior is not None:
            raise ScannerInputError(
                f"Multiple inputs represent {spec.network} {spec.timepoint}: "
                f"{prior} and {spec.path}")
        seen_slots[slot] = spec.path
        specs.append(spec)

    if not explicit_inputs and not allow_partial_input_dir:
        expected_slots = {
            (network, timepoint)
            for network in ("PRONET", "PRESCIENT")
            for timepoint in TIMEPOINTS
        }
        missing_slots = sorted(
            expected_slots.difference(seen_slots),
            key=lambda slot: (
                slot[0],
                TIMEPOINT_ORDER.get(slot[1], len(TIMEPOINT_ORDER)),
            ),
        )
        if missing_slots:
            rendered = ", ".join(
                f"{network}/{timepoint}"
                for network, timepoint in missing_slots
            )
            raise ScannerInputError(
                "Canonical input directory is incomplete; missing "
                f"{len(missing_slots)} of 36 network/timepoint file(s): "
                f"{rendered}. Use --allow-partial-input-dir only for an "
                "intentional partial run."
            )

    return sorted(
        specs,
        key=lambda spec: (
            spec.network,
            TIMEPOINT_ORDER.get(spec.timepoint, len(TIMEPOINT_ORDER)),
            spec.path.name.casefold(),
        ),
    )


def load_form_map(path: Path | str) -> dict[str, str]:
    """Load ``grouped_variables.json`` (or a direct variable/form map)."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ScannerInputError(f"Variable-to-form map does not exist: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScannerInputError(
            f"Could not parse variable-to-form map {resolved}: {exc}") from exc

    if isinstance(raw, dict) and "var_forms" in raw:
        raw = raw["var_forms"]
    if not isinstance(raw, dict) or not raw:
        raise ScannerInputError(
            f"Variable-to-form map at {resolved} must be a nonempty JSON object "
            "or contain a nonempty 'var_forms' object")

    result: dict[str, str] = {}
    for variable, form in raw.items():
        variable_text = str(variable).strip()
        form_text = str(form).strip()
        if variable_text and form_text:
            result[variable_text] = form_text
    if not result:
        raise ScannerInputError(
            f"Variable-to-form map at {resolved} contains no usable entries")
    return result



def load_important_form_vars(
        path: Path | str | None, *, required: bool = False
        ) -> dict[str, dict[str, str]]:
    """Load the administrative-variable roles used for form presence."""
    if path is None:
        if required:
            raise ScannerInputError("--important-form-vars is required but missing")
        return {}
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        if required:
            raise ScannerInputError(
                f"Important-form metadata does not exist: {resolved}")
        return {}
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScannerInputError(
            f"Could not parse important-form metadata {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScannerInputError(
            f"Important-form metadata must contain a JSON object: {resolved}")
    result: dict[str, dict[str, str]] = {}
    for form, info in raw.items():
        if not isinstance(info, dict):
            continue
        result[str(form)] = {
            key: str(info.get(key, "") or "").strip()
            for key in _ADMIN_FIELD_KEYS
        }
    return result


def _prescient_completion_variable(completion_var: str) -> str:
    if not completion_var:
        return ""
    result = (completion_var + "_rpms").replace("_hc", "")
    result = result.replace("onboarding", "checkin")
    result = result.replace("end_of_12month_study_pe", "checkin")
    return result.replace("end_of_12month_study_p", "checkin")


def _completion_variables(
        form_info: Mapping[str, object], network: str) -> set[str]:
    base = str(form_info.get("completion_var", "") or "").strip()
    result = {base} if base else set()
    if base and str(network).strip().upper() == "PRESCIENT":
        result.add(_prescient_completion_variable(base))
    return {value for value in result if value}


def _administrative_variables(
        form_info: Mapping[str, object], network: str) -> set[str]:
    result = {
        str(form_info.get(key, "") or "").strip()
        for key in _ADMIN_FIELD_KEYS
    }
    result.update(_completion_variables(form_info, network))
    return {value for value in result if value}


def _is_missing_like(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or _is_project_missing_code(text)


def _matches_redcap_code(value: object, codes: Iterable[int]) -> bool:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return False
    return number.is_finite() and number in {
        Decimal(code) for code in codes
    }

def _mapped_form(column: str, form_map: Mapping[str, str]) -> tuple[str, str] | None:
    """Return ``(form, base variable)`` with REDCap checkbox fallback."""
    form = form_map.get(column)
    base = column
    if form is None and "___" in column:
        base = column.split("___", 1)[0]
        form = form_map.get(base)
    if not form:
        return None
    return str(form), base


def _is_project_missing_code(text: str) -> bool:
    if text.casefold() in _MISSING_TEXT:
        return True
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return False
    return number.is_finite() and number in _MISSING_NUMBERS


def normalize_value(
        value: object, *, include_missing_codes: bool = False,
        numeric_equivalence: bool = False) -> Hashable:
    """Return a comparison token, preserving unavailable substantive slots."""
    if value is None:
        return MISSING_VALUE_TOKEN
    text = str(value).strip()
    if not text:
        return MISSING_VALUE_TOKEN
    if _is_project_missing_code(text) and not include_missing_codes:
        return MISSING_VALUE_TOKEN
    if numeric_equivalence:
        try:
            number = Decimal(text)
        except (InvalidOperation, ValueError):
            pass
        else:
            if number.is_finite():
                if number == 0:
                    number = Decimal(0)
                return ("number", number.normalize())
    return ("text", text)

def _read_header(path: Path) -> list[str]:
    try:
        header = pd.read_csv(
            path, nrows=0, dtype=str, keep_default_na=False,
            encoding="utf-8-sig", on_bad_lines="error")
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise ScannerInputError(f"Could not parse combined CSV {path}: {exc}") from exc
    return list(header.columns)


def collect_form_observations(
        specs: Sequence[InputSpec], form_map: Mapping[str, str], *,
        important_form_vars: Mapping[str, Mapping[str, object]] | None = None,
        participant_column: str = "subjectid",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        include_missing_codes: bool = False,
        include_completion_fields: bool = False,
        numeric_equivalence: bool = False) -> ObservationIndex:
    """Read combined CSVs and index substantive forms by participant.

    Participant observations retain unavailable substantive slots. After all
    files are scanned, a form/timepoint key is kept only when at least one
    participant has a real substantive value on it.
    """
    if chunk_size < 1:
        raise ScannerInputError("--chunk-size must be a positive integer")
    metadata = important_form_vars or {}
    observations: ObservationIndex = defaultdict(dict)
    for spec in specs:
        header = _read_header(spec.path)
        header_set = set(header)
        file_observations: ObservationIndex = defaultdict(dict)
        populated_keys: set[FormKey] = set()
        if participant_column not in header_set:
            raise ScannerInputError(
                f"Participant column {participant_column!r} is missing from {spec.path}")
        matching_columns: dict[str, list[str]] = defaultdict(list)
        substantive_columns: dict[str, list[str]] = defaultdict(list)
        for column in header:
            if column == participant_column:
                continue
            mapped = _mapped_form(column, form_map)
            if mapped is None:
                continue
            form, base_variable = mapped
            form_info = metadata.get(form, {})
            administrative = _administrative_variables(form_info, spec.network)
            completion_variables = _completion_variables(form_info, spec.network)
            is_completion = (
                base_variable in completion_variables
                or column in completion_variables
                or base_variable.casefold().endswith(
                    ("_complete", "_complete_rpms"))
                or column.casefold().endswith(
                    ("_complete", "_complete_rpms"))
            )
            is_administrative = (
                base_variable in administrative or column in administrative)
            if is_administrative or is_completion:
                if include_completion_fields and is_completion:
                    matching_columns[form].append(column)
                continue
            matching_columns[form].append(column)
            substantive_columns[form].append(column)
        if not substantive_columns:
            raise ScannerInputError(
                f"None of the columns in {spec.path} are substantive form fields")
        missing_marker_columns: dict[str, str] = {}
        missing_completion_columns: dict[str, str] = {}
        for form in substantive_columns:
            form_info = metadata.get(form, {})
            missing_var = str(form_info.get("missing_var", "") or "").strip()
            if missing_var and missing_var in header_set:
                missing_marker_columns[form] = missing_var
            if spec.network == "PRESCIENT":
                completion_candidates = _completion_variables(
                    form_info, spec.network)
                for candidate in sorted(completion_candidates):
                    if candidate.endswith("_rpms") and candidate in header_set:
                        missing_completion_columns[form] = candidate
                        break
        selected_column_set = {participant_column}
        for columns in matching_columns.values():
            selected_column_set.update(columns)
        selected_column_set.update(missing_marker_columns.values())
        selected_column_set.update(missing_completion_columns.values())
        projected_columns = [
            column for column in header if column in selected_column_set
        ]
        column_positions = {
            column: position
            for position, column in enumerate(projected_columns)
        }
        matching_positions = {
            form: tuple(column_positions[column] for column in columns)
            for form, columns in matching_columns.items()
            if form in substantive_columns
        }
        substantive_positions = {
            form: tuple(column_positions[column] for column in columns)
            for form, columns in substantive_columns.items()
        }
        missing_marker_positions = {
            form: column_positions[column]
            for form, column in missing_marker_columns.items()
        }
        missing_completion_positions = {
            form: column_positions[column]
            for form, column in missing_completion_columns.items()
        }
        seen_subjects: set[str] = set()
        try:
            chunks = pd.read_csv(
                spec.path,
                usecols=list(selected_column_set),
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
                on_bad_lines="error",
                chunksize=chunk_size,
            )
            for chunk in chunks:
                subjects = chunk[participant_column].astype(str).str.strip()
                if (subjects == "").any():
                    row_number = int((subjects == "").idxmax()) + 2
                    raise ScannerInputError(
                        f"Blank participant ID in {spec.path} near CSV row {row_number}")
                duplicates = set(
                    subjects[subjects.duplicated(keep=False)].tolist())
                duplicates.update(set(subjects).intersection(seen_subjects))
                if duplicates:
                    sample = f"{len(duplicates)} duplicate participant ID value(s)"
                    raise ScannerInputError(
                        f"Duplicate participant row(s) in {spec.path}: {sample}")
                seen_subjects.update(subjects.tolist())
                for row_values, subject in zip(
                        chunk.itertuples(index=False, name=None), subjects):
                    participant = (spec.network, subject)
                    for form, positions in substantive_positions.items():
                        missing_position = missing_marker_positions.get(form)
                        if (
                            missing_position is not None
                            and _matches_redcap_code(
                                row_values[missing_position], (1,))
                        ):
                            continue
                        completion_position = missing_completion_positions.get(form)
                        if (
                            completion_position is not None
                            and _matches_redcap_code(
                                row_values[completion_position], (3, 4))
                        ):
                            continue
                        key = (spec.timepoint, form)
                        tokens = [
                            normalize_value(
                                row_values[position],
                                include_missing_codes=include_missing_codes,
                                numeric_equivalence=numeric_equivalence,
                            )
                            for position in matching_positions.get(form, ())
                        ]
                        if not tokens:
                            continue
                        file_observations[key][participant] = Counter(tokens)
                        if any(
                            not _is_missing_like(row_values[position])
                            for position in positions
                        ):
                            populated_keys.add(key)
        except ScannerInputError:
            raise
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            raise ScannerInputError(
                f"Could not parse combined CSV {spec.path}: {exc}") from exc

        # One canonical file is one network/timepoint scope. Drop forms that
        # are wholly unavailable in that scope before retaining observations,
        # while preserving unavailable participant rows for populated forms.
        for key in populated_keys:
            observations[key].update(file_observations[key])

    return {
        form_key: dict(participants)
        for form_key, participants in observations.items()
    }

def multiset_overlap(
        values_a: Mapping[Hashable, int],
        values_b: Mapping[Hashable, int]
        ) -> tuple[int, int, int, float | None]:
    """Return real matches, effective sizes, and Dice percentage.

    Missing slots cancel one-for-one only when both forms have one available.
    Any surplus missing slots remain in that form's effective size as
    unmatched values.
    """
    raw_size_a = int(sum(values_a.values()))
    raw_size_b = int(sum(values_b.values()))
    if raw_size_a <= 0 or raw_size_b <= 0:
        raise ValueError("Both value multisets must be nonempty")
    paired_missing = min(
        int(values_a.get(MISSING_VALUE_TOKEN, 0)),
        int(values_b.get(MISSING_VALUE_TOKEN, 0)),
    )
    size_a = raw_size_a - paired_missing
    size_b = raw_size_b - paired_missing
    real_a = {
        value: count for value, count in values_a.items()
        if value != MISSING_VALUE_TOKEN
    }
    real_b = {
        value: count for value, count in values_b.items()
        if value != MISSING_VALUE_TOKEN
    }
    smaller, larger = (
        (real_a, real_b) if len(real_a) <= len(real_b)
        else (real_b, real_a)
    )
    matches = int(sum(
        min(int(count), int(larger.get(value, 0)))
        for value, count in smaller.items()
    ))
    denominator = size_a + size_b
    percentage = 200.0 * matches / float(denominator) if denominator > 0 else None
    return matches, size_a, size_b, percentage

def _prepare_value_profile(
        values: Mapping[Hashable, int]) -> _PreparedValueProfile:
    """Precompute values reused by every self-pair involving one profile."""
    raw_size = int(sum(values.values()))
    if raw_size <= 0:
        raise ValueError("Value multiset must be nonempty")
    missing_count = int(values.get(MISSING_VALUE_TOKEN, 0))
    real_values = {
        value: int(count)
        for value, count in values.items()
        if value != MISSING_VALUE_TOKEN
    }
    return _PreparedValueProfile(
        raw_size=raw_size,
        missing_count=missing_count,
        nonmissing_size=raw_size - missing_count,
        real_values=real_values,
        real_items=tuple(real_values.items()),
    )


def _add_weighted_prepared_overlap(
        profile_a: _PreparedValueProfile,
        profile_b: _PreparedValueProfile,
        weight: int, score_histogram: Counter[Fraction],
        matching_histogram: Counter[int], size_a_histogram: Counter[int],
        size_b_histogram: Counter[int]) -> bool:
    """Add a repeated self-pair outcome without rebuilding profile invariants."""
    if weight <= 0:
        return False
    if profile_a is profile_b:
        matches = profile_a.nonmissing_size
        if matches <= 0:
            return False
        score_histogram[Fraction(1, 1)] += weight
        matching_histogram[matches] += weight
        size_a_histogram[matches] += weight
        size_b_histogram[matches] += weight
        return True
    paired_missing = min(profile_a.missing_count, profile_b.missing_count)
    size_a = profile_a.raw_size - paired_missing
    size_b = profile_b.raw_size - paired_missing
    denominator = size_a + size_b
    if denominator <= 0:
        return False
    items, lookup = (
        (profile_a.real_items, profile_b.real_values)
        if len(profile_a.real_items) <= len(profile_b.real_items)
        else (profile_b.real_items, profile_a.real_values)
    )
    matches = int(sum(
        min(count, int(lookup.get(value, 0)))
        for value, count in items
    ))
    score_histogram[Fraction(2 * matches, denominator)] += weight
    matching_histogram[matches] += weight
    size_a_histogram[size_a] += weight
    size_b_histogram[size_b] += weight
    return True

def _form_sort_key(form_key: FormKey) -> tuple[int, str, str]:
    timepoint, form = form_key
    return (
        TIMEPOINT_ORDER.get(timepoint, len(TIMEPOINT_ORDER)),
        timepoint.casefold(),
        form.casefold(),
    )


def _weighted_median(histogram: Mapping[Fraction | int, int]) -> float:
    """Return the ordinary median represented by a value/weight histogram."""
    total = int(sum(histogram.values()))
    if total <= 0:
        raise ValueError("Cannot calculate a weighted median without values")
    ranks = ((total - 1) // 2, total // 2)
    selected: list[Fraction | int] = []
    cumulative = 0
    for value in sorted(histogram):
        cumulative += int(histogram[value])
        while len(selected) < 2 and cumulative > ranks[len(selected)]:
            selected.append(value)
        if len(selected) == 2:
            break
    return (float(selected[0]) + float(selected[1])) / 2.0


def _add_weighted_overlap(
        values_a: Mapping[Hashable, int], values_b: Mapping[Hashable, int],
        weight: int, score_histogram: Counter[Fraction],
        matching_histogram: Counter[int], size_a_histogram: Counter[int],
        size_b_histogram: Counter[int]) -> bool:
    """Add one overlap outcome with multiplicity; return whether it scored."""
    if weight <= 0:
        return False
    matches, size_a, size_b, percentage = multiset_overlap(values_a, values_b)
    if percentage is None:
        return False
    score_histogram[Fraction(2 * matches, size_a + size_b)] += weight
    matching_histogram[matches] += weight
    size_a_histogram[size_a] += weight
    size_b_histogram[size_b] += weight
    return True


def summarize_form_matches(
        observations: Mapping[FormKey, Mapping[ParticipantKey, ValueCounter]], *,
        include_unshared_pairs: bool = True) -> pd.DataFrame:
    """Summarize same-source-form pairs across equal or different timepoints.

    Different timepoints remain matched within participant. Exact identity
    self-pairs use distinct participant records.
    """
    form_keys = sorted(observations, key=_form_sort_key)
    records: list[dict[str, object]] = []
    for left_index, left_key in enumerate(form_keys):
        left_participants = observations[left_key]
        for right_key in form_keys[left_index:]:
            if left_key[1] != right_key[1]:
                continue
            right_participants = observations[right_key]
            score_histogram: Counter[Fraction] = Counter()
            matching_histogram: Counter[int] = Counter()
            size_left_histogram: Counter[int] = Counter()
            size_right_histogram: Counter[int] = Counter()
            scored_participants: set[ParticipantKey] = set()

            if left_key == right_key:
                groups: dict[
                    frozenset[tuple[Hashable, int]],
                    tuple[_PreparedValueProfile, list[ParticipantKey]],
                ] = {}
                for participant in sorted(left_participants):
                    values = left_participants[participant]
                    fingerprint = frozenset(values.items())
                    if fingerprint not in groups:
                        groups[fingerprint] = (
                            _prepare_value_profile(values), [])
                    groups[fingerprint][1].append(participant)
                profile_groups = list(groups.values())
                scored_group_indices: set[int] = set()
                for group_index, (profile_left, members_left) in enumerate(
                        profile_groups):
                    for right_index in range(group_index, len(profile_groups)):
                        profile_right, members_right = profile_groups[right_index]
                        if group_index == right_index:
                            weight = len(members_left) * (len(members_left) - 1) // 2
                        else:
                            weight = len(members_left) * len(members_right)
                        if _add_weighted_prepared_overlap(
                                profile_left, profile_right, weight,
                                score_histogram, matching_histogram,
                                size_left_histogram, size_right_histogram):
                            scored_group_indices.add(group_index)
                            scored_group_indices.add(right_index)
                for group_index in scored_group_indices:
                    scored_participants.update(profile_groups[group_index][1])
            else:
                common = sorted(
                    set(left_participants).intersection(right_participants))
                for participant in common:
                    if _add_weighted_overlap(
                            left_participants[participant],
                            right_participants[participant],
                            1,
                            score_histogram,
                            matching_histogram,
                            size_left_histogram,
                            size_right_histogram):
                        scored_participants.add(participant)

            record_pairs_compared = int(sum(score_histogram.values()))
            if record_pairs_compared == 0 and not include_unshared_pairs:
                continue
            timepoint_left, form_left = left_key
            timepoint_right, form_right = right_key
            record: dict[str, object] = {
                "form_1": form_left,
                "timepoint_1": timepoint_left,
                "form_2": form_right,
                "timepoint_2": timepoint_right,
                "median_match_percentage": "",
                "mean_match_percentage": "",
                "record_pairs_compared": record_pairs_compared,
                "unique_participants_compared": len(scored_participants),
                "participant_records_form_1": len(left_participants),
                "participant_records_form_2": len(right_participants),
                "median_matching_values": "",
                "median_usable_values_form_1": "",
                "median_usable_values_form_2": "",
                "networks_represented": "|".join(sorted({
                    network for network, _subject in scored_participants
                })),
            }
            if record_pairs_compared:
                weighted_score_sum = sum(
                    score * weight
                    for score, weight in score_histogram.items()
                )
                if left_key == right_key:
                    pooled_size_histogram = (
                        size_left_histogram + size_right_histogram)
                    median_size_left = _weighted_median(
                        pooled_size_histogram)
                    median_size_right = median_size_left
                else:
                    median_size_left = _weighted_median(
                        size_left_histogram)
                    median_size_right = _weighted_median(
                        size_right_histogram)
                record.update({
                    "median_match_percentage": round(
                        100.0 * _weighted_median(score_histogram), 6),
                    "mean_match_percentage": round(
                        100.0 * float(weighted_score_sum)
                        / record_pairs_compared, 6),
                    "median_matching_values": round(
                        _weighted_median(matching_histogram), 6),
                    "median_usable_values_form_1": round(
                        median_size_left, 6),
                    "median_usable_values_form_2": round(
                        median_size_right, 6),
                })
            records.append(record)
    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)

def _load_config(path: Path, explicitly_requested: bool) -> dict[str, object]:
    if not path.is_file():
        if explicitly_requested:
            raise ScannerInputError(f"Explicit config JSON does not exist: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScannerInputError(f"Could not parse config JSON {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ScannerInputError(f"Config JSON must contain an object: {path}")
    return config


def _configured_path(
        config: Mapping[str, object], config_path: Path, key: str) -> Path | None:
    raw_paths = config.get("paths", {})
    if not isinstance(raw_paths, Mapping):
        return None
    raw_value = raw_paths.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    result = Path(raw_value).expanduser()
    if not result.is_absolute():
        result = config_path.parent / result
    return result.resolve()


def _testing_enabled(config: Mapping[str, object]) -> bool:
    """Validate and return the repository's strict testing-output switch."""
    if "testing_enabled" not in config:
        return False
    value = config["testing_enabled"]
    if not isinstance(value, str) or value not in {"True", "False"}:
        raise ScannerInputError(
            "config.testing_enabled must be exactly the string 'True' or "
            "'False' when present"
        )
    return value == "True"


def _refuse_output_collision(
        output_path: Path, *, specs: Sequence[InputSpec],
        form_map_path: Path,
        important_form_vars_path: Path | None,
        config_path: Path, protect_config: bool) -> None:
    """Prevent the aggregate output from replacing any source artifact."""
    protected_paths = {spec.path.resolve() for spec in specs}
    protected_paths.add(form_map_path.resolve())
    if important_form_vars_path is not None:
        protected_paths.add(important_form_vars_path.resolve())
    if protect_config:
        protected_paths.add(config_path.resolve())
    resolved_output = output_path.resolve()
    if resolved_output in protected_paths:
        raise ScannerInputError(
            "Output path collides with an input, metadata, form map, or "
            f"config file: {resolved_output}"
        )

def _atomic_write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", delete=False,
                prefix=f".{output_path.name}.{os.getpid()}.", suffix=".tmp",
                dir=output_path.parent) as handle:
            temporary_path = Path(handle.name)
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare only instances of the same source form across equal or "
            "different timepoints; write median and mean encoded-value overlap "
            "percentages."))
    parser.add_argument(
        "--config",
        help="Pipeline config.json used for default input, mapping, and output paths.")
    parser.add_argument(
        "--form-map", "--grouped-variables", dest="form_map",
        help=(
            "grouped_variables.json containing var_forms, or a direct JSON "
            "variable-to-form map (default: configured dependencies path)."))
    parser.add_argument(
        "--important-form-vars",
        help=(
            "important_form_vars.json identifying administrative and missing "
            "fields (default: beside the form map or in dependencies)."))
    parser.add_argument(
        "--input", action="append", default=[], metavar="CSV",
        help=(
            "Explicit canonical combined CSV; repeatable and glob-aware. "
            "When supplied, --input-dir is ignored."))
    parser.add_argument(
        "--input-dir", "--combined-dir", dest="input_dir",
        help="Directory containing canonical AMPSCZ combined CSVs.")
    parser.add_argument(
        "--allow-partial-input-dir", action="store_true",
        help=(
            "Allow a directory scan with fewer than all 36 canonical inputs; "
            "intended for deliberate partial analyses only."))
    parser.add_argument(
        "--output",
        help=(
            "Summary CSV (default: configured output path/"
            "form_value_match_summary.csv)."))
    parser.add_argument(
        "--participant-column", default="subjectid",
        help="Participant identifier column (default: %(default)s).")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="CSV rows read per chunk (default: %(default)s).")
    parser.add_argument(
        "--include-missing-codes", action="store_true",
        help="Include nonblank project missing-value sentinels such as -9 and 999.")
    parser.add_argument(
        "--include-completion-fields", action="store_true",
        help="Include mapped *_complete fields in each form's value multiset.")
    parser.add_argument(
        "--numeric-equivalence", action="store_true",
        help="Treat numerically equivalent text spellings (1, 1.0, 01) as equal.")
    parser.add_argument(
        "--omit-unshared-pairs", action="store_true",
        help=(
            "Omit form pairs with no scoreable record pair; by default they "
            "are listed with blank percentages and a pair count of zero."))
    return parser


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.chunk_size < 1:
            raise ScannerInputError("--chunk-size must be a positive integer")
        if not str(args.participant_column).strip():
            raise ScannerInputError("--participant-column cannot be blank")

        config_path = (
            Path(args.config).expanduser().resolve()
            if args.config else _default_config_path())
        config = _load_config(config_path, explicitly_requested=bool(args.config))
        testing_enabled = _testing_enabled(config)

        configured_dependencies = _configured_path(
            config, config_path, "dependencies_path")
        form_map_path = (
            Path(args.form_map).expanduser().resolve()
            if args.form_map else
            (configured_dependencies or config_path.parent / "dependencies")
            / "grouped_variables.json"
        )
        form_map = load_form_map(form_map_path)

        if args.important_form_vars:
            important_form_vars_path: Path | None = Path(
                args.important_form_vars).expanduser().resolve()
        else:
            adjacent_metadata = (
                form_map_path.parent / "important_form_vars.json").resolve()
            configured_metadata = (
                (configured_dependencies / "important_form_vars.json").resolve()
                if configured_dependencies is not None else None
            )
            important_form_vars_path = next(
                (
                    candidate for candidate in (
                        adjacent_metadata, configured_metadata)
                    if candidate is not None and candidate.is_file()
                ),
                None,
            )
        important_form_vars = load_important_form_vars(
            important_form_vars_path,
            required=bool(args.important_form_vars),
        )

        input_dir = (
            Path(args.input_dir).expanduser().resolve()
            if args.input_dir else
            _configured_path(config, config_path, "combined_csv_path")
        )
        specs = discover_inputs(
            args.input,
            input_dir,
            allow_partial_input_dir=args.allow_partial_input_dir,
        )

        configured_output = _configured_path(config, config_path, "output_path")
        if (
            configured_output is not None
            and testing_enabled
        ):
            configured_output = configured_output / "testing"
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output else
            (configured_output or Path.cwd().resolve())
            / "form_value_match_summary.csv"
        )

        _refuse_output_collision(
            output_path,
            specs=specs,
            form_map_path=form_map_path,
            important_form_vars_path=important_form_vars_path,
            config_path=config_path,
            protect_config=config_path.is_file(),
        )
        observations = collect_form_observations(
            specs,
            form_map,
            important_form_vars=important_form_vars,
            participant_column=str(args.participant_column),
            chunk_size=args.chunk_size,
            include_missing_codes=args.include_missing_codes,
            include_completion_fields=args.include_completion_fields,
            numeric_equivalence=args.numeric_equivalence,
        )
        summary = summarize_form_matches(
            observations,
            include_unshared_pairs=not args.omit_unshared_pairs,
        )
        _atomic_write_csv(summary, output_path)
    except ScannerInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Error writing form-value match summary: {exc}", file=sys.stderr)
        return 2

    print(
        "Form-value match summary complete: "
        f"{len(summary)} form pair(s) written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

