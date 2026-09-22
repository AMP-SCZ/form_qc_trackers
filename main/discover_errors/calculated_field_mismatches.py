"""Recalculate REDCap calculated fields across combined CSV exports.

The scanner is intentionally standalone: it does not mutate combined CSVs and
is not called by the normal QC pipeline.  It translates the current REDCap data
dictionary once, loads the relevant portions of every combined CSV, recalculates
every available calculated field, and writes only mismatches or evaluation
failures.

The default ``direct`` comparison evaluates every formula against an immutable
snapshot of the values stored in the input CSVs.  An opt-in ``propagated`` mode
recalculates same-event dependencies in order, but explicit-event references
still use the immutable snapshot.  This distinction is deliberate: the current
study dictionary contains cyclic cross-event calculated-field dependencies, so
silently applying a partial recursive closure would make results depend on
iteration order. Direct-mode downstream mismatches and calculations whose
branching eligibility cannot be safely established are retained but marked
``semantic_uncertain``. The selected modes are recorded in diagnostics.
"""

from __future__ import annotations

import argparse
from collections import ChainMap, Counter, defaultdict
import csv
from dataclasses import dataclass, field
from decimal import Decimal, DecimalException, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from process_variables.transform_calculated_fields import (  # noqa: E402
    CALCULATION_COLUMN,
    FIELD_TYPE_COLUMN,
    FORM_COLUMN,
    VARIABLE_COLUMN,
    CalculationSchemaError,
    CalculationTranslationError,
    TransformCalculatedFields,
    _discover_data_dictionary,
    evaluate_converted_calculation,
    iter_field_references,
    redcap_truthy,
    translate_redcap_calculation,
)


SCHEMA_VERSION = 2
DEFAULT_NUMERIC_TOLERANCE = Decimal("1e-10")
DEFAULT_CHUNK_SIZE = 1_000
CROSS_EVENT_MODE = "stored_snapshot"

NETWORK_ORDER = {"PRONET": 0, "PRESCIENT": 1, "UNKNOWN": 2}
TIMEPOINTS = (
    "screening", "baseline",
    *(f"month{x}" for x in range(1, 13)),
    "month18", "month24", "floating", "conversion",
)
TIMEPOINT_ORDER = {value: index for index, value in enumerate(TIMEPOINTS)}

CANONICAL_FILE_RE = re.compile(
    r"^AMPSCZ-combined-redcap_"
    r"(?P<timepoint>screening|baseline|month_(?:[1-9]|1[0-2]|18|24)|"
    r"floating_forms|conversion)_"
    r"(?P<network>ProNET|PRONET|PRESCIENT)-day1to1\.csv$",
    re.IGNORECASE,
)

IDENTITY_COLUMNS = (
    "subjectid", "redcap_event_name", "event_name",
    "redcap_repeat_instrument", "redcap_repeat_instance",
    "network", "timepoint", "chrcrit_part",
)

OUTPUT_COLUMNS = (
    "network", "timepoint", "subjectid", "redcap_event_name",
    "redcap_repeat_instrument", "redcap_repeat_instance",
    "variable", "form", "stored_value", "recalculated_value",
    "status", "error", "numeric_difference",
    "structural_blank_dependencies", "cross_event_mode",
    "comparison_mode", "semantic_risk", "actionability",
    "branching_eligibility", "branching_eligibility_reason",
    "mismatch_kind", "source_file", "source_row",
)


class ScannerInputError(ValueError):
    """An input/configuration problem that must fail before output is written."""


@dataclass(frozen=True)
class InputSpec:
    path: Path
    network: str
    timepoint: str
    canonical: bool


@dataclass
class LoadedRow:
    network: str
    timepoint: str
    subjectid: str
    event_name: str
    repeat_instrument: str
    repeat_instance: str
    values: dict[str, Any]
    available_columns: frozenset[str]
    source_file: str
    source_row: int

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        event_identity = self.event_name or f"__timepoint__:{self.timepoint}"
        return (
            self.network, self.subjectid, event_identity,
            self.repeat_instrument, self.repeat_instance,
        )

    @property
    def is_repeating(self) -> bool:
        return bool(self.repeat_instrument or self.repeat_instance)


@dataclass(frozen=True)
class CalculationOutcome:
    status: str
    value: Any = ""
    error: str = ""
    structural_dependencies: tuple[str, ...] = ()


@dataclass
class ScanArtifacts:
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    structural_diagnostics: dict[tuple[Any, ...], dict[str, Any]] = field(
        default_factory=dict)
    counters: Counter[str] = field(default_factory=Counter)

    def add_diagnostic(
            self, key: tuple[Any, ...], diagnostic: dict[str, Any]) -> None:
        self.structural_diagnostics.setdefault(key, diagnostic)


@dataclass
class LoadedInputs:
    rows: list[LoadedRow]
    input_details: list[dict[str, Any]]
    file_schemas: dict[str, frozenset[str]]
    event_schemas: dict[tuple[str, str], frozenset[str]]
    timepoint_schemas: dict[tuple[str, str], frozenset[str]]
    network_schemas: dict[str, frozenset[str]]
    quarantined_identities: frozenset[tuple[str, str, str, str, str]]
    quarantined_values: frozenset[
        tuple[str, str, str, str, str, str]]
    row_identity_diagnostics: list[dict[str, Any]]


def _is_real_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        missing = pd.isna(value)
        # numpy.bool_ is intentionally accepted here.  Array-like results do
        # not have an unambiguous truth value and fall through to ``False``.
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _normalized_value(value: Any) -> tuple[str, Any]:
    if _is_real_missing(value):
        return "blank", ""
    if isinstance(value, bool):
        return "number", Decimal(1 if value else 0)
    text = str(value)
    stripped = text.strip()
    try:
        number = Decimal(stripped)
    except (InvalidOperation, ValueError):
        return "text", text
    if not number.is_finite():
        # Literal strings such as ``NaN`` are meaningful REDCap sentinels;
        # only actual floating/pandas NaN values were classified as missing.
        return "text", text
    return "number", number


def values_match(
        left: Any, right: Any, *,
        numeric_tolerance: Decimal | str | float = DEFAULT_NUMERIC_TOLERANCE
        ) -> bool:
    """Compare values without false mismatches from numeric representation.

    Thus ``1``, ``1.0``, ``01.000`` and ``1e0`` compare equal. Actual missing
    scalars and whitespace-only cells compare as blank, while the literal text
    ``"NaN"`` remains text and never silently becomes missing.
    """
    try:
        tolerance = Decimal(str(numeric_tolerance))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("numeric tolerance must be a finite non-negative value") from exc
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("numeric tolerance must be a finite non-negative value")

    left_kind, left_value = _normalized_value(left)
    right_kind, right_value = _normalized_value(right)
    if left_kind != right_kind:
        return False
    if left_kind == "number":
        if left_value == right_value:
            return True
        try:
            return abs(left_value - right_value) <= tolerance
        except DecimalException:
            # Extremely large hostile exponents can exceed the active Decimal
            # context during subtraction. Unequal values fail closed instead
            # of crashing the entire audit.
            return False
    return left_value == right_value


def _numeric_difference(left: Any, right: Any) -> str:
    left_kind, left_value = _normalized_value(left)
    right_kind, right_value = _normalized_value(right)
    if left_kind != "number" or right_kind != "number":
        return ""
    try:
        return str(abs(left_value - right_value))
    except DecimalException:
        return "unrepresentable"


def _display_value(value: Any) -> str:
    if _is_real_missing(value):
        return ""
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def _normalize_network(value: Any) -> str:
    normalized = str(value or "").strip().replace("-", "").replace("_", "")
    folded = normalized.casefold()
    if folded == "pronet":
        return "PRONET"
    if folded == "prescient":
        return "PRESCIENT"
    return "UNKNOWN"


def _normalize_timepoint(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = normalized.replace("floating_forms", "floating")
    match = re.fullmatch(r"month_?(\d+)", normalized)
    if match:
        return f"month{int(match.group(1))}"
    if normalized in {"screening", "baseline", "floating", "conversion"}:
        return normalized
    return normalized


def _timepoint_from_event(event_name: str) -> str:
    event = re.sub(r"_arm_\d+$", "", str(event_name).strip(), flags=re.I)
    return _normalize_timepoint(event)


def _cohort_arm(value: Any) -> str:
    cohort = str(value or "").strip().casefold()
    if cohort in {"1", "1.0", "chr"}:
        return "1"
    if cohort in {"2", "2.0", "hc"}:
        return "2"
    return ""


def _event_name_for_timepoint(timepoint: str, arm: str) -> str:
    if not timepoint or not arm:
        return ""
    prefix = (
        "floating_forms" if timepoint == "floating"
        else re.sub(r"^month(\d+)$", r"month_\1", timepoint)
    )
    return f"{prefix}_arm_{arm}"


def _parse_input_name(path: Path) -> InputSpec | None:
    match = CANONICAL_FILE_RE.fullmatch(path.name)
    if match is None:
        return None
    return InputSpec(
        path=path.resolve(),
        network=_normalize_network(match.group("network")),
        timepoint=_normalize_timepoint(match.group("timepoint")),
        canonical=True,
    )


def _expand_explicit_inputs(values: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        expanded = Path(value).expanduser()
        if any(character in value for character in "*?["):
            parent = expanded.parent if str(expanded.parent) else Path.cwd()
            paths.extend(parent.glob(expanded.name))
        else:
            paths.append(expanded)
    return paths


def discover_inputs(
        explicit_inputs: Sequence[str], input_dir: Path | None,
        excluded_paths: Iterable[Path] = (), *,
        require_complete_directory: bool = False) -> list[InputSpec]:
    excluded = {path.expanduser().resolve() for path in excluded_paths}
    specs: list[InputSpec] = []
    if explicit_inputs:
        candidates = _expand_explicit_inputs(explicit_inputs)
        if not candidates:
            raise ScannerInputError("No explicit input CSV matched")
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if not resolved.is_file():
                raise ScannerInputError(f"Input CSV does not exist: {resolved}")
            parsed = _parse_input_name(resolved)
            specs.append(parsed or InputSpec(
                resolved, "UNKNOWN", "", canonical=False))
    else:
        if input_dir is None:
            raise ScannerInputError(
                "No input CSVs were supplied and no input directory is configured")
        directory = input_dir.expanduser().resolve()
        if not directory.is_dir():
            raise ScannerInputError(f"Input directory does not exist: {directory}")
        for candidate in sorted(directory.iterdir(), key=lambda path: path.name.casefold()):
            if not candidate.is_file():
                continue
            parsed = _parse_input_name(candidate)
            if parsed is not None:
                if candidate.resolve() in excluded:
                    raise ScannerInputError(
                        "An output/source path collides with canonical input: "
                        f"{candidate.resolve()}")
                specs.append(parsed)
        if not specs:
            raise ScannerInputError(
                f"No canonical combined input CSVs were found in {directory}")

    unique: dict[Path, InputSpec] = {}
    for spec in specs:
        if spec.path in excluded:
            raise ScannerInputError(
                f"An input path collides with an output/source path: {spec.path}")
        unique.setdefault(spec.path, spec)
    ordered = sorted(
        unique.values(),
        key=lambda spec: (
            NETWORK_ORDER.get(spec.network, 99),
            TIMEPOINT_ORDER.get(spec.timepoint, 99),
            spec.path.name.casefold(), str(spec.path)),
    )

    canonical_slots: dict[tuple[str, str], Path] = {}
    for spec in ordered:
        if not spec.canonical:
            continue
        slot = (spec.network, spec.timepoint)
        previous = canonical_slots.get(slot)
        if previous is not None and previous != spec.path:
            raise ScannerInputError(
                f"Ambiguous combined CSVs for {slot[0]} {slot[1]}: "
                f"{previous}, {spec.path}")
        canonical_slots[slot] = spec.path
    if require_complete_directory and not explicit_inputs:
        expected = {
            (network, timepoint)
            for network in ("PRONET", "PRESCIENT")
            for timepoint in TIMEPOINTS
        }
        missing = sorted(
            expected.difference(canonical_slots),
            key=lambda slot: (
                NETWORK_ORDER.get(slot[0], 99),
                TIMEPOINT_ORDER.get(slot[1], 99)),
        )
        if missing:
            rendered = ", ".join(
                f"{network}/{timepoint}" for network, timepoint in missing)
            raise ScannerInputError(
                "Canonical input directory is incomplete; missing "
                f"{len(missing)} of 36 network/timepoint file(s): {rendered}. "
                "Use --allow-partial-input-dir only for an intentional test run.")
    return ordered


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScannerInputError(f"Could not read config JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScannerInputError(f"Config root must be an object: {path}")
    return value


def _config_path_value(
        config: Mapping[str, Any], config_path: Path, key: str) -> Path | None:
    try:
        raw = config["paths"][key]
    except (KeyError, TypeError):
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _dictionary_from_config(
        config: Mapping[str, Any], config_path: Path) -> Path | None:
    dependencies = _config_path_value(
        config, config_path, "dependencies_path")
    if dependencies is None:
        return None
    dictionary_dir = dependencies / "data_dictionary"
    exact = dictionary_dir / "current_data_dictionary.csv"
    if exact.is_file():
        return exact.resolve()
    matches = sorted(dictionary_dir.glob("*current_data_dictionary*.csv"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ScannerInputError(
            "Multiple current data dictionaries were found under the "
            f"configured dependencies path: {dictionary_dir}")
    return None


def _load_important_form_vars(
        path: Path | None, *, required: bool) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        if required:
            raise ScannerInputError(
                f"Important-form variable map does not exist: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScannerInputError(
            f"Could not read important-form variable map {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScannerInputError(
            f"Important-form variable map root must be an object: {path}")
    result = {
        str(form): dict(info)
        for form, info in value.items() if isinstance(info, dict)
    }
    if required and not result:
        raise ScannerInputError(
            f"Important-form variable map is empty or has no object entries: {path}")
    return result


def _load_forms_per_timepoint(
        path: Path | None, *, required: bool,
        ) -> dict[str, dict[str, list[str]]]:
    if path is None or not path.is_file():
        if required:
            raise ScannerInputError(
                f"Forms-per-timepoint map does not exist: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScannerInputError(
            f"Could not read forms-per-timepoint map {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScannerInputError(
            f"Forms-per-timepoint map root must be an object: {path}")
    if required:
        missing_cohorts = [
            cohort for cohort in ("CHR", "HC")
            if not isinstance(value.get(cohort), dict)
        ]
        if missing_cohorts:
            raise ScannerInputError(
                "Forms-per-timepoint map is missing cohort schedule(s) "
                f"{missing_cohorts}: {path}")
    result: dict[str, dict[str, list[str]]] = {}
    for cohort in ("CHR", "HC"):
        raw_schedule = value.get(cohort, {})
        if not isinstance(raw_schedule, dict):
            continue
        result[cohort] = {
            _normalize_timepoint(timepoint): [str(form) for form in forms]
            for timepoint, forms in raw_schedule.items()
            if isinstance(forms, list)
        }
    if required and not any(result.values()):
        raise ScannerInputError(
            f"Forms-per-timepoint map contains no schedule entries: {path}")
    return result


def _read_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            first_line = source.readline()
    except OSError as exc:
        raise ScannerInputError(f"Could not read input CSV {path}: {exc}") from exc
    if not first_line:
        raise ScannerInputError(f"Input CSV is empty: {path}")
    try:
        header = next(csv.reader([first_line]))
    except (csv.Error, StopIteration) as exc:
        raise ScannerInputError(f"Malformed CSV header in {path}: {exc}") from exc
    duplicates = sorted(
        name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise ScannerInputError(
            f"Input CSV has duplicate header(s) {duplicates[:20]}: {path}")
    if "subjectid" not in header:
        raise ScannerInputError(
            f"Required column 'subjectid' is missing from input CSV: {path}")
    return header


def _runtime_reference_key(reference: Mapping[str, Any]) -> str:
    variable = str(reference["variable"])
    choice = reference.get("choice")
    return variable if choice is None else f"{variable}___{choice}"


def build_safe_branching_rules(
        calculations: Mapping[str, Mapping[str, Any]],
        ) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Compile only branching rules safe to use as target eligibility.

    Rules depending on any calculated field are deliberately ignored. The
    current dictionary contains hundreds of self-referential calc visibility
    rules; using a stale target to decide whether to audit that same target can
    hide precisely the mismatch this program is meant to find.
    """
    calculated_fields = set(calculations)
    rules: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, list[str]] = defaultdict(list)
    for variable, entry in sorted(calculations.items()):
        raw = str(entry.get("branching_logic", "")).strip()
        if not raw:
            continue
        try:
            converted, node = translate_redcap_calculation(raw)
            nodes = list(iter_field_references(node))
        except CalculationTranslationError:
            diagnostics["translation_unsupported"].append(variable)
            continue
        if any(reference.variable in calculated_fields for reference in nodes):
            diagnostics["calculated_dependency_ignored"].append(variable)
            continue
        references = []
        for reference in nodes:
            references.append({
                "variable": reference.variable,
                "event": reference.event,
                "choice": reference.choice,
                "instance": reference.instance,
                "kind": "field",
                "display": reference.display,
            })
        rules[variable] = {
            "variable": variable,
            "converted_calculation": converted,
            "references": references,
        }
    return rules, {
        key: sorted(value) for key, value in sorted(diagnostics.items())
    }


def _prescient_completion_variable(form_info: Mapping[str, Any]) -> str:
    completion = str(form_info.get("completion_var", "")).strip()
    if not completion:
        return ""
    completion = (completion + "_rpms").replace("_hc", "")
    completion = completion.replace("onboarding", "checkin")
    completion = completion.replace(
        "end_of_12month_study_pe", "checkin")
    return completion.replace("end_of_12month_study_p", "checkin")


def _relevant_columns(
        calculations: Mapping[str, Mapping[str, Any]],
        important_form_vars: Mapping[str, Mapping[str, Any]] | None = None,
        branching_rules: Mapping[str, Mapping[str, Any]] | None = None,
        ) -> set[str]:
    relevant = set(calculations)
    for entry in calculations.values():
        relevant.update(
            _runtime_reference_key(reference)
            for reference in entry["references"])
        form = str(entry.get("form", "")).strip()
        if form:
            relevant.add(f"{form}_complete")
    for form_info in (important_form_vars or {}).values():
        if not isinstance(form_info, Mapping):
            continue
        for value in (
            form_info.get("missing_var", ""),
            form_info.get("completion_var", ""),
            _prescient_completion_variable(form_info),
        ):
            field_name = str(value).strip()
            if field_name:
                relevant.add(field_name)
    for rule in (branching_rules or {}).values():
        relevant.update(
            _runtime_reference_key(reference)
            for reference in rule.get("references", ()))
    relevant.update(IDENTITY_COLUMNS)
    return relevant


def _row_event_name(
        values: Mapping[str, Any], spec: InputSpec) -> str:
    for column in ("redcap_event_name", "event_name"):
        value = str(values.get(column, "")).strip()
        if value:
            return value
    return _event_name_for_timepoint(
        spec.timepoint, _cohort_arm(values.get("chrcrit_part", "")))


def _propagate_subject_arms(rows: Sequence[LoadedRow]) -> None:
    """Fill sparse event names only when a subject has unambiguous arm data.

    Some real PRESCIENT records legitimately move from arm 1 to arm 2 over
    time. Explicit row-level event names are therefore authoritative and mixed
    arm histories are retained; they are never collapsed into one subject arm.
    """
    event_arms: dict[tuple[str, str], set[str]] = defaultdict(set)
    cohort_arms: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        event_arm = _event_arm(row.event_name)
        cohort_arm = _cohort_arm(row.values.get("chrcrit_part", ""))
        if event_arm:
            event_arms[(row.network, row.subjectid)].add(event_arm)
        if cohort_arm:
            cohort_arms[(row.network, row.subjectid)].add(cohort_arm)

    resolved: dict[tuple[str, str], str] = {}
    keys = set(event_arms).union(cohort_arms)
    for key in sorted(keys):
        explicit = event_arms.get(key, set())
        cohort = cohort_arms.get(key, set())
        if len(explicit) == 1:
            resolved[key] = next(iter(explicit))
        elif not explicit and len(cohort) == 1:
            resolved[key] = next(iter(cohort))

    for row in rows:
        arm = resolved.get((row.network, row.subjectid), "")
        if not row.event_name and row.timepoint and arm:
            row.event_name = _event_name_for_timepoint(row.timepoint, arm)


def _semantic_value_key(value: Any) -> tuple[str, Any]:
    """Return an exact semantic key for duplicate-row reconciliation.

    Decimal equality intentionally collapses representation-only differences
    such as ``1``, ``1.0`` and ``1e0``.  Unlike mismatch comparison, duplicate
    identity reconciliation does not use a tolerance: tolerance-based grouping
    is not transitive and could therefore make the winner depend on row order.
    """
    kind, normalized = _normalized_value(value)
    if kind == "number" and normalized == 0:
        normalized = Decimal(0)
    return kind, normalized


def _values_semantically_equal(left: Any, right: Any) -> bool:
    return _semantic_value_key(left) == _semantic_value_key(right)


def _merge_duplicate_identity_rows(
        rows: Sequence[LoadedRow],
        ) -> tuple[LoadedRow, set[str], int]:
    """Merge all non-conflicting cells for one logical REDCap row.

    A field absent from one file's schema is treated as unobserved, not blank.
    A field present but empty is an observed blank.  This distinction lets a
    canonical and a generic export safely contribute complementary columns.
    Only genuinely conflicting fields are quarantined; unrelated fields at
    the same event remain usable by cross-event calculations.
    """
    ordered = sorted(rows, key=lambda item: (item.source_file, item.source_row))
    winner = ordered[0]
    merged_values = dict(winner.values)
    all_value_keys = set().union(*(row.values for row in ordered))
    conflicts: set[str] = set()
    distinct_row_signatures: set[tuple[Any, ...]] = set()

    for row in ordered:
        signature: list[tuple[str, tuple[str, Any]]] = []
        for key in sorted(all_value_keys):
            if key in row.available_columns:
                signature.append((key, _semantic_value_key(
                    row.values.get(key, ""))))
        distinct_row_signatures.add(tuple(signature))

    for key in sorted(all_value_keys):
        observations = [
            (row, row.values.get(key, ""))
            for row in ordered if key in row.available_columns
        ]
        if not observations:
            continue
        first_value = observations[0][1]
        if any(
                not _values_semantically_equal(first_value, value)
                for _, value in observations[1:]):
            conflicts.add(key)
            merged_values.pop(key, None)
            continue

        # Preserve the deterministic winner's representation when observed;
        # otherwise take the first observed representation. Blank values stay
        # omitted, matching the normal row-loading representation.
        selected = next((
            value for candidate, value in observations
            if candidate is winner), first_value)
        if _is_real_missing(selected):
            merged_values.pop(key, None)
        else:
            merged_values[key] = selected

    winner.values = merged_values
    winner.available_columns = frozenset().union(*(
        candidate.available_columns for candidate in ordered))
    return winner, conflicts, len(distinct_row_signatures)


def _canonical_repeat_identity(
        instrument: Any, instance: Any, *,
        source_file: Path, source_row: int) -> tuple[str, str]:
    repeat_instrument = str(instrument or "").strip()
    repeat_instance = str(instance or "").strip()
    if bool(repeat_instrument) != bool(repeat_instance):
        raise ScannerInputError(
            "Repeating-row identity requires both instrument and instance at "
            f"CSV row {source_row} in {source_file}")
    if not repeat_instance:
        return "", ""
    try:
        number = Decimal(repeat_instance)
    except (InvalidOperation, ValueError) as exc:
        raise ScannerInputError(
            f"Invalid repeat instance {repeat_instance!r} at CSV row "
            f"{source_row} in {source_file}") from exc
    if (
        not number.is_finite()
        or number != number.to_integral_value()
        or number < 1
    ):
        raise ScannerInputError(
            f"Invalid repeat instance {repeat_instance!r} at CSV row "
            f"{source_row} in {source_file}; expected a positive integer")
    return repeat_instrument, str(int(number))


def load_rows(
        specs: Sequence[InputSpec], relevant_columns: set[str],
        chunk_size: int) -> LoadedInputs:
    """Read relevant columns and return deterministic, deduplicated rows."""
    if chunk_size < 1:
        raise ScannerInputError("--chunk-size must be a positive integer")

    loaded: list[LoadedRow] = []
    input_details: list[dict[str, Any]] = []
    schemas: dict[str, frozenset[str]] = {}
    for spec in specs:
        header = _read_header(spec.path)
        schema = frozenset(header)
        schemas[str(spec.path)] = schema
        selected = [column for column in header if column in relevant_columns]
        if "subjectid" not in selected:
            # Defensive; _read_header already guarantees it is in the header.
            selected.insert(0, "subjectid")
        row_count = 0
        actual_networks: Counter[str] = Counter()
        actual_timepoints: Counter[str] = Counter()
        try:
            chunks = pd.read_csv(
                spec.path, usecols=selected, dtype=str,
                keep_default_na=False, encoding="utf-8-sig",
                on_bad_lines="error", chunksize=chunk_size,
            )
            for chunk in chunks:
                columns = list(chunk.columns)
                for values_tuple in chunk.itertuples(index=False, name=None):
                    row_count += 1
                    values = {
                        column: value
                        for column, value in zip(columns, values_tuple)
                        if not _is_real_missing(value)
                    }
                    subjectid = str(values.get("subjectid", "")).strip()
                    if not subjectid:
                        raise ScannerInputError(
                            f"Blank subjectid at CSV row {row_count + 1} "
                            f"in {spec.path}")
                    event_name = _row_event_name(values, spec)
                    raw_network = str(values.get("network", "")).strip()
                    row_network = _normalize_network(
                        raw_network or spec.network)
                    raw_timepoint = str(values.get("timepoint", "")).strip()
                    row_timepoint = _normalize_timepoint(
                        raw_timepoint or spec.timepoint
                        or _timepoint_from_event(event_name))
                    actual_networks[row_network] += 1
                    actual_timepoints[row_timepoint] += 1
                    if spec.canonical:
                        if raw_network and row_network != spec.network:
                            raise ScannerInputError(
                                "Row network metadata conflicts with canonical "
                                f"filename at CSV row {row_count + 1} in "
                                f"{spec.path}")
                        if raw_timepoint and row_timepoint != spec.timepoint:
                            raise ScannerInputError(
                                "Row timepoint metadata conflicts with canonical "
                                f"filename at CSV row {row_count + 1} in "
                                f"{spec.path}")
                        if (
                            event_name
                            and _timepoint_from_event(event_name) != spec.timepoint
                        ):
                            raise ScannerInputError(
                                "Row event metadata conflicts with canonical "
                                f"filename at CSV row {row_count + 1} in "
                                f"{spec.path}")
                    repeat_instrument, repeat_instance = (
                        _canonical_repeat_identity(
                            values.get("redcap_repeat_instrument", ""),
                            values.get("redcap_repeat_instance", ""),
                            source_file=spec.path,
                            source_row=row_count + 1,
                        ))
                    loaded.append(LoadedRow(
                        network=row_network,
                        timepoint=row_timepoint,
                        subjectid=subjectid,
                        event_name=event_name,
                        repeat_instrument=repeat_instrument,
                        repeat_instance=repeat_instance,
                        values=values,
                        available_columns=schema,
                        source_file=str(spec.path),
                        source_row=row_count + 1,
                    ))
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            if isinstance(exc, ScannerInputError):
                raise
            raise ScannerInputError(f"Could not parse input CSV {spec.path}: {exc}") from exc
        input_details.append({
            "source_file": str(spec.path),
            "network": spec.network,
            "timepoint": spec.timepoint,
            "row_count": row_count,
            "column_count": len(header),
            "calculated_columns_present": 0,
            "actual_network_row_counts": dict(sorted(actual_networks.items())),
            "actual_timepoint_row_counts": dict(sorted(
                actual_timepoints.items())),
        })

    _propagate_subject_arms(loaded)

    event_schema_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    timepoint_schema_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    network_schema_sets: dict[str, set[str]] = defaultdict(set)
    for row in loaded:
        network_schema_sets[row.network].update(row.available_columns)
        if row.timepoint:
            timepoint_schema_sets[(row.network, row.timepoint)].update(
                row.available_columns)
        if row.event_name:
            event_schema_sets[(row.network, row.event_name)].update(
                row.available_columns)

    by_identity: dict[
        tuple[str, str, str, str, str], list[LoadedRow]] = defaultdict(list)
    for row in loaded:
        by_identity[row.identity].append(row)

    retained: list[LoadedRow] = []
    duplicate_diagnostics: list[dict[str, Any]] = []
    quarantined: set[tuple[str, str, str, str, str]] = set()
    quarantined_values: set[
        tuple[str, str, str, str, str, str]] = set()
    for identity, rows in sorted(by_identity.items()):
        if len(rows) == 1:
            retained.append(rows[0])
            continue
        winner, conflicts, distinct_count = _merge_duplicate_identity_rows(rows)
        retained.append(winner)
        if conflicts:
            quarantined.add(identity)
            quarantined_values.update(
                (*identity, variable) for variable in conflicts)
            duplicate_diagnostics.append({
                "type": "conflicting_duplicate_identity",
                "network": identity[0],
                "subjectid": identity[1],
                "event_name": identity[2],
                "repeat_instrument": identity[3],
                "repeat_instance": identity[4],
                "row_count": len(rows),
                "distinct_row_count": distinct_count,
                "conflicting_variables": sorted(conflicts),
            })
        else:
            duplicate_diagnostics.append({
                "type": "semantically_equivalent_rows_collapsed",
                "network": identity[0],
                "subjectid": identity[1],
                "event_name": identity[2],
                "repeat_instrument": identity[3],
                "repeat_instance": identity[4],
                "row_count": len(rows),
            })
    retained.sort(key=_loaded_row_sort_key)
    return LoadedInputs(
        rows=retained,
        input_details=input_details,
        file_schemas=schemas,
        event_schemas={
            key: frozenset(value)
            for key, value in sorted(event_schema_sets.items())
        },
        timepoint_schemas={
            key: frozenset(value)
            for key, value in sorted(timepoint_schema_sets.items())
        },
        network_schemas={
            key: frozenset(value)
            for key, value in sorted(network_schema_sets.items())
        },
        quarantined_identities=frozenset(quarantined),
        quarantined_values=frozenset(quarantined_values),
        row_identity_diagnostics=duplicate_diagnostics,
    )


def _loaded_row_sort_key(row: LoadedRow) -> tuple[Any, ...]:
    return (
        NETWORK_ORDER.get(row.network, 99),
        TIMEPOINT_ORDER.get(row.timepoint, 99),
        row.subjectid, row.event_name,
        row.repeat_instrument, _repeat_sort_key(row.repeat_instance),
        row.source_file, row.source_row,
    )


def _repeat_sort_key(value: str) -> tuple[int, Any]:
    try:
        number = Decimal(str(value))
        if number.is_finite():
            return 0, number
        return 1, str(value)
    except (InvalidOperation, ValueError):
        return 1, str(value)


def _event_arm(event_name: str) -> str:
    match = re.search(r"_arm_(\d+)$", str(event_name), flags=re.I)
    return match.group(1) if match else ""


def _instance_keys(value: str) -> tuple[Any, ...]:
    if value == "":
        return ()
    keys: list[Any] = [value]
    try:
        number = Decimal(value)
        if number == number.to_integral_value():
            integer = int(number)
            keys.extend((integer, str(integer)))
    except (InvalidOperation, ValueError):
        pass
    return tuple(dict.fromkeys(keys))


def build_stored_snapshot(
        rows: Sequence[LoadedRow]) -> tuple[
            dict[tuple[str, str], dict[Any, Any]],
            dict[tuple[str, str, str], dict[Any, Any]],
            dict[tuple[str, str], frozenset[str]]]:
    """Index immutable imported values by network, subject, and event."""
    by_subject: dict[tuple[str, str], dict[Any, Any]] = defaultdict(dict)
    current_instance_values: dict[
        tuple[str, str, str], dict[Any, Any]] = defaultdict(dict)
    available_events: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        subject_key = (row.network, row.subjectid)
        if row.event_name:
            available_events[subject_key].add(row.event_name)
        if not row.is_repeating:
            if row.event_name:
                by_subject[subject_key][row.event_name] = row.values
            continue

        for instance in _instance_keys(row.repeat_instance):
            for variable, value in row.values.items():
                if row.event_name:
                    by_subject[subject_key][
                        (row.event_name, variable, instance)] = value
                    by_subject[subject_key][
                        (row.event_name, instance, variable)] = value
                current_instance_values[
                    (row.network, row.subjectid, row.event_name)][
                        (variable, instance)] = value
                current_instance_values[
                    (row.network, row.subjectid, row.event_name)][
                        (instance, variable)] = value

    return (
        dict(by_subject),
        dict(current_instance_values),
        {key: frozenset(value) for key, value in available_events.items()},
    )


def _schema_for_explicit_event(
        row: LoadedRow, event_name: str,
        loaded: LoadedInputs) -> frozenset[str] | None:
    schema = loaded.event_schemas.get((row.network, event_name))
    if schema is not None:
        return schema
    return loaded.timepoint_schemas.get((
        row.network, _timepoint_from_event(event_name)))


def _schedule_supports_structural_blank(
        row: LoadedRow, event_name: str, source_form: str,
        forms_per_timepoint: Mapping[
            str, Mapping[str, Sequence[str]]] | None,
        ) -> bool:
    """Whether the applicability schedule explicitly excludes a source form.

    A column appearing somewhere in a network is not, by itself, evidence that
    its absence at this event represents a REDCap blank. Export-schema drift
    looks identical. We therefore classify a missing column as structural only
    when the configured cohort/timepoint schedule affirmatively excludes its
    source instrument; otherwise the calculation fails closed as unevaluable.
    """
    form = str(source_form or "").strip()
    if not form or not forms_per_timepoint:
        return False
    effective_event = event_name or row.event_name
    arm = _event_arm(effective_event)
    cohort = "chr" if arm == "1" else "hc" if arm == "2" else ""
    timepoint = (
        _timepoint_from_event(effective_event)
        if effective_event else row.timepoint)
    if not cohort or not timepoint:
        return False
    scheduled = forms_per_timepoint.get(cohort, {}).get(timepoint)
    if not isinstance(scheduled, Sequence) or isinstance(scheduled, str):
        return False
    scheduled_names = {str(name).strip().casefold() for name in scheduled}
    return form.casefold() not in scheduled_names


def _reference_value_is_quarantined(
        row: LoadedRow, event_name: str | None, runtime_key: str,
        instance: Any, source_form: str, loaded: LoadedInputs) -> bool:
    """Check field-level duplicate conflicts without poisoning an event.

    Non-instance explicit references resolve from the event's nonrepeating
    row. Instance references may resolve from any repeat instrument, so only
    conflicts for that variable and instance are considered blocking.
    """
    if event_name is None:
        current_conflict = (
            *row.identity, runtime_key) in loaded.quarantined_values
        if not row.is_repeating or source_form == row.repeat_instrument:
            return current_conflict
        base_conflict = (
            row.network, row.subjectid, row.event_name, "", "", runtime_key
        ) in loaded.quarantined_values
        return current_conflict or base_conflict
    event = str(event_name)
    if instance is None:
        return (
            row.network, row.subjectid, event, "", "", runtime_key
        ) in loaded.quarantined_values
    instance_keys = {str(value) for value in _instance_keys(str(instance))}
    return any(
        network == row.network
        and subjectid == row.subjectid
        and candidate_event == event
        and repeat_instance in instance_keys
        and candidate_variable == runtime_key
        for (
            network, subjectid, candidate_event, _repeat_instrument,
            repeat_instance, candidate_variable,
        ) in loaded.quarantined_values
    )


def _validate_formula_references(
        row: LoadedRow, variable: str, entry: Mapping[str, Any],
        loaded: LoadedInputs, artifacts: ScanArtifacts,
        subject_events: frozenset[str],
        forms_per_timepoint: Mapping[
            str, Mapping[str, Sequence[str]]] | None = None,
        same_event_schema: frozenset[str] | None = None,
        ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return blocking structural problems and legitimate structural blanks."""
    blocking: list[str] = []
    structural_blanks: list[str] = []
    for reference in entry["references"]:
        runtime_key = _runtime_reference_key(reference)
        event_name = reference.get("event")
        instance = reference.get("instance")
        display = str(reference.get("display") or runtime_key)
        source_form = str(reference.get("source_form", "")).strip()

        if _reference_value_is_quarantined(
                row, event_name, runtime_key, instance, source_form, loaded):
            blocking.append(display)
            diagnostic_key = (
                "quarantined_reference_value", row.network, row.subjectid,
                str(event_name or row.event_name), variable, runtime_key,
                str(instance or ""),
            )
            artifacts.add_diagnostic(diagnostic_key, {
                "type": "duplicate_conflict_for_dependency",
                "network": row.network,
                "subjectid": row.subjectid,
                "event_name": str(event_name or row.event_name),
                "target_variable": variable,
                "source_variable": runtime_key,
                "repeat_instance": str(instance or ""),
            })
            continue

        if event_name is None:
            effective_schema = same_event_schema or row.available_columns
            if runtime_key not in effective_schema:
                if (
                    runtime_key in loaded.network_schemas.get(
                        row.network, frozenset())
                    and _schedule_supports_structural_blank(
                        row, row.event_name, source_form,
                        forms_per_timepoint)
                ):
                    structural_blanks.append(display)
                else:
                    blocking.append(display)
                    diagnostic_key = (
                    "missing_current_column", row.source_file,
                        variable, runtime_key,
                    )
                    artifacts.add_diagnostic(diagnostic_key, {
                        "type": "missing_dependency_column",
                        "target_variable": variable,
                        "missing_column": runtime_key,
                        "source_file": row.source_file,
                    })
            if instance is not None and (
                    "redcap_repeat_instance" not in row.available_columns):
                blocking.append(display)
                diagnostic_key = (
                    "missing_repeat_metadata", row.source_file, variable,
                )
                artifacts.add_diagnostic(diagnostic_key, {
                    "type": "missing_repeat_metadata",
                    "target_variable": variable,
                    "source_file": row.source_file,
                })
            continue

        event_name = str(event_name)
        current_arm = _event_arm(row.event_name)
        reference_arm = _event_arm(event_name)
        if (
            reference_arm and not current_arm
            and event_name not in subject_events
        ):
            blocking.append(display)
            diagnostic_key = (
                "unknown_subject_arm", row.network, row.subjectid, variable,
            )
            artifacts.add_diagnostic(diagnostic_key, {
                "type": "unknown_subject_arm_for_cross_event_calculation",
                "network": row.network,
                "subjectid": row.subjectid,
                "target_variable": variable,
            })
            continue
        if (
            current_arm and reference_arm and current_arm != reference_arm
            and event_name not in subject_events
        ):
            # Cohort-specific formulas commonly include an explicit reference
            # to the other arm. It is a REDCap blank only when this subject has
            # no such event. Mixed-arm histories are valid in the source data;
            # when the exact event exists, continue and resolve its real value.
            structural_blanks.append(display)
            continue

        schema = _schema_for_explicit_event(row, event_name, loaded)
        if schema is None:
            blocking.append(display)
            diagnostic_key = (
                "missing_event_schema", row.network, event_name, variable,
            )
            artifacts.add_diagnostic(diagnostic_key, {
                "type": "missing_cross_event_input",
                "network": row.network,
                "event_name": event_name,
                "target_variable": variable,
                "source_variable": runtime_key,
            })
            continue
        if runtime_key not in schema:
            if (
                runtime_key in loaded.network_schemas.get(
                    row.network, frozenset())
                and _schedule_supports_structural_blank(
                    row, event_name, source_form, forms_per_timepoint)
            ):
                structural_blanks.append(display)
            else:
                blocking.append(display)
                diagnostic_key = (
                    "missing_event_column", row.network, event_name,
                    variable, runtime_key,
                )
                artifacts.add_diagnostic(diagnostic_key, {
                    "type": "missing_cross_event_dependency_column",
                    "network": row.network,
                    "event_name": event_name,
                    "target_variable": variable,
                    "missing_column": runtime_key,
                })
            continue
        if instance is not None and "redcap_repeat_instance" not in schema:
            blocking.append(display)
            diagnostic_key = (
                "missing_cross_event_repeat_metadata", row.network,
                event_name, variable,
            )
            artifacts.add_diagnostic(diagnostic_key, {
                "type": "missing_cross_event_repeat_metadata",
                "network": row.network,
                "event_name": event_name,
                "target_variable": variable,
            })
            continue
        if event_name not in subject_events:
            # The event/column exists in the supplied export set but this
            # subject has no row at that event. REDCap resolves that as blank.
            structural_blanks.append(display)

    return tuple(dict.fromkeys(blocking)), tuple(dict.fromkeys(structural_blanks))


def _target_form_is_definitively_inactive(
        row: LoadedRow, variable: str, entry: Mapping[str, Any],
        form_info: Mapping[str, Any] | None,
        forms_per_timepoint: Mapping[str, Mapping[str, Sequence[str]]] | None,
        *, include_inactive_forms: bool) -> str:
    """Identify a blank target on a form that REDCap did not instantiate.

    A blank completion field is stronger evidence than a population heuristic:
    it means the target instrument has no saved status at this row/event. Any
    isolated calculated values on that uninstantiated form are therefore not
    treated as an active-form mismatch unless ``--include-inactive-forms`` is
    selected explicitly.
    """
    form = str(entry.get("form", "")).strip()
    if not form:
        return ""

    current_arm = _event_arm(row.event_name)
    cohort = "chr" if current_arm == "1" else "hc" if current_arm == "2" else ""
    if cohort and row.timepoint and forms_per_timepoint:
        cohort_schedule = forms_per_timepoint.get(cohort, {})
        scheduled = cohort_schedule.get(row.timepoint)
        if isinstance(scheduled, Sequence) and not isinstance(scheduled, str):
            if form.casefold() not in {
                    str(name).strip().casefold() for name in scheduled}:
                return "unscheduled_form"
    form_tokens = set(filter(None, re.split(r"[^a-z0-9]+", form.casefold())))
    if ((current_arm == "1" and "hc" in form_tokens)
            or (current_arm == "2" and "chr" in form_tokens)):
        return "opposite_cohort_form"

    info = form_info if isinstance(form_info, Mapping) else {}
    missing_variable = str(info.get("missing_var", "")).strip()
    if missing_variable:
        kind, value = _normalized_value(row.values.get(missing_variable, ""))
        if kind == "number" and value == 1:
            return "explicit_missing_button"
    if row.network == "PRESCIENT":
        rpms_completion = _prescient_completion_variable(info)
        if rpms_completion:
            kind, value = _normalized_value(
                row.values.get(rpms_completion, ""))
            if kind == "number" and value in {Decimal(3), Decimal(4)}:
                return "prescient_missing_completion_code"

    if include_inactive_forms:
        return ""
    completion_field = str(info.get(
        "completion_var", "")).strip() or f"{form}_complete"
    if (
        completion_field in row.available_columns
        and _is_real_missing(row.values.get(completion_field, ""))
    ):
        return "uninstantiated_form"
    return ""


def _semantic_risk(
        entry: Mapping[str, Any], stored: Any, recalculated: Any, *,
        comparison_mode: str,
        extra_risks: Sequence[str] = ()) -> str:
    risks: list[str] = []
    if "datediff" in entry.get("functions", ()):
        risks.append("datediff_version_sensitive")
    if any(
            reference.get("event") is not None
            and reference.get("kind") == "calc"
            for reference in entry.get("references", ())):
        risks.append("cross_event_calc_uses_stored_snapshot")
    if (
        comparison_mode == "direct"
        and entry.get("same_event_calc_dependencies")
    ):
        # A direct calculation can disagree solely because it consumed a stale
        # stored upstream calculation. Keep the row for review but do not call
        # it a confirmed downstream defect.
        risks.append("same_event_calc_dependency_uses_stored_snapshot")
    stored_kind, stored_value = _normalized_value(stored)
    expected_kind, expected_value = _normalized_value(recalculated)
    if (
        (stored_kind == "blank" and expected_kind == "number"
         and expected_value == 0)
        or (expected_kind == "blank" and stored_kind == "number"
            and stored_value == 0)
    ):
        risks.append("blank_zero_persistence_uncalibrated")
    risks.extend(str(risk) for risk in extra_risks if str(risk))
    return "|".join(dict.fromkeys(risks))


def _result_row(
        row: LoadedRow, variable: str, entry: Mapping[str, Any], *,
        recalculated: Any, status: str, error: str = "",
        structural_blanks: Sequence[str] = (),
        comparison_mode: str = "direct",
        branching_eligibility: str = "visible",
        branching_eligibility_reason: str = "",
        extra_semantic_risks: Sequence[str] = ()) -> dict[str, Any]:
    stored = row.values.get(variable, "")
    semantic_risk = (
        _semantic_risk(
            entry, stored, recalculated,
            comparison_mode=comparison_mode,
            extra_risks=extra_semantic_risks)
        if status == "mismatch" else "")
    if status == "mismatch":
        actionability = (
            "semantic_uncertain"
            if any(risk in semantic_risk.split("|") for risk in (
                "blank_zero_persistence_uncalibrated",
                "datediff_version_sensitive",
                "cross_event_calc_uses_stored_snapshot",
                "same_event_calc_dependency_uses_stored_snapshot",
                "branching_eligibility_not_assessed",
            )) else "actionable"
        )
    elif status == "evaluation_error":
        actionability = "review_evaluation_error"
    else:
        actionability = "not_evaluable"
    stored_blank = _is_real_missing(stored)
    recalculated_blank = _is_real_missing(recalculated)
    mismatch_kind = (
        "evaluation_failure" if status != "mismatch"
        else "stored_blank_expected_nonblank"
        if stored_blank and not recalculated_blank
        else "stored_nonblank_expected_blank"
        if recalculated_blank and not stored_blank
        else "both_nonblank_difference"
    )
    return {
        "network": row.network,
        "timepoint": row.timepoint,
        "subjectid": row.subjectid,
        "redcap_event_name": row.event_name,
        "redcap_repeat_instrument": row.repeat_instrument,
        "redcap_repeat_instance": row.repeat_instance,
        "variable": variable,
        "form": entry.get("form", ""),
        "stored_value": _display_value(stored),
        "recalculated_value": _display_value(recalculated),
        "status": status,
        "error": error,
        "numeric_difference": (
            _numeric_difference(stored, recalculated)
            if status == "mismatch" else ""),
        "structural_blank_dependencies": "|".join(structural_blanks),
        "cross_event_mode": CROSS_EVENT_MODE,
        "comparison_mode": comparison_mode,
        "semantic_risk": semantic_risk,
        "actionability": actionability,
        "branching_eligibility": branching_eligibility,
        "branching_eligibility_reason": branching_eligibility_reason,
        "mismatch_kind": mismatch_kind,
        "source_file": Path(row.source_file).name,
        "source_row": row.source_row,
    }


def scan_calculated_fields(
        loaded: LoadedInputs,
        calculations: Mapping[str, Mapping[str, Any]],
        evaluation_order: Sequence[str], *,
        numeric_tolerance: Decimal,
        blank_result_as_zero: bool,
        comparison_mode: str,
        include_inactive_forms: bool = False,
        important_form_vars: Mapping[str, Mapping[str, Any]] | None = None,
        forms_per_timepoint: Mapping[
            str, Mapping[str, Sequence[str]]] | None = None,
        branching_rules: Mapping[str, Mapping[str, Any]] | None = None,
        branching_rule_diagnostics: Mapping[
            str, Sequence[str]] | None = None,
        mismatch_sink: Callable[[Mapping[str, Any]], None] | None = None,
        ) -> ScanArtifacts:
    """Evaluate each available calculated field without mutating inputs.

    When ``mismatch_sink`` is supplied, rows are emitted in final deterministic
    order as each participant row completes. This keeps production memory
    proportional to one input network plus one participant's results instead
    of retaining hundreds of thousands of output dictionaries.
    """
    if comparison_mode not in {"direct", "propagated"}:
        raise ValueError(f"Unknown comparison mode: {comparison_mode}")

    artifacts = ScanArtifacts()
    subject_data, current_instances, events_by_subject = (
        build_stored_snapshot(loaded.rows))
    converted = {
        variable: entry for variable, entry in calculations.items()
        if entry.get("status") == "converted"
    }
    ordered_variables = [
        variable for variable in evaluation_order if variable in converted]
    ordered_variables.extend(sorted(set(converted).difference(
        ordered_variables)))
    uncertain_branching_reason: dict[str, str] = {}
    for reason, variables in (branching_rule_diagnostics or {}).items():
        for variable in variables:
            uncertain_branching_reason.setdefault(str(variable), str(reason))

    base_rows = {
        (row.network, row.subjectid, row.event_name): row
        for row in loaded.rows if not row.is_repeating and row.event_name
    }
    repeat_registry = {
        (row.network, row.event_name, row.timepoint, row.repeat_instrument)
        for row in loaded.rows if row.is_repeating and row.repeat_instrument
    }
    fields_by_form: dict[str, set[str]] = defaultdict(set)
    for variable, entry in converted.items():
        target_form = str(entry.get("form", "")).strip()
        if target_form:
            fields_by_form[target_form].add(variable)
        for reference in entry.get("references", ()):
            source_form = str(reference.get("source_form", "")).strip()
            if source_form:
                fields_by_form[source_form].add(
                    _runtime_reference_key(reference))

    for row in loaded.rows:
        # Duplicate identities may merge complementary file schemas. Select
        # targets from that merged schema, not merely the winning source file.
        available_targets = tuple(
            variable for variable in ordered_variables
            if variable in row.available_columns)
        if not available_targets:
            continue

        base_row = base_rows.get((
            row.network, row.subjectid, row.event_name))
        base_values: Mapping[str, Any] = (
            base_row.values
            if row.is_repeating and base_row is not None else {})
        if row.is_repeating:
            imported_repeat_values = dict(base_values)
            # Empty cells were omitted while loading. Materialize blanks for
            # fields belonging to this repeat instrument before overlaying its
            # saved values, while retaining nonrepeat values from the base row.
            imported_repeat_values.update({
                field_name: ""
                for field_name in fields_by_form.get(
                    row.repeat_instrument, set())
            })
            imported_repeat_values.update(row.values)
            imported_current = imported_repeat_values
        else:
            imported_current = row.values
        propagated_current = dict(imported_current)
        same_event_schema = row.available_columns
        if row.is_repeating and base_row is not None:
            same_event_schema = frozenset(
                set(row.available_columns).union(base_row.available_columns))
        unavailable: set[str] = set()
        row_results: list[dict[str, Any]] = []
        subject_key = (row.network, row.subjectid)
        event_data: Mapping[Any, Any] = ChainMap(
            current_instances.get(
                (row.network, row.subjectid, row.event_name), {}),
            subject_data.get(subject_key, {}),
        )
        subject_events = events_by_subject.get(subject_key, frozenset())

        for variable in available_targets:
            entry = converted[variable]
            form = str(entry.get("form", "")).strip()
            if (
                row.is_repeating
                and form != row.repeat_instrument
            ):
                artifacts.counters["other_repeat_instrument_targets_skipped"] += 1
                continue
            if (
                not row.is_repeating
                and form
                and (
                    row.network, row.event_name, row.timepoint, form
                ) in repeat_registry
            ):
                artifacts.counters[
                    "base_row_repeat_instrument_targets_skipped"] += 1
                continue
            artifacts.counters["evaluations_considered"] += 1
            if (*row.identity, variable) in loaded.quarantined_values:
                artifacts.counters["conflicting_target_values_skipped"] += 1
                artifacts.add_diagnostic(
                    ("quarantined_target_value", *row.identity, variable), {
                        "type": "duplicate_conflict_for_target",
                        "network": row.network,
                        "subjectid": row.subjectid,
                        "event_name": row.event_name,
                        "repeat_instrument": row.repeat_instrument,
                        "repeat_instance": row.repeat_instance,
                        "target_variable": variable,
                    })
                unavailable.add(variable)
                continue
            inactive_reason = _target_form_is_definitively_inactive(
                row, variable, entry,
                (important_form_vars or {}).get(
                    str(entry.get("form", "")).strip(), {}),
                forms_per_timepoint,
                include_inactive_forms=include_inactive_forms,
            )
            if inactive_reason:
                artifacts.counters[
                    f"inactive_target_skipped__{inactive_reason}"] += 1
                # A definitively inactive same-event dependency contributes a
                # legitimate blank. Leave the imported blank in propagated
                # state instead of blocking every downstream calculation.
                continue
            branching_eligibility = "visible"
            branching_reason = ""
            branching_rule = (branching_rules or {}).get(variable)
            if branching_rule is not None:
                eligibility_blocking, _ = _validate_formula_references(
                    row, variable, branching_rule, loaded, artifacts,
                    subject_events, forms_per_timepoint,
                    same_event_schema=same_event_schema)
                if eligibility_blocking:
                    artifacts.counters[
                        "branching_eligibility_not_evaluable"] += 1
                    branching_eligibility = "unknown"
                    branching_reason = "missing_branching_dependency"
                else:
                    eligibility = evaluate_converted_calculation(
                        str(branching_rule["converted_calculation"]),
                        imported_current,
                        event_data,
                        blank_result_as_zero=False,
                    )
                    if eligibility.status != "ok":
                        artifacts.counters[
                            "branching_eligibility_errors"] += 1
                        branching_eligibility = "unknown"
                        branching_reason = "branching_evaluation_error"
                    elif not redcap_truthy(eligibility.value):
                        artifacts.counters[
                            "inactive_target_skipped__branching_logic_false"
                        ] += 1
                        continue
                    else:
                        artifacts.counters[
                            "branching_eligibility_visible"] += 1
            elif str(entry.get("branching_logic", "")).strip():
                # Unsupported and calc-dependent rules cannot safely suppress
                # an audit. Evaluate the calculation, but preserve this third
                # eligibility state on every resulting row.
                branching_eligibility = "unknown"
                branching_reason = uncertain_branching_reason.get(
                    variable, "branching_rule_not_safely_translated")
                artifacts.counters[
                    "branching_eligibility_not_assessed"] += 1

            extra_risks = (
                ("branching_eligibility_not_assessed",)
                if branching_eligibility == "unknown" else ())
            blocking, structural_blanks = _validate_formula_references(
                row, variable, entry, loaded, artifacts, subject_events,
                forms_per_timepoint,
                same_event_schema=same_event_schema)
            if blocking:
                artifacts.counters["structurally_unevaluable"] += 1
                unavailable.add(variable)
                continue

            if comparison_mode == "propagated":
                blocked_by = sorted(set(entry.get(
                    "same_event_calc_dependencies", ())).intersection(
                        unavailable))
                if blocked_by:
                    unavailable.add(variable)
                    artifacts.counters["blocked_dependencies"] += 1
                    row_results.append(_result_row(
                        row, variable, entry, recalculated="",
                        status="blocked_dependency",
                        error=("Same-event recalculation blocked by: "
                               + ", ".join(blocked_by)),
                        structural_blanks=structural_blanks,
                        comparison_mode=comparison_mode,
                        branching_eligibility=branching_eligibility,
                        branching_eligibility_reason=branching_reason,
                        extra_semantic_risks=extra_risks,
                    ))
                    continue
                current_values: Mapping[str, Any] = propagated_current
            else:
                current_values = imported_current

            result = evaluate_converted_calculation(
                str(entry["converted_calculation"]),
                current_values,
                event_data,
                blank_result_as_zero=blank_result_as_zero,
            )
            if result.status != "ok":
                unavailable.add(variable)
                artifacts.counters["evaluation_errors"] += 1
                row_results.append(_result_row(
                    row, variable, entry, recalculated="",
                    status="evaluation_error", error=result.error,
                    structural_blanks=structural_blanks,
                    comparison_mode=comparison_mode,
                    branching_eligibility=branching_eligibility,
                    branching_eligibility_reason=branching_reason,
                    extra_semantic_risks=extra_risks,
                ))
                continue

            if comparison_mode == "propagated":
                propagated_current[variable] = result.value
            artifacts.counters["evaluated"] += 1
            stored = imported_current.get(variable, "")
            if values_match(
                    stored, result.value,
                    numeric_tolerance=numeric_tolerance):
                artifacts.counters["matches"] += 1
                continue

            artifacts.counters["mismatches"] += 1
            row_results.append(_result_row(
                row, variable, entry, recalculated=result.value,
                status="mismatch",
                structural_blanks=structural_blanks,
                comparison_mode=comparison_mode,
                branching_eligibility=branching_eligibility,
                branching_eligibility_reason=branching_reason,
                extra_semantic_risks=extra_risks,
            ))

        row_results.sort(key=_mismatch_sort_key)
        if mismatch_sink is None:
            artifacts.mismatches.extend(row_results)
        else:
            for mismatch in row_results:
                mismatch_sink(mismatch)

    artifacts.mismatches.sort(key=_mismatch_sort_key)
    return artifacts


def _mismatch_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        NETWORK_ORDER.get(str(item["network"]), 99),
        TIMEPOINT_ORDER.get(str(item["timepoint"]), 99),
        str(item["subjectid"]), str(item["redcap_event_name"]),
        str(item["redcap_repeat_instrument"]),
        _repeat_sort_key(str(item["redcap_repeat_instance"])),
        str(item["variable"]), str(item["status"]),
        str(item["source_file"]), int(item["source_row"]),
    )


def _merge_schema_maps(
        maps: Iterable[Mapping[Any, frozenset[str]]]) -> dict[Any, frozenset[str]]:
    combined: dict[Any, set[str]] = defaultdict(set)
    for mapping in maps:
        for key, values in mapping.items():
            combined[key].update(values)
    return {
        key: frozenset(values)
        for key, values in sorted(combined.items(), key=lambda item: str(item[0]))
    }


def merge_loaded_metadata(parts: Sequence[LoadedInputs]) -> LoadedInputs:
    """Merge diagnostics metadata without retaining participant row values."""
    return LoadedInputs(
        rows=[],
        input_details=[
            detail for part in parts for detail in part.input_details],
        file_schemas={
            key: value for part in parts
            for key, value in part.file_schemas.items()},
        event_schemas=_merge_schema_maps(
            part.event_schemas for part in parts),
        timepoint_schemas=_merge_schema_maps(
            part.timepoint_schemas for part in parts),
        network_schemas=_merge_schema_maps(
            part.network_schemas for part in parts),
        quarantined_identities=frozenset(
            identity for part in parts
            for identity in part.quarantined_identities),
        quarantined_values=frozenset(
            identity for part in parts
            for identity in part.quarantined_values),
        row_identity_diagnostics=sorted(
            (diagnostic for part in parts
             for diagnostic in part.row_identity_diagnostics),
            key=lambda item: json.dumps(
                item, sort_keys=True, ensure_ascii=False,
                separators=(",", ":")),
        ),
    )


def merge_scan_metadata(parts: Sequence[ScanArtifacts]) -> ScanArtifacts:
    """Merge counters/diagnostics after mismatch rows have been spooled."""
    merged = ScanArtifacts()
    for part in parts:
        merged.counters.update(part.counters)
        for key, diagnostic in part.structural_diagnostics.items():
            merged.add_diagnostic(key, diagnostic)
    return merged


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_stage(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    # A pandas DataFrame duplicates the entire mismatch list in memory and can
    # make a large but otherwise valid audit fail during the final write. Rows
    # are already deterministically sorted, so stream them directly instead.
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=OUTPUT_COLUMNS,
            extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json_stage(value: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(
            value, output, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )
        output.write("\n")


def _write_artifacts_transactionally(
        rows: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any],
        output_path: Path, diagnostics_path: Path) -> None:
    destinations = (output_path, diagnostics_path)
    if output_path == diagnostics_path:
        raise ScannerInputError("CSV output and diagnostics paths must differ")

    staged = {
        destination: destination.with_name(
            f".{destination.name}.{os.getpid()}.stage")
        for destination in destinations
    }
    backups = {
        destination: destination.with_name(
            f".{destination.name}.{os.getpid()}.backup")
        for destination in destinations
    }
    touched: list[Path] = []
    committed = False
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
        _write_csv_stage(rows, staged[output_path])
        _write_json_stage(diagnostics, staged[diagnostics_path])
        for destination in destinations:
            backup = backups[destination]
            if destination.exists():
                os.replace(destination, backup)
            touched.append(destination)
            os.replace(staged[destination], destination)
        committed = True
    except Exception:
        for destination in reversed(touched):
            backup = backups[destination]
            try:
                destination.unlink(missing_ok=True)
                if backup.exists():
                    os.replace(backup, destination)
            except OSError:
                pass
        raise
    finally:
        cleanup = list(staged.values())
        if committed:
            cleanup.extend(backups.values())
        for path in cleanup:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _write_spooled_artifacts_transactionally(
        spool_paths: Sequence[Path], diagnostics: Mapping[str, Any],
        output_path: Path, diagnostics_path: Path) -> None:
    """Commit pre-sorted network CSV spools as one deterministic artifact."""
    destinations = (output_path, diagnostics_path)
    if output_path == diagnostics_path:
        raise ScannerInputError("CSV output and diagnostics paths must differ")
    staged = {
        destination: destination.with_name(
            f".{destination.name}.{os.getpid()}.stage")
        for destination in destinations
    }
    backups = {
        destination: destination.with_name(
            f".{destination.name}.{os.getpid()}.backup")
        for destination in destinations
    }
    touched: list[Path] = []
    committed = False
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
        with staged[output_path].open("wb") as combined:
            for index, spool_path in enumerate(spool_paths):
                with spool_path.open("rb") as source:
                    if index:
                        source.readline()  # discard subsequent CSV header
                    shutil.copyfileobj(source, combined, length=1024 * 1024)
        _write_json_stage(diagnostics, staged[diagnostics_path])
        for destination in destinations:
            backup = backups[destination]
            if destination.exists():
                os.replace(destination, backup)
            touched.append(destination)
            os.replace(staged[destination], destination)
        committed = True
    except Exception:
        for destination in reversed(touched):
            backup = backups[destination]
            try:
                destination.unlink(missing_ok=True)
                if backup.exists():
                    os.replace(backup, destination)
            except OSError:
                pass
        raise
    finally:
        cleanup = list(staged.values())
        if committed:
            cleanup.extend(backups.values())
        for path in cleanup:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _translation_diagnostics(
        calculations: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variable, entry in sorted(calculations.items()):
        if entry.get("status") == "converted":
            continue
        rows.append({
            "variable": variable,
            "form": entry.get("form", ""),
            "status": entry.get("status", "invalid"),
            "error_code": entry.get("error_code", ""),
            "error": entry.get("error", ""),
        })
    return rows


def build_diagnostics(
        *, dictionary_path: Path, loaded: LoadedInputs,
        calculations: Mapping[str, Mapping[str, Any]],
        artifacts: ScanArtifacts, numeric_tolerance: Decimal,
        blank_result_as_zero: bool, comparison_mode: str,
        include_inactive_forms: bool, chunk_size: int,
        important_form_vars_path: Path | None,
        important_form_vars_count: int,
        forms_per_timepoint_path: Path | None,
        forms_per_timepoint_count: int,
        applicability_maps_required: bool,
        branching_rule_count: int,
        branching_rule_diagnostics: Mapping[str, Sequence[str]],
        retained_row_count: int | None = None,
        output_row_count: int | None = None,
        ) -> dict[str, Any]:
    input_details: list[dict[str, Any]] = []
    calculated_fields = set(calculations)
    for detail in loaded.input_details:
        copied = dict(detail)
        schema = loaded.file_schemas.get(copied["source_file"], frozenset())
        copied["calculated_columns_present"] = len(
            calculated_fields.intersection(schema))
        input_details.append(copied)

    structural_diagnostics = sorted(
        artifacts.structural_diagnostics.values(),
        key=lambda item: json.dumps(
            item, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")),
    )
    converted_fields = {
        variable for variable, entry in calculations.items()
        if entry.get("status") == "converted"
    }
    all_input_columns = set().union(*loaded.file_schemas.values()) if (
        loaded.file_schemas) else set()
    covered_targets = sorted(converted_fields.intersection(all_input_columns))
    uncovered_targets = sorted(converted_fields.difference(all_input_columns))
    calculated_targets_with_branching = sum(
        bool(str(entry.get("branching_logic", "")).strip())
        for entry in calculations.values())
    excluded_branching_count = sum(
        len(variables) for variables in branching_rule_diagnostics.values())
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_data_dictionary": str(dictionary_path),
        "source_data_dictionary_sha256": _sha256_file(dictionary_path),
        "important_form_vars": (
            str(important_form_vars_path)
            if important_form_vars_path is not None else ""),
        "important_form_vars_count": important_form_vars_count,
        "forms_per_timepoint": (
            str(forms_per_timepoint_path)
            if forms_per_timepoint_path is not None else ""),
        "forms_per_timepoint_slot_count": forms_per_timepoint_count,
        "applicability_maps_required": applicability_maps_required,
        "safe_branching_rule_count": branching_rule_count,
        "branching_logic": {
            "calculated_targets_with_rule_count": (
                calculated_targets_with_branching),
            "safe_rule_count": branching_rule_count,
            "excluded_rule_count": excluded_branching_count,
            "eligibility_states": ["visible", "hidden", "unknown"],
            "unknown_result_actionability": "semantic_uncertain",
        },
        "comparison_mode": comparison_mode,
        "cross_event_mode": CROSS_EVENT_MODE,
        "branching_logic_mode": "safe_noncalculated_tri_state_eligibility",
        "inactive_form_mode": (
            "include_all" if include_inactive_forms
            else "skip_blank_target_when_form_completion_is_blank"),
        "blank_result_as_zero": blank_result_as_zero,
        "numeric_tolerance": str(numeric_tolerance),
        "chunk_size": chunk_size,
        "mismatch_output_mode": "deterministic_row_streaming",
        "inputs": input_details,
        "summary": {
            "input_file_count": len(input_details),
            "input_row_count": sum(
                int(item["row_count"]) for item in input_details),
            "retained_row_count": (
                len(loaded.rows)
                if retained_row_count is None else retained_row_count),
            "calculated_field_count": len(calculations),
            "converted_calculated_field_count": sum(
                entry.get("status") == "converted"
                for entry in calculations.values()),
            "converted_target_covered_count": len(covered_targets),
            "converted_target_uncovered_count": len(uncovered_targets),
            "translation_issue_count": sum(
                entry.get("status") != "converted"
                for entry in calculations.values()),
            "output_row_count": (
                len(artifacts.mismatches)
                if output_row_count is None else output_row_count),
            **dict(sorted(artifacts.counters.items())),
        },
        "translation_diagnostics": _translation_diagnostics(calculations),
        "branching_rule_diagnostics": dict(branching_rule_diagnostics),
        "converted_target_coverage": {
            "covered_count": len(covered_targets),
            "uncovered_count": len(uncovered_targets),
            "uncovered_variables": [
                {
                    "variable": variable,
                    "form": calculations[variable].get("form", ""),
                }
                for variable in uncovered_targets
            ],
            "by_network": {
                network: {
                    "covered_count": len(converted_fields.intersection(schema)),
                    "uncovered_count": len(converted_fields.difference(schema)),
                }
                for network, schema in sorted(loaded.network_schemas.items())
            },
        },
        "calculation_catalog": [
            {
                "variable": variable,
                "form": entry.get("form", ""),
                "original_calculation": entry.get(
                    "original_calculation", ""),
            }
            for variable, entry in sorted(calculations.items())
        ],
        "structural_diagnostics": structural_diagnostics,
        "runtime_notes": [
            (
                "Canonical input files are evaluated and spooled by network. "
                "When generic explicit files are mixed with canonical files, "
                "they are loaded together for identity/cross-file resolution; "
                "all identities and value lookups remain keyed by actual row "
                "network, and output remains deterministically sorted."
            ),
            (
                "Direct mode evaluates every target against the immutable "
                "imported snapshot. Mismatches that depend on a same-event "
                "calculated field are labeled semantic_uncertain because a "
                "stale upstream stored value can create a downstream result."
                if comparison_mode == "direct" else
                "Propagated mode updates same-event calculated dependencies "
                "in translator order; explicit cross-event values remain an "
                "immutable stored snapshot."
            ),
            (
                "Cross-event calculated-field dependencies are not recursively "
                "materialized because the current study dictionary contains "
                "cross-event dependency cycles."
            ),
            (
                "Branching logic is evaluated as visible, hidden, or unknown. "
                "Only safely translated rules without calculated-field "
                "dependencies may suppress a target. Unknown eligibility is "
                "retained and labeled semantic_uncertain."
            ),
            (
                "Unless --include-inactive-forms is selected, targets are "
                "skipped when their instrument completion field exists and "
                "is blank, indicating an uninstantiated form."
            ),
        ],
    }
    if loaded.row_identity_diagnostics:
        result["row_identity_diagnostics"] = sorted(
            loaded.row_identity_diagnostics,
            key=lambda item: json.dumps(
                item, sort_keys=True, ensure_ascii=False,
                separators=(",", ":")),
        )
    return result


def _parse_tolerance(value: str) -> Decimal:
    try:
        tolerance = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ScannerInputError(
            "--numeric-tolerance must be a finite non-negative number") from exc
    if not tolerance.is_finite() or tolerance < 0:
        raise ScannerInputError(
            "--numeric-tolerance must be a finite non-negative number")
    return tolerance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate REDCap calculated fields across combined CSV files "
            "and write values that do not match the stored export."))
    parser.add_argument(
        "--config",
        help="Pipeline config.json used for default input/output paths.")
    parser.add_argument(
        "--data-dictionary",
        help=(
            "Current REDCap data-dictionary CSV. If omitted, the pipeline "
            "dependencies folder is searched."))
    parser.add_argument(
        "--important-form-vars",
        help=(
            "important_form_vars.json used to exclude forms explicitly "
            "marked missing. Defaults to the dictionary dependency folder."))
    parser.add_argument(
        "--forms-per-timepoint",
        help=(
            "forms_per_timepoint.json used for cohort/timepoint target "
            "applicability. Defaults to the dictionary dependency folder."))
    parser.add_argument(
        "--input", action="append", default=[], metavar="CSV",
        help=(
            "Explicit combined CSV (repeatable; glob patterns accepted). "
            "When supplied, --input-dir is ignored."))
    parser.add_argument(
        "--input-dir", "--combined-dir", dest="input_dir",
        help=(
            "Directory containing canonical AMPSCZ combined CSVs. Discovery "
            "is non-recursive and accepts only the 36 canonical filenames."))
    parser.add_argument(
        "--allow-partial-input-dir", action="store_true",
        help=(
            "Allow fewer than all 36 canonical network/timepoint files when "
            "using --input-dir. Intended for controlled smoke tests only."))
    parser.add_argument(
        "--allow-missing-applicability-maps", action="store_true",
        help=(
            "Allow a production directory run without important_form_vars.json "
            "or forms_per_timepoint.json. Intended only for isolated tests; "
            "explicit --input and --allow-partial-input-dir runs already use "
            "this test-mode behavior."))
    parser.add_argument(
        "--output",
        help="Mismatch CSV (default: configured output/calculated_field_mismatches.csv).")
    parser.add_argument(
        "--diagnostics",
        help="Diagnostics JSON (default: alongside the mismatch CSV).")
    parser.add_argument(
        "--numeric-tolerance", default=str(DEFAULT_NUMERIC_TOLERANCE),
        help="Inclusive finite numeric comparison tolerance (default: %(default)s).")
    parser.add_argument(
        "--blank-result-as-zero", action="store_true",
        help=(
            "Treat an exact blank formula result as persisted numeric zero. "
            "Use only after validating this REDCap-version behavior."))
    parser.add_argument(
        "--comparison-mode", choices=("direct", "propagated"),
        default="direct",
        help=(
            "direct uses the imported snapshot for every formula; propagated "
            "updates same-event calculated dependencies (default: %(default)s)."))
    parser.add_argument(
        "--include-inactive-forms", action="store_true",
        help=(
            "Also evaluate calculated targets whose instrument completion "
            "field is blank. By default these unsaved forms are "
            "skipped to avoid unscheduled-form false flags."))
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="CSV rows read per pandas chunk (default: %(default)s).")
    parser.add_argument(
        "--fail-on-mismatch", action="store_true",
        help=(
            "Return exit status 1 when the output contains an actionable "
            "mismatch; semantic-uncertain rows do not trigger failure."))
    return parser


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tolerance = _parse_tolerance(args.numeric_tolerance)
        if args.chunk_size < 1:
            raise ScannerInputError("--chunk-size must be a positive integer")

        config_path = (
            Path(args.config).expanduser().resolve()
            if args.config else _default_config_path())
        if args.config and not config_path.is_file():
            raise ScannerInputError(
                f"Explicit config JSON does not exist: {config_path}")
        config = _load_config(config_path)

        configured_output = _config_path_value(
            config, config_path, "output_path")
        testing_enabled = str(config.get(
            "testing_enabled", "False")).strip().casefold() == "true"
        if configured_output is not None and testing_enabled:
            configured_output = configured_output / "testing"
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output else
            (configured_output or Path.cwd().resolve())
            / "calculated_field_mismatches.csv"
        )
        diagnostics_path = (
            Path(args.diagnostics).expanduser().resolve()
            if args.diagnostics else output_path.with_name(
                f"{output_path.stem}_diagnostics.json"))
        if output_path == diagnostics_path:
            raise ScannerInputError(
                "CSV output and diagnostics paths must differ")

        try:
            if args.data_dictionary:
                dictionary_path = Path(
                    args.data_dictionary).expanduser().resolve()
            else:
                configured_dictionary = _dictionary_from_config(
                    config, config_path)
                if args.config and configured_dictionary is None:
                    raise ScannerInputError(
                        "The explicit config does not resolve to exactly one "
                        "current data dictionary; supply --data-dictionary or "
                        "fix paths.dependencies_path")
                dictionary_path = (
                    configured_dictionary or _discover_data_dictionary())
        except FileNotFoundError as exc:
            raise ScannerInputError(str(exc)) from exc
        if not dictionary_path.is_file():
            raise ScannerInputError(
                f"Data dictionary does not exist: {dictionary_path}")
        if dictionary_path in {output_path, diagnostics_path}:
            raise ScannerInputError(
                "The data dictionary cannot also be an output path")
        if config_path in {output_path, diagnostics_path}:
            raise ScannerInputError(
                "The config file cannot also be an output path")

        configured_dependencies = _config_path_value(
            config, config_path, "dependencies_path")
        default_form_map = dictionary_path.parent.parent / "important_form_vars.json"
        if (
            not default_form_map.is_file()
            and configured_dependencies is not None
            and not args.data_dictionary
        ):
            default_form_map = configured_dependencies / "important_form_vars.json"
        form_map_candidate = (
            Path(args.important_form_vars).expanduser().resolve()
            if args.important_form_vars else
            default_form_map.resolve()
        )
        default_schedule = dictionary_path.parent.parent / "forms_per_timepoint.json"
        if (
            not default_schedule.is_file()
            and configured_dependencies is not None
            and not args.data_dictionary
        ):
            default_schedule = configured_dependencies / "forms_per_timepoint.json"
        schedule_candidate = (
            Path(args.forms_per_timepoint).expanduser().resolve()
            if args.forms_per_timepoint else
            default_schedule.resolve()
        )

        input_dir = (
            Path(args.input_dir).expanduser().resolve()
            if args.input_dir else _config_path_value(
                config, config_path, "combined_csv_path"))
        specs = discover_inputs(
            args.input, input_dir,
            excluded_paths=(
                output_path, diagnostics_path, dictionary_path, config_path,
                form_map_candidate, schedule_candidate,
            ),
            require_complete_directory=(
                not args.input and not args.allow_partial_input_dir),
        )

        # Complete directory scans are production audits: applicability maps
        # are mandatory because their absence can create large false-positive
        # sets. Explicit inputs and partial-directory scans are deliberate test
        # modes for backward compatibility. The escape hatch is intentionally
        # named and recorded rather than silently weakening a production run.
        require_applicability_maps = (
            not args.input
            and not args.allow_partial_input_dir
            and not args.allow_missing_applicability_maps
        )
        form_map_path = (
            form_map_candidate
            if form_map_candidate.is_file()
            or args.important_form_vars
            or require_applicability_maps else None)
        schedule_path = (
            schedule_candidate
            if schedule_candidate.is_file()
            or args.forms_per_timepoint
            or require_applicability_maps else None)
        important_form_vars = _load_important_form_vars(
            form_map_path,
            required=bool(args.important_form_vars)
            or require_applicability_maps)
        forms_per_timepoint = _load_forms_per_timepoint(
            schedule_path,
            required=bool(args.forms_per_timepoint)
            or require_applicability_maps)

        try:
            dictionary = pd.read_csv(
                dictionary_path, dtype=str, keep_default_na=False,
                encoding="utf-8-sig", on_bad_lines="error")
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            raise ScannerInputError(
                f"Could not parse data dictionary {dictionary_path}: {exc}") from exc

        try:
            transformer = TransformCalculatedFields(dictionary)
            calculations = transformer.convert_all_calculated_fields()
        except CalculationSchemaError as exc:
            raise ScannerInputError(str(exc)) from exc

        branching_rules, branching_rule_diagnostics = (
            build_safe_branching_rules(calculations))

        relevant_columns = _relevant_columns(
            calculations, important_form_vars, branching_rules)
        if any(not spec.canonical for spec in specs):
            # A generic explicit file can contain either network (or both).
            # Loading it separately under the filename-derived UNKNOWN bucket
            # breaks duplicate reconciliation and cross-file lookups with a
            # canonical file. One mixed input group is still network-isolated
            # at every identity and lookup key, and its rows sort by the actual
            # per-row network metadata.
            canonical_networks = {
                spec.network for spec in specs if spec.canonical
            }
            fallback_network = (
                next(iter(canonical_networks))
                if len(canonical_networks) == 1 else "UNKNOWN")
            mixed_specs = [
                spec if spec.canonical or fallback_network == "UNKNOWN"
                else InputSpec(
                    spec.path, fallback_network, spec.timepoint,
                    canonical=False)
                for spec in specs
            ]
            spec_groups = [("ACTUAL_ROW_NETWORKS", mixed_specs)]
        else:
            grouped_specs: dict[str, list[InputSpec]] = defaultdict(list)
            for spec in specs:
                grouped_specs[spec.network].append(spec)
            spec_groups = sorted(
                grouped_specs.items(),
                key=lambda item: NETWORK_ORDER.get(item[0], 99))

        spool_paths: list[Path] = []
        spool_directory: Path | None = None
        loaded_parts: list[LoadedInputs] = []
        artifact_parts: list[ScanArtifacts] = []
        retained_row_count = 0
        total_output_rows = 0
        has_actionable_mismatch = False
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            spool_directory = Path(tempfile.mkdtemp(
                prefix=f".{output_path.name}.{os.getpid()}.",
                suffix=".spool", dir=output_path.parent))
            for index, (_, group_specs) in enumerate(spec_groups):
                loaded_part = load_rows(
                    group_specs, relevant_columns, args.chunk_size)
                spool_path = spool_directory / f"network_{index:02d}.csv"
                # Register the path before opening/writing it so even a partial
                # write or evaluator exception is cleaned by the outer finally.
                spool_paths.append(spool_path)
                group_output_rows = 0
                group_has_actionable = False
                with spool_path.open(
                        "w", encoding="utf-8", newline="") as spool_output:
                    spool_writer = csv.DictWriter(
                        spool_output, fieldnames=OUTPUT_COLUMNS,
                        extrasaction="ignore", lineterminator="\n")
                    spool_writer.writeheader()

                    def emit_mismatch(row: Mapping[str, Any]) -> None:
                        nonlocal group_output_rows, group_has_actionable
                        spool_writer.writerow(row)
                        group_output_rows += 1
                        if row.get("actionability") == "actionable":
                            group_has_actionable = True

                    artifacts_part = scan_calculated_fields(
                        loaded_part, calculations,
                        transformer.evaluation_order,
                        numeric_tolerance=tolerance,
                        blank_result_as_zero=args.blank_result_as_zero,
                        comparison_mode=args.comparison_mode,
                        include_inactive_forms=args.include_inactive_forms,
                        important_form_vars=important_form_vars,
                        forms_per_timepoint=forms_per_timepoint,
                        branching_rules=branching_rules,
                        branching_rule_diagnostics=(
                            branching_rule_diagnostics),
                        mismatch_sink=emit_mismatch,
                    )
                retained_row_count += len(loaded_part.rows)
                total_output_rows += group_output_rows
                has_actionable_mismatch = (
                    has_actionable_mismatch or group_has_actionable)

                # Retain only non-participant diagnostics after this isolated
                # network has been deterministically spooled.
                loaded_part.rows = []
                loaded_parts.append(loaded_part)
                artifact_parts.append(artifacts_part)

            loaded = merge_loaded_metadata(loaded_parts)
            artifacts = merge_scan_metadata(artifact_parts)
            diagnostics = build_diagnostics(
                dictionary_path=dictionary_path,
                loaded=loaded,
                calculations=calculations,
                artifacts=artifacts,
                numeric_tolerance=tolerance,
                blank_result_as_zero=args.blank_result_as_zero,
                comparison_mode=args.comparison_mode,
                include_inactive_forms=args.include_inactive_forms,
                chunk_size=args.chunk_size,
                important_form_vars_path=form_map_path,
                important_form_vars_count=len(important_form_vars),
                forms_per_timepoint_path=schedule_path,
                forms_per_timepoint_count=sum(
                    len(schedule)
                    for schedule in forms_per_timepoint.values()),
                applicability_maps_required=require_applicability_maps,
                branching_rule_count=len(branching_rules),
                branching_rule_diagnostics=branching_rule_diagnostics,
                retained_row_count=retained_row_count,
                output_row_count=total_output_rows,
            )
            _write_spooled_artifacts_transactionally(
                spool_paths, diagnostics, output_path, diagnostics_path)
        finally:
            for spool_path in spool_paths:
                try:
                    spool_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if spool_directory is not None:
                shutil.rmtree(spool_directory, ignore_errors=True)
    except ScannerInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Error writing calculated-field audit: {exc}", file=sys.stderr)
        return 2

    print(
        "Calculated-field audit complete: "
        f"{total_output_rows} output row(s) written to {output_path}")
    if args.fail_on_mismatch and has_actionable_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
