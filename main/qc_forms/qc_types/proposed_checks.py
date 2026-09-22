"""Dictionary-derived QC checks that are not active in the legacy pipeline.

``ProposedChecks`` deliberately has the same callable result contract as the
other QC classes: construct it with the current combined dataframe, timepoint,
network, and ``form_check_info``; calling the instance returns canonical QC
row dictionaries.  It is dataframe-wide (like ``CrossChecks``) so the data
dictionary, schema, checkbox groups, calculations, and cross-row state are not
rescanned once per participant.

Every finding is routed only to the ``Proposed Checks`` report.  The module is
kept in one file on purpose: the dictionary compiler, generic validators, and
the curated clinical relationships can be reviewed and promoted independently
without changing any existing production checker.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import sys
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)

import pandas as pd

try:  # Production package layout.
    from qc_forms.form_check import FormCheck
    from utils.branching_logic_eval import evaluate_branching_logic
    from process_variables.transform_calculated_fields import (
        TransformCalculatedFields,
        evaluate_converted_calculation,
    )
    from process_variables.transform_branching_logic import (
        TransformBranchingLogic,
    )
except ModuleNotFoundError:  # Flat local-test workspace.
    from form_check import FormCheck
    from utils.branching_logic_eval import evaluate_branching_logic
    from process_variables.transform_calculated_fields import (
        TransformCalculatedFields,
        evaluate_converted_calculation,
    )
    from process_variables.transform_branching_logic import TransformBranchingLogic


REPORT_NAME = "Proposed Checks"
# "na"/"n/a" are the free-text no-data sentinel sites enter where a governed
# missing code is unavailable. utils.py _MISSING_CODE_LIST does not list them,
# so without this they read as substantive values and emit domain/format/range
# findings whose entire content is the sentinel (the legacy pipeline reports
# the same values as "Date is in improper format. It was written as NA").
MISSING_TEXT = frozenset({"", "nan", "nat", "none", "null", "na", "n/a"})
STUDY_MISSING_CODES = frozenset({"-3", "-3.0", "-9", "-9.0", "-99", "-99.0", "999", "999.0"})
DATE_MISSING_CODES = frozenset({"1901-01-01", "1903-03-03", "1909-03-03", "1909-09-09"})
SYSTEM_COLUMNS = frozenset({
    "subjectid", "record_id", "redcap_event_name", "event_name",
    "redcap_repeat_instrument", "redcap_repeat_instance", "network",
    "timepoint", "visit_status", "visit_status_string",
    "recruitment_status_v2", "site", "cohort",
})

# Proposed Checks exposes structured current-row evidence to reviewers.  The
# variable name is always retained, but values that can identify a participant
# or contain unrestricted text are redacted before they leave this checker.
# Several immutable identity fields predate reliable dictionary Identifier?
# metadata, so keep the explicit denylist in addition to FieldSpec.identifier.
SENSITIVE_EVIDENCE_VARIABLES = frozenset({
    "chrdemo_dob", "chrcrit_dob", "chrguid_dob", "chrguid_mob",
    "chrguid_yob", "chrdemo_sexassigned", "chrguid_sex_chr",
    "chrguid_sex_hc", "chrguid_guid", "chrguid_pseudoguid",
})
SENSITIVE_EVIDENCE_NAME = re.compile(
    r"(?:^chrguid_|^chrsaliva_id\d+[a-z]*$|(?:^|_)barcode(?:_|$)|"
    r"(?:^|_)(?:record|study|participant|subject|sibling|specimen|sample)_?id"
    r"(?:\d|_|$)|(?:^|_)(?:name|signature|username|email|phone|address)(?:_|$))",
    flags=re.IGNORECASE,
)
MAX_EVIDENCE_VALUE_CHARS = 120

# Clinically reviewed SCID endpoint inventory for this dictionary version.
# The exact formula AST remains authoritative; these sets classify outputs,
# not their source logic. Prefixes are added when looking up the field.
SCID_TERMINAL_ENDPOINTS = frozenset({
    "chrscid_a25", "chrscid_a51", "chrscid_a70", "chrscid_a91",
    "chrscid_a108", "chrscid_a129", "chrscid_a138", "chrscid_a153",
    "chrscid_a170", "chrscid_a192", "chrscid_a198", "chrscid_a206",
    "chrscid_a213", "chrscid_a221", "chrscid_c10", "chrscid_c14",
    "chrscid_c26", "chrscid_c37", "chrscid_c44", "chrscid_c71",
    "chrscid_c78", "chrscid_d9", "chrscid_as20", "chrscid_as43",
    "chrscid_as50_52", "chrscid_as71", "chrscid_as103", "chrscid_as124",
})
SCID_SUD_COUNT_ENDPOINTS = frozenset({
    "chrscid_e13_14", "chrscid_e33",
    "chrscid_e136_137", "chrscid_e138_139", "chrscid_e140_141",
    "chrscid_e142_143", "chrscid_e144_145", "chrscid_e146_147",
    "chrscid_e148_149", "chrscid_e150_151",
    "chrscid_e299_301", "chrscid_e303_305", "chrscid_e307_309",
    "chrscid_e311_313", "chrscid_e315_317", "chrscid_e319_321",
    "chrscid_e323_325", "chrscid_e327_329",
})
SCID_MANUAL_DIAGNOSIS_FIELD = "chrscid_cur_diagnosis"
# These are structured clinician-entered terminal/summary decisions rather
# than REDCap calculations.  They must never be silently mistaken for a
# deterministic formula endpoint: the dictionary currently contains no
# approved criteria/exclusion/override predicate for them.
# Exact deterministic bridges between clinician-entered SCID criteria and the
# resulting structured episode/disorder selections.  The target is a typed
# diagnosis/episode field, so any declared choice is a positive selection;
# the criteria components use 3 as their declared threshold/true code.
SCID_MANUAL_CHAINS = {
    "chrscid_d4": ("chrscid_d2", "chrscid_d3"),
    "chrscid_d10": ("chrscid_d9",),
    "chrscid_d24": ("chrscid_d21", "chrscid_d22", "chrscid_d23"),
    "chrscid_d29": ("chrscid_d26", "chrscid_d27", "chrscid_d28"),
    "chrscid_d40": ("chrscid_d37", "chrscid_d38", "chrscid_d39"),
    "chrscid_c51": ("chrscid_c48", "chrscid_c49", "chrscid_c50"),
    "chrscid_a193": ("chrscid_a192",),
}
SCHIZOTYPAL_CALC_ENDPOINTS = frozenset({
    "chrschizotypal_szt22a5", "chrschizotypal_threshold_2",
    "chrschizotypal_summ",
})

# Only these clinician-authored PSYCHS warning predicates have been reviewed
# as hard, same-symptom contradictions.  The dictionary contains many other
# descriptive branches that are instructions, missingness reminders, or
# workflow prompts; replaying them wholesale would create false positives.
PSYCHS_SCREEN_VALUE_ALERT_SUFFIXES = frozenset({
    "a3_err", "a4_err", "b1_err", "b2_err", "b3_err", "b6a_err",
    "b7_err", "b7_err2", "b7_err3", "b11_err", "b11_err2",
    "b11_err3", "b15_err", "b15_err2", "b15_err3", "d1_err",
    "d2_err1", "d2_err2",
})
PSYCHS_SCREEN_DATE_ALERT_SUFFIXES = frozenset({
    "a1_date_err4", "a1_date_err5",
    "a6_date_err2", "a6_date_err3",
    "a6_date_err6", "a6_date_err7",
    "a10_date_err", "a10_date_err2",
    "a14_date_err", "a14_date_err2",
    "b5_date_err", "b5_date_err2", "b5_dateerror",
    "b5_dateerror2", "b5_dateerror3",
    "d8_date_err", "a8_err3", "d8_err", "d8_err2",
    "d9_date_err", "d9_err", "d9_err2",
    "d22_date_err", "d22_err", "d22_err2", "d22_err3",
    "d23_date_err", "d23_err", "d23_err2",
})
PSYCHS_FOLLOWUP_VALUE_ALERT_SUFFIXES = frozenset({
    "c3_err", "c4_err", "d2_err1", "d2_err2",
})
PSYCHS_FOLLOWUP_DATE_ALERT_SUFFIXES = frozenset({
    "c1_date_err4", "c1_date_err5",
    "c6_date_err2", "c6_date_err3",
    "c6_date_err6", "c6_date_err7",
    "c10_date_err", "c10_date_err2",
    "c14_date_err", "c14_date_err2",
    "d8_date_err", "c8_err3", "d8_err", "d8_err2",
    "d9_date_err", "d9_err", "d9_err2",
    "d22_date_err", "d22_err", "d22_err2", "d22_err3",
    "d23_date_err", "d23_err", "d23_err2",
})

SCID_BINARY_SUBTYPE_GROUPS = {
    "chrscid_a3": ("chrscid_a4", "chrscid_a5"),
    "chrscid_a6": ("chrscid_a7", "chrscid_a8"),
    "chrscid_a9": ("chrscid_a10", "chrscid_a11"),
    "chrscid_a13": ("chrscid_a14", "chrscid_a15"),
    "chrscid_a17": ("chrscid_a18", "chrscid_a19", "chrscid_a20", "chrscid_a21"),
    "chrscid_a29": ("chrscid_a30", "chrscid_a31"),
    "chrscid_a32": ("chrscid_a33", "chrscid_a34"),
    "chrscid_a35": ("chrscid_a36", "chrscid_a37"),
    "chrscid_a39": ("chrscid_a40", "chrscid_a41"),
    "chrscid_a43": ("chrscid_a44", "chrscid_a45", "chrscid_a46", "chrscid_a47"),
    "chrscid_a54": ("chrscid_a55", "chrscid_a56"),
    "chrscid_a63": ("chrscid_a64", "chrscid_a65"),
    "chrscid_a72": ("chrscid_a73", "chrscid_a74"),
    "chrscid_a80": ("chrscid_a81", "chrscid_a82"),
    "chrscid_a92": ("chrscid_a93", "chrscid_a94"),
    "chrscid_a101": ("chrscid_a102", "chrscid_a103"),
}

SCID_SUBSTANCE_RATING_BASES = (
    "chrscid_sedhypanxiol_rating", "chrscid_cannabis_rating",
    "chrscid_stimulants_rating", "chrscid_opioids_rating",
    "chrscid_hallucinogens_rating", "chrscid_phencyclidine_rating",
    "chrscid_inhalants_rating", "chrscid_othersub_rating",
)

DD_VARIABLE = "Variable / Field Name"
DD_FORM = "Form Name"
DD_TYPE = "Field Type"
DD_LABEL = "Field Label"
DD_CHOICES = "Choices, Calculations, OR Slider Labels"
DD_NOTE = "Field Note"
DD_VALIDATION = "Text Validation Type OR Show Slider Number"
DD_MIN = "Text Validation Min"
DD_MAX = "Text Validation Max"
DD_IDENTIFIER = "Identifier?"
DD_BRANCH = "Branching Logic (Show field only if...)"
DD_REQUIRED = "Required Field?"
DD_MATRIX = "Matrix Group Name"
DD_RANKING = "Matrix Ranking?"
DD_ANNOTATION = "Field Annotation"
REQUIRED_DICTIONARY_COLUMNS = frozenset({
    DD_VARIABLE, DD_FORM, DD_TYPE, DD_LABEL, DD_CHOICES, DD_NOTE,
    DD_VALIDATION, DD_MIN, DD_MAX, DD_IDENTIFIER, DD_BRANCH, DD_REQUIRED,
    DD_MATRIX, DD_RANKING, DD_ANNOTATION,
})


class ProposedCheckConfigurationError(RuntimeError):
    """The dictionary or dependency snapshot cannot support safe QC."""


class ProposedCheckInputError(RuntimeError):
    """The current combined export violates the proposed-check contract."""


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _is_blank(value: Any) -> bool:
    return _text(value).casefold() in MISSING_TEXT


def _decimal(value: Any) -> Decimal | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _canonical_code(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    number = _decimal(text)
    if number is None:
        return text
    integral = number.to_integral_value()
    if number == integral:
        return str(integral)
    return format(number.normalize(), "f")


def _is_missing_value(value: Any) -> bool:
    """Whether ``value`` is blank or a governed study/date sentinel."""

    return (
        _is_blank(value)
        or _canonical_code(value) in STUDY_MISSING_CODES
        or _text(value)[:10] in DATE_MISSING_CODES
    )


def _is_missing_checkbox_column(variable: str) -> bool:
    """Whether an expanded checkbox column represents a missing-code choice."""

    if "___" not in variable:
        return False
    return _canonical_code(variable.split("___", 1)[1]) in STUDY_MISSING_CODES


def _codes_equal(left: Any, right: Any) -> bool:
    return _canonical_code(left) == _canonical_code(right)


def _is_yes(value: Any) -> bool:
    return _canonical_code(value) == "1"


def _is_missing_observation(variable: str, value: Any) -> bool:
    """Whether a scalar or selected expanded choice is governed missingness."""

    return (
        _is_missing_value(value)
        or (_is_missing_checkbox_column(variable) and _is_yes(value))
    )


def _is_no(value: Any) -> bool:
    return _canonical_code(value) == "0"


def _parse_choices(raw: Any) -> tuple[tuple[str, str], ...]:
    """Parse REDCap choices without assuming integer-only codes."""

    value = _text(raw)
    if not value:
        return ()
    choices = []
    for token in re.split(r"\s*\|\s*", value):
        if not token.strip():
            continue
        code, separator, label = token.partition(",")
        if not separator:
            # A malformed choice is retained with a blank label so schema QC
            # can report it instead of silently losing the domain member.
            choices.append((_canonical_code(code), ""))
        else:
            choices.append((_canonical_code(code), label.strip()))
    return tuple(choices)


def _annotation_value(annotation: str, tag: str) -> str:
    match = re.search(
        rf"(?im){re.escape(tag)}\s*=\s*(?:['\"]([^'\"]*)['\"]|([^\s]+))",
        annotation or "",
    )
    return _text(match.group(1) or match.group(2)) if match else ""


def _strict_datetime(value: Any, validation: str) -> datetime | None:
    text = _text(value)
    patterns = {
        "date_ymd": (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
        "datetime_ymd": (
            r"\d{4}-\d{2}-\d{2}(?: |T)\d{2}:\d{2}(?::\d{2})?",
            None,
        ),
        "time": (r"\d{2}:\d{2}(?::\d{2})?", None),
    }
    pattern_info = patterns.get(validation)
    if not text or pattern_info is None or not re.fullmatch(pattern_info[0], text):
        return None
    try:
        if validation == "date_ymd":
            return datetime.strptime(text, pattern_info[1])
        if validation == "datetime_ymd":
            return datetime.fromisoformat(text.replace("T", " "))
        return datetime.strptime(text, "%H:%M:%S" if text.count(":") == 2 else "%H:%M")
    except ValueError:
        return None


def _values_match(left: Any, right: Any, tolerance: Decimal = Decimal("1e-10")) -> bool:
    if _is_blank(left) and _is_blank(right):
        return True
    left_number, right_number = _decimal(left), _decimal(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= tolerance
    return _text(left) == _text(right)


@dataclass(frozen=True)
class FieldSpec:
    variable: str
    form: str
    field_type: str
    label: str
    choices: tuple[tuple[str, str], ...]
    note: str
    validation: str
    minimum: str
    maximum: str
    identifier: bool
    branching_logic: str
    required: bool
    matrix: str
    ranking: str
    annotation: str
    calculation: str

    @property
    def choice_codes(self) -> frozenset[str]:
        return frozenset(code for code, _ in self.choices)

    @property
    def is_descriptive(self) -> bool:
        return self.field_type == "descriptive"

    @property
    def is_calculation(self) -> bool:
        return self.field_type == "calc"

    @property
    def is_numeric(self) -> bool:
        return self.validation in {"number", "integer"}

    @property
    def is_temporal(self) -> bool:
        return self.validation in {"date_ymd", "datetime_ymd", "time"}


@dataclass(frozen=True)
class DictionaryCatalog:
    frame: pd.DataFrame
    fields: Mapping[str, FieldSpec]
    forms: Mapping[str, tuple[str, ...]]
    calculations: Mapping[str, Mapping[str, Any]]
    calculation_order: tuple[str, ...]
    diagnostics: tuple[tuple[str, str, str], ...]

    @classmethod
    def from_frame(cls, source: pd.DataFrame) -> "DictionaryCatalog":
        missing = sorted(REQUIRED_DICTIONARY_COLUMNS - set(source.columns))
        if missing:
            raise ProposedCheckConfigurationError(
                "Data dictionary is missing required column(s): " + ", ".join(missing))
        frame = source.copy().fillna("")
        fields: dict[str, FieldSpec] = {}
        forms: dict[str, list[str]] = defaultdict(list)
        diagnostics: list[tuple[str, str, str]] = []
        normalized_names: dict[str, str] = {}

        for position, (_, row) in enumerate(frame.iterrows(), start=2):
            variable = _text(row[DD_VARIABLE])
            form = _text(row[DD_FORM])
            field_type = _text(row[DD_TYPE]).casefold()
            if not variable or not form or not field_type:
                diagnostics.append((
                    "DD-SCHEMA-004", variable or f"dictionary_row_{position}",
                    f"Dictionary row {position} has a blank variable, form, or field type."))
                continue
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,99}", variable):
                diagnostics.append((
                    "DD-SCHEMA-004", variable,
                    f"Variable name {variable!r} is not a valid lowercase REDCap name."))
            folded = variable.casefold()
            if folded in normalized_names:
                diagnostics.append((
                    "DD-SCHEMA-004", variable,
                    f"Variable name duplicates {normalized_names[folded]!r} case-insensitively."))
                continue
            normalized_names[folded] = variable
            choices = _parse_choices(row[DD_CHOICES])
            codes = [code for code, _ in choices]
            if any(not code for code in codes) or len(codes) != len(set(codes)):
                diagnostics.append((
                    "DD-SCHEMA-004", variable,
                    "Choice list contains a blank or duplicate normalized code."))
            minimum, maximum = _text(row[DD_MIN]), _text(row[DD_MAX])
            if minimum and maximum:
                low, high = _decimal(minimum), _decimal(maximum)
                validation = _text(row[DD_VALIDATION]).casefold()
                if validation in {"number", "integer"} and (
                        low is None or high is None or low > high):
                    diagnostics.append((
                        "DD-SCHEMA-006", variable,
                        f"Numeric bounds are invalid or reversed: {minimum!r}, {maximum!r}."))
            spec = FieldSpec(
                variable=variable,
                form=form,
                field_type=field_type,
                label=_text(row[DD_LABEL]),
                choices=choices,
                note=_text(row[DD_NOTE]),
                validation=_text(row[DD_VALIDATION]).casefold(),
                minimum=minimum,
                maximum=maximum,
                identifier=_text(row[DD_IDENTIFIER]).casefold() == "y",
                branching_logic=_text(row[DD_BRANCH]),
                required=_text(row[DD_REQUIRED]).casefold() == "y",
                matrix=_text(row[DD_MATRIX]),
                ranking=_text(row[DD_RANKING]),
                annotation=_text(row[DD_ANNOTATION]),
                calculation=_text(row[DD_CHOICES]) if field_type == "calc" else "",
            )
            fields[variable] = spec
            forms[form].append(variable)

        calculations: dict[str, Mapping[str, Any]] = {}
        calculation_order: tuple[str, ...] = ()
        if any(spec.is_calculation for spec in fields.values()):
            try:
                transformer = TransformCalculatedFields(frame)
                calculations = transformer()
                calculation_order = tuple(transformer.evaluation_order)
                for variable, metadata in transformer.excluded_conversions.items():
                    diagnostics.append((
                        "DD-CALC-002", variable,
                        f"Calculation could not be compiled: {metadata.get('error', '')}"))
            except Exception as exc:
                diagnostics.append((
                    "DD-CALC-004", "calculation_graph",
                    f"Calculation catalog generation failed: {type(exc).__name__}: {exc}"))

        return cls(
            frame=frame,
            fields=fields,
            forms={form: tuple(values) for form, values in forms.items()},
            calculations=calculations,
            calculation_order=calculation_order,
            diagnostics=tuple(diagnostics),
        )


class _OverlayRow:
    """Read-through row wrapper used for topological calculation evaluation."""

    def __init__(self, source: Any, overlay: Mapping[str, Any]):
        self._source = source
        self._overlay = overlay

    def __getattr__(self, name: str) -> Any:
        if name in self._overlay:
            return self._overlay[name]
        return getattr(self._source, name)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class ProposedChecks(FormCheck):
    """Evaluate missing dictionary-derived and curated checks once per export."""

    REPORT_NAME = REPORT_NAME
    _catalog_cache: dict[str, DictionaryCatalog] = {}
    _longitudinal_state: dict[tuple[str, str], dict[str, Any]] = {}
    _event_values: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    _saliva_barcodes: dict[tuple[str, str], tuple[str, str, str]] = {}
    _guid_identifiers: dict[tuple[str, str, str], str] = {}
    _configuration_seen: set[tuple[str, str, str]] = set()
    _configuration_audited: set[tuple[str, int]] = set()

    def __init__(
            self, combined_df: pd.DataFrame, timepoint: str, network: str,
            form_check_info: Mapping[str, Any], *,
            catalog: DictionaryCatalog | None = None):
        self._injected_curated_rules = list(
            form_check_info.get("_proposed_checks_curated_rules", []))
        super().__init__(timepoint, network, form_check_info)
        self._raw = combined_df.copy(deep=False)
        self._raw.index = pd.RangeIndex(len(self._raw))
        self._row_cache: dict[int, Any] = {}
        self._emitted: set[tuple[Any, ...]] = set()
        self.rule_metrics: dict[str, dict[str, int]] = defaultdict(
            lambda: {"candidates": 0, "emitted": 0, "suppressed": 0})
        self.catalog = catalog or self._resolve_catalog(form_check_info)
        self._var_forms = dict(self.grouped_vars.get("var_forms", {}))
        for variable, spec in self.catalog.fields.items():
            self._var_forms.setdefault(variable, spec.form)
        self._form_mask_cache: dict[tuple[str, bool], pd.Series] = {}
        self._branch_cache: dict[tuple[str, tuple[int, ...]], tuple[pd.Series, str]] = {}
        self._blank_series_cache: dict[str, pd.Series] = {}
        self._observed_series_cache: dict[tuple[str, bool], pd.Series] = {}
        self._branch_transformer = None
        # Longitudinal PSYCHS history is allowed to consume a calculated
        # diagnosis only after the stored value has matched an independent
        # formula recomputation in this same pass.  This prevents one bad
        # calculated field from cascading into later-visit false flags.
        self._verified_calculation_values: dict[tuple[int, str], Any] = {}
        self._scheduled_forms_value = self._compute_scheduled_forms()
        forms_with_values: set[str] = set()
        for column in self._raw.columns:
            base = column.split("___", 1)[0]
            spec = self.catalog.fields.get(base)
            if spec is None:
                continue
            if "___" in column:
                has_value = (
                    not _is_missing_checkbox_column(column)
                    and self._raw[column].map(_is_yes).any())
            else:
                values = self._raw[column]
                has_value = bool((~values.map(_is_missing_value)).any())
            if has_value:
                forms_with_values.add(spec.form)
        self._active_forms = self._scheduled_forms_value | forms_with_values
        self._generic_specs = tuple(
            spec for spec in self.catalog.fields.values()
            if spec.form in self._active_forms)
        self._validate_input()
        self._run()

    def __call__(self) -> list[dict[str, Any]]:
        return self.final_output_list

    @classmethod
    def reset_run_state(cls) -> None:
        """Clear stateful registries before a complete QC pipeline run."""

        cls._longitudinal_state.clear()
        cls._event_values.clear()
        cls._saliva_barcodes.clear()
        cls._guid_identifiers.clear()
        cls._configuration_seen.clear()
        cls._configuration_audited.clear()
        # The canonical dictionary may be replaced between scheduler runs in
        # the same Python process. Reload it once at the start of each run,
        # then reuse the compiled catalog for every network/timepoint.
        cls.clear_catalog_cache()

    @classmethod
    def clear_catalog_cache(cls) -> None:
        cls._catalog_cache.clear()

    def _resolve_catalog(
            self, form_check_info: Mapping[str, Any]) -> DictionaryCatalog:
        injected = form_check_info.get("_proposed_checks_catalog")
        if injected is not None:
            return injected
        injected_frame = form_check_info.get("_proposed_checks_dictionary")
        if injected_frame is not None:
            if not isinstance(injected_frame, pd.DataFrame):
                raise ProposedCheckConfigurationError(
                    "_proposed_checks_dictionary must be a pandas DataFrame")
            return DictionaryCatalog.from_frame(injected_frame)

        config = getattr(self.utils, "config_info", {})
        dependency_path = ""
        if isinstance(config, Mapping):
            paths = config.get("paths", {})
            if isinstance(paths, Mapping):
                dependency_path = _text(paths.get("dependencies_path", ""))
        key = dependency_path or "__utils_default_dictionary__"
        cls = type(self)
        if key not in cls._catalog_cache:
            try:
                frame = self.utils.read_data_dictionary()
            except Exception as exc:
                raise ProposedCheckConfigurationError(
                    "Utils.read_data_dictionary() could not load the canonical "
                    f"REDCap data dictionary: {type(exc).__name__}: {exc}") from exc
            if not isinstance(frame, pd.DataFrame):
                raise ProposedCheckConfigurationError(
                    "Utils.read_data_dictionary() must return a pandas DataFrame")
            cls._catalog_cache = {key: DictionaryCatalog.from_frame(frame)}
        return cls._catalog_cache[key]

    def _validate_input(self) -> None:
        if not isinstance(self._raw, pd.DataFrame):
            raise ProposedCheckInputError("ProposedChecks requires a pandas DataFrame")
        if self._raw.empty:
            return
        required = {
            "subjectid", "visit_status", "visit_status_string",
            "recruitment_status_v2",
        }
        missing = sorted(required - set(self._raw.columns))
        if missing:
            raise ProposedCheckInputError(
                "Combined export is missing runtime column(s): " + ", ".join(missing))
        blank_subject = self._raw["subjectid"].map(_is_blank)
        if blank_subject.any():
            raise ProposedCheckInputError("Combined export contains blank subjectid values")
        duplicate = self._raw["subjectid"].duplicated(keep=False)
        if duplicate.any():
            values = self._raw.loc[duplicate, "subjectid"].astype(str).unique()[:10]
            raise ProposedCheckInputError(
                "ProposedChecks requires one row per subject/timepoint; duplicates: "
                + ", ".join(values))

    def _run(self) -> None:
        if self._raw.empty:
            return
        audit_key = (self.network, id(self.catalog))
        if audit_key not in self._configuration_audited:
            self._run_dictionary_configuration_checks()
            self._configuration_audited.add(audit_key)
        self._run_schema_checks()
        self._run_domain_checks()
        self._run_branch_checks()
        self._run_numeric_checks()
        self._run_format_checks()
        self._run_action_checks()
        self._run_calculation_checks()
        self._run_curated_relationships()
        self._run_injected_curated_rules()
        self._update_longitudinal_state()

    # ------------------------------------------------------------------
    # Canonical row/report helpers

    def _row_at(self, position: int):
        if position not in self._row_cache:
            self._row_cache[position] = next(
                self._raw.iloc[[position]].itertuples(index=False))
        return self._row_cache[position]

    def _first_known_position(self) -> int | None:
        for position, subject in enumerate(self._raw["subjectid"]):
            if subject in self.subject_info:
                return position
        return None

    def _forms_for(self, variables: Sequence[str], fallback: str = "configuration") -> list[str]:
        forms = []
        for variable in variables:
            base = variable.split("___", 1)[0]
            form = self._var_forms.get(variable) or self._var_forms.get(base)
            if form and form not in forms:
                forms.append(form)
        return forms or [fallback]

    def _evidence_variables(self, variables: Sequence[str]) -> list[str]:
        """Expand declared operands to the concrete exported columns shown.

        REDCap checkbox fields normally have no base column in the combined
        export. A rule may still declare the base field, so expose each
        available ``field___code`` column and its 0/1 value instead of showing
        the misleading ``<unavailable>`` value for the base name.
        """

        expanded: list[str] = []
        for raw_variable in variables:
            variable = str(raw_variable).strip()
            if not variable:
                continue
            if variable in self._raw:
                expanded.append(variable)
                continue
            spec = self.catalog.fields.get(variable)
            if spec is not None and spec.field_type == "checkbox":
                checkbox_columns = [
                    f"{variable}___{code}" for code, _ in spec.choices
                    if f"{variable}___{code}" in self._raw]
                if checkbox_columns:
                    expanded.extend(checkbox_columns)
                    continue
            expanded.append(variable)
        return list(dict.fromkeys(expanded))

    def _redact_evidence_value(self, variable: str) -> bool:
        base = variable.split("___", 1)[0]
        spec = self.catalog.fields.get(base)
        if (base in SYSTEM_COLUMNS
                or base in SENSITIVE_EVIDENCE_VARIABLES
                or SENSITIVE_EVIDENCE_NAME.search(base)):
            return True
        if spec is None:
            # An exported field without dictionary governance cannot be
            # classified safely. Configuration anchors that are not exported
            # are rendered as <unavailable> before this decision is needed.
            return variable in self._raw
        if spec.identifier or "@USERNAME" in spec.annotation:
            return True
        # Structured codes, numbers and assessment dates are useful evidence.
        # Unvalidated text/notes can contain names, signatures, IDs or clinical
        # narrative and therefore stay out of routine tracker output.
        if (spec.field_type in {"text", "notes", "textarea", "file", "signature"}
                and not (spec.is_numeric or spec.is_temporal)):
            return True
        return False

    @staticmethod
    def _sanitize_evidence_text(value: Any) -> str:
        text = str(value)
        text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
        # " | " is the canonical multi-flag delimiter in tracker workbooks;
        # never allow a raw value to create a phantom flag or misalign review
        # state during workbook import. Semicolons delimit value pairs.
        text = text.replace("|", "\u00a6").replace(";", "\uFF1B")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > MAX_EVIDENCE_VALUE_CHARS:
            text = text[:MAX_EVIDENCE_VALUE_CHARS - 3].rstrip() + "..."
        return text

    def _variable_value_summary(
            self, position: int, variables: Sequence[str]) -> str:
        pairs: list[str] = []
        for variable in self._evidence_variables(variables):
            safe_variable = self._sanitize_evidence_text(variable)
            if variable not in self._raw:
                rendered = "<unavailable>"
            elif self._redact_evidence_value(variable):
                rendered = "[REDACTED]"
            else:
                value = self._raw[variable].iat[position]
                if _is_blank(value):
                    rendered = "<blank>"
                else:
                    rendered = self._sanitize_evidence_text(value) or "<blank>"
            pairs.append(f"{safe_variable}={rendered}")
        return "; ".join(pairs)

    def _emit(
            self, position: int, check_id: str, variables: Sequence[str],
            message: str, *, forms: Sequence[str] | None = None,
            severity: str = "Error", priority: bool = False) -> None:
        if position < 0 or position >= len(self._raw):
            return
        row = self._row_at(position)
        subject = getattr(row, "subjectid", "")
        if subject not in self.subject_info:
            self.rule_metrics[check_id]["suppressed"] += 1
            return
        variables = list(dict.fromkeys(str(value) for value in variables if value))
        if not variables:
            variables = [check_id.lower().replace("-", "_")]
        resolved_forms = list(forms or self._forms_for(variables))
        identity = (
            self.network, subject, self.timepoint, check_id,
            tuple(resolved_forms), tuple(variables), message,
        )
        self.rule_metrics[check_id]["candidates"] += 1
        if identity in self._emitted:
            self.rule_metrics[check_id]["suppressed"] += 1
            return
        self._emitted.add(identity)
        output = self.create_row_output(
            row, resolved_forms, variables,
            f"[{check_id}] {message}",
            {
                "reports": [self.REPORT_NAME],
                "check_id": check_id,
                "severity": severity,
                "nda_excluder": False,
                "priority": priority,
                "priority_item": priority,
            },
        )
        # Keep observed evidence separate from error_message. The latter is a
        # lifecycle/reviewer-state key, so embedding changing values there
        # would make one continuing issue look resolved and newly opened. The
        # report formatter decorates only the Proposed Checks workbook view.
        output["proposed_variable_values"] = self._variable_value_summary(
            position, variables)
        self.final_output_list.append(output)
        self.rule_metrics[check_id]["emitted"] += 1

    def _emit_mask(
            self, mask: pd.Series, check_id: str, variables: Sequence[str],
            message: str | Callable[[Any], str], **kwargs) -> None:
        normalized = mask.reindex(self._raw.index, fill_value=False).fillna(False).astype(bool)
        for position in normalized[normalized].index:
            row = self._row_at(int(position))
            rendered = message(row) if callable(message) else message
            self._emit(int(position), check_id, variables, rendered, **kwargs)

    def _emit_configuration(
            self, check_id: str, variable: str, message: str,
            *, form: str = "configuration", severity: str = "Config error") -> None:
        key = (self.network, check_id, f"{variable}|{message}")
        if key in self._configuration_seen:
            return
        self._configuration_seen.add(key)
        position = self._first_known_position()
        if position is None:
            return
        self._emit(
            position, check_id, [variable], message,
            forms=[form], severity=severity, priority=True)

    # ------------------------------------------------------------------
    # Applicability and normalized Series helpers

    def _blank_mask(self, variable: str) -> pd.Series:
        if variable in self._blank_series_cache:
            return self._blank_series_cache[variable]
        if variable not in self._raw:
            result = pd.Series(True, index=self._raw.index, dtype=bool)
        else:
            values = self._raw[variable]
            result = values.isna() | values.astype(str).str.strip().str.casefold().isin(MISSING_TEXT)
        self._blank_series_cache[variable] = result
        return result

    def _observed_mask(self, variable: str, *, missing_codes_are_blank: bool = True) -> pd.Series:
        key = (variable, missing_codes_are_blank)
        if key in self._observed_series_cache:
            return self._observed_series_cache[key]
        observed = ~self._blank_mask(variable)
        if _is_missing_checkbox_column(variable):
            observed &= False
        if missing_codes_are_blank and variable in self._raw:
            observed &= ~self._raw[variable].map(_is_missing_value)
        self._observed_series_cache[key] = observed
        return observed

    def _clinical_numeric(self, variable: str) -> tuple[pd.Series, pd.Series]:
        """Return numeric values plus a mask excluding study missing codes.

        A numeric conversion alone is unsafe for clinical ordinal fields:
        REDCap missing sentinels such as ``-9`` parse cleanly but are not a
        severity, count, or date/year value that can support a contradiction.
        """

        numeric = pd.to_numeric(self._raw[variable], errors="coerce")
        valid = self._observed_mask(
            variable, missing_codes_are_blank=True) & numeric.notna()
        return numeric, valid

    def _numeric_series(self, variable: str) -> pd.Series:
        """Parse substantive numeric values while masking blanks/sentinels."""

        numeric = pd.to_numeric(self._raw[variable], errors="coerce")
        return numeric.where(self._observed_mask(variable))

    def _cohort_for_subject(self, subject: Any) -> str:
        return _text(self.subject_info.get(subject, {}).get("cohort", "")).casefold()

    def _scheduled_mask(self, form: str) -> pd.Series:
        result = pd.Series(False, index=self._raw.index, dtype=bool)
        for position, subject in enumerate(self._raw["subjectid"]):
            cohort = self._cohort_for_subject(subject)
            scheduled = self.forms_per_tp.get(cohort, {}).get(self.timepoint, [])
            result.iat[position] = form in scheduled
        return result

    def _completion_variable(self, form: str) -> str:
        metadata = self.important_form_vars.get(form, {})
        completion = _text(metadata.get("completion_var", ""))
        if self.network == "PRESCIENT" and completion:
            completion = (completion + "_rpms").replace(
                "_hc", "").replace("onboarding", "checkin")
        return completion

    def _form_mask(self, form: str, *, due: bool) -> pd.Series:
        key = (form, due)
        if key in self._form_mask_cache:
            return self._form_mask_cache[key]
        mask = self._scheduled_mask(form)
        metadata = self.important_form_vars.get(form, {})
        missing_var = _text(metadata.get("missing_var", ""))
        if missing_var in self._raw and self.timepoint != "floating":
            mask &= ~self._raw[missing_var].map(_is_yes)
        if due:
            completion = self._completion_variable(form)
            complete = (
                self._raw[completion].map(lambda value: _canonical_code(value) == "2")
                if completion in self._raw else pd.Series(False, index=self._raw.index))
            moved = pd.Series(False, index=self._raw.index, dtype=bool)
            if self.timepoint in self.tp_list:
                current_index = self.tp_list.index(self.timepoint)
                for position, value in enumerate(self._raw["visit_status"]):
                    normalized = _text(value).replace("_", "")
                    moved.iat[position] = (
                        normalized in self.tp_list
                        and self.tp_list.index(normalized) > current_index)
            if self.timepoint == "floating":
                complete[:] = True
            mask &= complete | moved
        self._form_mask_cache[key] = mask
        return mask

    def _branch_state(
            self, spec: FieldSpec, candidate_mask: pd.Series) -> tuple[pd.Series, str]:
        positions = tuple(int(value) for value in candidate_mask[candidate_mask].index)
        result = pd.Series(False, index=self._raw.index, dtype=bool)
        if not spec.branching_logic:
            result.loc[list(positions)] = True
            return result, "known"
        if spec.variable in self.excl_bl:
            return result, "excluded"
        metadata = self.conv_bl.get(spec.variable)
        expression = metadata.get("converted_branching_logic", "") if isinstance(
            metadata, Mapping) else ""
        if not isinstance(expression, str) or not expression:
            # A deployment normally supplies the pre-generated branch catalog.
            # This fallback keeps the proposed module independently testable
            # and prevents a newly-added dictionary row from being silently
            # treated as hidden until dependency regeneration catches up.
            try:
                if self._branch_transformer is None:
                    self._branch_transformer = TransformBranchingLogic(
                        self.catalog.frame,
                        utils=self.utils,
                        config_info=self.config_info,
                    )
                expression = self._branch_transformer.branching_logic_redcap_to_python(
                    spec.branching_logic)
            except Exception as exc:
                return result, f"missing:{type(exc).__name__}:{exc}"
        key = (expression, positions)
        if key in self._branch_cache:
            return self._branch_cache[key]
        try:
            for position in positions:
                result.iat[position] = bool(evaluate_branching_logic(
                    expression, curr_row=self._row_at(position), instance=self))
        except Exception as exc:
            state = f"error:{type(exc).__name__}:{exc}"
            self._branch_cache[key] = (result, state)
            return result, state
        self._branch_cache[key] = (result, "known")
        return result, "known"

    def _datediff_branch_state(
            self, spec: FieldSpec,
            candidate_mask: pd.Series) -> tuple[pd.Series, str]:
        """Evaluate reviewed REDCap predicates containing simple datediff().

        The shared branching evaluator intentionally rejects function calls.
        PSYCHS hard-error rows nevertheless use REDCap's unsigned field-to-
        field ``datediff``.  Resolve only that narrow, audited signature to a
        numeric literal per row, then pass the remaining expression through
        the normal converter and safe evaluator.
        """

        result = pd.Series(False, index=self._raw.index, dtype=bool)
        call_pattern = re.compile(
            r"datediff\(\s*\[([A-Za-z][A-Za-z0-9_]*)\]\s*,\s*"
            r"\[([A-Za-z][A-Za-z0-9_]*)\]\s*,\s*['\"]([d])['\"]\s*,\s*"
            r"['\"](?:ymd|ymd h:m|ymd h:m:s)['\"]"
            r"(?:\s*,\s*(true|false|1|0))?\s*\)",
            flags=re.IGNORECASE)
        if "datediff" not in spec.branching_logic.casefold():
            return self._branch_state(spec, candidate_mask)
        if not call_pattern.search(spec.branching_logic):
            return result, "unsupported_datediff_signature"
        try:
            if self._branch_transformer is None:
                self._branch_transformer = TransformBranchingLogic(
                    self.catalog.frame,
                    utils=self.utils,
                    config_info=self.config_info,
                )
            for position in candidate_mask[candidate_mask].index:
                failure = None

                def replace_call(match: re.Match) -> str:
                    nonlocal failure
                    start_variable, end_variable, _, signed = match.groups()
                    start = self._parsed_series(start_variable).iat[int(position)]
                    end = self._parsed_series(end_variable).iat[int(position)]
                    if pd.isna(start) or pd.isna(end):
                        failure = "unparseable_datediff_operand"
                        return "0"
                    days = (end - start).total_seconds() / 86400
                    if _text(signed).casefold() not in {"true", "1"}:
                        days = abs(days)
                    return repr(float(days))

                substituted = call_pattern.sub(
                    replace_call, spec.branching_logic)
                if failure:
                    return result, failure
                if "datediff" in substituted.casefold():
                    return result, "unsupported_datediff_signature"
                expression = (
                    self._branch_transformer.branching_logic_redcap_to_python(
                        substituted))
                result.iat[int(position)] = bool(evaluate_branching_logic(
                    expression, curr_row=self._row_at(int(position)),
                    instance=self))
        except Exception as exc:
            return result, f"error:{type(exc).__name__}:{exc}"
        return result, "known"

    def _branch_sources_known(self, spec: FieldSpec) -> pd.Series:
        """Return rows whose locally referenced branch inputs are substantive.

        A false branch caused by a blank or governed sentinel cannot support a
        participant-level stale-data finding.  Expanded checkbox references
        are resolved to their exported ``field___choice`` columns when present.
        """

        known = pd.Series(True, index=self._raw.index, dtype=bool)
        references = re.findall(
            r"\[([A-Za-z][A-Za-z0-9_]*)(?:\(([^\]\)]+)\))?\]",
            spec.branching_logic,
        )
        for reference, choice in references:
            candidate = reference
            if choice:
                expanded = f"{reference}___{_canonical_code(choice)}"
                if expanded in self._raw:
                    candidate = expanded
            if candidate in self._raw:
                known &= self._observed_mask(candidate)
            elif reference in self.catalog.fields:
                # A locally governed source column is unavailable, so the
                # participant relationship cannot be evaluated safely.
                known &= False
        return known

    @staticmethod
    def _branch_references(spec: FieldSpec) -> tuple[str, ...]:
        """Return the exact local columns referenced by a REDCap predicate."""

        references: list[str] = []
        for variable, choice in re.findall(
                r"\[([A-Za-z][A-Za-z0-9_]*)(?:\(([^\]\)]+)\))?\]",
                spec.branching_logic):
            resolved = (
                f"{variable}___{_canonical_code(choice)}" if choice else variable)
            if resolved not in references:
                references.append(resolved)
        return tuple(references)

    def _valid_operand_mask(self, variable: str) -> pd.Series:
        """Require a substantive, parseable, in-domain rule operand.

        Descriptive REDCap warnings often rely on JavaScript-like coercion.
        Replaying them is safe only after independently proving every operand
        is a real value in its declared domain.  Blank and sentinel values
        therefore suppress, rather than trigger, an authored warning.
        """

        parent = variable.split("___", 1)[0]
        spec = self.catalog.fields.get(parent)
        if variable not in self._raw or spec is None:
            return pd.Series(False, index=self._raw.index, dtype=bool)
        observed = self._observed_mask(variable, missing_codes_are_blank=True)
        if "___" in variable:
            return observed & self._raw[variable].map(
                lambda value: _canonical_code(value) in {"0", "1"})
        if spec.field_type in {"radio", "dropdown"} and spec.choice_codes:
            return observed & self._raw[variable].map(
                lambda value: _canonical_code(value) in spec.choice_codes)
        if spec.field_type == "yesno":
            return observed & self._raw[variable].map(
                lambda value: _canonical_code(value) in {"0", "1"})
        if spec.is_temporal:
            return observed & self._parsed_series(variable).notna()
        if spec.is_numeric:
            return observed & pd.to_numeric(
                self._raw[variable], errors="coerce").notna()
        return observed

    def _applicable_operand_mask(
            self, references: Sequence[str], candidate: pd.Series) -> pd.Series:
        """Require each operand and its own REDCap branch to be applicable."""

        def proof_mask(variable: str) -> pd.Series:
            proof = self._valid_operand_mask(variable)
            parent = variable.split("___", 1)[0]
            field = self.catalog.fields.get(parent)
            if field is not None and field.is_calculation:
                verified = pd.Series([
                    (int(position), parent) in self._verified_calculation_values
                    for position in self._raw.index],
                    index=self._raw.index, dtype=bool)
                proof &= verified
            return proof

        applicable = candidate.copy()
        for variable in references:
            applicable &= proof_mask(variable)
            parent = variable.split("___", 1)[0]
            source = self.catalog.fields.get(parent)
            if source is None or not source.branching_logic:
                continue
            for branch_source in self._branch_references(source):
                applicable &= proof_mask(branch_source)
            visible, state = self._branch_state(source, applicable)
            if state != "known":
                return pd.Series(False, index=self._raw.index, dtype=bool)
            applicable &= visible & self._branch_sources_known(source)
        return applicable

    # ------------------------------------------------------------------
    # Dictionary/configuration and export schema

    def _run_dictionary_configuration_checks(self) -> None:
        for check_id, variable, message in self.catalog.diagnostics:
            form = self.catalog.fields.get(variable).form if variable in self.catalog.fields else "configuration"
            self._emit_configuration(check_id, variable, message, form=form)

        all_fields = set(self.catalog.fields)
        for variable, spec in self.catalog.fields.items():
            references = re.findall(
                r"\[([A-Za-z][A-Za-z0-9_]*)(?:\([^\]]+\))?\]",
                "\n".join((spec.branching_logic, spec.calculation, spec.annotation)),
            )
            for reference in references:
                if reference not in all_fields and reference not in {
                        "event_name", "redcap_event_name", "record_id"} and not re.fullmatch(
                            r"[A-Za-z][A-Za-z0-9_]*_arm_\d+", reference):
                    self._emit_configuration(
                        "DD-SCHEMA-005", variable,
                        f"Dictionary expression references unknown field {reference!r}.",
                        form=spec.form)

            if spec.field_type in {"radio", "dropdown", "yesno"} and spec.choices:
                for literal in re.findall(
                        r"\[" + re.escape(variable) + r"\]\s*(?:=|<>)\s*['\"]([^'\"]+)['\"]",
                        spec.branching_logic):
                    if _canonical_code(literal) not in spec.choice_codes:
                        self._emit_configuration(
                            "DD-SCHEMA-010", variable,
                            f"Branch compares {variable} with undeclared code {literal!r}.",
                            form=spec.form)

        # Repeated-slot source lint is intentionally restricted to actual
        # repeated-record families.  Treating every digit in a variable name
        # as a slot number misclassifies item numbers, ages, month labels, and
        # REDCap event-arm suffixes as copy/paste defects.
        slot_patterns = (
            re.compile(r"^(chrpharm_med)(\d{1,2})"),
            re.compile(r"^(chrmed_cond)(\d{1,2})"),
            re.compile(r"^(chrpsychsoc_treat)(\d{1,2})"),
            re.compile(r"^(chrrul_resource)(\d{1,2})"),
            re.compile(r"^(chrcoen_study)(\d{1,2})"),
            re.compile(r"^(chrae_)(?:ae|aes|tp|dr|sig|add)(\d{1,2})"),
        )

        def repeated_slot(value: str) -> tuple[str, int] | None:
            for pattern in slot_patterns:
                match = pattern.search(value)
                if match:
                    return match.group(1), int(match.group(2))
            return None

        for variable, spec in self.catalog.fields.items():
            target = repeated_slot(variable)
            if target is None or not spec.branching_logic:
                continue
            target_family, target_index = target
            suspicious = []
            for reference in re.findall(r"\[([A-Za-z][A-Za-z0-9_]*)", spec.branching_logic):
                source = repeated_slot(reference)
                if (source is not None and source[0] == target_family
                        and abs(source[1] - target_index) > 1):
                    suspicious.append(reference)
            if suspicious:
                self._emit_configuration(
                    "DD-SCHEMA-007", variable,
                    "Repeated-slot branch contains nonadjacent reference(s): "
                    + ", ".join(sorted(set(suspicious))), form=spec.form,
                    severity="Config warning")

    def _compute_scheduled_forms(self) -> set[str]:
        forms = set()
        for subject in self._raw["subjectid"]:
            cohort = self._cohort_for_subject(subject)
            forms.update(self.forms_per_tp.get(cohort, {}).get(self.timepoint, []))
        return forms

    def _scheduled_forms(self) -> set[str]:
        cached = getattr(self, "_scheduled_forms_value", None)
        return set(cached if cached is not None else self._compute_scheduled_forms())

    def _run_schema_checks(self) -> None:
        scheduled = self._scheduled_forms()
        excluded = set(self.general_check_vars.get("excluded_vars", {}).get(self.network, []))
        available = set(self._raw.columns)
        for form in sorted(scheduled):
            expected = []
            for variable in self.catalog.forms.get(form, ()):
                spec = self.catalog.fields[variable]
                if (spec.is_descriptive or variable in excluded
                        or (self.network == "PRONET" and spec.identifier)):
                    continue
                if spec.field_type == "checkbox":
                    expected.extend(f"{variable}___{code}" for code, _ in spec.choices)
                else:
                    expected.append(variable)
            missing = sorted(set(expected) - available)
            if missing:
                rendered = ", ".join(missing[:20])
                suffix = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
                self._emit_configuration(
                    "DD-SCHEMA-001", form,
                    f"Scheduled form export is missing {len(missing)} expected field column(s): "
                    f"{rendered}{suffix}.", form=form)

        # FORM-007 (expected completion column is absent) was removed with the
        # rest of the form-completion scope.

        known = set(self.catalog.fields) | SYSTEM_COLUMNS
        known.update(
            f"{spec.variable}___{code}"
            for spec in self.catalog.fields.values() if spec.field_type == "checkbox"
            for code, _ in spec.choices)
        known.update(self._completion_variable(form) for form in self.important_form_vars)
        unknown = sorted(
            column for column in available
            if column not in known
            and not column.endswith("_complete")
            and not column.endswith("_complete_rpms"))
        # Unknown combined columns are commonly approved upstream/site helper
        # columns. Report one reviewable schema row rather than one participant
        # flag per column.
        if unknown:
            rendered = ", ".join(unknown[:25])
            suffix = f" (+{len(unknown) - 25} more)" if len(unknown) > 25 else ""
            self._emit_configuration(
                "DD-SCHEMA-003", "unexpected_export_columns",
                f"Export contains {len(unknown)} column(s) not classified by the dictionary/system allowlist: "
                f"{rendered}{suffix}.", severity="Config warning")

    # ------------------------------------------------------------------
    # Generic domains, requiredness, branches, numeric and format checks

    def _run_domain_checks(self) -> None:
        for spec in self._generic_specs:
            if spec.field_type in {"radio", "dropdown"} and spec.variable in self._raw:
                allowed = spec.choice_codes
                if not allowed:
                    continue
                values = self._raw[spec.variable]
                invalid = self._observed_mask(spec.variable) & ~values.map(
                    lambda value: _canonical_code(value) in allowed)
                self._emit_mask(
                    invalid, "DD-DOMAIN-001", [spec.variable],
                    lambda row, variable=spec.variable, codes=sorted(allowed):
                    f"Value {_text(getattr(row, variable))!r} is outside the declared domain {codes!r}.")

            elif spec.field_type == "yesno" and spec.variable in self._raw:
                invalid = self._observed_mask(spec.variable) & ~self._raw[spec.variable].map(
                    lambda value: _canonical_code(value) in {"0", "1"})
                self._emit_mask(
                    invalid, "DD-DOMAIN-002", [spec.variable],
                    lambda row, variable=spec.variable:
                    f"Yes/No value {_text(getattr(row, variable))!r} is not canonical 0 or 1.")

            elif spec.field_type == "checkbox":
                columns = [
                    f"{spec.variable}___{code}" for code, _ in spec.choices
                    if f"{spec.variable}___{code}" in self._raw]
                for column in columns:
                    invalid = self._observed_mask(column) & ~self._raw[column].map(
                        lambda value: _canonical_code(value) in {"0", "1"})
                    self._emit_mask(
                        invalid, "DD-DOMAIN-003", [column],
                        lambda row, variable=column:
                        f"Expanded checkbox value {_text(getattr(row, variable))!r} is not blank, 0, or 1.")

                none_code = _annotation_value(spec.annotation, "@NONEOFTHEABOVE")
                none_column = f"{spec.variable}___{_canonical_code(none_code)}"
                other_columns = [
                    column for column in columns
                    if column != none_column
                    and not _is_missing_checkbox_column(column)]
                if none_code and none_column in self._raw and other_columns:
                    conflict = self._raw[none_column].map(_is_yes) & pd.concat(
                        [self._raw[column].map(_is_yes) for column in other_columns], axis=1).any(axis=1)
                    self._emit_mask(
                        conflict, "DD-DOMAIN-005", [none_column] + other_columns,
                        "None-of-the-above is selected together with another checkbox choice.")

            hidden_code = _annotation_value(spec.annotation, "@HIDECHOICE")
            # A hidden choice that is itself a governed missing code is a
            # missingness report, not a dictionary-version defect. Every other
            # generic check reaches its value through ``_observed_mask``, which
            # already drops sentinels; this one compares the raw code, so it
            # needs the guard stated explicitly.
            if _canonical_code(hidden_code) in STUDY_MISSING_CODES:
                hidden_code = ""
            if hidden_code and spec.variable in self._raw:
                hidden = self._raw[spec.variable].map(
                    lambda value, expected=hidden_code: _codes_equal(value, expected))
                self._emit_mask(
                    hidden, "DD-DOMAIN-006", [spec.variable],
                    f"Hidden choice {hidden_code!r} is present and requires dictionary-version review.",
                    severity="Warning")

        # Curated checkbox states whose labels alone are not sufficient for a
        # global inference.
        for base, exclusive_code in {
                "chrchs_covidvac": "0",
                "chrblood_drawprep": "8",
                "chrsaliva_food": "4",
        }.items():
            exclusive = f"{base}___{exclusive_code}"
            peers = [
                column for column in self._raw
                if column.startswith(base + "___") and column != exclusive
                and not _is_missing_checkbox_column(column)]
            if exclusive in self._raw and peers:
                mask = self._raw[exclusive].map(_is_yes) & pd.concat(
                    [self._raw[column].map(_is_yes) for column in peers], axis=1).any(axis=1)
                self._emit_mask(
                    mask, "DD-DOMAIN-005", [exclusive] + peers,
                    "Exclusive No/None choice is selected with a substantive choice.")

    def _run_branch_checks(self) -> None:
        for spec in self._generic_specs:
            if spec.is_descriptive or spec.is_calculation or not spec.branching_logic:
                continue
            if spec.field_type == "checkbox":
                columns = [
                    f"{spec.variable}___{code}" for code, _ in spec.choices
                    if (f"{spec.variable}___{code}" in self._raw
                        and _canonical_code(code) not in STUDY_MISSING_CODES)]
                observed = (
                    pd.concat([self._raw[column].map(_is_yes) for column in columns], axis=1).any(axis=1)
                    if columns else pd.Series(False, index=self._raw.index))
            elif spec.variable in self._raw:
                observed = self._observed_mask(spec.variable)
                columns = [spec.variable]
            else:
                continue

            relevant = self._form_mask(spec.form, due=False)
            candidate = relevant & observed
            visible, state = self._branch_state(spec, candidate)
            if state != "known":
                continue

            # A false branch is actionable only when every referenced local
            # source is substantive. Blank/sentinel controllers are not used
            # to manufacture a hidden-with-data finding.
            known_sources = self._branch_sources_known(spec)
            hidden_with_data = relevant & observed & known_sources & ~visible
            if "@HIDDEN" not in spec.annotation or "@DEFAULT" not in spec.annotation:
                self._emit_mask(
                    hidden_with_data, "DD-BRANCH-001",
                    [*columns, *self._branch_references(spec)],
                    "Field contains participant-entered data while its branch is false.",
                    severity="Warning")

    def _run_numeric_checks(self) -> None:
        active_ranges = self.variable_ranges.get(self.network, {})
        for spec in self._generic_specs:
            if not spec.is_numeric or spec.variable not in self._raw:
                continue
            observed = self._observed_mask(spec.variable)
            numeric = self._numeric_series(spec.variable)
            finite = numeric.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
            malformed = observed & ~finite
            self._emit_mask(
                malformed, "DD-NUM-001", [spec.variable],
                lambda row, variable=spec.variable:
                f"Numeric field contains non-finite or malformed value {_text(getattr(row, variable))!r}.")

            valid = observed & finite
            if spec.validation == "integer":
                noninteger = valid & numeric.map(lambda value: float(value).is_integer() is False)
                self._emit_mask(
                    noninteger, "DD-NUM-002", [spec.variable],
                    lambda row, variable=spec.variable:
                    f"Integer field contains non-integral value {_text(getattr(row, variable))!r}.")
                valid &= ~noninteger

            low, high = _decimal(spec.minimum), _decimal(spec.maximum)
            out_of_range = pd.Series(False, index=self._raw.index, dtype=bool)
            if low is not None:
                out_of_range |= valid & numeric.lt(float(low))
            if high is not None:
                out_of_range |= valid & numeric.gt(float(high))
            if (low is not None or high is not None) and spec.variable not in active_ranges:
                # Preserve the stable two-sided ID used by the design while
                # distinguishing the currently-dropped one-sided case.
                check_id = "DD-NUM-003" if low is not None and high is not None else "DD-NUM-004"
                self._emit_mask(
                    out_of_range, check_id, [spec.variable],
                    lambda row, variable=spec.variable, lo=spec.minimum, hi=spec.maximum:
                    f"Value {_text(getattr(row, variable))!r} violates inclusive bound(s) "
                    f"minimum={lo or '<none>'}, maximum={hi or '<none>'}.")

    def _run_format_checks(self) -> None:
        id_for_validation = {
            "date_ymd": "DD-FMT-001",
            "datetime_ymd": "DD-FMT-002",
            "time": "DD-FMT-003",
        }
        now = datetime.now()
        for spec in self._generic_specs:
            if spec.variable not in self._raw:
                continue
            observed = self._observed_mask(spec.variable)
            if spec.validation in id_for_validation:
                parsed = self._raw[spec.variable].map(
                    lambda value, validation=spec.validation: _strict_datetime(value, validation))
                invalid = observed & parsed.isna()
                self._emit_mask(
                    invalid, id_for_validation[spec.validation], [spec.variable],
                    lambda row, variable=spec.variable, validation=spec.validation:
                    f"Value {_text(getattr(row, variable))!r} is not a real canonical {validation} value.")
                valid = observed & ~invalid

                if "@NOTFUTURE" in spec.annotation:
                    future = valid & parsed.map(lambda value: value is not None and value > now)
                    self._emit_mask(
                        future, "DD-FMT-005", [spec.variable],
                        "Value is later than the current local date/time.")

                low = _strict_datetime(spec.minimum, spec.validation) if spec.minimum else None
                high = _strict_datetime(spec.maximum, spec.validation) if spec.maximum else None
                outside = pd.Series(False, index=self._raw.index, dtype=bool)
                if low is not None:
                    outside |= valid & parsed.map(lambda value: value is not None and value < low)
                if high is not None:
                    outside |= valid & parsed.map(lambda value: value is not None and value > high)
                if low is not None or high is not None:
                    self._emit_mask(
                        outside, "DD-FMT-006", [spec.variable],
                        f"Temporal value violates declared bound(s) {spec.minimum!r}–{spec.maximum!r}.")

            elif spec.validation == "alpha_only":
                invalid = observed & ~self._raw[spec.variable].astype(str).str.fullmatch(
                    r"[^\W\d_]+(?:[ '\-][^\W\d_]+)*", na=False)
                self._emit_mask(
                    invalid, "DD-FMT-004", [spec.variable],
                    "Alpha-only field contains a nonalphabetic value.")

    # ------------------------------------------------------------------
    # Field actions, calculations, and completion-domain validation

    def _run_action_checks(self) -> None:
        for spec in self._generic_specs:
            if spec.variable not in self._raw or spec.is_descriptive or spec.is_calculation:
                continue
            observed = self._observed_mask(spec.variable)
            readonly = "@READONLY" in spec.annotation
            default = _annotation_value(spec.annotation, "@DEFAULT")
            if readonly and default and not default.startswith("["):
                wrong = observed & ~self._raw[spec.variable].map(
                    lambda value, expected=default: _codes_equal(value, expected))
                self._emit_mask(
                    wrong, "DD-ACTION-003", [spec.variable],
                    f"Read-only field does not retain declared value {default!r}.")
            elif "@HIDDEN" in spec.annotation and default and not default.startswith("["):
                wrong = observed & ~self._raw[spec.variable].map(
                    lambda value, expected=default: _codes_equal(value, expected))
                self._emit_mask(
                    wrong, "DD-ACTION-004", [spec.variable],
                    f"Hidden initialization field differs from declared default {default!r}.",
                    severity="Warning")

    def _substantive_variables(self, form: str) -> list[str]:
        variables = []
        for variable in self.catalog.forms.get(form, ()):
            spec = self.catalog.fields[variable]
            if (spec.is_descriptive or spec.is_calculation
                    or "@USERNAME" in spec.annotation
                    or ("@HIDDEN" in spec.annotation and "@DEFAULT" in spec.annotation)):
                continue
            if spec.field_type == "checkbox":
                variables.extend(
                    f"{variable}___{code}" for code, _ in spec.choices
                    if (f"{variable}___{code}" in self._raw
                        and _canonical_code(code) not in STUDY_MISSING_CODES))
            elif variable in self._raw:
                variables.append(variable)
        metadata = self.important_form_vars.get(form, {})
        for excluded in (
                metadata.get("missing_var", ""),
                metadata.get("missing_spec_var", ""),
                self._completion_variable(form)):
            if excluded in variables:
                variables.remove(excluded)
        return variables

    def _substantive_mask(self, form: str) -> pd.Series:
        variables = self._substantive_variables(form)
        if not variables:
            return pd.Series(False, index=self._raw.index, dtype=bool)
        masks: list[pd.Series] = []
        for variable in variables:
            parent = variable.split("___", 1)[0]
            spec = self.catalog.fields.get(parent)
            if spec is not None and spec.field_type == "checkbox" and "___" in variable:
                # REDCap exports every checkbox option as 0/1.  An unchecked
                # option is therefore structurally present but is not
                # substantive form data.
                masks.append(self._raw[variable].map(_is_yes))
            else:
                masks.append(
                    self._observed_mask(variable, missing_codes_are_blank=True))
        return pd.concat(
            masks, axis=1).any(axis=1)

    def _run_calculation_checks(self) -> None:
        if not self.catalog.calculations:
            return
        ordered = list(self.catalog.calculation_order)
        ordered.extend(variable for variable in self.catalog.calculations if variable not in ordered)
        # Store only the fields referenced cross-event, not an entire very-wide
        # combined row, in the process-wide event snapshot.
        cross_event_needed = {
            reference.get("variable")
            for metadata in self.catalog.calculations.values()
            for reference in metadata.get("cross_event_dependencies", [])
            if isinstance(reference, Mapping) and reference.get("variable")}

        for position in self._raw.index:
            row = self._row_at(int(position))
            subject = getattr(row, "subjectid", "")
            if subject not in self.subject_info:
                continue
            overlay: dict[str, Any] = {}
            proxy = _OverlayRow(row, overlay)
            event_data = dict(self._event_values.get((self.network, subject), {}))
            current_event_aliases = self._event_aliases(subject, row)
            # Explicit references to the current REDCap event must resolve from
            # the current row (and its topologically recomputed overlay), not
            # wait until the snapshot is persisted after this loop.
            for alias in current_event_aliases:
                event_data[alias] = proxy
            for variable in ordered:
                metadata = self.catalog.calculations.get(variable, {})
                if variable not in self._raw or metadata.get("status") != "converted":
                    continue
                spec = self.catalog.fields.get(variable)
                if (spec is None or spec.form not in self._active_forms
                        or not self._form_mask(spec.form, due=False).iat[position]):
                    continue
                branch, state = self._branch_state(
                    spec, pd.Series(
                        [index == position for index in self._raw.index],
                        index=self._raw.index, dtype=bool))
                if state != "known":
                    self._emit_configuration(
                        "DD-CALC-003", variable,
                        f"Calculation target visibility is uncertain ({state}).",
                        form=spec.form, severity="Config warning")
                    continue
                if not branch.iat[position]:
                    if (spec.form == "scid5_psychosis_mood_substance_abuse"
                            and variable in SCID_TERMINAL_ENDPOINTS
                            and self._branch_sources_known(spec).iat[position]
                            and self._observed_mask(variable).iat[position]):
                        self._emit(
                            int(position), "SCID-DX-004", [variable],
                            "SCID endpoint is not applicable/hidden but retains a value.",
                            forms=[spec.form], severity="Warning")
                    continue
                converted = metadata.get("converted_calculation", "")
                dependencies = list(metadata.get("referenced_variables", []))
                missing_event_scopes = sorted({
                    reference.get("event")
                    for reference in metadata.get("cross_event_dependencies", [])
                    if reference.get("event") not in event_data})
                if missing_event_scopes:
                    self._emit_configuration(
                        "DD-CALC-003", variable,
                        "Calculation cannot be compared because explicit REDCap event "
                        f"scope(s) are unavailable: {', '.join(missing_event_scopes)}.",
                        form=spec.form, severity="Config warning")
                    continue
                unknown_dependencies = [
                    dependency for dependency in dependencies
                    if ((dependency in overlay
                         and _is_missing_observation(
                             dependency, overlay[dependency]))
                        or (dependency not in overlay
                            and (not hasattr(row, dependency)
                                 or _is_missing_observation(
                                     dependency, getattr(row, dependency)))))]
                if unknown_dependencies:
                    # Missing source data cannot support a deterministic
                    # comparison, including a SCID diagnosis contradiction.
                    continue
                outcome = evaluate_converted_calculation(
                    converted, proxy, event_data=event_data)
                if outcome.status != "ok":
                    self._emit_configuration(
                        "DD-CALC-002", variable,
                        f"Calculation evaluation failed: {outcome.error}", form=spec.form)
                    continue
                expected = outcome.value
                overlay[variable] = expected
                stored = getattr(row, variable)
                if _is_missing_value(stored) or _is_missing_value(expected):
                    continue
                if _values_match(stored, expected):
                    self._verified_calculation_values[(int(position), variable)] = expected
                    continue

                if spec.form == "scid5_psychosis_mood_substance_abuse" and variable in SCID_TERMINAL_ENDPOINTS:
                    expected_positive = self._positive_endpoint(expected)
                    stored_positive = self._positive_endpoint(stored)
                    if expected_positive is True and stored_positive is False:
                        check_id = "SCID-DX-001"
                        message = (
                            f"Exact dictionary criteria evaluate as Met ({expected!r}), but "
                            f"the stored diagnosis/endpoint is {stored!r}.")
                    elif stored_positive is True and expected_positive is False:
                        check_id = "SCID-DX-002"
                        message = (
                            f"Stored diagnosis/endpoint is positive ({stored!r}), but exact "
                            f"dictionary criteria evaluate as Not met ({expected!r}).")
                    else:
                        check_id = "SCID-DX-005"
                        message = (
                            f"Stored SCID endpoint {stored!r} differs from exact formula "
                            f"recomputation {expected!r}.")
                elif variable in SCHIZOTYPAL_CALC_ENDPOINTS:
                    check_id = "SCID-SZT-001"
                    message = (
                        f"Stored schizotypal calculated value {stored!r} differs from "
                        f"exact dictionary recomputation {expected!r}.")
                else:
                    check_id = "DD-CALC-001"
                    message = (
                        f"Stored calculated value {stored!r} differs from recomputed value "
                        f"{expected!r}.")
                self._emit(
                    int(position), check_id, [variable, *dependencies], message,
                    forms=[spec.form])

            snapshot = {
                variable: overlay.get(variable, getattr(row, variable))
                for variable in cross_event_needed if hasattr(row, variable)}
            if snapshot:
                registry = self._event_values.setdefault((self.network, subject), {})
                for alias in current_event_aliases:
                    registry[alias] = snapshot

    def _event_aliases(self, subject: Any, row: Any) -> set[str]:
        aliases = {self.timepoint}
        explicit = _text(getattr(row, "redcap_event_name", ""))
        if explicit:
            aliases.add(explicit)
        cohort = self._cohort_for_subject(subject)
        arm = "2" if cohort == "hc" else "1"
        token = self.timepoint.casefold().replace("_", "")
        match = re.fullmatch(r"month(\d+)", token)
        event_token = f"month_{match.group(1)}" if match else token
        if event_token in {"screening", "baseline", "conversion"} or match:
            aliases.add(f"{event_token}_arm_{arm}")
        return aliases

    @staticmethod
    def _positive_endpoint(value: Any) -> bool | None:
        if _is_blank(value) or _canonical_code(value) in STUDY_MISSING_CODES:
            return None
        code = _canonical_code(value)
        # All targets in SCID_TERMINAL_ENDPOINTS use the reviewed SCID
        # calculation convention 3=criteria met. Do not globalize Yes=1 to
        # SCID: many source items use 1 for a different state.
        if code == "3":
            return True
        if code in {"0", "1", "2"}:
            return False
        return None

    # FORM-001 (completion-status domain validation) was removed: the Proposed
    # Checks report is scoped to data-quality contradictions and carries no
    # form-completion findings. ``_completion_variable`` is retained because
    # ``_form_mask`` and ``_substantive_variables`` still need it for
    # applicability gating, which emits nothing.

    def _run_injected_curated_rules(self) -> None:
        rules = self._injected_curated_rules
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            predicate = rule.get("predicate")
            if not callable(predicate):
                continue
            variables = list(rule.get("variables", []))
            if any(variable not in self._raw for variable in variables):
                continue
            for position in self._raw.index:
                row = self._row_at(int(position))
                if any(_is_missing_observation(
                           variable, self._raw[variable].iat[int(position)])
                       for variable in variables):
                    continue
                try:
                    violated = bool(predicate(row))
                except Exception as exc:
                    self._emit_configuration(
                        "CURATED-CONFIG-001", _text(rule.get("check_id", "curated_rule")),
                        f"Injected curated predicate failed: {type(exc).__name__}: {exc}")
                    continue
                if violated:
                    self._emit(
                        int(position), _text(rule.get("check_id", "CURATED-UNNAMED")),
                        variables, _text(rule.get("message", "Curated relationship failed.")),
                        forms=[_text(rule.get("form", "configuration"))],
                        severity=_text(rule.get("severity", "Error")) or "Error")

    # ------------------------------------------------------------------
    # Curated deterministic clinical/cross-form relationships

    def _has_columns(self, *variables: str) -> bool:
        return all(variable in self._raw for variable in variables)

    def _code_mask(self, variable: str, codes: Iterable[Any]) -> pd.Series:
        if variable not in self._raw:
            return pd.Series(False, index=self._raw.index, dtype=bool)
        allowed = {_canonical_code(value) for value in codes}
        return self._raw[variable].map(_canonical_code).isin(allowed)

    def _any_observed(self, variables: Sequence[str]) -> pd.Series:
        present = [variable for variable in variables if variable in self._raw]
        if not present:
            return pd.Series(False, index=self._raw.index, dtype=bool)
        masks = []
        for variable in present:
            parent = variable.split("___", 1)[0]
            spec = self.catalog.fields.get(parent)
            masks.append(
                (pd.Series(False, index=self._raw.index, dtype=bool)
                 if _is_missing_checkbox_column(variable)
                 else self._raw[variable].map(_is_yes))
                if ("___" in variable and spec is not None
                    and spec.field_type == "checkbox")
                else self._observed_mask(variable))
        return pd.concat(masks, axis=1).any(axis=1)

    def _all_observed(self, variables: Sequence[str]) -> pd.Series:
        if not variables or any(variable not in self._raw for variable in variables):
            return pd.Series(False, index=self._raw.index, dtype=bool)
        masks = []
        for variable in variables:
            parent = variable.split("___", 1)[0]
            spec = self.catalog.fields.get(parent)
            masks.append(
                (pd.Series(False, index=self._raw.index, dtype=bool)
                 if _is_missing_checkbox_column(variable)
                 else self._raw[variable].map(_is_yes))
                if ("___" in variable and spec is not None
                    and spec.field_type == "checkbox")
                else self._observed_mask(variable))
        return pd.concat(masks, axis=1).all(axis=1)

    def _parsed_series(self, variable: str) -> pd.Series:
        if variable not in self._raw:
            return pd.Series(pd.NaT, index=self._raw.index)
        spec = self.catalog.fields.get(variable)
        validation = spec.validation if spec else ""
        if validation in {"date_ymd", "datetime_ymd", "time"}:
            parsed = self._raw[variable].map(
                lambda value: None
                if (_canonical_code(value) in STUDY_MISSING_CODES
                    or _text(value)[:10] in DATE_MISSING_CODES)
                else _strict_datetime(value, validation))
            # An all-missing object Series otherwise contains None values, and
            # pandas cannot order None against None. Normalize to datetime64.
            return pd.to_datetime(parsed, errors="coerce")
        cleaned = self._raw[variable].map(
            lambda value: None
            if (_canonical_code(value) in STUDY_MISSING_CODES
                or _text(value)[:10] in DATE_MISSING_CODES)
            else value)
        return pd.to_datetime(cleaned, errors="coerce")

    def _combine_series_date_time(
            self, dates: pd.Series, times: pd.Series) -> pd.Series:
        """Combine parsed date and time series without coercing malformed data."""
        combined = pd.Series(pd.NaT, index=self._raw.index, dtype="datetime64[ns]")
        known = dates.notna() & times.notna()
        if known.any():
            combined.loc[known] = pd.to_datetime(
                dates.loc[known].dt.strftime("%Y-%m-%d") + " "
                + times.loc[known].dt.strftime("%H:%M:%S"), errors="coerce")
        return combined

    def _combine_date_time(self, date_variable: str, time_variable: str) -> pd.Series | None:
        if date_variable not in self._raw or time_variable not in self._raw:
            return None
        return self._combine_series_date_time(
            self._parsed_series(date_variable), self._parsed_series(time_variable))

    def _rule_equal(
            self, check_id: str, left: str, right: str, message: str,
            *, forms: Sequence[str] | None = None, severity: str = "Error") -> None:
        if not self._has_columns(left, right):
            return
        comparable = self._observed_mask(left) & self._observed_mask(right)
        mismatch = comparable & ~pd.Series(
            [_values_match(a, b) for a, b in zip(self._raw[left], self._raw[right])],
            index=self._raw.index)
        self._emit_mask(
            mismatch, check_id, [left, right], message,
            forms=forms, severity=severity)

    def _rule_order(
            self, check_id: str, earlier: str, later: str, message: str,
            *, strict: bool = False, forms: Sequence[str] | None = None,
            severity: str = "Error") -> None:
        if not self._has_columns(earlier, later):
            return
        left, right = self._parsed_series(earlier), self._parsed_series(later)
        comparable = left.notna() & right.notna()
        violation = comparable & (right.le(left) if strict else right.lt(left))
        self._emit_mask(
            violation, check_id, [earlier, later], message,
            forms=forms, severity=severity)

    def _rule_forbids(
            self, check_id: str, controller: str, trigger_codes: Iterable[Any],
            forbidden: Sequence[str], message: str, *,
            forms: Sequence[str] | None = None, severity: str = "Error") -> None:
        if controller not in self._raw:
            return
        substantive_codes = [
            code for code in trigger_codes if not _is_missing_value(code)]
        trigger = self._observed_mask(controller) & self._code_mask(
            controller, substantive_codes)
        for variable in forbidden:
            if variable not in self._raw:
                continue
            self._emit_mask(
                trigger & self._observed_mask(variable), check_id,
                [controller, variable], message,
                forms=forms, severity=severity)

    def _checkbox_selected(self, base: str, codes: Iterable[Any]) -> pd.Series:
        selected = pd.Series(False, index=self._raw.index, dtype=bool)
        substantive_codes = [code for code in codes if not _is_missing_value(code)]
        for code in substantive_codes:
            column = f"{base}___{_canonical_code(code)}"
            if column in self._raw:
                selected |= self._raw[column].map(_is_yes)
        if base in self._raw:
            # Test fixtures and some non-REDCap extracts retain the compact
            # checkbox value. Support them without changing export semantics.
            selected |= self._code_mask(base, substantive_codes)
        return selected

    def _run_curated_relationships(self) -> None:
        self._run_consent_status_identity_checks()
        self._run_eligibility_conversion_checks()
        self._run_cssrs_psychs_checks()
        self._run_scid_schizotypal_checks()
        self._run_episode_and_medication_checks()
        self._run_health_specimen_checks()
        self._run_procedure_and_device_checks()
        self._run_scale_demographic_assist_checks()
        self._run_administrative_scale_extensions()
        self._run_final_deterministic_extensions()

    def _run_consent_status_identity_checks(self) -> None:
        self._rule_equal(
            "CONSENT-001", "enrollmentnote_dateofconsent", "chric_consent_date",
            "Enrollment-note consent date does not equal governed initial-consent date.",
            forms=["enrollment_note", "informed_consent_run_sheet"])
        self._rule_order(
            "CONSENT-002", "chric_consent_date", "chric_reconsent_date",
            "Reconsent date precedes initial consent.",
            forms=["informed_consent_run_sheet", "informed_reconsent"])

        if self._has_columns("chric_reconsent_date", "chric_reconsent_age"):
            dob_var = next((value for value in (
                "chrdemo_dob", "chrcrit_dob", "chrguid_full_dob")
                if value in self._raw), None)
            if dob_var:
                reconsent = self._parsed_series("chric_reconsent_date")
                dob = self._parsed_series(dob_var)
                reported = self._numeric_series("chric_reconsent_age")
                expected = pd.Series(
                    [((end.year - start.year) - ((end.month, end.day) < (start.month, start.day)))
                     if pd.notna(start) and pd.notna(end) else float("nan")
                     for start, end in zip(dob, reconsent)], index=self._raw.index)
                mismatch = reported.notna() & expected.notna() & reported.ne(expected)
                self._emit_mask(
                    mismatch, "CONSENT-003",
                    ["chric_reconsent_age", "chric_reconsent_date", dob_var],
                    "Reconsent age does not equal DOB-derived age on reconsent date.",
                    forms=["informed_reconsent", "sociodemographics"])

        if "chric_consent_date" in self._raw:
            for variable, spec in self.catalog.fields.items():
                if (variable not in self._raw
                        or variable == "chric_consent_date"
                        or "interview_date" not in variable
                        or spec.form in {
                            "adverse_events", "revised_status_form", "status_form",
                            "missing_data"}):
                    continue
                self._rule_order(
                    "CONSENT-004", "chric_consent_date", variable,
                    "Assessment/acquisition interview date precedes initial consent.",
                    forms=["informed_consent_run_sheet", spec.form])
            self._rule_order(
                "CONSENT-007", "chric_consent_date", "chrmri_entry_date",
                "MRI acquisition date precedes initial consent.",
                forms=["informed_consent_run_sheet", "mri_run_sheet"])

        # Every form with an explicit interview/acquisition and entry date.
        by_form: dict[str, dict[str, str]] = defaultdict(dict)
        for variable, spec in self.catalog.fields.items():
            if variable not in self._raw:
                continue
            if "interview_date" in variable or variable.endswith("_acquisition_date"):
                by_form[spec.form]["interview"] = variable
            if variable.endswith("_entry_date") or variable == "chr_statusform_dateofentry":
                by_form[spec.form]["entry"] = variable
        for form, pair in by_form.items():
            if set(pair) == {"interview", "entry"}:
                self._rule_order(
                    "CONSENT-005", pair["interview"], pair["entry"],
                    "Data-entry date precedes interview/acquisition date.", forms=[form])

        if self._has_columns("chr_statusform_screenfail", "chr_statusform_earlywithdrawal"):
            conflict = self._raw["chr_statusform_screenfail"].map(_is_yes) & self._raw[
                "chr_statusform_earlywithdrawal"].map(_is_yes)
            self._emit_mask(
                conflict, "STATUS-001",
                ["chr_statusform_screenfail", "chr_statusform_earlywithdrawal"],
                "Screen failure and early withdrawal are both Yes.",
                forms=["revised_status_form"])
        self._rule_forbids(
            "STATUS-003", "chr_statusform_reentry", [2],
            ["chr_statusform_reentry_date"],
            "Re-entry No cannot retain a re-entry date.", forms=["revised_status_form"])
        self._rule_order(
            "STATUS-002", "chric_consent_date", "chr_subject_date_of_withdrawal",
            "Withdrawal date precedes consent/enrollment.",
            forms=["informed_consent_run_sheet", "revised_status_form"])
        self._rule_order(
            "STATUS-003", "chr_subject_date_of_withdrawal", "chr_statusform_reentry_date",
            "Re-entry date precedes withdrawal date.", forms=["revised_status_form"])
        self._rule_order(
            "STATUS-003", "statusform_withdrawal", "statusform_reentrydate1",
            "Legacy re-entry date precedes governed withdrawal date.", forms=["status_form"])

        # Do not gate an entire downstream modality form from the initial
        # consent alone.  Reconsent can legitimately change permission, and
        # return/termination records remain valid after a decline.  Only
        # dated, effective-consent acquisition predicates are safe enough for
        # participant output; those are handled by their specific workflows.

        # Cohort-specific GUID sex source and demographic agreement.
        if self._has_columns("chrguid_sex_chr", "chrguid_sex_hc"):
            chr_observed = self._observed_mask("chrguid_sex_chr")
            hc_observed = self._observed_mask("chrguid_sex_hc")
            invalid = pd.Series(False, index=self._raw.index, dtype=bool)
            for position, subject in enumerate(self._raw["subjectid"]):
                cohort = self._cohort_for_subject(subject)
                invalid.iat[position] = (
                    (cohort == "chr" and hc_observed.iat[position])
                    or (cohort == "hc" and chr_observed.iat[position]))
            self._emit_mask(
                invalid, "IDENT-001", ["chrguid_sex_chr", "chrguid_sex_hc"],
                "The cohort-inappropriate GUID sex field is populated.",
                forms=["guid_form"])
            if "chrdemo_sexassigned" in self._raw:
                mismatch = pd.Series(False, index=self._raw.index, dtype=bool)
                for position, subject in enumerate(self._raw["subjectid"]):
                    source = "chrguid_sex_chr" if self._cohort_for_subject(subject) == "chr" else "chrguid_sex_hc"
                    mismatch.iat[position] = (
                        self._observed_mask(source).iat[position]
                        and self._observed_mask("chrdemo_sexassigned").iat[position]
                        and not _codes_equal(
                            self._raw[source].iat[position],
                            self._raw["chrdemo_sexassigned"].iat[position]))
                self._emit_mask(
                    mismatch, "IDENT-002",
                    ["chrguid_sex_chr", "chrguid_sex_hc", "chrdemo_sexassigned"],
                    "Cohort-specific GUID sex disagrees with demographic sex assigned at birth.",
                    forms=["guid_form", "sociodemographics"])

        if self._has_columns("chrguid_guid_pseudo", "chrguid_guid", "chrguid_pseudoguid"):
            real = self._observed_mask("chrguid_guid")
            pseudo = self._observed_mask("chrguid_pseudoguid")
            invalid = (
                (self._code_mask("chrguid_guid_pseudo", [1]) & pseudo)
                | (self._code_mask("chrguid_guid_pseudo", [2]) & real))
            self._emit_mask(
                invalid, "IDENT-004",
                ["chrguid_guid_pseudo", "chrguid_guid", "chrguid_pseudoguid"],
                "Real/pseudo GUID selection does not match which identifier is populated.",
                forms=["guid_form"])

        if self._has_columns("chrguid_middlename1", "chrguid_middlename"):
            mismatch = (
                self._code_mask("chrguid_middlename1", [0])
                & self._observed_mask("chrguid_middlename"))
            self._emit_mask(
                mismatch, "IDENT-008",
                ["chrguid_middlename1", "chrguid_middlename"],
                "Middle-name Yes/No does not agree with middle-name presence.", forms=["guid_form"])

        if "chrdemo_sibling_id" in self._raw:
            self_id = self._raw["subjectid"].astype(str).str.strip()
            sibling = self._raw["chrdemo_sibling_id"].astype(str).str.strip()
            self._emit_mask(
                self._observed_mask("chrdemo_sibling_id") & sibling.eq(self_id),
                "IDENT-006", ["chrdemo_sibling_id", "subjectid"],
                "Sibling foreign key equals the participant's own ID.",
                forms=["sociodemographics"])
            if "chrdemo_sibling" in self._raw:
                roster = {_text(value).casefold() for value in self.subject_info}
                unresolved = (
                    self._code_mask("chrdemo_sibling", [2, 3, 4, 5, 6])
                    & self._observed_mask("chrdemo_sibling_id")
                    & ~sibling.str.casefold().isin(roster))
                self._emit_mask(
                    unresolved, "IDENT-011",
                    ["chrdemo_sibling", "chrdemo_sibling_id"],
                    "Declared study-sibling ID does not resolve to another participant in the run roster.",
                    forms=["sociodemographics"])

        self._run_guid_dob_reconstruction()

    def _run_guid_dob_reconstruction(self) -> None:
        """Validate GUID birth components against the governed DOB source.

        GUID month/day/year fields are separate numeric values rather than a
        REDCap date field, so the generic format engine cannot establish that
        they form a real calendar date or agree with the participant's DOB.
        """
        components = ("chrguid_mob", "chrguid_dob", "chrguid_yob")
        if not self._has_columns(*components):
            return

        month = self._numeric_series("chrguid_mob")
        day = self._numeric_series("chrguid_dob")
        year = self._numeric_series("chrguid_yob")
        complete = month.notna() & day.notna() & year.notna()
        reconstructed = pd.Series(pd.NaT, index=self._raw.index, dtype="datetime64[ns]")
        invalid = pd.Series(False, index=self._raw.index, dtype=bool)
        for position, values in enumerate(zip(year, month, day)):
            if not complete.iat[position]:
                continue
            try:
                # Decimal-looking values such as 2.0 are permitted only when
                # they are integral, consistent with the dictionary fields.
                parts = tuple(int(value) for value in values)
                if any(float(value) != integer for value, integer in zip(values, parts)):
                    invalid.iat[position] = True
                    continue
                reconstructed.iat[position] = pd.Timestamp(date(*parts))
            except (TypeError, ValueError, OverflowError):
                invalid.iat[position] = True
        self._emit_mask(
            invalid, "IDENT-003", list(components),
            "Supplied GUID month/day/year components do not form a real DOB.",
            forms=["guid_form"])

        governed_dob = next((variable for variable in ("chrdemo_dob", "chrcrit_dob")
                             if variable in self._raw), None)
        if governed_dob is None:
            return
        source = self._parsed_series(governed_dob)
        mismatch = complete & reconstructed.notna() & source.notna() & reconstructed.ne(source)
        self._emit_mask(
            mismatch, "IDENT-003", [*components, governed_dob],
            "GUID month/day/year components disagree with the governed date of birth.",
            forms=["guid_form", self.catalog.fields[governed_dob].form])

    def _run_eligibility_conversion_checks(self) -> None:
        """Reconcile cohort eligibility and the conversion evidence chain."""
        if "chrcrit_part" in self._raw:
            mismatch = pd.Series(False, index=self._raw.index, dtype=bool)
            for position, subject in enumerate(self._raw["subjectid"]):
                if not self._observed_mask("chrcrit_part").iat[position]:
                    continue
                cohort = self._cohort_for_subject(subject)
                code = _canonical_code(self._raw["chrcrit_part"].iat[position])
                mismatch.iat[position] = bool(
                    code and ((cohort == "chr" and code != "1")
                              or (cohort == "hc" and code != "2")))
            self._emit_mask(
                mismatch, "ELIG-001", ["chrcrit_part"],
                "Eligibility participant type disagrees with the exported cohort/arm.",
                forms=["inclusionexclusion_criteria_review"])

        dob_variable = next((variable for variable in (
            "chrdemo_dob", "chrcrit_dob", "chrguid_dob")
            if variable in self._raw), None)
        if self._has_columns("chrcrit_part", "chrcrit_inc1"):
            age = pd.Series(float("nan"), index=self._raw.index, dtype=float)
            age_variables: list[str] = []
            if dob_variable and "chrcrit_date" in self._raw:
                dob = self._parsed_series(dob_variable)
                criterion_date = self._parsed_series("chrcrit_date")
                age = pd.Series([
                    (end.year - start.year
                     - ((end.month, end.day) < (start.month, start.day)))
                    if pd.notna(start) and pd.notna(end) else float("nan")
                    for start, end in zip(dob, criterion_date)],
                    index=self._raw.index, dtype=float)
                age_variables.extend((dob_variable, "chrcrit_date"))
            for fallback in (
                    "chrcrit_age", "screening_age",
                    "chrdemo_age_yrs_chr", "chrdemo_age_yrs_hc"):
                if fallback in self._raw:
                    age = age.fillna(self._numeric_series(fallback))
                    age_variables.append(fallback)
            eligible_age = age.between(12, 30)
            applicable = (
                age.notna() & self._code_mask("chrcrit_part", [1, 2])
                & self._valid_operand_mask("chrcrit_inc1"))
            self._emit_mask(
                applicable & self._code_mask("chrcrit_inc1", [1]).ne(eligible_age),
                "ELIG-008",
                [*age_variables, "chrcrit_part", "chrcrit_inc1"],
                "Eligibility age criterion disagrees with completed age 12-30 on the criterion date.",
                forms=["sociodemographics", "inclusionexclusion_criteria_review"])
            included = pd.Series([
                _text(self.subject_info.get(subject, {}).get(
                    "inclusion_status", "")).casefold() == "included"
                for subject in self._raw["subjectid"]], index=self._raw.index)
            self._emit_mask(
                included & age.notna() & ~eligible_age,
                "ELIG-009", age_variables or ["chrcrit_inc1"],
                "Included participant is outside the authored 12-30 eligibility-age range.",
                forms=["inclusionexclusion_criteria_review"])

        if self._has_columns("chrcrit_included", "chrcrit_excluded"):
            known = self._observed_mask("chrcrit_included") & self._observed_mask(
                "chrcrit_excluded")
            included = self._code_mask("chrcrit_included", [1])
            excluded = self._code_mask("chrcrit_excluded", [1])
            self._emit_mask(
                known & included.eq(excluded), "ELIG-002",
                ["chrcrit_included", "chrcrit_excluded"],
                "Included and excluded calculations are not complementary.",
                forms=["inclusionexclusion_criteria_review"])

        if self._has_columns("chrschizotypal_summ", "chrcrit_excl6"):
            self._emit_mask(
                self._code_mask("chrschizotypal_summ", [1])
                & self._observed_mask("chrcrit_excl6")
                & ~self._code_mask("chrcrit_excl6", [1]),
                "ELIG-003", ["chrschizotypal_summ", "chrcrit_excl6"],
                "Schizotypal criteria are met but the Cluster A exclusion is not Yes.",
                forms=["scid5_schizotypal_personality_sciddpq",
                       "inclusionexclusion_criteria_review"])

        if self._has_columns("chrap_total", "chrcrit_excl1"):
            total = self._numeric_series("chrap_total")
            observed = total.notna() & self._observed_mask("chrcrit_excl1")
            wrong = observed & (
                (total.gt(50) & ~self._code_mask("chrcrit_excl1", [1]))
                | (total.le(50) & ~self._code_mask("chrcrit_excl1", [0])))
            self._emit_mask(
                wrong, "ELIG-004", ["chrap_total", "chrcrit_excl1"],
                "Lifetime antipsychotic total and antipsychotic exclusion disagree.",
                forms=["lifetime_ap_exposure_screen",
                       "inclusionexclusion_criteria_review"])

        if self._has_columns("chrtbi_severe_inj", "chrcrit_excl4"):
            severe = self._numeric_series("chrtbi_severe_inj")
            known = severe.notna() & self._observed_mask("chrcrit_excl4")
            wrong = known & (
                (severe.ge(7) & ~self._code_mask("chrcrit_excl4", [1]))
                | (severe.lt(7) & ~self._code_mask("chrcrit_excl4", [0])))
            self._emit_mask(
                wrong, "ELIG-005", ["chrtbi_severe_inj", "chrcrit_excl4"],
                "TBI severe-injury score and CNS/TBI exclusion disagree.",
                forms=["traumatic_brain_injury_screen",
                       "inclusionexclusion_criteria_review"])

        if self._has_columns("chrcrit_part", "chrcrit_excl8"):
            verified_family_psychosis = pd.Series(
                False, index=self._raw.index, dtype=bool)
            roles = ["mother", "father"]
            roles.extend(f"sibling{number}" for number in range(1, 10))
            roles.extend(f"child{number}" for number in range(1, 5))
            source_variables: list[str] = []
            for role in roles:
                diagnosis = f"chrfigs_{role}_pdx"
                agreement = f"{diagnosis}_agree"
                if not self._has_columns(diagnosis, agreement):
                    continue
                source_variables.extend((diagnosis, agreement))
                for position in self._raw.index:
                    verified = self._verified_calculation_values.get(
                        (int(position), diagnosis))
                    if (_canonical_code(verified) in {"2", "3"}
                            and self._valid_operand_mask(agreement).iat[position]
                            and self._code_mask(agreement, [1]).iat[position]):
                        verified_family_psychosis.iat[position] = True
            self._emit_mask(
                self._code_mask("chrcrit_part", [2])
                & self._valid_operand_mask("chrcrit_excl8")
                & self._code_mask("chrcrit_excl8", [0])
                & verified_family_psychosis,
                "ELIG-010", ["chrcrit_excl8", *source_variables],
                "HC family-history exclusion is No despite an agreed, formula-verified probable/definite first-degree psychosis diagnosis.",
                forms=["family_interview_for_genetic_studies_figs",
                       "inclusionexclusion_criteria_review"])

        if "chrconv_conv" not in self._raw:
            return
        converted = self._code_mask("chrconv_conv", [1])
        self._rule_order(
            "CONV-002", "chric_consent_date", "chrconv_interview_date",
            "Conversion date precedes consent/screening.",
            forms=["informed_consent_run_sheet", "conversion_form"])
        for entry in ("chrconv_nopsychs_date", "chrconv_inst2"):
            if entry in self._raw:
                self._rule_order(
                    "CONV-002", "chrconv_interview_date", entry,
                    "Conversion date is later than its supporting/entry date.",
                    forms=["conversion_form"])

        hc = pd.Series(
            [self._cohort_for_subject(subject) == "hc"
             for subject in self._raw["subjectid"]], index=self._raw.index)
        self._emit_mask(
            converted & hc, "CONV-003", ["chrconv_conv"],
            "Healthy-control participant is marked converted; high-priority review is required.",
            forms=["conversion_form"], severity="Warning")

        if "chrconv_consensus_outcome" in self._raw:
            self._emit_mask(
                converted & self._code_mask("chrconv_consensus_outcome", [0]),
                "CONV-007", ["chrconv_conv", "chrconv_consensus_outcome"],
                "Participant is marked converted without a positive consensus outcome.",
                forms=["conversion_form"], severity="Warning")

    def _cssrs_maximum(
            self, prefix: str, item_stub: str, target: str,
            yes_code: Any, check_id: str, form: str) -> None:
        items = [f"{prefix}{item_stub}{number}" for number in range(1, 6)]
        if target not in self._raw or not all(item in self._raw for item in items):
            return
        expected = pd.Series(0, index=self._raw.index, dtype=int)
        known_items: list[pd.Series] = []
        for number, item in enumerate(items, 1):
            spec = self.catalog.fields.get(item)
            allowed = (
                spec.choice_codes - STUDY_MISSING_CODES
                if spec is not None and spec.choice_codes else frozenset())
            known = self._observed_mask(item, missing_codes_are_blank=True)
            if allowed:
                known &= self._raw[item].map(_canonical_code).isin(allowed)
            known_items.append(known)
            expected.loc[self._code_mask(item, [yes_code])] = number
        observed = pd.to_numeric(self._raw[target], errors="coerce")
        any_yes = expected.gt(0)
        all_known = pd.concat(known_items, axis=1).all(axis=1)
        target_observed = self._observed_mask(target, missing_codes_are_blank=True)
        mismatch = all_known & target_observed & (
            (any_yes & observed.ne(expected)) | ~any_yes)
        self._emit_mask(
            mismatch, check_id, [*items, target],
            "C-SSRS maximum ideation severity does not equal the highest endorsed item (or is retained when none are endorsed).",
            forms=[form])

    def _cssrs_behavior_pair(
            self, controller: str, count: str, *, yes: Any,
            check_id: str, form: str) -> None:
        if not self._has_columns(controller, count):
            return
        numeric = pd.to_numeric(self._raw[count], errors="coerce")
        count_numeric = self._observed_mask(count) & numeric.notna()
        controller_yes = self._code_mask(controller, [yes])
        controller_known = self._observed_mask(controller)
        wrong = (
            (controller_yes & count_numeric & numeric.le(0))
            | (controller_known & ~controller_yes & count_numeric & numeric.ne(0))
        )
        self._emit_mask(
            wrong, check_id, [controller, count],
            "C-SSRS behavior Yes/No does not agree with its attempt/event count.",
            forms=[form])

    def _run_cssrs_psychs_checks(self) -> None:
        self._cssrs_maximum(
            "chrcssrsb_", "si", "chrcssrsb_idintsvl", 1,
            "CSSRS-001", "cssrs_baseline")
        self._cssrs_maximum(
            "chrcssrsb_", "css_sim", "chrcssrsb_css_sipmms", 1,
            "CSSRS-001", "cssrs_baseline")
        self._cssrs_maximum(
            "chrcssrsfu_", "css_sim", "chrcssrsfu_css_sipmms", 1,
            "CSSRS-001", "cssrs_followup")

        # Recent endorsement is a subset of lifetime endorsement. The exported
        # baseline lifetime fields use Yes=1/No=2; recent fields use Yes=1/No=0.
        for number in range(1, 6):
            lifetime = f"chrcssrsb_si{number}l"
            recent = f"chrcssrsb_css_sim{number}"
            if self._has_columns(lifetime, recent):
                self._emit_mask(
                    self._observed_mask(lifetime)
                    & self._code_mask(recent, [1])
                    & ~self._code_mask(lifetime, [1]),
                    "CSSRS-002", [lifetime, recent],
                    "Recent suicidal ideation is Yes while the corresponding lifetime item is not Yes.",
                    forms=["cssrs_baseline"])

        behavior_windows = [
            ("chrcssrsb_sb1l", "chrcssrsb_cssrs_actual"),
            ("chrcssrsb_sbnssibl", "chrcssrsb_cssrs_nssi"),
            ("chrcssrsb_sb3l", "chrcssrsb_cssrs_yrs_ia"),
            ("chrcssrsb_sb4l", "chrcssrsb_cssrs_yrs_aa"),
            ("chrcssrsb_sb5l", "chrcssrsb_cssrs_yrs_pab"),
            ("chrcssrsb_sb6l", "chrcssrsb_cssrs_yrs_sb"),
        ]
        for lifetime, recent in behavior_windows:
            if self._has_columns(lifetime, recent):
                self._emit_mask(
                    self._observed_mask(lifetime)
                    & self._code_mask(recent, [1])
                    & ~self._code_mask(lifetime, [1]),
                    "CSSRS-002", [lifetime, recent],
                    "Recent suicidal behavior is Yes while the corresponding lifetime behavior is not Yes.",
                    forms=["cssrs_baseline"])

        baseline_pairs = [
            ("chrcssrsb_sb1l", "chrcssrsb_snmacatl", 1),
            ("chrcssrsb_sb3l", "chrcssrsb_nminatl", 1),
            ("chrcssrsb_sb4l", "chrcssrsb_nmabatl", 1),
            ("chrcssrsb_sb5l", "chrcssrsb_nmapab", 1),
            ("chrcssrsb_cssrs_actual", "chrcssrsb_cssrs_num_attempt", 1),
            ("chrcssrsb_cssrs_yrs_ia", "chrcssrsb_cssrs_yrs_nia", 1),
            ("chrcssrsb_cssrs_yrs_aa", "chrcssrsb_cssrs_yrs_naa", 1),
            ("chrcssrsb_cssrs_yrs_pab", "chrcssrsb_cssrs_yrs_npab", 1),
        ]
        follow_pairs = [
            ("chrcssrsfu_cssrs_actual", "chrcssrsfu_cssrs_num_attempt", 1),
            ("chrcssrsfu_cssrs_yrs_ia", "chrcssrsfu_cssrs_yrs_nia", 1),
            ("chrcssrsfu_cssrs_yrs_aa", "chrcssrsfu_cssrs_yrs_naa", 1),
        ]
        for controller, count, yes in baseline_pairs:
            self._cssrs_behavior_pair(
                controller, count, yes=yes, check_id="CSSRS-003",
                form="cssrs_baseline")
        for controller, count, yes in follow_pairs:
            self._cssrs_behavior_pair(
                controller, count, yes=yes, check_id="CSSRS-003",
                form="cssrs_followup")

        lifetime_recent_counts = [
            ("chrcssrsb_snmacatl", "chrcssrsb_cssrs_num_attempt"),
            ("chrcssrsb_nminatl", "chrcssrsb_cssrs_yrs_nia"),
            ("chrcssrsb_nmabatl", "chrcssrsb_cssrs_yrs_naa"),
            ("chrcssrsb_nmapab", "chrcssrsb_cssrs_yrs_npab"),
        ]
        for lifetime, recent in lifetime_recent_counts:
            if not self._has_columns(lifetime, recent):
                continue
            lifetime_value, lifetime_valid = self._clinical_numeric(lifetime)
            recent_value, recent_valid = self._clinical_numeric(recent)
            self._emit_mask(
                lifetime_valid & recent_valid
                & lifetime_value.lt(recent_value),
                "CSSRS-006", [lifetime, recent],
                "Lifetime suicidal-behavior count is lower than the recent-window count.",
                forms=["cssrs_baseline"])

        for prefix, form in (("chrcssrsb_", "cssrs_baseline"),
                             ("chrcssrsfu_", "cssrs_followup")):
            actual = f"{prefix}cssrs_actual"
            lethality = [
                f"{prefix}{suffix}" for suffix in (
                    "cssrs_mlmlllt1", "cssrs_mlmrllt1", "actlthl3",
                    "cssrs_plmrlt1", "cssrs_plmllt1", "potlthl3")
                if f"{prefix}{suffix}" in self._raw]
            if actual in self._raw and lethality:
                lethality_observed = pd.concat(
                    [self._observed_mask(field) for field in lethality], axis=1
                ).any(axis=1)
                self._emit_mask(
                    self._code_mask(actual, [0]) & lethality_observed,
                    "CSSRS-007", [actual, *lethality],
                    "C-SSRS lethality data are present without an actual attempt.",
                    forms=[form])

            most_lethal = f"{prefix}cssrs_mlmlllt1"
            most_recent = f"{prefix}cssrs_mlmrllt1"
            first_attempt = f"{prefix}actlthl3"
            actual_ratings = [most_lethal, most_recent, first_attempt]
            if self._has_columns(*actual_ratings):
                values = [self._clinical_numeric(field) for field in actual_ratings]
                all_valid = pd.concat(
                    [valid for _, valid in values], axis=1).all(axis=1)
                contradiction = (
                    values[0][0].lt(values[1][0])
                    | values[0][0].lt(values[2][0]))
                self._emit_mask(
                    all_valid & contradiction, "CSSRS-008", actual_ratings,
                    "C-SSRS most-lethal actual lethality is lower than the most-recent or first-attempt rating.",
                    forms=[form])

            count = f"{prefix}cssrs_num_attempt"
            potential_ratings = [
                f"{prefix}cssrs_plmllt1", f"{prefix}cssrs_plmrlt1",
                f"{prefix}potlthl3"]
            if count in self._raw:
                one_attempt = self._code_mask(count, [1])
                for ratings, label in (
                        (actual_ratings, "actual"),
                        (potential_ratings, "potential")):
                    if not self._has_columns(*ratings):
                        continue
                    parsed = [self._clinical_numeric(field) for field in ratings]
                    valid = pd.concat(
                        [item[1] for item in parsed], axis=1).all(axis=1)
                    unequal = ~(
                        parsed[0][0].eq(parsed[1][0])
                        & parsed[0][0].eq(parsed[2][0]))
                    self._emit_mask(
                        one_attempt & valid & unequal, "CSSRS-009",
                        [count, *ratings],
                        f"C-SSRS reports one attempt but its first, most-recent, and most-lethal {label} lethality ratings differ.",
                        forms=[form])

        if self._has_columns(
                "chrcssrsfu_skip_aa___0", "chrcssrsfu_cssrs_actual"):
            self._emit_mask(
                self._valid_operand_mask("chrcssrsfu_skip_aa___0")
                & self._raw["chrcssrsfu_skip_aa___0"].map(_is_yes)
                & self._code_mask("chrcssrsfu_cssrs_actual", [1]),
                "CSSRS-010",
                ["chrcssrsfu_skip_aa___0", "chrcssrsfu_cssrs_actual"],
                "C-SSRS actual-attempt section is marked not applicable while an actual attempt is endorsed.",
                forms=["cssrs_followup"])

        baseline_components = [
            "chrcssrsb_sb1l", "chrcssrsb_sb3l", "chrcssrsb_sb4l", "chrcssrsb_sb5l"]
        if self._has_columns("chrcssrsb_sb6l", *baseline_components):
            expected = pd.concat(
                [self._code_mask(field, [1]) for field in baseline_components], axis=1
            ).any(axis=1)
            known = pd.concat(
                [self._observed_mask(field) for field in baseline_components], axis=1
            ).all(axis=1) & self._observed_mask("chrcssrsb_sb6l")
            stored = self._code_mask("chrcssrsb_sb6l", [1])
            self._emit_mask(
                known & stored.ne(expected), "CSSRS-004",
                ["chrcssrsb_sb6l", *baseline_components],
                "Lifetime suicidal-behavior aggregate does not equal the OR of actual, interrupted, aborted, and preparatory behavior.",
                forms=["cssrs_baseline"])

        for prefix, form in (("chrcssrsb_", "cssrs_baseline"),
                             ("chrcssrsfu_", "cssrs_followup")):
            aggregate = f"{prefix}cssrs_yrs_sb"
            components = [
                f"{prefix}cssrs_actual", f"{prefix}cssrs_yrs_ia",
                f"{prefix}cssrs_yrs_aa", f"{prefix}cssrs_yrs_pab"]
            if aggregate not in self._raw:
                continue
            existing = [field for field in components if field in self._raw]
            if len(existing) < 3:
                continue
            expected = pd.concat(
                [self._code_mask(field, [1]) for field in existing], axis=1).any(axis=1)
            all_known = pd.concat(
                [self._observed_mask(field) for field in existing], axis=1).all(axis=1)
            stored = self._code_mask(aggregate, [1])
            if form == "cssrs_followup":
                violation = all_known & self._observed_mask(aggregate) & stored.ne(expected)
            else:
                # Baseline component window is 3 months but aggregate is 1 year:
                # only component Yes -> aggregate Yes is deterministic.
                violation = (
                    all_known & self._observed_mask(aggregate)
                    & expected & ~stored)
            self._emit_mask(
                violation, "CSSRS-005", [aggregate, *existing],
                "C-SSRS behavior aggregate is inconsistent with its component behaviors.",
                forms=[form])

        # PSYCHS severity-window ordering, discovered from authored a0/b0/c0/d0
        # symptom families rather than a hand-maintained list.
        groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        for variable, spec in self.catalog.fields.items():
            if variable not in self._raw or not spec.form.startswith("psychs_"):
                continue
            match = re.match(r"^(.*?)([abcd])0$", variable)
            if match:
                groups[(spec.form, match.group(1))][match.group(2)] = variable
        for (form, _), family in groups.items():
            if (form in {"psychs_p1p8", "psychs_p9ac32"}
                    and {"a", "b", "d"}.issubset(family)):
                a, a_valid = self._clinical_numeric(family["a"])
                b, b_valid = self._clinical_numeric(family["b"])
                d, d_valid = self._clinical_numeric(family["d"])
                self._emit_mask(
                    a_valid & b_valid & a.lt(b),
                    "PSYCHS-001", [family["a"], family["b"]],
                    "PSYCHS screening lifetime severity is lower than past-year severity.",
                    forms=[form])
                self._emit_mask(
                    a_valid & d_valid & a.lt(d),
                    "PSYCHS-001", [family["a"], family["d"]],
                    "PSYCHS screening lifetime severity is lower than past-month severity.",
                    forms=[form])
                self._emit_mask(
                    b_valid & d_valid & b.lt(d),
                    "PSYCHS-001", [family["b"], family["d"]],
                    "PSYCHS screening past-year severity is lower than past-month severity.",
                    forms=[form])

        # Every clearly named onset/threshold/offset triplet receives exact
        # chronological checks when all members are present.
        for variable, spec in self.catalog.fields.items():
            if variable not in self._raw or not spec.form.startswith("psychs_"):
                continue
            lowered = variable.lower()
            if "onset" not in lowered and not lowered.endswith("_on"):
                continue
            candidates = {
                lowered.replace("onset", "offset"),
                lowered[:-3] + "off" if lowered.endswith("_on") else "",
            }
            offset = next((candidate for candidate in candidates if candidate in self._raw), None)
            if offset:
                self._rule_order(
                    "PSYCHS-002", variable, offset,
                    "PSYCHS symptom offset does not occur strictly after onset.",
                    strict=True, forms=[spec.form])
            interview = {
                "psychs_p1p8": "chrpsychs_scr_interview_date",
                "psychs_p9ac32": "chrpsychs_scr_interview_date",
                "psychs_p1p8_fu": "chrpsychs_fu_interview_date",
                "psychs_p9ac32_fu": "chrpsychs_fu_interview_date",
                "psychs_p1p8_fu_hc": "hcpsychs_fu_interview_date",
                "psychs_p9ac32_fu_hc": "hcpsychs_fu_interview_date",
            }.get(spec.form)
            if interview:
                if interview in self._raw:
                    self._rule_order(
                        "PSYCHS-003", variable, interview,
                        "PSYCHS onset is later than the interview date.", forms=[spec.form])

        self._run_reviewed_psychs_dictionary_alerts()
        self._run_psychs_followup_template_date_checks()
        self._run_psychs_global_reduction_checks()

    @staticmethod
    def _dictionary_alert_message(label: str) -> str:
        message = re.sub(r"<[^>]+>", " ", label)
        for encoded, plain in (
                ("&gt;", ">"), ("&lt;", "<"), ("&ge;", ">="),
                ("&le;", "<="), ("&amp;", "&"), ("&nbsp;", " ")):
            message = message.replace(encoded, plain)
        return re.sub(r"\s+", " ", message).strip()

    def _run_reviewed_psychs_dictionary_alerts(self) -> None:
        """Replay only audited, hard PSYCHS same-symptom predicates.

        Every operand must be substantive, valid, and visible under its own
        branch.  This deliberately excludes missingness prompts, approval
        reminders, and the unsafe follow-up c0/d0 warning whose validity
        depends on the actual visit interval (handled by PSYCHS-004).
        """

        prefixes = {
            "chrpsychs_scr_": (
                PSYCHS_SCREEN_VALUE_ALERT_SUFFIXES,
                PSYCHS_SCREEN_DATE_ALERT_SUFFIXES),
            "chrpsychs_fu_": (
                PSYCHS_FOLLOWUP_VALUE_ALERT_SUFFIXES,
                PSYCHS_FOLLOWUP_DATE_ALERT_SUFFIXES),
            "hcpsychs_fu_": (
                PSYCHS_FOLLOWUP_VALUE_ALERT_SUFFIXES,
                PSYCHS_FOLLOWUP_DATE_ALERT_SUFFIXES),
        }
        for spec in self.catalog.fields.values():
            if (not spec.is_descriptive or not spec.branching_logic
                    or spec.form not in self._active_forms):
                continue
            classification = None
            for prefix, (value_suffixes, date_suffixes) in prefixes.items():
                match = re.fullmatch(
                    re.escape(prefix) + r"\d+(.+)", spec.variable)
                if match is None:
                    continue
                suffix = match.group(1)
                if suffix in value_suffixes:
                    classification = "value"
                elif suffix in date_suffixes:
                    classification = "date"
                break
            if classification is None:
                continue
            references = self._branch_references(spec)
            if not references or any(
                    reference not in self._raw for reference in references):
                continue
            candidate = self._form_mask(spec.form, due=True)
            candidate = self._applicable_operand_mask(references, candidate)
            if not candidate.any():
                continue
            violation, state = (
                self._datediff_branch_state(spec, candidate)
                if "datediff" in spec.branching_logic.casefold()
                else self._branch_state(spec, candidate))
            if state != "known":
                self._emit_configuration(
                    "PSYCHS-DD-001", spec.variable,
                    f"Reviewed PSYCHS predicate could not be evaluated ({state}).",
                    form=spec.form)
                continue
            message = self._dictionary_alert_message(spec.label)
            self._emit_mask(
                candidate & violation,
                "PSYCHS-005" if classification == "value" else "PSYCHS-006",
                references,
                message or "PSYCHS answers violate an authored instrument rule.",
                forms=[spec.form])

    def _run_psychs_global_reduction_checks(self) -> None:
        """Validate the authored GRD E.3 30-percent/12-month definition."""

        for prefix, form in (
                ("chrpsychs_scr_", "psychs_p9ac32"),
                ("chrpsychs_fu_", "psychs_p9ac32_fu"),
                ("hcpsychs_fu_", "psychs_p9ac32_fu_hc")):
            controller = prefix + "e3"
            start, stop = prefix + "e3_start", prefix + "e3_stop"
            if self._has_columns(controller, start, stop):
                start_value, start_valid = self._clinical_numeric(start)
                stop_value, stop_valid = self._clinical_numeric(stop)
                applicable = (
                    self._form_mask(form, due=True)
                    & self._code_mask(controller, [1])
                    & start_valid & stop_valid & start_value.gt(0))
                self._emit_mask(
                    applicable & stop_value.div(start_value).gt(0.70),
                    "PSYCHS-007", [controller, start, stop],
                    "PSYCHS global-reduction scores do not demonstrate the authored decrease of at least 30%.",
                    forms=[form])

            start_date = prefix + "e3_start_date"
            stop_date = prefix + "e3_stop_date"
            if self._has_columns(controller, start_date, stop_date):
                start_value = self._parsed_series(start_date)
                stop_value = self._parsed_series(stop_date)
                applicable = (
                    self._form_mask(form, due=True)
                    & self._code_mask(controller, [1])
                    & start_value.notna() & stop_value.notna())
                elapsed = (stop_value.dt.normalize()
                           - start_value.dt.normalize()).dt.days
                self._emit_mask(
                    applicable & (elapsed.ne(365) | stop_value.le(start_value)),
                    "PSYCHS-007", [controller, start_date, stop_date],
                    "PSYCHS global-reduction dates are not a forward 365-day interval.",
                    forms=[form])

    def _run_psychs_followup_template_date_checks(self) -> None:
        """Apply documented P1-P8 date invariants to identical P9-P15 fields."""

        for prefix, form in (
                ("chrpsychs_fu_", "psychs_p9ac32_fu"),
                ("hcpsychs_fu_", "psychs_p9ac32_fu_hc")):
            interview_variable = prefix + "interview_date"
            if interview_variable not in self._raw:
                continue
            form_candidate = self._form_mask(form, due=True)
            interview = self._parsed_series(interview_variable)
            for number in range(9, 16):
                lower_bound_pairs = (
                    (f"{prefix}{number}d8_date", f"{prefix}{number}c1_on6"),
                    (f"{prefix}{number}d8_date", f"{prefix}{number}c10_on"),
                    (f"{prefix}{number}d9_date", f"{prefix}{number}c1_on6"),
                    (f"{prefix}{number}d9_date", f"{prefix}{number}c10_on"),
                    (f"{prefix}{number}d22_date", f"{prefix}{number}c1_on3"),
                    (f"{prefix}{number}d22_date", f"{prefix}{number}c14_on"),
                    (f"{prefix}{number}d23_date", f"{prefix}{number}c1_on3"),
                    (f"{prefix}{number}d23_date", f"{prefix}{number}c14_on"),
                )
                for current, lower in lower_bound_pairs:
                    if not self._has_columns(current, lower):
                        continue
                    applicable = self._applicable_operand_mask(
                        (current, lower), form_candidate)
                    current_date = self._parsed_series(current)
                    lower_date = self._parsed_series(lower)
                    self._emit_mask(
                        applicable & current_date.notna() & lower_date.notna()
                        & current_date.lt(lower_date),
                        "PSYCHS-006", [current, lower],
                        "PSYCHS current qualifying date precedes its governed symptom-history onset.",
                        forms=[form])

                current_dates = (
                    (f"{prefix}{number}d8_date", 90),
                    (f"{prefix}{number}d9_date", None),
                    (f"{prefix}{number}d22_date", 365),
                    (f"{prefix}{number}d23_date", None),
                )
                for current, maximum_age in current_dates:
                    if current not in self._raw:
                        continue
                    applicable = self._applicable_operand_mask(
                        (current, interview_variable), form_candidate)
                    current_date = self._parsed_series(current)
                    self._emit_mask(
                        applicable & current_date.notna() & interview.notna()
                        & current_date.gt(interview),
                        "PSYCHS-006", [current, interview_variable],
                        "PSYCHS current qualifying date is after the interview date.",
                        forms=[form])
                    if maximum_age is not None:
                        elapsed = (interview.dt.normalize()
                                   - current_date.dt.normalize()).dt.days
                        self._emit_mask(
                            applicable & current_date.notna() & interview.notna()
                            & elapsed.gt(maximum_age),
                            "PSYCHS-006", [current, interview_variable],
                            f"PSYCHS current qualifying date is more than {maximum_age} days before interview.",
                            forms=[form])

    def _run_scid_structured_manual_chains(self) -> None:
        """Check deterministic clinician-entered SCID diagnosis chains.

        These terminal fields are not REDCap calculations. Compare a populated
        target only when all immediate source criteria are known; blank,
        sentinel, or otherwise indeterminate values do not produce findings.
        """

        form = "scid5_psychosis_mood_substance_abuse"
        for target, components in SCID_MANUAL_CHAINS.items():
            if target not in self._raw or not all(component in self._raw for component in components):
                continue
            target_spec = self.catalog.fields.get(target)
            if target_spec is None:
                continue
            candidate = self._form_mask(form, due=False) | self._observed_mask(target)
            branch, branch_state = self._branch_state(target_spec, candidate)
            if branch_state != "known":
                self._emit_configuration(
                    "SCID-DX-006", target,
                    f"Structured SCID endpoint branch cannot be evaluated ({branch_state}).",
                    form=form)
                continue

            target_observed = self._observed_mask(target, missing_codes_are_blank=True)
            self._emit_mask(
                candidate & self._branch_sources_known(target_spec)
                & ~branch & target_observed, "SCID-DX-004",
                [target, *components],
                "SCID structured diagnosis/episode field is not applicable but retains a value.",
                forms=[form], severity="Warning")

            applicable = candidate & branch
            component_known = []
            component_met = []
            for component in components:
                known = self._observed_mask(component, missing_codes_are_blank=True)
                component_known.append(known)
                component_met.append(self._code_mask(component, [3]))
            all_known = pd.concat(component_known, axis=1).all(axis=1)
            all_met = pd.concat(component_met, axis=1).all(axis=1)
            not_met = all_known & ~all_met
            self._emit_mask(
                applicable & target_observed & not_met, "SCID-DX-002",
                [target, *components],
                "Structured SCID diagnosis/episode is present but its criteria are definitively Not met.",
                forms=[form])

    def _run_scid_schizotypal_checks(self) -> None:
        expected_endpoints = set(SCID_TERMINAL_ENDPOINTS) | set(SCID_SUD_COUNT_ENDPOINTS)
        missing = sorted(expected_endpoints - set(self.catalog.fields))
        uncompiled = sorted(
            variable for variable in expected_endpoints
            if variable in self.catalog.fields
            and self.catalog.calculations.get(variable, {}).get("status") != "converted")
        if missing:
            self._emit_configuration(
                "SCID-DX-006", missing,
                "Required SCID endpoint manifest entries are absent from the selected dictionary.")
        if uncompiled:
            self._emit_configuration(
                "SCID-DX-006", uncompiled,
                "Required SCID endpoint calculations could not be compiled into executable rules.")
        if "chrscid_cur_diagnosis" in self.catalog.fields:
            self._emit_configuration(
                "SCID-DX-006", ["chrscid_cur_diagnosis"],
                "Clinician-entered final SCID diagnosis requires an approved diagnosis-to-criteria/override mapping before deterministic comparison.")
        self._run_scid_structured_manual_chains()
        self._run_scid_symptom_consistency_checks()

        missing_schizotypal = sorted(
            variable for variable in SCHIZOTYPAL_CALC_ENDPOINTS
            if variable not in self.catalog.fields
            or self.catalog.calculations.get(variable, {}).get("status") != "converted")
        if missing_schizotypal:
            self._emit_configuration(
                "SCID-SZT-002", missing_schizotypal,
                "Schizotypal diagnostic calculation cannot be recomputed from the "
                "selected dictionary.", form="scid5_schizotypal_personality_sciddpq")

        scid_form = "scid5_psychosis_mood_substance_abuse"
        interview_var = (
            "chrscid_interview_date" if "chrscid_interview_date" in self._raw
            else next((variable for variable, spec in self.catalog.fields.items()
                       if variable in self._raw and spec.form == scid_form
                       and spec.is_temporal and "interview_date" in variable), None))
        dob_var = next((variable for variable in (
            "chrdemo_dob", "chrcrit_dob", "chrguid_full_dob", "chrguid_dob")
            if variable in self._raw), None)
        participant_age = None
        if interview_var and dob_var:
            interview_date = self._parsed_series(interview_var)
            dob = self._parsed_series(dob_var)
            participant_age = pd.Series([
                ((end.year - start.year) - ((end.month, end.day) < (start.month, start.day)))
                if pd.notna(start) and pd.notna(end) else float("nan")
                for start, end in zip(dob, interview_date)], index=self._raw.index)
        if participant_age is None or not participant_age.notna().any():
            fallback_age = self._participant_age(interview_var)
            if fallback_age.notna().any():
                participant_age = fallback_age

        onset_age_variables = [
            "chrscid_c56_c65", "chrscid_c58_c66", "chrscid_a52", "chrscid_a109",
            "chrscid_d45_d49_d54_d56", "chrscid_d61_d65", "chrscid_e19_37",
            *[f"chrscid_e{number}" for number in
              (164, 168, 172, 176, 180, 184, 188, 192)],
        ]
        for variable in onset_age_variables:
            if variable not in self._raw:
                continue
            value = pd.to_numeric(self._raw[variable], errors="coerce")
            spec = self.catalog.fields.get(variable)
            declaration = " ".join((
                " ".join(" ".join(choice) for choice in spec.choices) if spec else "",
                spec.note if spec else "",
                spec.annotation if spec else ""))
            declared_99_unknown = bool(re.search(r"(?<!\d)99(?!\d)", declaration))
            valid = self._observed_mask(variable, missing_codes_are_blank=True) & value.notna()
            if declared_99_unknown:
                valid &= value.ne(99)
            violation = valid & value.lt(0)
            if participant_age is not None:
                violation |= valid & participant_age.notna() & value.gt(participant_age)
            self._emit_mask(
                violation, "SCID-AGE-001", [variable],
                "SCID onset age is negative or exceeds age at the SCID interview.",
                forms=["scid5_psychosis_mood_substance_abuse"])

        interview_year = (
            self._parsed_series(interview_var).dt.year if interview_var else None)
        birth_year = None
        if dob_var:
            birth_year = self._parsed_series(dob_var).dt.year
        for variable in ("chrguid_yob", "chrdemo_yob", "chrcrit_yob"):
            if birth_year is not None:
                break
            if variable in self._raw:
                birth_year = self._numeric_series(variable)
                break
        last_met_onset = {
            300: "chrscid_e164_302", 304: "chrscid_e168_306",
            308: "chrscid_e172_310", 312: "chrscid_e176_314",
            316: "chrscid_e180_318", 320: "chrscid_e184_322",
            324: "chrscid_e188_326", 328: "chrscid_e192_330",
        }
        for number in (300, 304, 308, 312, 316, 320, 324, 328):
            variable = f"chrscid_e{number}"
            if variable not in self._raw:
                continue
            year = self._numeric_series(variable)
            violation = pd.Series(False, index=self._raw.index, dtype=bool)
            if interview_year is not None:
                violation |= year.notna() & interview_year.notna() & year.gt(interview_year)
            if birth_year is not None:
                violation |= year.notna() & birth_year.notna() & year.lt(birth_year)
            self._emit_mask(
                violation, "SCID-YEAR-001", [variable],
                "SCID year-last-met is after interview year or before birth year.",
                forms=["scid5_psychosis_mood_substance_abuse"])
            onset_var = last_met_onset[number]
            if onset_var in self._raw and birth_year is not None:
                onset_age, onset_valid = self._clinical_numeric(onset_var)
                inferred_onset_year = birth_year + onset_age
                self._emit_mask(
                    year.notna() & onset_valid & birth_year.notna()
                    & year.lt(inferred_onset_year),
                    "SCID-YEAR-002", [variable, onset_var],
                    "SCID year-last-met precedes the birth-year-plus-onset-age estimate.",
                    forms=["scid5_psychosis_mood_substance_abuse"], severity="Warning")

    def _run_scid_symptom_consistency_checks(self) -> None:
        """Run exact SCID within-module symptom and overview invariants."""

        form = "scid5_psychosis_mood_substance_abuse"
        form_candidate = self._form_mask(form, due=True)

        # Each reviewed parent is an ordinal assertion that one of its binary
        # direction/subtype children occurred.  All children must be known and
        # branch-visible before an all-absent contradiction is possible.
        for parent, children in SCID_BINARY_SUBTYPE_GROUPS.items():
            variables = (parent, *children)
            if not self._has_columns(*variables):
                continue
            candidate = self._applicable_operand_mask(variables, form_candidate)
            all_absent = pd.concat(
                [self._code_mask(child, [0]) for child in children], axis=1
            ).all(axis=1)
            self._emit_mask(
                candidate & self._code_mask(parent, [2, 3]) & all_absent,
                "SCID-SYM-001", variables,
                "SCID symptom is rated subthreshold/threshold but every governed subtype is absent.",
                forms=[form])

        if self._has_columns("chrscid_a54", "chrscid_a55", "chrscid_a56"):
            variables = ("chrscid_a54", "chrscid_a55", "chrscid_a56")
            candidate = self._applicable_operand_mask(variables, form_candidate)
            self._emit_mask(
                candidate & self._code_mask("chrscid_a54", [2, 3])
                & self._code_mask("chrscid_a55", [1])
                & self._code_mask("chrscid_a56", [1]),
                "SCID-SYM-002", variables,
                "SCID elevated/expansive mood and irritable mood cannot both be marked present.",
                forms=[form])

        overview_controllers: list[str] = []
        for base in SCID_SUBSTANCE_RATING_BASES:
            spec = self.catalog.fields.get(base)
            columns = [f"{base}___{number}" for number in range(1, 5)]
            if spec is None or not self._has_columns(*columns):
                continue
            references = self._branch_references(spec)
            controller = next((reference for reference in references
                               if reference in self._raw), None)
            if controller is None:
                continue
            overview_controllers.append(controller)
            variables = (controller, *columns)
            candidate = self._applicable_operand_mask(variables, form_candidate)
            lifetime_absent = self._raw[columns[0]].map(_is_yes)
            lifetime_threshold = self._raw[columns[1]].map(_is_yes)
            year_absent = self._raw[columns[2]].map(_is_yes)
            year_threshold = self._raw[columns[3]].map(_is_yes)
            contradiction = (
                (lifetime_absent & lifetime_threshold)
                | (year_absent & year_threshold)
                | (lifetime_absent & year_threshold))
            self._emit_mask(
                candidate & self._code_mask(controller, [1]) & contradiction,
                "SCID-SUD-001", variables,
                "SCID substance overview contains mutually incompatible lifetime/past-year ratings.",
                forms=[form])

        no_lifetime = "chrscid_no_drug_lifetime___1"
        if no_lifetime in self._raw and overview_controllers:
            candidate = self._valid_operand_mask(no_lifetime)
            any_use = pd.Series(False, index=self._raw.index, dtype=bool)
            for controller in overview_controllers:
                any_use |= (
                    self._valid_operand_mask(controller)
                    & self._code_mask(controller, [1]))
            self._emit_mask(
                form_candidate & candidate
                & self._raw[no_lifetime].map(_is_yes) & any_use,
                "SCID-SUD-002", [no_lifetime, *overview_controllers],
                "SCID no-lifetime-drug-use selection conflicts with a positive substance overview.",
                forms=[form])

        for variable in ("chrscid_a26_53", "chrscid_a109_episodes", "chrscid_c57"):
            if variable not in self._raw:
                continue
            value, valid = self._clinical_numeric(variable)
            self._emit_mask(
                form_candidate & valid & (value.lt(0) | value.mod(1).ne(0)),
                "SCID-RANGE-001", [variable],
                "SCID episode count must be a nonnegative integer.",
                forms=[form])
        for variable in (
                "chrscid_c64", "chrscid_d43", "chrscid_d59", "chrscid_e16",
                "chrscid_e163", "chrscid_e167", "chrscid_e171", "chrscid_e175",
                "chrscid_e179", "chrscid_e183", "chrscid_e187", "chrscid_e191"):
            if variable not in self._raw:
                continue
            value, valid = self._clinical_numeric(variable)
            self._emit_mask(
                form_candidate & valid & value.lt(0), "SCID-RANGE-002",
                [variable], "SCID months-prior value cannot be negative.",
                forms=[form])

    def _slot_mask(self, form: str, stem: str) -> pd.Series:
        variables = [
            variable for variable, spec in self.catalog.fields.items()
            if spec.form == form and variable.startswith(stem)
            and not spec.is_descriptive and not spec.is_calculation]
        return self._any_observed(variables)

    def _episode_family(
            self, *, form: str, root: str | None, stem: str,
            slots: int, name_suffixes: Sequence[str],
            onset_suffixes: Sequence[str], offset_suffixes: Sequence[str],
            add_template: str | None, check_prefix: str) -> None:
        slot_masks: list[pd.Series] = []
        for number in range(1, slots + 1):
            slot_stem = stem.format(n=number)
            slot_masks.append(self._slot_mask(form, slot_stem))

        if root and root in self._raw and slot_masks:
            self._emit_mask(
                self._code_mask(root, [0, 2]) & pd.concat(slot_masks, axis=1).any(axis=1),
                f"{check_prefix}-001", [root],
                "Episode-family root is No but one or more episode slots contain data.",
                forms=[form])

        for number, current in enumerate(slot_masks, 1):
            slot_stem = stem.format(n=number)
            if add_template and number < slots:
                add = add_template.format(n=number)
                if add in self._raw:
                    next_mask = slot_masks[number]
                    self._emit_mask(
                        next_mask & self._observed_mask(add)
                        & ~self._code_mask(add, [1]),
                        f"{check_prefix}-003", [add],
                        f"Episode slot {number + 1} contains data but slot {number}'s add gate is not Yes.",
                        forms=[form])
            onset = next((slot_stem + suffix for suffix in onset_suffixes
                          if slot_stem + suffix in self._raw), None)
            offset = next((slot_stem + suffix for suffix in offset_suffixes
                           if slot_stem + suffix in self._raw), None)
            if onset and offset:
                self._rule_order(
                    f"{check_prefix}-005", onset, offset,
                    f"Episode slot {number} offset precedes onset.", forms=[form])

            # Stale offset dates remain contradictions. Blank offset dates are
            # intentionally outside Proposed Checks' current scope.
            status_fields = [
                variable for variable, spec in self.catalog.fields.items()
                if spec.form == form and variable.startswith(slot_stem)
                and (variable.endswith("_trans") or re.search(r"_mo\d+$", variable))]
            if offset and status_fields:
                for status in status_fields:
                    spec = self.catalog.fields[status]
                    continuing_codes = {
                        code for code, label in spec.choices
                        if any(word in label.lower() for word in ("present", "continuing", "ongoing"))}
                    if continuing_codes:
                        self._emit_mask(
                            self._code_mask(status, continuing_codes)
                            & self._observed_mask(offset),
                            f"{check_prefix}-006", [status, offset],
                            "Present/continuing episode state retains an offset date.",
                            forms=[form])

    def _run_episode_and_medication_checks(self) -> None:
        self._episode_family(
            form="health_conditions_medical_historypsychiatric_histo",
            root="chrmed_any", stem="chrmed_cond{n}", slots=15,
            name_suffixes=("_name",), onset_suffixes=("_onset",),
            offset_suffixes=("_offset",), add_template="chrmed_cond{n}_add",
            check_prefix="EP-MEDHX")
        self._episode_family(
            form="psychosocial_treatment_form", root=None,
            stem="chrpsychsoc_treat{n}", slots=15,
            name_suffixes=("_type",), onset_suffixes=("_onset",),
            offset_suffixes=("_offset",), add_template="chrpsychsoc_treat{n}_add",
            check_prefix="EP-PSYCHSOC")
        self._episode_family(
            form="resource_use_log", root="chrrul_any",
            stem="chrrul_resource{n}", slots=30,
            name_suffixes=("_code",), onset_suffixes=("_on",),
            offset_suffixes=("_off",), add_template="chrrul_resource{n}_add",
            check_prefix="EP-RESOURCE")

        # Adverse-event naming changes after early slots, so map each slot's
        # exact dictionary members while retaining one lifecycle engine.
        ae_variables: list[list[str]] = []
        for number in range(1, 31):
            variables = [
                variable for variable, spec in self.catalog.fields.items()
                if spec.form == "adverse_events"
                and (re.search(fr"(?:ae|diag|aes|tp|sig|sae|ssi|add){number}(?:_|$)", variable)
                     or variable.startswith(f"chr_ae{number}"))
                and not spec.is_descriptive]
            ae_variables.append(variables)
        ae_masks = [self._any_observed(variables) for variables in ae_variables]
        for number in range(1, 31):
            variables = ae_variables[number - 1]
            onset = next((field for field in (
                f"chrae_aes{number}date", f"chr_ae{number}date") if field in self._raw), None)
            offset = next((field for field in (
                f"chrae_aes{number}off", f"chr_ae{number}date_dr") if field in self._raw), None)
            if onset and offset:
                self._rule_order(
                    "EP-AE-005", onset, offset,
                    f"Adverse-event slot {number} resolution date precedes onset.",
                    forms=["adverse_events"])
            if number < 30:
                add = f"chrae_add{number}"
                if add in self._raw:
                    self._emit_mask(
                        ae_masks[number] & self._observed_mask(add)
                        & ~self._code_mask(add, [1]),
                        "EP-AE-003", [add],
                        f"Adverse-event slot {number + 1} contains data but the prior add gate is not Yes.",
                        forms=["adverse_events"])

        self._run_ae_lifecycle_extensions(ae_masks, ae_variables)

        # Medication course extensions not covered by the legacy checker:
        # chronology and exact regular/intermittent field exclusivity.
        for form, first, last, past in (
            ("current_pharmaceutical_treatment_floating_med_125", 1, 25, False),
            ("current_pharmaceutical_treatment_floating_med_2650", 26, 50, False),
            ("past_pharmaceutical_treatment", 1, 60, True),
        ):
            for number in range(first, last + 1):
                suffix = "_past" if past else ""
                base = f"chrpharm_med{number}"
                onset = f"{base}_onset{suffix}"
                offset = f"{base}_offset{suffix}"
                if self._has_columns(onset, offset):
                    self._rule_order(
                        "MED-DATE-001", onset, offset,
                        f"Medication slot {number} offset precedes onset.", forms=[form])
                use = f"{base}_use{suffix}"
                regular = [f"{base}_dosage{suffix}", f"{base}_comp{suffix}"]
                intermittent = [f"{base}_dosage_2{suffix}", f"{base}_frequency{suffix}"]
                if use in self._raw:
                    use_spec = self.catalog.fields.get(use)
                    regular_codes = {
                        code for code, label in (use_spec.choices if use_spec else ())
                        if "regular" in label.lower() and "intermitt" not in label.lower()}
                    intermittent_codes = {
                        code for code, label in (use_spec.choices if use_spec else ())
                        if "intermitt" in label.lower()}
                    for required in regular:
                        if required in self._raw and regular_codes:
                            self._emit_mask(
                                self._code_mask(use, intermittent_codes)
                                & self._observed_mask(required),
                                "MED-USE-001", [use, required],
                                "Intermittent medication use retains a regular-use field.",
                                forms=[form])
                    for required in intermittent:
                        if required in self._raw and intermittent_codes:
                            self._emit_mask(
                                self._code_mask(use, regular_codes)
                                & self._observed_mask(required),
                                "MED-USE-002", [use, required],
                                "Regular medication use retains an intermittent-use field.",
                                forms=[form])

                for controller, code, companion in (
                    (f"{base}_datasource{suffix}", 99, f"{base}_other{suffix}"),
                    (f"{base}_indication{suffix}", 8, f"{base}_other2{suffix}"),
                ):
                    if self._has_columns(controller, companion):
                        non_other = set(self.catalog.fields[controller].choice_codes) - {
                            _canonical_code(code)}
                        self._rule_forbids(
                            "MED-OTHER-001", controller, non_other, [companion],
                            "Medication Other text is present when Other is not selected.",
                            forms=[form])

        # Lifetime AP course cardinality and positive dose/day rules. Drug
        # roots are inferred from exact course-controller names (25 families).
        ap_controllers = sorted(
            variable for variable, spec in self.catalog.fields.items()
            if spec.form == "lifetime_ap_exposure_screen"
            and re.fullmatch(r"chrap_[a-z0-9]+course", variable))
        for controller in ap_controllers:
            base = controller[:-6]
            ever = base
            if ever not in self._raw or controller not in self._raw:
                continue
            count = self._numeric_series(controller)
            any_course = self._any_observed([
                f"{base}{kind}{number}"
                for number in range(1, 4)
                for kind in ("dose", "days", "dosecourse", "doseeq")
                if f"{base}{kind}{number}" in self._raw])
            self._emit_mask(
                self._code_mask(ever, [0]) & (
                    self._observed_mask(controller) | any_course),
                "AP-001", [ever, controller],
                "Antipsychotic Ever=No retains course count or course data.",
                forms=["lifetime_ap_exposure_screen"])
            self._emit_mask(
                self._code_mask(ever, [1]) & self._observed_mask(controller)
                & (~count.between(1, 3)),
                "AP-001", [ever, controller],
                "Supplied antipsychotic course count must be from 1 through 3 when Ever=Yes.",
                forms=["lifetime_ap_exposure_screen"])
            for number in range(1, 4):
                dose = f"{base}dose{number}"
                days = f"{base}days{number}"
                tuple_fields = [field for field in (
                    dose, days, f"{base}dosecourse{number}", f"{base}doseeq{number}")
                    if field in self._raw]
                if len(tuple_fields) >= 2:
                    present = self._any_observed(tuple_fields)
                    self._emit_mask(
                        self._code_mask(ever, [1]) & count.notna()
                        & count.lt(number) & present, "AP-002", tuple_fields,
                        f"Antipsychotic course {number} exists beyond the declared course count.",
                        forms=["lifetime_ap_exposure_screen"])
                for positive in (dose, days):
                    if positive in self._raw:
                        numeric = self._numeric_series(positive)
                        self._emit_mask(
                            self._observed_mask(positive) & numeric.notna() & numeric.le(0),
                            "AP-003", [positive],
                            "Antipsychotic course dose and days must be positive.",
                            forms=["lifetime_ap_exposure_screen"])

    def _run_ae_lifecycle_extensions(
            self, slot_masks: Sequence[pd.Series],
            slot_variables: Sequence[Sequence[str]]) -> None:
        """Apply the AE rules that cannot be inferred from generic branches.

        AE field names are irregular (notably the clinician-date fields), so
        this stays separate from the reusable repeated-episode helper.
        """
        if not slot_masks:
            return
        any_slot = pd.concat(slot_masks, axis=1).any(axis=1)
        if "chrae_aescreen" in self._raw:
            self._emit_mask(
                self._code_mask("chrae_aescreen", [0]) & any_slot,
                "EP-AE-001", ["chrae_aescreen"],
                "Adverse-event screen is No but one or more AE slots contain data.",
                forms=["adverse_events"])

        for number, current in enumerate(slot_masks, 1):
            onset = f"chrae_aes{number}date"
            clinician_gate = f"chrae_dr{number}"
            clinician_date = f"chr_ae{number}date_dr"
            clinician_initials = f"chrae_sig{number}_dr"
            if clinician_gate in self._raw:
                clinician_fields = [field for field in (clinician_date, clinician_initials)
                                   if field in self._raw]
                if clinician_fields:
                    self._emit_mask(
                        self._observed_mask(clinician_gate)
                        & ~self._code_mask(clinician_gate, [1])
                        & self._any_observed(clinician_fields),
                        "EP-AE-006", [clinician_gate, *clinician_fields],
                        "Clinician-review date/initials are present while the clinician block is not displayed.",
                        forms=["adverse_events"])
                if onset in self._raw and clinician_date in self._raw:
                    self._rule_order(
                        "EP-AE-007", onset, clinician_date,
                        f"Adverse-event slot {number} clinician-review date precedes onset.",
                        forms=["adverse_events"])

    def _run_health_specimen_checks(self) -> None:
        # Unit-pair and physiologic derivations. These explicitly correct the
        # dictionary's Fahrenheit +3 defect while the generic calc engine
        # continues to audit the stored REDCap formula itself.
        if self._has_columns("chrchs_height", "chrchs_heightunits", "chrchs_heightcm"):
            raw = self._numeric_series("chrchs_height")
            stored = self._numeric_series("chrchs_heightcm")
            expected = raw.where(self._code_mask("chrchs_heightunits", [1]), raw * 2.54)
            known = raw.notna() & stored.notna() & self._observed_mask("chrchs_heightunits")
            self._emit_mask(
                known & (stored - expected).abs().gt(0.05), "HEALTH-001",
                ["chrchs_height", "chrchs_heightunits", "chrchs_heightcm"],
                "Height-centimeter derivation disagrees with value and units.",
                forms=["current_health_status"])
        if self._has_columns("chrchs_weight", "chrchs_weightunits", "chrchs_weightkg"):
            raw = self._numeric_series("chrchs_weight")
            stored = self._numeric_series("chrchs_weightkg")
            expected = raw.where(self._code_mask("chrchs_weightunits", [1]),
                                 raw / 2.2046226218)
            known = raw.notna() & stored.notna() & self._observed_mask("chrchs_weightunits")
            self._emit_mask(
                known & (stored - expected).abs().gt(0.05), "HEALTH-001",
                ["chrchs_weight", "chrchs_weightunits", "chrchs_weightkg"],
                "Weight-kilogram derivation disagrees with value and units.",
                forms=["current_health_status"])
        if self._has_columns("chrchs_heightcm", "chrchs_weightkg", "chrchs_bmi"):
            height = self._numeric_series("chrchs_heightcm") / 100
            weight = self._numeric_series("chrchs_weightkg")
            bmi = self._numeric_series("chrchs_bmi")
            expected = weight / height.pow(2)
            known = height.gt(0) & weight.notna() & bmi.notna()
            self._emit_mask(
                known & (bmi - expected).abs().gt(0.11), "HEALTH-002",
                ["chrchs_heightcm", "chrchs_weightkg", "chrchs_bmi"],
                "BMI does not equal weight-kg divided by height-meters squared.",
                forms=["current_health_status"])
        if self._has_columns("chrchs_bodytemp", "chrchs_bodytempunits", "chrchs_bodytempf"):
            raw = self._numeric_series("chrchs_bodytemp")
            stored = self._numeric_series("chrchs_bodytempf")
            expected = raw.where(self._code_mask("chrchs_bodytempunits", [2]),
                                 raw * 9 / 5 + 32)
            known = raw.notna() & stored.notna() & self._observed_mask("chrchs_bodytempunits")
            self._emit_mask(
                known & (stored - expected).abs().gt(0.11), "HEALTH-003",
                ["chrchs_bodytemp", "chrchs_bodytempunits", "chrchs_bodytempf"],
                "Fahrenheit temperature is not C x 9/5 + 32 (or the entered Fahrenheit value).",
                forms=["current_health_status"])
        if self._has_columns("chrchs_systolic", "chrchs_diastolic"):
            systolic = self._numeric_series("chrchs_systolic")
            diastolic = self._numeric_series("chrchs_diastolic")
            self._emit_mask(
                systolic.notna() & diastolic.notna() & systolic.le(diastolic),
                "HEALTH-004", ["chrchs_systolic", "chrchs_diastolic"],
                "Systolic blood pressure is not greater than diastolic blood pressure.",
                forms=["current_health_status"])
        self._rule_order(
            "HEALTH-005", "chrchs_sleep", "chrchs_wake",
            "Wake datetime does not follow bedtime.", forms=["current_health_status"])
        self._rule_order(
            "HEALTH-005", "chrchs_wake", "chrchs_interview_date",
            "Wake datetime is after the current-health interview.",
            forms=["current_health_status"])

        # Explicit health context gates whose companion fields are not always
        # dictionary-required.
        for controller, yes_codes, companions in (
            ("chrchs_tob", [1], ["chrchs_tobuse"]),
            ("chrchs_mar", [1], ["chrchs_mardate"]),
            ("chrchs_mens", [1], ["chrchs_mensdate"]),
            ("chrchs_covidinfhosp", [1], ["chrchs_covidinfhospdays"]),
        ):
            self._rule_forbids(
                "HEALTH-006", controller, [0], companions,
                "Health-status No response retains a detail/date companion.",
                forms=["current_health_status"])
        if self._has_columns("chrchs_covidinfdays", "chrchs_covidinfhospdays"):
            total = self._numeric_series("chrchs_covidinfdays")
            hospital = self._numeric_series("chrchs_covidinfhospdays")
            self._emit_mask(
                total.notna() & hospital.notna() & hospital.gt(total),
                "HEALTH-007", ["chrchs_covidinfdays", "chrchs_covidinfhospdays"],
                "COVID hospitalization days exceed total COVID illness days.",
                forms=["current_health_status"])
        if self._checkbox_selected("chrchs_covidvac", [0]).any():
            vaccine_details = [
                field for field in ("chrchs_covidvac_other", "chrchs_covidvacnum",
                                    "chrchs_covidvac1date", "chrchs_covidvac2date")
                if field in self._raw]
            self._emit_mask(
                self._checkbox_selected("chrchs_covidvac", [0])
                & self._any_observed(vaccine_details),
                "HEALTH-008", ["chrchs_covidvac", *vaccine_details],
                "COVID vaccine No is selected with vaccine details.",
                forms=["current_health_status"])
        self._rule_order(
            "HEALTH-009", "chrchs_covidvac1date", "chrchs_covidvac2date",
            "Second vaccine date precedes first vaccine date.",
            forms=["current_health_status"])

        # Blood collection and storage chronology.
        self._rule_order(
            "BLOOD-001", "chrblood_drawdate", "chrblood_labdate",
            "Blood sample reached the lab before it was drawn.",
            forms=["blood_sample_preanalytic_quality_assurance"])
        for freeze in ("chrblood_wbfrztime", "chrblood_serumfrztime",
                       "chrblood_plfrztime", "chrblood_bcfrztime"):
            self._rule_order(
                "BLOOD-002", "chrblood_drawdate", freeze,
                "Specimen freezer timestamp precedes the blood draw.",
                forms=["blood_sample_preanalytic_quality_assurance"])

        vial_families = [
            *(f"chrblood_wb{number}" for number in range(1, 4)),
            *(f"chrblood_se{number}" for number in range(1, 4)),
            *(f"chrblood_pl{number}" for number in range(1, 7)),
            "chrblood_bc1",
        ]
        vial_ids: list[str] = []
        primary_rack_positions: list[str] = []
        for base in vial_families:
            fields = [field for field in (
                f"{base}id", f"{base}vol", f"{base}box", f"{base}pos")
                if field in self._raw]
            if f"{base}id" in self._raw:
                vial_ids.append(f"{base}id")
            if f"{base}pos" in self._raw and base != "chrblood_bc1":
                primary_rack_positions.append(f"{base}pos")
        for position in range(len(vial_ids)):
            for other in range(position + 1, len(vial_ids)):
                left, right = vial_ids[position], vial_ids[other]
                duplicate = (
                    self._observed_mask(left) & self._observed_mask(right)
                    & self._raw[left].astype(str).str.strip().str.upper().eq(
                        self._raw[right].astype(str).str.strip().str.upper()))
                self._emit_mask(
                    duplicate, "BLOOD-004", [left, right],
                    "Two specimen vial IDs are duplicated within the visit.",
                    forms=["blood_sample_preanalytic_quality_assurance"])

        # Whole-blood, serum, and plasma specimens share chrblood_rack_barcode,
        # so a position collision among these fields is deterministic. Buffy
        # coat has its own box and only one declared vial in this dictionary.
        for position, left in enumerate(primary_rack_positions):
            for right in primary_rack_positions[position + 1:]:
                duplicate = (
                    self._observed_mask(left) & self._observed_mask(right)
                    & self._raw[left].astype(str).str.strip().eq(
                        self._raw[right].astype(str).str.strip()))
                self._emit_mask(
                    duplicate, "BLOOD-005", ["chrblood_rack_barcode", left, right],
                    "Two blood vials occupy the same position in the shared cryotube rack.",
                    forms=["blood_sample_preanalytic_quality_assurance"])

        for variable in ("chrblood_wholeblood_freeze", "chrblood_serum_freeze",
                         "chrblood_plasma_freeze", "chrblood_buffy_freeze"):
            if variable not in self._raw:
                continue
            value = self._numeric_series(variable)
            self._emit_mask(
                self._observed_mask(variable) & value.notna() & value.lt(0),
                "BLOOD-006", [variable],
                "Specimen processing time cannot be negative.",
                forms=["blood_sample_preanalytic_quality_assurance"])

        if self._has_columns("chrblood_spintemp", "chrblood_spinunit", "chrblood_spintempc"):
            temperature = self._numeric_series("chrblood_spintemp")
            stored_celsius = self._numeric_series("chrblood_spintempc")
            expected_celsius = temperature.where(
                self._code_mask("chrblood_spinunit", [1]),
                (temperature - 32) * 5 / 9).round(1)
            known = temperature.notna() & stored_celsius.notna() & self._observed_mask("chrblood_spinunit")
            self._emit_mask(
                known & (stored_celsius - expected_celsius).abs().gt(0.11),
                "BLOOD-007", ["chrblood_spintemp", "chrblood_spinunit", "chrblood_spintempc"],
                "Centrifuge Celsius temperature disagrees with the source value and units.",
                forms=["blood_sample_preanalytic_quality_assurance"])

        # CBC result/reference family. 'inrange' actually asks whether the
        # result is outside the reported reference range (Yes=1).
        cbc_stems = (
            "rbc", "hct", "hgb", "mcv", "mch", "mchc", "platelets",
            "mpv", "wbc", "neut", "lymph", "monos", "eos", "baso")
        aliases = {"platelets": "plat", "monos": "mono"}
        for stem in cbc_stems:
            range_stem = aliases.get(stem, stem)
            value = f"chrcbc_{stem}"
            outside = f"chrcbc_{range_stem}inrange"
            low = f"chrcbc_{range_stem}_low"
            high = f"chrcbc_{range_stem}_high"
            if stem == "eos" and high not in self._raw and "chrccbc_eos_high" in self._raw:
                high = "chrccbc_eos_high"
            if value not in self._raw or outside not in self._raw:
                continue
            val = self._numeric_series(value)
            lo = self._numeric_series(low) if low in self._raw else None
            hi = self._numeric_series(high) if high in self._raw else None
            if lo is not None and hi is not None:
                outside_yes = self._code_mask(outside, [1])
                outside_no = self._code_mask(outside, [0])
                self._emit_mask(
                    lo.notna() & hi.notna() & lo.gt(hi), "CBC-002",
                    [low, high], "CBC reference low exceeds reference high.",
                    forms=["cbc_with_differential"])
                self._emit_mask(
                    outside_yes & val.notna() & lo.notna() & hi.notna()
                    & val.ge(lo) & val.le(hi),
                    "CBC-003", [value, outside, low, high],
                    "CBC result is marked outside range but lies within its entered bounds.",
                    forms=["cbc_with_differential"])
                self._emit_mask(
                    outside_no & (lo.notna() | hi.notna()), "CBC-004",
                    [outside, low, high],
                    "CBC result marked not outside range retains hidden reference bounds.",
                    forms=["cbc_with_differential"])
        self._rule_forbids(
            "CBC-005", "chrcbc_mpv_available", [0],
            ["chrcbc_mpv", "chrcbc_mpvinrange", "chrcbc_mpv_low", "chrcbc_mpv_high"],
            "MPV unavailable retains dependent MPV fields.",
            forms=["cbc_with_differential"])
        if self._has_columns("chrcbc_interview_date", "chrblood_drawdate"):
            cbc_date = self._parsed_series("chrcbc_interview_date")
            draw_date = self._parsed_series("chrblood_drawdate")
            self._emit_mask(
                cbc_date.notna() & draw_date.notna()
                & cbc_date.dt.normalize().ne(draw_date.dt.normalize()),
                "CBC-006", ["chrcbc_interview_date", "chrblood_drawdate"],
                "CBC collection date disagrees with the blood-draw calendar date.",
                forms=["cbc_with_differential",
                       "blood_sample_preanalytic_quality_assurance"])

        if self._has_columns("chrblood_interview_date", "chrblood_drawdate"):
            interview_date = self._parsed_series("chrblood_interview_date")
            draw_date = self._parsed_series("chrblood_drawdate")
            self._emit_mask(
                interview_date.notna() & draw_date.notna()
                & interview_date.dt.normalize().ne(draw_date.dt.normalize()),
                "BLOOD-012", ["chrblood_interview_date", "chrblood_drawdate"],
                "Blood form interview date disagrees with the draw calendar date.",
                forms=["blood_sample_preanalytic_quality_assurance"])

        for stem, last in (("wb", 3), ("se", 3), ("pl", 6)):
            for number in range(1, last + 1):
                variable = f"chrblood_{stem}{number}vol"
                if variable not in self._raw:
                    continue
                volume, valid = self._clinical_numeric(variable)
                self._emit_mask(
                    valid & volume.gt(1) & volume.le(1.1),
                    "BLOOD-013", [variable],
                    "Blood vial volume exceeds the current dictionary maximum of 1 mL but falls inside the legacy 1.1-mL tolerance.",
                    forms=["blood_sample_preanalytic_quality_assurance"])
        if "chrcbc_wbcsum" in self._raw:
            differential = [field for field in (
                "chrcbc_neut", "chrcbc_lymph", "chrcbc_monos",
                "chrcbc_eos", "chrcbc_baso") if field in self._raw]
            if differential:
                parsed = pd.concat(
                    [self._numeric_series(field) for field in differential], axis=1)
                sentinel = pd.concat([
                    self._raw[field].map(lambda value: _canonical_code(value) in {"-3", "-9"})
                    for field in differential], axis=1).any(axis=1)
                complete = parsed.notna().all(axis=1) & ~sentinel
                stored = self._numeric_series("chrcbc_wbcsum")
                self._emit_mask(
                    complete & self._observed_mask("chrcbc_wbcsum")
                    & (stored - parsed.sum(axis=1)).abs().gt(1e-10),
                    "CBC-007", [*differential, "chrcbc_wbcsum"],
                    "CBC differential sum does not equal the sum of its complete substantive components.",
                    forms=["cbc_with_differential"])

        # Saliva collection chronology and six complete, unique aliquot tuples.
        saliva_form = "daily_activity_and_saliva_sample_collection"
        for date_variable in (
                "chrsaliva_beddate", "chrsaliva_wakedate", "chrsaliva_coldate"):
            self._rule_order(
                "SALIVA-009", date_variable, "chrsaliva_interview_date",
                "Saliva sleep/collection date is later than the saliva assessment date.",
                forms=[saliva_form])
        self._rule_equal(
            "SALIVA-010", "chrsaliva_interview_date", "chrsaliva_coldate",
            "The two governed saliva collection-date fields disagree.",
            forms=[saliva_form])
        self._rule_order(
            "SALIVA-001", "chrsaliva_time1", "chrsaliva_time2",
            "Second saliva collection time precedes the first.",
            forms=["daily_activity_and_saliva_sample_collection"])
        self._rule_order(
            "SALIVA-001", "chrsaliva_time2", "chrsaliva_time3",
            "Third saliva collection time precedes the second.",
            forms=["daily_activity_and_saliva_sample_collection"])
        bed_datetime = self._combine_date_time("chrsaliva_beddate", "chrsaliva_bedtime")
        wake_datetime = self._combine_date_time("chrsaliva_wakedate", "chrsaliva_waketime")
        collection_date = self._parsed_series("chrsaliva_coldate")
        first_time = self._parsed_series("chrsaliva_time1")
        if bed_datetime is not None and wake_datetime is not None:
            known = bed_datetime.notna() & wake_datetime.notna()
            self._emit_mask(
                known & wake_datetime.le(bed_datetime), "SALIVA-006",
                ["chrsaliva_beddate", "chrsaliva_bedtime",
                 "chrsaliva_wakedate", "chrsaliva_waketime"],
                "Saliva wake datetime does not follow bedtime.",
                forms=["daily_activity_and_saliva_sample_collection"])
            if collection_date is not None and first_time is not None:
                first_collection = self._combine_series_date_time(collection_date, first_time)
                self._emit_mask(
                    wake_datetime.notna() & first_collection.notna()
                    & wake_datetime.gt(first_collection), "SALIVA-006",
                    ["chrsaliva_wakedate", "chrsaliva_waketime",
                     "chrsaliva_coldate", "chrsaliva_time1"],
                    "Saliva wake datetime is after the first collection.",
                    forms=["daily_activity_and_saliva_sample_collection"])
        saliva_ids: list[str] = []
        position_pairs: list[tuple[str, str]] = []
        for number in range(1, 4):
            collection = f"chrsaliva_time{number}"
            for aliquot in "ab":
                base = f"chrsaliva_{number}{aliquot}"
                fields = [
                    f"chrsaliva_id{number}{aliquot}",
                    f"chrsaliva_vol{number}{aliquot}",
                    f"chrsaliva_box{number}{aliquot}",
                    f"chrsaliva_pos{number}{aliquot}",
                    f"chrsaliva_frztime{number}{aliquot}",
                ]
                fields = [field for field in fields if field in self._raw]
                identifier = f"chrsaliva_id{number}{aliquot}"
                if identifier in self._raw:
                    saliva_ids.append(identifier)
                box, pos = f"chrsaliva_box{number}{aliquot}", f"chrsaliva_pos{number}{aliquot}"
                if self._has_columns(box, pos):
                    position_pairs.append((box, pos))
                freeze = f"chrsaliva_frztime{number}{aliquot}"
                if self._has_columns(collection, freeze):
                    self._rule_order(
                        "SALIVA-003", collection, freeze,
                        "Saliva aliquot freeze time precedes its collection time.",
                        forms=["daily_activity_and_saliva_sample_collection"])
        for position in range(len(saliva_ids)):
            for other in range(position + 1, len(saliva_ids)):
                left, right = saliva_ids[position], saliva_ids[other]
                duplicate = (
                    self._observed_mask(left) & self._observed_mask(right)
                    & self._raw[left].astype(str).str.strip().str.upper().eq(
                        self._raw[right].astype(str).str.strip().str.upper()))
                self._emit_mask(
                    duplicate, "SALIVA-004", [left, right],
                    "Two saliva barcodes are duplicated within the visit.",
                    forms=["daily_activity_and_saliva_sample_collection"])
        for position in range(len(position_pairs)):
            for other in range(position + 1, len(position_pairs)):
                box1, pos1 = position_pairs[position]
                box2, pos2 = position_pairs[other]
                duplicate = (
                    self._observed_mask(box1) & self._observed_mask(pos1)
                    & self._observed_mask(box2) & self._observed_mask(pos2)
                    & self._raw[box1].astype(str).str.strip().eq(
                        self._raw[box2].astype(str).str.strip())
                    & self._raw[pos1].astype(str).str.strip().eq(
                        self._raw[pos2].astype(str).str.strip()))
                self._emit_mask(
                    duplicate, "SALIVA-005", [box1, pos1, box2, pos2],
                    "Two saliva aliquots occupy the same position within a box.",
                    forms=["daily_activity_and_saliva_sample_collection"])

        # One shared freezer time applies to all three collection rounds.
        if "chrsaliva_frztime" in self._raw and collection_date is not None:
            shared_freeze = self._parsed_series("chrsaliva_frztime")
            final_time = self._parsed_series("chrsaliva_time3")
            if shared_freeze is not None and final_time is not None:
                final_collection = self._combine_series_date_time(collection_date, final_time)
                self._emit_mask(
                    shared_freeze.notna() & final_collection.notna()
                    & shared_freeze.lt(final_collection), "SALIVA-007",
                    ["chrsaliva_coldate", "chrsaliva_time3", "chrsaliva_frztime"],
                    "Shared saliva freezer timestamp precedes the final collection.",
                    forms=["daily_activity_and_saliva_sample_collection"])

        # The protocol stores the A and B rack identifiers separately. When
        # all IDs are recorded, each set must be internally stable; whether
        # the two racks must differ remains a storage-policy decision.
        a_racks = [field for field in ("chrsaliva_box1a", "chrsaliva_box2a", "chrsaliva_box3a")
                   if field in self._raw]
        b_racks = [field for field in ("chrsaliva_box1b", "chrsaliva_box2b", "chrsaliva_box3b")
                   if field in self._raw]
        for group, label in ((a_racks, "A"), (b_racks, "B")):
            for position, left in enumerate(group):
                for right in group[position + 1:]:
                    self._emit_mask(
                        self._observed_mask(left) & self._observed_mask(right)
                        & ~self._raw[left].astype(str).str.strip().eq(
                            self._raw[right].astype(str).str.strip()),
                        "SALIVA-008", [left, right],
                        f"Saliva {label}-specimen rack identifiers differ within one visit.",
                        forms=["daily_activity_and_saliva_sample_collection"])

        # A and B aliquots are authored as separate rack assignments, and the
        # dictionary explicitly forbids the paired racks from being identical.
        for number in range(1, 4):
            left, right = f"chrsaliva_box{number}a", f"chrsaliva_box{number}b"
            if not self._has_columns(left, right):
                continue
            self._emit_mask(
                self._observed_mask(left) & self._observed_mask(right)
                & self._raw[left].astype(str).str.strip().str.casefold().eq(
                    self._raw[right].astype(str).str.strip().str.casefold()),
                "SALIVA-011", [left, right],
                "Paired saliva A/B aliquots are assigned to the same rack.",
                forms=[saliva_form])

        allowed_wells = {
            f"{row}{column}"
            for rows, columns in (
                (("A", "C", "E", "G"), range(2, 13, 2)),
                (("B", "D", "F", "H"), range(1, 12, 2)))
            for row in rows for column in columns
        }
        for number in range(1, 4):
            for aliquot in "ab":
                variable = f"chrsaliva_pos{number}{aliquot}"
                if variable not in self._raw:
                    continue
                normalized = self._raw[variable].astype(str).str.strip().str.upper()
                self._emit_mask(
                    self._observed_mask(variable) & ~normalized.isin(allowed_wells),
                    "SALIVA-012", [variable],
                    "Saliva rack position is outside the authored checkerboard wells.",
                    forms=[saliva_form])

        # GCP source visibility, significance/action, and mutually exclusive
        # notification states for every analyte/vital family.
        gcp_families = {
            "rbc": "rbc", "hct": "hct", "hgb": "hgb", "mcv": "mcv",
            "mch": "mch", "mchc": "mchc", "plat": "plat",
            "wbc": "wbc", "neut": "neut", "lymph": "lymph",
            "mono": "mono", "eos": "eos", "baso": "baso",
        }
        for gcp, cbc in gcp_families.items():
            sig = f"chrgpc_{gcp}sig"
            action = f"chrgpc_act{gcp}"
            source = f"chrcbc_{cbc}inrange"
            if sig not in self._raw:
                continue
            if source in self._raw:
                self._emit_mask(
                    self._observed_mask(source)
                    & ~self._code_mask(source, [1]) & self._observed_mask(sig),
                    "GCP-CBC-001", [source, sig],
                    "GCP significance is populated while source CBC is not marked outside range.",
                    forms=["cbc_with_differential", "gcp_cbc_with_differential"])
            action_columns = [
                column for column in self._raw if column.startswith(action + "___")]
            any_action = (
                pd.concat([self._raw[column].map(_is_yes) for column in action_columns], axis=1).any(axis=1)
                if action_columns else pd.Series(False, index=self._raw.index, dtype=bool))
            self._emit_mask(
                self._code_mask(sig, [0]) & any_action,
                "GCP-CBC-002", [sig, action],
                "CBC result marked not clinically significant retains action selections.",
                forms=["gcp_cbc_with_differential"])
            notified = self._checkbox_selected(action, [1])
            unable = self._checkbox_selected(action, [2])
            self._emit_mask(
                notified & unable, "GCP-CBC-003", [action],
                "Notified participant and unable-to-notify actions are both selected.",
                forms=["gcp_cbc_with_differential"])
        for stem, source in (("systolic", "chrchs_systolic"),
                             ("diastolic", "chrchs_diastolic"),
                             ("temp", "chrchs_bodytempf")):
            sig, action = f"chrgpc_{stem}_sig", f"chrgpc_act{stem}"
            if sig not in self._raw:
                continue
            action_columns = [column for column in self._raw
                              if column.startswith(action + "___")]
            any_action = (
                pd.concat([self._raw[column].map(_is_yes) for column in action_columns], axis=1).any(axis=1)
                if action_columns else pd.Series(False, index=self._raw.index, dtype=bool))
            self._emit_mask(
                self._code_mask(sig, [0]) & any_action,
                "GCP-HEALTH-001", [sig, action],
                "Vital marked not clinically significant retains action selections.",
                forms=["gcp_current_health_status"])

        self._rule_order(
            "GCP-CBC-005", "chrblood_drawdate", "chrcbccs_review_date",
            "GCP CBC review date precedes the blood collection date.",
            forms=["blood_sample_preanalytic_quality_assurance",
                   "gcp_cbc_with_differential"])
        self._rule_order(
            "GCP-HEALTH-002", "chrchs_interview_date", "chrgpc_date",
            "GCP current-health review date precedes the current-health interview.",
            forms=["current_health_status", "gcp_current_health_status"])
        for variable, form, check_id in (
            ("chrcbccs_review_date", "gcp_cbc_with_differential", "GCP-CBC-005"),
            ("chrgpc_date", "gcp_current_health_status", "GCP-HEALTH-002"),
        ):
            if variable not in self._raw:
                continue
            review = self._parsed_series(variable)
            self._emit_mask(
                review.notna() & review.dt.date.gt(date.today()), check_id, [variable],
                "GCP review date is in the future.", forms=[form])

    def _run_procedure_and_device_checks(self) -> None:
        # EEG: exact run-status/note pairs, temporal order, and file routing.
        self._rule_order(
            "EEG-001", "chreeg_start", "chreeg_end",
            "EEG session end time is not after start time.", strict=True,
            forms=["eeg_run_sheet"])
        if "chreeg_zip" in self._raw:
            count = self._numeric_series("chreeg_zip")
            self._emit_mask(
                self._observed_mask("chreeg_zip")
                & (count.isna() | count.lt(0) | count.mod(1).ne(0)),
                "EEG-003", ["chreeg_zip"],
                "EEG ZIP count must be a nonnegative integer.",
                forms=["eeg_run_sheet"])
            # EEG-004 (a completed acquisition holding zero ZIP files) was
            # removed: it reports absent data against a completion status.
            routed = pd.Series(False, index=self._raw.index)
            destinations = [destination for destination in (
                "chreeg_hub", "chreeg_drive") if destination in self._raw]
            routing_known = self._all_observed(destinations)
            for destination in destinations:
                routed |= self._raw[destination].map(_is_yes)
            self._emit_mask(
                count.gt(0) & routing_known & ~routed, "EEG-005",
                ["chreeg_zip", "chreeg_hub", "chreeg_drive"],
                "EEG ZIP files are present with neither Network Hub upload nor local-drive archive confirmed.",
                forms=["eeg_run_sheet"], severity="Warning")
            # The reverse EEG-005 branch (upload/archive confirmed with no ZIP
            # files) was removed: its defect is that the files are absent.

        # MRI session and every sequence status/QC/comment family.
        self._rule_order(
            "MRI-001", "chrmri_starttime", "chrmri_endtime",
            "MRI session end time is not after start time.", strict=True,
            forms=["mri_run_sheet"])
        if self._has_columns("chrmri_session_year", "chrmri_session_month", "chrmri_session_day"):
            session_parts = [
                "chrmri_session_year", "chrmri_session_month", "chrmri_session_day"]
            year = self._numeric_series("chrmri_session_year")
            month = self._numeric_series("chrmri_session_month")
            day = self._numeric_series("chrmri_session_day")
            invalid = pd.Series(False, index=self._raw.index, dtype=bool)
            future = pd.Series(False, index=self._raw.index, dtype=bool)
            complete_parts = self._all_observed(session_parts)
            for position, parts in enumerate(zip(year, month, day)):
                if not complete_parts.iat[position]:
                    continue
                try:
                    session_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
                    future.iat[position] = session_date > date.today()
                except (TypeError, ValueError, OverflowError):
                    invalid.iat[position] = True
            self._emit_mask(
                invalid, "MRI-002",
                ["chrmri_session_year", "chrmri_session_month", "chrmri_session_day"],
                "MRI session year/month/day do not form a real calendar date.",
                forms=["mri_run_sheet"])
            self._emit_mask(
                future, "MRI-002",
                ["chrmri_session_year", "chrmri_session_month", "chrmri_session_day"],
                "MRI acquisition date is in the future.", forms=["mri_run_sheet"])

        sequence_controls: list[str] = []
        for variable, spec in self.catalog.fields.items():
            if spec.form != "mri_run_sheet" or variable not in self._raw:
                continue
            labels = " ".join(label.lower() for _, label in spec.choices)
            if not spec.choices or not any(token in labels for token in (
                    "no scans", "not acquired", "none acquired")):
                continue
            sequence_controls.append(variable)
            no_codes = {
                code for code, label in spec.choices
                if any(token in label.lower() for token in (
                    "no scans", "not acquired", "none acquired"))}
            candidates = [
                f"{variable}_qc", variable.replace("_ref_num", "_ref_qc"),
                variable.replace("_num", "_qc")]
            qc = next((candidate for candidate in candidates if candidate in self._raw), None)
            if qc:
                self._emit_mask(
                    self._code_mask(variable, no_codes) & self._observed_mask(qc),
                    "MRI-003", [variable, qc],
                    "Not-acquired MRI sequence retains a QC status.",
                    forms=["mri_run_sheet"])
        if sequence_controls and "chrmri_consent" in self._raw:
            acquired_any = pd.Series(False, index=self._raw.index, dtype=bool)
            for variable in sequence_controls:
                spec = self.catalog.fields[variable]
                no_codes = {
                    code for code, label in spec.choices
                    if any(token in label.lower() for token in (
                        "no scans", "not acquired", "none acquired"))}
                acquired_any |= self._observed_mask(variable) & ~self._code_mask(variable, no_codes)
            self._emit_mask(
                acquired_any & self._code_mask("chrmri_consent", [0]),
                "MRI-005", ["chrmri_consent", *sequence_controls[:8]],
                "MRI sequences were acquired while MRI consent is No.",
                forms=["mri_run_sheet"], severity="Critical")

        # Incidental finding directionality and notification contradiction.
        self._rule_forbids(
            "MRI-IF-002", "chrif_any_finding", [0],
            ["chrif_source_find", "chrif_descrip_find", "chrif_fu_action"],
            "Incidental-finding No retains finding details/actions.",
            forms=["mri_incidental_findings_run_sheet"])
        if "chrif_fu_action" in self.catalog.fields:
            action_spec = self.catalog.fields["chrif_fu_action"]
            notified = [code for code, label in action_spec.choices
                        if "notified" in label.lower() and "unable" not in label.lower()]
            unable = [code for code, label in action_spec.choices
                      if "unable" in label.lower()]
            self._emit_mask(
                self._checkbox_selected("chrif_fu_action", notified)
                & self._checkbox_selected("chrif_fu_action", unable),
                "MRI-IF-003", ["chrif_fu_action"],
                "Participant-notified and unable-to-notify actions are both selected.",
                forms=["mri_incidental_findings_run_sheet"])

        # Speech and PSYCHS A/V run sheets share the same operational grammar.
        av_maps = [
            ("speech_sampling_run_sheet", "chrspeech_", {
                "language": "other_lang", "record": "record_yn", "upload": "upload",
                "pause": "pause", "deviation": "deviation", "quality": "quality"}),
            ("psychs_av_recording_run_sheet", "chrpsychs_av_", {
                "language": "other_lang", "record": "record_yn", "upload": "upload",
                "pause": "pause", "deviation": "deviation", "quality": "quality"}),
        ]
        for form, prefix, names in av_maps:
            english, language = prefix + "english", prefix + names["language"]
            self._rule_forbids(
                "AV-001", english, [1], [language],
                "English recording retains a non-English language value.", forms=[form])
            record, upload = prefix + names["record"], prefix + names["upload"]
            if self._has_columns(record, upload):
                self._emit_mask(
                    self._code_mask(upload, [1]) & self._observed_mask(record)
                    & ~self._code_mask(record, [1]),
                    "AV-002", [record, upload],
                    "Recording upload is Yes without a successful recording workflow.",
                    forms=[form])
                self._emit_mask(
                    self._code_mask(record, [1]) & self._code_mask(upload, [0]),
                    "AV-002", [record, upload],
                    "Successful recording was not uploaded.", forms=[form], severity="Warning")
        # MindLAMP onboarding/check-in/end state and percentages.
        if "chrdig_detail_change" in self._raw:
            new_values = [
                field for field in ("chrdig_new_study_vers", "chrdig_new_lamp_id",
                                    "chrdig_new_phone_mod", "chrdig_new_phone_soft_version")
                if field in self._raw]
            self._emit_mask(
                self._code_mask("chrdig_detail_change", [0])
                & self._any_observed(["chrdig_change_date", *new_values]),
                "MINDLAMP-002", ["chrdig_detail_change", "chrdig_change_date", *new_values],
                "MindLAMP detail change No retains change date/new-value data.",
                forms=["digital_biomarkers_mindlamp_checkin"])
            self._rule_order(
                "MINDLAMP-003", "chrdig_change_date", "chrdig_interview_date",
                "MindLAMP change date is after the check-in interview.",
                forms=["digital_biomarkers_mindlamp_checkin"])
        for variable, form in (
            ("chrdig_monthly_percent", "digital_biomarkers_mindlamp_checkin"),
            ("chrdbe_percent_surveys", "digital_biomarkers_mindlamp_end_of_12month_study_p"),
        ):
            if variable in self._raw:
                value = self._numeric_series(variable)
                self._emit_mask(
                    self._observed_mask(variable)
                    & (value.isna() | ~value.between(0, 100)),
                    "MINDLAMP-004", [variable],
                    "MindLAMP survey percentage must be finite and from 0 through 100.",
                    forms=[form])
        if "chrdig_number_tech" in self._raw:
            count = self._numeric_series("chrdig_number_tech")
            self._emit_mask(
                self._observed_mask("chrdig_number_tech")
                & (count.isna() | count.lt(0) | count.mod(1).ne(0)),
                "MINDLAMP-005", ["chrdig_number_tech"],
                "MindLAMP technical-missing count must be a nonnegative integer.",
                forms=["digital_biomarkers_mindlamp_checkin"])
        self._rule_forbids(
            "MINDLAMP-006", "chrdbe_premature", [0], ["chrdbe_premature_notes"],
            "Regular MindLAMP ending retains a premature-ending reason.",
            forms=["digital_biomarkers_mindlamp_end_of_12month_study_p"])
        if self._has_columns("chrdbe_premature", "chrdbe_regular"):
            applicable = self._all_observed(
                ["chrdbe_premature", "chrdbe_regular"])
            self._emit_mask(
                applicable & self._raw["chrdbe_premature"].map(_is_yes).eq(
                    self._raw["chrdbe_regular"].map(_is_yes)),
                "MINDLAMP-006", ["chrdbe_premature", "chrdbe_regular"],
                "MindLAMP end record must select exactly one of premature or regular ending.",
                forms=["digital_biomarkers_mindlamp_end_of_12month_study_p"])

        # Axivity device prerequisites. Device continuity is handled in the
        # longitudinal state method after these same-row checks.
        for form, prefix in (
            ("digital_biomarkers_axivity_onboarding", "chrax_"),
            ("digital_biomarkers_axivity_checkin", "chraxci_"),
            ("digital_biomarkers_axivity_end_of_12month_study_pe", "chraxe_"),
        ):
            config_date = prefix + "config_date"
            self._rule_order(
                "AXIVITY-002", config_date, prefix + "interview_date",
                "Axivity configuration date is after interview date.", forms=[form])

    def _participant_age(self, interview: str | None = None) -> pd.Series:
        result = pd.Series(float("nan"), index=self._raw.index, dtype=float)
        for variable in ("chrdemo_age_yrs_chr", "chrdemo_age_yrs_hc",
                         "chrcrit_age", "screening_age"):
            if variable in self._raw:
                value = self._numeric_series(variable)
                result = result.fillna(value)
        if interview and interview in self._raw:
            visit = self._parsed_series(interview)
            for dob_name in ("chrdemo_dob", "chrguid_dob", "chrcrit_dob"):
                if dob_name not in self._raw:
                    continue
                dob = self._parsed_series(dob_name)
                derived = pd.Series([
                    (end.year - start.year
                     - ((end.month, end.day) < (start.month, start.day)))
                    if pd.notna(start) and pd.notna(end) else float("nan")
                    for start, end in zip(dob, visit)],
                    index=self._raw.index, dtype=float)
                result = result.fillna(derived)
        for key in ("age", "subject_age", "interview_age"):
            values = pd.Series(
                [self.subject_info.get(subject, {}).get(key)
                 for subject in self._raw["subjectid"]],
                index=self._raw.index,
            )
            numeric = pd.to_numeric(values, errors="coerce")
            numeric = numeric.where(~values.map(_is_missing_value))
            result = result.fillna(numeric)
        return result

    def _run_scale_demographic_assist_checks(self) -> None:
        # Global Functioning and SOFAS deterministic score orderings.
        for form, low, current, high in (
            ("global_functioning_role_scale", "chrgfr_gf_role_low",
             "chrgfr_gf_role_scole", "chrgfr_gf_role_high"),
            ("global_functioning_social_scale", "chrgfs_gf_social_low",
             "chrgfs_gf_social_scale", "chrgfs_gf_social_high"),
        ):
            if not self._has_columns(low, current, high):
                continue
            low_value = self._numeric_series(low)
            current_value = self._numeric_series(current)
            high_value = self._numeric_series(high)
            known = low_value.notna() & current_value.notna() & high_value.notna()
            self._emit_mask(
                known & ((low_value > current_value) | (current_value > high_value)),
                "FUNCTION-001", [low, current, high],
                "Global Functioning scores violate low <= current <= high.", forms=[form])
        if self._has_columns("chrsofas_lowscore", "chrsofas_currscore"):
            low = self._numeric_series("chrsofas_lowscore")
            current = self._numeric_series("chrsofas_currscore")
            self._emit_mask(
                low.notna() & current.notna() & low.gt(current),
                "SOFAS-001", ["chrsofas_lowscore", "chrsofas_currscore"],
                "SOFAS lowest-past-year score exceeds current score.",
                forms=["sofas_screening"])
        if self._has_columns("chrsofas_lowscore", "chrsofas_currscore12mo"):
            low = self._numeric_series("chrsofas_lowscore")
            year = self._numeric_series("chrsofas_currscore12mo")
            self._emit_mask(
                low.notna() & year.notna() & low.gt(year),
                "SOFAS-001", ["chrsofas_lowscore", "chrsofas_currscore12mo"],
                "SOFAS lowest-past-year score exceeds 12-month score.",
                forms=["sofas_screening"])
        self._rule_order(
            "SOFAS-002", "chrsofas_premorbdate", "chrsofas_interview_date",
            "SOFAS premorbid date is after the interview date.",
            forms=["sofas_screening"])
        # IQ instrument is age-specific. WASI-II's governed range is not encoded
        # in this dictionary, so it remains configuration/review rather than an
        # invented hard limit.
        if "chriq_assessment" in self._raw:
            age = self._participant_age("chriq_interview_date")
            assessment = self._raw["chriq_assessment"].map(_canonical_code)
            invalid = (
                (assessment.eq("2") & age.notna() & ~age.between(16, 30))
                | (assessment.eq("3") & age.notna() & ~age.between(12, 15))
                | (assessment.eq("4") & age.notna() & ~age.between(16, 90)))
            self._emit_mask(
                invalid, "IQ-001", ["chriq_assessment"],
                "Selected IQ instrument is outside its approved age range.",
                forms=["iq_assessment_wasiii_wiscv_waisiv"])

            dob_variable = next((variable for variable in (
                "chrdemo_dob", "chrcrit_dob", "chrguid_dob")
                if variable in self._raw), None)
            if (dob_variable and self._has_columns(
                    "chric_consent_date", "chriq_vocab_raw")):
                dob = self._parsed_series(dob_variable)
                consent = self._parsed_series("chric_consent_date")
                age_at_consent = pd.Series([
                    (end.year - start.year
                     - ((end.month, end.day) < (start.month, start.day)))
                    if pd.notna(start) and pd.notna(end) else float("nan")
                    for start, end in zip(dob, consent)],
                    index=self._raw.index, dtype=float)
                vocabulary, vocabulary_valid = self._clinical_numeric(
                    "chriq_vocab_raw")
                self._emit_mask(
                    age_at_consent.notna() & age_at_consent.lt(14)
                    & assessment.isin({"1", "2", "3"})
                    & vocabulary_valid & ~vocabulary.between(0, 53),
                    "IQ-005", [dob_variable, "chric_consent_date",
                               "chriq_assessment", "chriq_vocab_raw"],
                    "IQ vocabulary raw score is outside the authored 0-53 range for participants younger than 14 at consent.",
                    forms=["informed_consent_run_sheet",
                           "iq_assessment_wasiii_wiscv_waisiv"])

        if self._has_columns("chrpreiq_reading_task", "chrpreiq_total_raw"):
            task = self._raw["chrpreiq_reading_task"].map(_canonical_code)
            raw_score, raw_valid = self._clinical_numeric("chrpreiq_total_raw")
            allowed = {"1": (0, 70), "3": (0, 50), "4": (0, 40)}
            invalid = pd.Series(False, index=self._raw.index, dtype=bool)
            for code, (minimum, maximum) in allowed.items():
                invalid |= task.eq(code) & raw_valid & ~raw_score.between(
                    minimum, maximum)
            self._emit_mask(
                invalid, "PREIQ-001",
                ["chrpreiq_reading_task", "chrpreiq_total_raw"],
                "Premorbid-IQ raw score is outside the authored task-specific range.",
                forms=["premorbid_iq_reading_accuracy"])
        if self._has_columns(
                "chrpreiq_reading_task", "chrpreiq_standard_score"):
            standard, standard_valid = self._clinical_numeric(
                "chrpreiq_standard_score")
            self._emit_mask(
                self._code_mask("chrpreiq_reading_task", [1])
                & standard_valid & ~standard.between(55, 145),
                "PREIQ-002",
                ["chrpreiq_reading_task", "chrpreiq_standard_score"],
                "WRAT5 standardized score is outside the authored 55-145 range.",
                forms=["premorbid_iq_reading_accuracy"])

        if "chrpenn_complete" in self._raw:
            self._rule_forbids(
                "PENN-001", "chrpenn_complete", [3], ["chrpenn_json_upload"],
                "PennCNB not performed retains a JSON result upload.", forms=["penncnb"])

        if self._has_columns("chrpas_pmod_adult3v1", "chrpas_pmod_adult3v3"):
            both = self._observed_mask("chrpas_pmod_adult3v1") & self._observed_mask(
                "chrpas_pmod_adult3v3")
            self._emit_mask(
                both, "PAS-001", ["chrpas_pmod_adult3v1", "chrpas_pmod_adult3v3"],
                "Both mutually exclusive Premorbid Adjustment adult marriage paths are scored.",
                forms=["premorbid_adjustment_scale"])

        # Reading/IQ and result-file workflow presence.
        # Pubertal menarche gate and age bound.
        if "chrpds_pds_f5b_p" in self._raw:
            self._rule_forbids(
                "PDS-001", "chrpds_pds_f5b_p", [1], ["chrpds_pds_f6_p"],
                "Menarche No/unknown retains an age at menarche.",
                forms=["pubertal_developmental_scale"])
            if "chrpds_pds_f6_p" in self._raw:
                menarche_age = self._numeric_series("chrpds_pds_f6_p")
                age = self._participant_age("chrpds_interview_date")
                self._emit_mask(
                    menarche_age.notna() & (menarche_age.lt(0)
                                             | (age.notna() & menarche_age.gt(age))),
                    "PDS-002", ["chrpds_pds_f6_p"],
                    "Age at menarche is negative or exceeds age at interview.",
                    forms=["pubertal_developmental_scale"])

        # ASSIST ten-family semantics. Family-specific domains still come from
        # DD-DOMAIN; these are only the exact cross-question implications.
        for number in range(1, 11):
            lifetime = f"chrassist_whoassist_use{number}"
            if lifetime not in self._raw:
                continue
            downstream = [
                f"chrassist_whoassist_{stem}{number}"
                for stem in ("often", "urge", "prob", "fail", "concern", "control")
                if f"chrassist_whoassist_{stem}{number}" in self._raw]
            self._emit_mask(
                self._code_mask(lifetime, [0]) & self._any_observed(downstream),
                "ASSIST-001", [lifetime, *downstream],
                f"ASSIST family {number} lifetime-use No retains downstream answers.",
                forms=["assist"])

        # Explicit scale summary semantics beyond generic calculation replay.
        bprs_items = [
            variable for variable, spec in self.catalog.fields.items()
            if spec.form == "bprs" and variable in self._raw
            and variable.startswith("chrbprs_bprs_")
            and variable != "chrbprs_bprs_total"
            and spec.field_type in {"radio", "dropdown"}
            and {"1", "2", "3", "4", "5", "6", "7"}.issubset(spec.choice_codes)]
        if bprs_items and "chrbprs_bprs_total" in self._raw:
            values = pd.concat([
                self._numeric_series(field) for field in bprs_items
            ], axis=1)
            stored = self._numeric_series("chrbprs_bprs_total")
            sentinel = values.isin([-3, -9, 999]).any(axis=1)
            complete = values.notna().all(axis=1) & ~sentinel
            expected = values.sum(axis=1)
            self._emit_mask(
                complete & self._observed_mask("chrbprs_bprs_total")
                & (stored - expected).abs().gt(1e-9),
                "BPRS-001", [*bprs_items, "chrbprs_bprs_total"],
                "BPRS total does not equal the sum of 24 complete substantive ratings.",
                forms=["bprs"])
        if self._has_columns("chrbprs_bprs_susp", "chrbprs_bprs_unus"):
            suspiciousness, suspiciousness_valid = self._clinical_numeric(
                "chrbprs_bprs_susp")
            unusual, unusual_valid = self._clinical_numeric("chrbprs_bprs_unus")
            contradiction = (
                (suspiciousness.between(3, 5) & unusual.eq(1))
                | (suspiciousness.between(6, 7) & unusual.between(2, 3)))
            self._emit_mask(
                suspiciousness_valid & unusual_valid & contradiction,
                "BPRS-002", ["chrbprs_bprs_susp", "chrbprs_bprs_unus"],
                "BPRS suspiciousness requires the corresponding authored unusual-thought-content rating.",
                forms=["bprs"])
        if self._has_columns("chrcdss_calg1", "chrcdss_calg6"):
            self._emit_mask(
                self._code_mask("chrcdss_calg1", [0])
                & self._observed_mask("chrcdss_calg6")
                & ~self._code_mask("chrcdss_calg6", [0]),
                "CDSS-001", ["chrcdss_calg1", "chrcdss_calg6"],
                "CDSS depression is absent while morning-depression item indicates depression.",
                forms=["cdss"], severity="Warning")
        oasis_items = [f"chroasis_oasis_{number}" for number in range(1, 6)]
        if self._has_columns(*oasis_items, "chroasis_oasis_total10"):
            values = pd.concat([
                self._numeric_series(field) for field in oasis_items
            ], axis=1)
            stored = self._numeric_series("chroasis_oasis_total10")
            sentinel = values.isin([-3, -9, 999]).any(axis=1)
            complete = values.notna().all(axis=1) & ~sentinel
            expected = values.sum(axis=1)
            self._emit_mask(
                complete & self._observed_mask("chroasis_oasis_total10")
                & (stored - expected).abs().gt(1e-9),
                "OASIS-001", [*oasis_items, "chroasis_oasis_total10"],
                "OASIS total does not equal the sum of five complete substantive items.",
                forms=["oasis"])
            no_anxiety = self._code_mask("chroasis_oasis_1", [0])
            contradiction = no_anxiety & (
                self._numeric_series("chroasis_oasis_2").gt(0)
                | self._numeric_series("chroasis_oasis_4").gt(0)
                | self._numeric_series("chroasis_oasis_5").gt(0))
            self._emit_mask(
                contradiction, "OASIS-002",
                ["chroasis_oasis_1", "chroasis_oasis_2",
                 "chroasis_oasis_4", "chroasis_oasis_5"],
                "OASIS reports no anxiety but positive intensity or interference.",
                forms=["oasis"])

        # Perceived-discrimination Other and recruitment exact subtype paths.
        recruitment_map = {
            1: "physician", 2: "nonphysician", 3: "child", 4: "community",
            5: "psychiatry", 6: "college", 7: "school", 8: "adult",
            9: "judicial", 10: "consumer", 11: "self", 15: "other",
        }
        if "chrrecruit" in self._raw:
            for code, suffix in recruitment_map.items():
                detail = f"chrrecruit_{suffix}"
                if detail in self._raw:
                    self._emit_mask(
                        self._observed_mask("chrrecruit")
                        & ~self._code_mask("chrrecruit", [code])
                        & self._observed_mask(detail),
                        "RECRUIT-001", ["chrrecruit", detail],
                        "Recruitment subtype is populated for a different primary source.",
                        forms=["recruitment_source"])
                other = f"{detail}_other"
                if self._has_columns(detail, other):
                    other_codes = [
                        choice for choice, label in self.catalog.fields[detail].choices
                        if "other" in label.lower()]

        # Health-condition parent/checkbox/Other families, using each field's
        # own Other code (heart is 4; most peers are 99).
        for token in ("alle", "lung", "arth", "endo", "bloo", "deve", "skin",
                      "heart", "gast", "neur", "vira"):
            parent, choices, other = (
                f"chrhealth_{token}", f"chrhealth_{token}cond",
                f"chrhealth_{token}oth" if token != "neur" else "chrhealth_neuroth")
            if parent not in self._raw:
                continue
            choice_columns = [column for column in self._raw
                              if column.startswith(choices + "___")]
            selected = pd.Series(False, index=self._raw.index, dtype=bool)
            for column in choice_columns:
                selected |= self._raw[column].map(_is_yes)
            self._emit_mask(
                self._code_mask(parent, [0])
                & (selected | (other in self._raw and self._observed_mask(other))),
                "HEALTHCOND-001", [parent, choices, other],
                "Health-condition parent No retains condition choices/Other text.",
                forms=["health_conditions_genetics_fluid_biomarkers"])
            spec = self.catalog.fields.get(choices)
            if spec and other in self._raw:
                other_codes = [code for code, label in spec.choices
                               if "other" in label.lower()]
                if other_codes:
                    selected_other = self._checkbox_selected(choices, other_codes)
                    checkbox_known = self._all_observed(choice_columns)
                    self._emit_mask(
                        checkbox_known & ~selected_other & self._observed_mask(other),
                        "HEALTHCOND-002", [choices, other],
                        "Health-condition Other text is present while Other is not selected.",
                        forms=["health_conditions_genetics_fluid_biomarkers"])
        # Co-enrollment five-slot topology, chronology, and Other companions.
        if "chrcoen_other_studies" in self._raw:
            slot_masks: list[pd.Series] = []
            for number in range(1, 6):
                prefix = f"chrcoen_study{number}"
                fields = [column for column in self._raw if column.startswith(prefix)]
                current = self._any_observed(fields)
                slot_masks.append(current)
                self._rule_order(
                    "COENROLL-002", f"{prefix}_start", f"{prefix}_stop",
                    f"Co-enrollment study {number} stop precedes start.",
                    forms=["coenrollment_form"])
                for suffix in ("name", "reason1", "reason2", "reason3", "reason4",
                               "reason5", "reason6", "reason7"):
                    controller, companion = f"{prefix}_{suffix}", f"{prefix}_{suffix}_other"
                    spec = self.catalog.fields.get(controller)
                    if not spec or companion not in self._raw:
                        continue
                    other_codes = [code for code, label in spec.choices
                                   if "other" in label.lower()]
            self._emit_mask(
                self._code_mask("chrcoen_other_studies", [0])
                & pd.concat(slot_masks, axis=1).any(axis=1),
                "COENROLL-001", ["chrcoen_other_studies"],
                "No other studies is selected but co-enrollment slots contain data.",
                forms=["coenrollment_form"])

        # TBI reporter applicability, declared versus numeric counts, age order,
        # and summary requirements. Informant disagreement is deliberately not
        # treated as a hard contradiction.
        if "chrtbi_sourceinfo" in self._raw:
            for reporter, allowed_sources in (("subject", [1, 3]), ("parent", [2, 3])):
                head = f"chrtbi_{reporter}_head_injury" if reporter == "subject" else "chrtbi_parent_headinjury"
                details = [column for column in self._raw
                           if column.startswith(f"chrtbi_{reporter}_") and column != head]
                applicable = self._code_mask("chrtbi_sourceinfo", allowed_sources)
                self._emit_mask(
                    self._observed_mask("chrtbi_sourceinfo")
                    & ~applicable & self._any_observed([head, *details]),
                    "TBI-001", ["chrtbi_sourceinfo", head],
                    f"{reporter.title()} TBI block contains data although that reporter is not a source.",
                    forms=["traumatic_brain_injury_screen"])
                times = f"chrtbi_{reporter}_times"
                if head in self._raw and times in self._raw:
                    self._emit_mask(
                        applicable & self._code_mask(head, [0])
                        & self._any_observed([times, *details]),
                        "TBI-002", [head, times],
                        f"{reporter.title()} reports no head injury but injury details are present.",
                        forms=["traumatic_brain_injury_screen"])
        if self._has_columns("chrtbi_age_first_inj", "chrtbi_age_recent_inj"):
            first = self._numeric_series("chrtbi_age_first_inj")
            recent = self._numeric_series("chrtbi_age_recent_inj")
            age = self._participant_age("chrtbi_interview_date")
            self._emit_mask(
                first.notna() & recent.notna() & first.gt(recent),
                "TBI-003", ["chrtbi_age_first_inj", "chrtbi_age_recent_inj"],
                "Age at first head injury exceeds age at most recent injury.",
                forms=["traumatic_brain_injury_screen"])
            self._emit_mask(
                (first.notna() & (first.lt(0) | (age.notna() & first.gt(age))))
                | (recent.notna() & (recent.lt(0) | (age.notna() & recent.gt(age)))),
                "TBI-004", ["chrtbi_age_first_inj", "chrtbi_age_recent_inj"],
                "TBI summary injury age is negative or exceeds age at interview.",
                forms=["traumatic_brain_injury_screen"])
        if "chrtbi_number_injs" in self._raw:
            total = self._numeric_series("chrtbi_number_injs")
            self._emit_mask(
                self._observed_mask("chrtbi_number_injs")
                & (total.isna() | total.lt(0) | total.mod(1).ne(0)),
                "TBI-005", ["chrtbi_number_injs"],
                "TBI summary injury count must be a nonnegative integer.",
                forms=["traumatic_brain_injury_screen"])

        # Stable FIGS role constants and direct chronology.
        for variable, expected in (("chrfigs_mother_sex", 2),
                                   ("chrfigs_father_sex", 1)):
            if variable in self._raw:
                self._emit_mask(
                    self._observed_mask(variable) & ~self._code_mask(variable, [expected]),
                    "FIGS-001", [variable],
                    "FIGS governed mother/father sex-role constant is incorrect.",
                    forms=["family_interview_for_genetic_studies_figs"])
        for prefix in ("chrpps_f", "chrpps_m"):
            dob = prefix + "dob"
            if dob in self._raw:
                participant_dob = next((field for field in (
                    "chrdemo_dob", "chrguid_dob") if field in self._raw), None)
                if participant_dob:
                    self._rule_order(
                        "PPS-001", dob, participant_dob,
                        "Biologic-parent DOB is not earlier than participant DOB.",
                        forms=["psychosis_polyrisk_score", "sociodemographics"])
        if "chrpps_sixmo" in self.catalog.fields and "chrpps_othstress" in self._raw:
            other_codes = [code for code, label in self.catalog.fields["chrpps_sixmo"].choices
                           if "other" in label.lower()]

        # Legacy status form explicit gates.
        self._rule_forbids(
            "STATUS-LEGACY-002", "statusform_reentrydate", [2],
            ["statusform_reentrydate1"],
            "Legacy re-entry No retains a re-entry date.", forms=["status_form"])

    def _run_final_deterministic_extensions(self) -> None:
        """Close deterministic gaps that require form-specific state models.

        These rules deliberately avoid clinical equivalence assumptions.  They
        cover chronology, explicit controller/detail semantics, repeated-state
        topology, and governed lookup tables that cannot be inferred from one
        dictionary row in isolation.
        """
        # --------------------------------------------------------------
        # Current health: all context timestamps must precede the authored
        # interview, and illness/vaccine state must agree with its details.
        health_form = "current_health_status"
        for variable in ("chrchs_ate", "chrchs_tobuse", "chrchs_mardate"):
            self._rule_order(
                "HEALTH-019", variable, "chrblood_drawdate",
                "Current-health pre-draw timestamp is later than the blood draw.",
                forms=[health_form,
                       "blood_sample_preanalytic_quality_assurance"])
        for variable in (
                "chrchs_ate", "chrchs_tobuse", "chrchs_mardate",
                "chrchs_mensdate", "chrchs_sickdate", "chrchs_covidinfdate",
                "chrchs_covidvac1date", "chrchs_covidvac2date"):
            self._rule_order(
                "HEALTH-010", variable, "chrchs_interview_date",
                "Current-health context timestamp is later than the interview.",
                forms=[health_form])

        self._rule_forbids(
            "HEALTH-011", "chrchs_sick", [0], ["chrchs_sickdate"],
            "No sickness is reported but a sickness date remains populated.",
            forms=[health_form])
        self._rule_forbids(
            "HEALTH-012", "chrchs_covidinf", [0],
            ["chrchs_covidinfdate", "chrchs_covidinfdays",
             "chrchs_covidinfhosp", "chrchs_covidinfhospdays"],
            "No COVID infection is reported but infection/hospital details remain.",
            forms=[health_form])
        if self._has_columns(
                "chrchs_covidinfdate", "chrchs_interview_date",
                "chrchs_covidinfdays"):
            infection = self._parsed_series("chrchs_covidinfdate")
            interview = self._parsed_series("chrchs_interview_date")
            days = self._numeric_series("chrchs_covidinfdays")
            elapsed = (interview.dt.normalize() - infection.dt.normalize()).dt.days
            self._emit_mask(
                days.notna() & (days.lt(0) | (elapsed.notna() & days.gt(elapsed))),
                "HEALTH-013",
                ["chrchs_covidinfdate", "chrchs_covidinfdays",
                 "chrchs_interview_date"],
                "COVID illness days are negative or exceed elapsed days through interview.",
                forms=[health_form])
        if "chrchs_covidinfhospdays" in self._raw:
            hospital_days = self._numeric_series("chrchs_covidinfhospdays")
            self._emit_mask(
                self._observed_mask("chrchs_covidinfhospdays")
                & (hospital_days.isna() | hospital_days.lt(0)),
                "HEALTH-013", ["chrchs_covidinfhospdays"],
                "COVID hospitalization days must be a nonnegative number.",
                forms=[health_form])

        vaccine_yes = self._checkbox_selected("chrchs_covidvac", [1, 2, 3, 4, 5])
        if "chrchs_covidvacnum" in self._raw:
            vaccine_count = self._numeric_series("chrchs_covidvacnum")
            self._emit_mask(
                vaccine_yes & self._observed_mask("chrchs_covidvacnum")
                & (vaccine_count.isna() | vaccine_count.lt(1)
                   | vaccine_count.mod(1).ne(0)),
                "HEALTH-014", ["chrchs_covidvac", "chrchs_covidvacnum"],
                "A supplied COVID vaccine dose count must be a positive integer.",
                forms=[health_form])

        # --------------------------------------------------------------
        # Current medication state: declared introduction month, terminal
        # stopped state, offset, first/last-use chronology, and the 25->26
        # continuation bridge.  Past courses have no monthly trajectory and
        # are intentionally left to the generic/legacy course checks.
        medication_months = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24)
        for number in range(1, 51):
            form = (
                "current_pharmaceutical_treatment_floating_med_125"
                if number <= 25 else
                "current_pharmaceutical_treatment_floating_med_2650")
            base = f"chrpharm_med{number}"
            name, introduction = f"{base}_name", f"{base}_tp"
            if name not in self._raw:
                continue
            states = [(month, f"{base}_mo{month}") for month in medication_months
                      if f"{base}_mo{month}" in self._raw]
            if introduction in self._raw and states:
                intro = self._numeric_series(introduction)
                before_intro = pd.Series(False, index=self._raw.index, dtype=bool)
                for month, state in states:
                    before_intro |= (
                        intro.notna() & intro.ge(month)
                        & self._observed_mask(state))
                self._emit_mask(
                    before_intro, "MED-STATE-002",
                    [introduction, *[state for _, state in states]],
                    f"Medication slot {number} contains a month state at/before its declared introduction.",
                    forms=[form])

            stopped = pd.Series(False, index=self._raw.index, dtype=bool)
            later_after_stop = pd.Series(False, index=self._raw.index, dtype=bool)
            for _, state in states:
                later_after_stop |= stopped & self._observed_mask(state)
                stopped |= self._code_mask(state, [2])
            if states:
                self._emit_mask(
                    later_after_stop, "MED-STATE-003",
                    [state for _, state in states],
                    f"Medication slot {number} has a later state after a terminal Stopped state.",
                    forms=[form])
            offset = f"{base}_offset"
            if offset in self._raw and states:
                state_known = pd.concat(
                    [self._observed_mask(state) for _, state in states], axis=1
                ).any(axis=1)
                intermittent_candidates = [field for field in (
                    "chrpharm_interm_meds" if number == 1 else "",
                    f"chrpharm_interm_meds_{number}",
                    f"chrpharm_interm_meds_{number - 1}") if field in self._raw]
                intermittent_stop = (
                    pd.concat([self._code_mask(field, [1])
                               for field in intermittent_candidates], axis=1).any(axis=1)
                    if intermittent_candidates else
                    pd.Series(False, index=self._raw.index, dtype=bool))
                self._emit_mask(
                    self._observed_mask(offset) & state_known
                    & ~stopped & ~intermittent_stop,
                    "MED-STATE-004", [offset, *intermittent_candidates],
                    f"Medication slot {number} has an offset without a stopped/intermittent-stopped state.",
                    forms=[form])

            onset = f"{base}_onset"
            first_dose, last_use = (
                f"chrpharm_firstdose_med{number}",
                f"chrpharm_lastuse_med{number}")
            for endpoint in (first_dose, last_use):
                self._rule_order(
                    "MED-DATE-002", onset, endpoint,
                    f"Medication slot {number} first/last-use date precedes onset.",
                    forms=[form])
            modification = "chrpharm_date_mod" if number <= 25 else "chrpharm_date_mod_2"
            for endpoint in (first_dose, last_use, offset):
                self._rule_order(
                    "MED-DATE-003", endpoint, modification,
                    f"Medication slot {number} course date is after the medication assessment/modification date.",
                    forms=[form])

        slot_26_fields = [field for field in self._raw
                          if field.startswith("chrpharm_med26_")]
        if "chrpharm_med25_add" in self._raw and slot_26_fields:
            self._emit_mask(
                self._any_observed(slot_26_fields)
                & self._observed_mask("chrpharm_med25_add")
                & ~self._code_mask("chrpharm_med25_add", [1]),
                "MED-TOPO-002", ["chrpharm_med25_add", *slot_26_fields[:8]],
                "Medication slot 26 is populated without the slot-25 bridge gate.",
                forms=["current_pharmaceutical_treatment_floating_med_125",
                       "current_pharmaceutical_treatment_floating_med_2650"])

        # --------------------------------------------------------------
        # MRI, Axivity, and MindLAMP operational state.
        if self._has_columns("chrmri_session_num", "chrmri_confirm"):
            first_session = self._code_mask("chrmri_session_num", [1])
            later_session = self._code_mask("chrmri_session_num", [2, 3, 4, 5, 6, 7])
            self._emit_mask(
                first_session & self._observed_mask("chrmri_confirm")
                & ~self._code_mask("chrmri_confirm", [3]),
                "MRI-006", ["chrmri_session_num", "chrmri_confirm"],
                "First MRI session must mark prior-scanner/coil confirmation N/A.",
                forms=["mri_run_sheet"])
            self._emit_mask(
                later_session & self._code_mask("chrmri_confirm", [3]),
                "MRI-006", ["chrmri_session_num", "chrmri_confirm"],
                "Later MRI session cannot mark prior-scanner/coil confirmation N/A.",
                forms=["mri_run_sheet"])

        for form, base in (
                ("digital_biomarkers_axivity_checkin", "chraxci_qaqc1"),
                ("digital_biomarkers_axivity_end_of_12month_study_pe", "chraxe_qaqc1")):
            columns = [f"{base}___{code}" for code in (1, 2, 3)
                       if f"{base}___{code}" in self._raw]
            if columns:
                selected_count = pd.concat(
                    [self._raw[column].map(_is_yes) for column in columns], axis=1
                ).sum(axis=1)
                self._emit_mask(
                    selected_count.gt(1), "AXIVITY-005", columns,
                    "Axivity QA/QC selects more than one of Yes, No, or N/A.",
                    forms=[form])

        for form, exchange, interview in (
                ("digital_biomarkers_axivity_onboarding", "chrax_date_exchanged",
                 "chrax_interview_date"),
                ("digital_biomarkers_axivity_checkin", "chraxci_date_exchanged",
                 "chraxci_interview_date"),
                ("digital_biomarkers_axivity_end_of_12month_study_pe",
                 "chraxe_dev_exchange", "chraxe_interview_date")):
            self._rule_order(
                "AXIVITY-006", "chric_consent_date", exchange,
                "Axivity exchange date precedes effective consent.",
                forms=["informed_consent_run_sheet", form])
            self._rule_order(
                "AXIVITY-006", exchange, interview,
                "Axivity exchange date is after its interview/handoff date.",
                forms=[form])

        if self._has_columns("chrdig_still_study", "chrdig_terminate"):
            self._emit_mask(
                self._code_mask("chrdig_still_study", [0])
                & self._observed_mask("chrdig_terminate")
                & ~self._code_mask("chrdig_terminate", [1]),
                "MINDLAMP-007", ["chrdig_still_study", "chrdig_terminate"],
                "Participant no longer in the MindLAMP study lacks termination confirmation.",
                forms=["digital_biomarkers_mindlamp_checkin"])
            self._emit_mask(
                self._code_mask("chrdig_still_study", [1])
                & self._code_mask("chrdig_terminate", [1]),
                "MINDLAMP-007", ["chrdig_still_study", "chrdig_terminate"],
                "MindLAMP termination is recorded while the participant is still in study.",
                forms=["digital_biomarkers_mindlamp_checkin"])

        # --------------------------------------------------------------
        # FIGS roster counts.  The 1,900+ symptom/diagnosis matrix cells are
        # already covered by generated domain/branch/requiredness checks; this
        # layer reconciles the roster controls that govern those rows.
        for count_field, role, maximum in (
                ("chrfigs_numsib", "sibling", 9),
                ("chrfigs_numchild", "child", 4)):
            if count_field not in self._raw:
                continue
            count = self._numeric_series(count_field)
            self._emit_mask(
                self._observed_mask(count_field)
                & (count.isna() | count.lt(0) | count.mod(1).ne(0)
                   | count.gt(maximum)),
                "FIGS-002", [count_field],
                f"FIGS {role} count must be an integer from 0 through {maximum}.",
                forms=["family_interview_for_genetic_studies_figs"])
            for number in range(1, maximum + 1):
                prefix = f"chrfigs_{role}{number}_"
                fields = [field for field in self._raw if field.startswith(prefix)]
                if not fields:
                    continue
                any_row = self._any_observed(fields)
                self._emit_mask(
                    count.notna() & count.lt(number) & any_row,
                    "FIGS-003", [count_field, *fields[:10]],
                    f"FIGS {role} {number} contains data beyond the declared {role} count.",
                    forms=["family_interview_for_genetic_studies_figs"])

        # --------------------------------------------------------------
        # WAIS-IV is present in the governed dependency tables but the active
        # CognitionChecks path currently executes only WASI-II (assessment 1).
        if "chriq_assessment" in self._raw:
            assessment = self._raw["chriq_assessment"].map(_canonical_code)
            raw_table = self.cognition_csvs.get("iq_raw_conversion_wais")
            fsiq_table = self.cognition_csvs.get("fsiq_conversion_wais")
            if isinstance(raw_table, pd.DataFrame) and not raw_table.empty:
                ages = self._participant_age("chriq_interview_date")

                def in_authored_range(value: float, token: Any) -> bool:
                    text = _text(token)
                    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", text)
                    if match:
                        return float(match.group(1)) <= value <= float(match.group(2))
                    number = _decimal(text)
                    return number is not None and float(number) == value

                score_pairs = (
                    ("chriq_vocab_raw", "chriq_scaled_vocab", 0),
                    ("chriq_matrix_raw", "chriq_scaled_matrix", 1),
                )
                for position in self._raw.index:
                    if assessment.iat[position] != "2" or pd.isna(ages.iat[position]):
                        continue
                    age_months = int(float(ages.iat[position]) * 12)
                    group_start = next((column for column in range(0, raw_table.shape[1], 3)
                                        if in_authored_range(
                                            age_months, raw_table.iloc[0, column])), None)
                    if group_start is None:
                        self._emit(
                            int(position), "IQ-003", ["chriq_assessment"],
                            "WAIS-IV age is not represented in the governed raw-score conversion table.",
                            forms=["iq_assessment_wasiii_wiscv_waisiv"],
                            severity="Config warning")
                        continue
                    expected_scaled: dict[str, float] = {}
                    for raw_field, scaled_field, offset_index in score_pairs:
                        if not self._has_columns(raw_field, scaled_field):
                            continue
                        if not self._observed_mask(raw_field).iat[position]:
                            continue
                        raw_value = pd.to_numeric(
                            pd.Series([self._raw[raw_field].iat[position]]),
                            errors="coerce").iat[0]
                        if pd.isna(raw_value):
                            continue
                        expected = None
                        for table_row in range(2, len(raw_table)):
                            if in_authored_range(
                                    float(raw_value),
                                    raw_table.iloc[table_row, group_start + offset_index]):
                                expected = pd.to_numeric(
                                    pd.Series([raw_table.iloc[table_row, group_start + 2]]),
                                    errors="coerce").iat[0]
                                break
                        if expected is None or pd.isna(expected):
                            self._emit(
                                int(position), "IQ-003", [raw_field],
                                "WAIS-IV raw score is absent from the governed age-specific conversion table.",
                                forms=["iq_assessment_wasiii_wiscv_waisiv"],
                                severity="Warning")
                            continue
                        expected_scaled[scaled_field] = float(expected)
                        stored = pd.to_numeric(
                            pd.Series([self._raw[scaled_field].iat[position]]),
                            errors="coerce").iat[0]
                        if (self._observed_mask(scaled_field).iat[position]
                                and (pd.isna(stored)
                                     or abs(float(stored) - float(expected)) > 1e-9)):
                            self._emit(
                                int(position), "IQ-003", [raw_field, scaled_field],
                                "WAIS-IV scaled score differs from the governed age/raw-score conversion.",
                                forms=["iq_assessment_wasiii_wiscv_waisiv"])

                    if len(expected_scaled) == 2 and "chriq_scaled_sum" in self._raw:
                        expected_sum = sum(expected_scaled.values())
                        stored_sum = pd.to_numeric(
                            pd.Series([self._raw["chriq_scaled_sum"].iat[position]]),
                            errors="coerce").iat[0]
                        if (self._observed_mask("chriq_scaled_sum").iat[position]
                                and (pd.isna(stored_sum)
                                     or abs(float(stored_sum) - expected_sum) > 1e-9)):
                            self._emit(
                                int(position), "IQ-003",
                                [*expected_scaled, "chriq_scaled_sum"],
                                "WAIS-IV scaled-score sum differs from the two governed subtest scores.",
                                forms=["iq_assessment_wasiii_wiscv_waisiv"])
                        if (isinstance(fsiq_table, pd.DataFrame)
                                and self._has_columns("chriq_fsiq")):
                            sum_column = "chriq_scaled_sum"
                            fsiq_column = "chriq_fsiq"
                            matches = fsiq_table[
                                pd.to_numeric(fsiq_table[sum_column], errors="coerce").eq(
                                    expected_sum)] if (
                                        sum_column in fsiq_table
                                        and fsiq_column in fsiq_table) else pd.DataFrame()
                            if not matches.empty:
                                expected_fsiq = float(matches.iloc[0][fsiq_column])
                                stored_fsiq = pd.to_numeric(
                                    pd.Series([self._raw["chriq_fsiq"].iat[position]]),
                                    errors="coerce").iat[0]
                                if (self._observed_mask("chriq_fsiq").iat[position]
                                        and (pd.isna(stored_fsiq)
                                             or abs(float(stored_fsiq) - expected_fsiq) > 1e-9)):
                                    self._emit(
                                        int(position), "IQ-003",
                                        ["chriq_scaled_sum", "chriq_fsiq"],
                                        "WAIS-IV FSIQ differs from the governed scaled-sum conversion.",
                                        forms=["iq_assessment_wasiii_wiscv_waisiv"])
            elif assessment.eq("2").any():
                self._emit_configuration(
                    "IQ-004", "iq_raw_conversion_wais",
                    "WAIS-IV is selected but its governed raw-score conversion table is unavailable.",
                    form="iq_assessment_wasiii_wiscv_waisiv")
            if assessment.isin({"3", "4"}).any():
                self._emit_configuration(
                    "IQ-004", "wiscv_waisr_conversion_tables",
                    "WISC-V/WAIS-R is selected, but no governed raw-to-scaled/FSIQ lookup table is configured; conversion values cannot be deterministically verified.",
                    form="iq_assessment_wasiii_wiscv_waisiv",
                    severity="Config warning")

    def _run_administrative_scale_extensions(self) -> None:
        """Run deterministic administrative/scale checks not expressible in DD rules."""
        # Co-enrollment uses named successive gates rather than an *_add
        # family. Do not require a stop date because an external study may be
        # ongoing, but every populated slot needs its stable core identity.
        coenrollment_masks: list[pd.Series] = []
        for number in range(1, 6):
            prefix = f"chrcoen_study{number}"
            fields = [variable for variable in self._raw
                      if variable.startswith(prefix + "_")]
            current = self._any_observed(fields)
            coenrollment_masks.append(current)
            if number > 1:
                gate = f"chrcoen_other_studies{number}"
                if gate in self._raw:
                    self._emit_mask(
                        current & self._observed_mask(gate)
                        & ~self._code_mask(gate, [1]),
                        "COENROLL-005", [gate],
                        f"Co-enrollment study slot {number} is populated without its preceding more-studies gate.",
                        forms=["coenrollment_form"])

            status = f"{prefix}_amp_status"
            if status not in self._raw:
                continue
            if number == 1 and f"{prefix}_timepoint" in self._raw:
                self._rule_forbids(
                    "COENROLL-006", status,
                    set(self.catalog.fields[status].choice_codes) - {"15"},
                    [f"{prefix}_timepoint"],
                    "Completed-timepoint detail is retained for another co-enrollment status.",
                    forms=["coenrollment_form"])
        for left_number in range(1, 6):
            left = [f"chrcoen_study{left_number}_{suffix}"
                    for suffix in ("name", "id", "start")]
            if not self._has_columns(*left):
                continue
            for right_number in range(left_number + 1, 6):
                right = [f"chrcoen_study{right_number}_{suffix}"
                         for suffix in ("name", "id", "start")]
                if not self._has_columns(*right):
                    continue
                same = self._all_observed([*left, *right])
                for left_field, right_field in zip(left, right):
                    same &= self._raw[left_field].astype(str).str.strip().str.casefold().eq(
                        self._raw[right_field].astype(str).str.strip().str.casefold())
                self._emit_mask(
                    same, "COENROLL-007", [*left, *right],
                    "Two co-enrollment slots contain the same normalized study/ID/start-date tuple.",
                    forms=["coenrollment_form"], severity="Warning")

        # If a reporter says there was one, two, or 3+ injuries, the declared
        # numeric summary must agree whenever the available reporters give a
        # single unambiguous category. Details are already branch-checked; this
        # explicit pass also prevents stale details under a No head-injury gate.
        reporter_categories = []
        for reporter, head, times in (
            ("subject", "chrtbi_subject_head_injury", "chrtbi_subject_times"),
            ("parent", "chrtbi_parent_headinjury", "chrtbi_parent_times"),
        ):
            details = [variable for variable in self._raw
                       if variable.startswith(f"chrtbi_{reporter}_")
                       and variable not in {head, times}]
            if head in self._raw and details:
                self._emit_mask(
                    self._code_mask(head, [0]) & self._any_observed(details),
                    "TBI-006", [head, *details],
                    f"{reporter.title()} TBI details are populated despite a No head-injury response.",
                    forms=["traumatic_brain_injury_screen"])
            if times in self._raw:
                reporter_categories.append(times)
                if head in self._raw:
                    self._emit_mask(
                        self._valid_operand_mask(head)
                        & self._valid_operand_mask(times)
                        & self._code_mask(head, [1])
                        & self._code_mask(times, [0]),
                        "TBI-008", [head, times],
                        f"{reporter.title()} reports a head injury but gives an injury count of zero.",
                        forms=["traumatic_brain_injury_screen"])

        if self._has_columns("chrtbi_sourceinfo", "chrtbi_severe_inj"):
            source = self._raw["chrtbi_sourceinfo"].map(_canonical_code)
            selected_masks: list[pd.Series] = []
            yes_masks: list[pd.Series] = []
            no_masks: list[pd.Series] = []
            for codes, head in (
                    ({"1", "3"}, "chrtbi_subject_head_injury"),
                    ({"2", "3"}, "chrtbi_parent_headinjury")):
                if head not in self._raw:
                    continue
                selected = source.isin(codes)
                selected_masks.append(selected)
                yes_masks.append(selected & self._valid_operand_mask(head)
                                 & self._code_mask(head, [1]))
                no_masks.append(selected & self._valid_operand_mask(head)
                                & self._code_mask(head, [0]))
            if selected_masks:
                any_selected_yes = pd.concat(yes_masks, axis=1).any(axis=1)
                subject_no = (
                    self._valid_operand_mask("chrtbi_subject_head_injury")
                    & self._code_mask("chrtbi_subject_head_injury", [0])
                    if "chrtbi_subject_head_injury" in self._raw
                    else pd.Series(False, index=self._raw.index, dtype=bool))
                parent_no = (
                    self._valid_operand_mask("chrtbi_parent_headinjury")
                    & self._code_mask("chrtbi_parent_headinjury", [0])
                    if "chrtbi_parent_headinjury" in self._raw
                    else pd.Series(False, index=self._raw.index, dtype=bool))
                all_selected_known_no = (
                    (source.eq("1") & subject_no)
                    | (source.eq("2") & parent_no)
                    | (source.eq("3") & subject_no & parent_no))
                severity, severity_valid = self._clinical_numeric("chrtbi_severe_inj")
                contradiction = (
                    (severity.eq(1) & any_selected_yes)
                    | (severity.gt(1) & all_selected_known_no))
                self._emit_mask(
                    severity_valid & contradiction, "TBI-009",
                    ["chrtbi_sourceinfo", "chrtbi_subject_head_injury",
                     "chrtbi_parent_headinjury", "chrtbi_severe_inj"],
                    "TBI most-severe-injury summary contradicts the selected reporter history.",
                    forms=["traumatic_brain_injury_screen"])

        tbi_age = self._participant_age("chrtbi_interview_date")
        for reporter in ("subject", "parent"):
            for suffix in ("", "2", "3", "4", "5"):
                variable = f"chrtbi_{reporter}_age{suffix}"
                if variable not in self._raw:
                    continue
                injury_age, injury_age_valid = self._clinical_numeric(variable)
                self._emit_mask(
                    injury_age_valid
                    & (injury_age.lt(0)
                       | (tbi_age.notna() & injury_age.gt(tbi_age))),
                    "TBI-010", [variable, "chrtbi_interview_date"],
                    "TBI injury age is negative or exceeds age at interview.",
                    forms=["traumatic_brain_injury_screen"])
        if "chrtbi_number_injs" in self._raw and reporter_categories:
            total = self._numeric_series("chrtbi_number_injs")
            for category in ("1", "2", "3"):
                selected = pd.concat(
                    [self._code_mask(variable, [category]) for variable in reporter_categories
                     if variable in self._raw], axis=1)
                # Agreement is deterministic only if every observed reporter
                # names the same category; reporter disagreement is review-only.
                observed = pd.concat(
                    [self._observed_mask(variable) for variable in reporter_categories
                     if variable in self._raw], axis=1)
                unanimous = selected.any(axis=1) & (
                    ~observed.any(axis=1) | selected.sum(axis=1).eq(observed.sum(axis=1)))
                invalid = total.notna() & (
                    (total.ne(int(category)) if category in {"1", "2"} else total.lt(3)))
                self._emit_mask(
                    unanimous & invalid, "TBI-007", [*reporter_categories, "chrtbi_number_injs"],
                    "TBI numeric injury summary disagrees with the unanimous reporter count category.",
                    forms=["traumatic_brain_injury_screen"])

        # A study-sibling category is an explicit foreign-key relationship,
        # unlike a free-text demographic sibling count.
        if self._has_columns("chrdemo_sibling", "chrdemo_sibling_id"):
            sibling_id = self._observed_mask("chrdemo_sibling_id")
            self._emit_mask(
                self._code_mask("chrdemo_sibling", [1]) & sibling_id, "IDENT-009",
                ["chrdemo_sibling", "chrdemo_sibling_id"],
                "Sibling Study ID is populated when no study sibling is declared.",
                forms=["sociodemographics"])

        # When a real DOB is available, manual age fallbacks must not coexist.
        # We do not infer a relationship between years and months alone because
        # their dictionary calculations intentionally use different rounding.
        if "chrdemo_dob" in self._raw:
            fallbacks = [field for field in (
                "chrdemo_age_mos3", "chrdemo_age_mos2") if field in self._raw]
            self._emit_mask(
                self._observed_mask("chrdemo_dob") & self._any_observed(fallbacks),
                "DEMO-AGE-001", ["chrdemo_dob", *fallbacks],
                "Manual demographic age fallback is retained despite a real DOB.",
                forms=["sociodemographics"])

        # The PAS dictionary exposes no controlling marriage-path field. The
        # only deterministic constraint is that V.1 and V.3 cannot both have
        # substantive scores; it is retained here for direct coverage.
        if self._has_columns("chrpas_pmod_adult3v1", "chrpas_pmod_adult3v3"):
            self._emit_mask(
                self._observed_mask("chrpas_pmod_adult3v1")
                & self._observed_mask("chrpas_pmod_adult3v3"), "PAS-002",
                ["chrpas_pmod_adult3v1", "chrpas_pmod_adult3v3"],
                "Both mutually exclusive PAS adult marriage paths are populated.",
                forms=["premorbid_adjustment_scale"])

    def _run_psychs_longitudinal_row(
            self, position: int, state: dict[str, Any]) -> None:
        """Check only date-ordered, formula-verified PSYCHS history facts."""

        subject = self._raw["subjectid"].iat[position]
        population = "hc" if self._cohort_for_subject(subject) == "hc" else "chr"
        screen_date_variable = "chrpsychs_scr_interview_date"
        follow_prefix = "hcpsychs_fu_" if population == "hc" else "chrpsychs_fu_"
        follow_date_variable = follow_prefix + "interview_date"
        screen_date = (
            self._parsed_series(screen_date_variable).iat[position]
            if screen_date_variable in self._raw else pd.NaT)
        follow_date = (
            self._parsed_series(follow_date_variable).iat[position]
            if follow_date_variable in self._raw else pd.NaT)
        is_followup = pd.notna(follow_date)
        current_date = follow_date if is_followup else screen_date
        if pd.isna(current_date):
            return

        history = state.setdefault("psychs_history", {
            "psychosis": {}, "bips": {}, "apss": {}, "last_date": pd.NaT,
        })
        prior_date = history.get("last_date", pd.NaT)
        has_strict_prior = bool(pd.notna(prior_date) and current_date > prior_date)

        if is_followup and has_strict_prior:
            singleton = pd.Series(False, index=self._raw.index, dtype=bool)
            singleton.iat[position] = True
            for number in range(1, 16):
                prior_psychosis = (
                    number in history["psychosis"]
                    and history["psychosis"][number] < current_date)
                prior_bips = (
                    number in history["bips"]
                    and history["bips"][number] < current_date)
                prior_apss = (
                    number in history["apss"]
                    and history["apss"][number] < current_date)

                previous_psychosis = f"{follow_prefix}{number}c6_prev"
                if (prior_psychosis and previous_psychosis in self._raw
                        and self._valid_operand_mask(previous_psychosis).iat[position]
                        and self._code_mask(previous_psychosis, [0]).iat[position]):
                    self._emit(
                        position, "PSYCHS-LONG-001", [previous_psychosis],
                        "PSYCHS says this symptom had not previously met psychosis criteria despite an earlier formula-verified positive visit.",
                        forms=self._forms_for([previous_psychosis]))

                first_conversion = f"{follow_prefix}{number}c6_conv"
                conversion_onset = f"{follow_prefix}{number}c6_on"
                if (first_conversion in self._raw
                        and self._valid_operand_mask(first_conversion).iat[position]
                        and self._code_mask(first_conversion, [1]).iat[position]):
                    previous_positive = (
                        previous_psychosis in self._raw
                        and self._valid_operand_mask(previous_psychosis).iat[position]
                        and self._code_mask(previous_psychosis, [1]).iat[position])
                    if prior_psychosis or previous_positive:
                        self._emit(
                            position, "PSYCHS-LONG-004",
                            [first_conversion, previous_psychosis],
                            "PSYCHS marks first-time psychosis conversion despite a definitive prior psychosis history.",
                            forms=self._forms_for(
                                [first_conversion, previous_psychosis]))
                    if conversion_onset in self._raw:
                        applicable = self._applicable_operand_mask(
                            (conversion_onset,), singleton)
                        onset = self._parsed_series(conversion_onset).iat[position]
                        if (applicable.iat[position] and pd.notna(onset)
                                and onset <= prior_date):
                            self._emit(
                                position, "PSYCHS-LONG-003", [conversion_onset],
                                "PSYCHS first-conversion onset is not strictly after the immediately previous valid visit.",
                                forms=self._forms_for([conversion_onset]))

                never_bips = f"{follow_prefix}{number}c7_free"
                if ((prior_bips or prior_psychosis) and never_bips in self._raw
                        and self._valid_operand_mask(never_bips).iat[position]
                        and self._code_mask(never_bips, [1]).iat[position]):
                    self._emit(
                        position, "PSYCHS-LONG-002", [never_bips],
                        "PSYCHS says there was no prior BIPS/psychosis history despite an earlier formula-verified positive visit.",
                        forms=self._forms_for([never_bips]))

                never_apss = f"{follow_prefix}{number}c11_free"
                if ((prior_apss or prior_psychosis) and never_apss in self._raw
                        and self._valid_operand_mask(never_apss).iat[position]
                        and self._code_mask(never_apss, [1]).iat[position]):
                    self._emit(
                        position, "PSYCHS-LONG-002", [never_apss],
                        "PSYCHS says there was no prior APSS/psychosis history despite an earlier formula-verified positive visit.",
                        forms=self._forms_for([never_apss]))

                # These fields explicitly say that a new since-prior onset is
                # equal to or after the previous visit; equality is valid.
                for suffix in ("c1_on3", "c1_on6", "c10_on", "c14_on"):
                    variable = f"{follow_prefix}{number}{suffix}"
                    if variable not in self._raw:
                        continue
                    applicable = self._applicable_operand_mask(
                        (variable,), singleton)
                    value = self._parsed_series(variable).iat[position]
                    if (applicable.iat[position] and pd.notna(value)
                            and value < prior_date):
                        self._emit(
                            position, "PSYCHS-LONG-003", [variable],
                            "PSYCHS since-prior-visit onset predates the immediately previous valid PSYCHS visit.",
                            forms=self._forms_for([variable]))

            aggregate = follow_prefix + "ac1_prev"
            prior_any_psychosis = any(
                value < current_date for value in history["psychosis"].values())
            if (prior_any_psychosis and aggregate in self._raw
                    and self._valid_operand_mask(aggregate).iat[position]
                    and self._code_mask(aggregate, [0]).iat[position]):
                self._emit(
                    position, "PSYCHS-LONG-001", [aggregate],
                    "PSYCHS aggregate says no symptom met psychosis criteria previously despite an earlier formula-verified positive symptom.",
                    forms=self._forms_for([aggregate]))

        # Add only calculations whose stored value matched independent formula
        # replay.  Current-visit positives are recorded after checking the
        # past-reference fields so they cannot satisfy their own history.
        if is_followup:
            source_prefix = follow_prefix
            suffixes = {"psychosis": "c6a", "bips": "c10", "apss": "c14"}
        else:
            source_prefix = "chrpsychs_scr_"
            suffixes = {"psychosis": "a6", "bips": "a10", "apss": "a14"}
        for domain, suffix in suffixes.items():
            for number in range(1, 16):
                variable = f"{source_prefix}{number}{suffix}"
                verified = self._verified_calculation_values.get(
                    (int(position), variable))
                if _canonical_code(verified) == "1":
                    history[domain].setdefault(number, current_date)
        if pd.isna(prior_date) or current_date > prior_date:
            history["last_date"] = current_date

    def _update_longitudinal_state(self) -> None:
        """Evaluate deterministic across-event state and retain this snapshot.

        ``qc_forms_main`` invokes ProposedChecks in visit order and calls
        ``reset_run_state`` once per complete run. Dates are nevertheless used
        as the precedence guard so an out-of-order floating record cannot turn
        an earlier value into a false longitudinal reversal.
        """
        if self._raw.empty:
            return
        interview_fields = [
            variable for variable, spec in self.catalog.fields.items()
            if variable in self._raw and "interview_date" in variable
            and spec.is_temporal]
        parsed_interviews = {
            variable: self._parsed_series(variable) for variable in interview_fields}
        interview_fields_by_form: dict[str, list[str]] = defaultdict(list)
        for variable in interview_fields:
            spec = self.catalog.fields.get(variable)
            if spec is not None:
                interview_fields_by_form[spec.form].append(variable)
        substantive_by_form = {
            form: self._substantive_mask(form)
            for form in interview_fields_by_form}

        immutable = (
            "chrdemo_dob", "chrguid_dob", "chrguid_mob", "chrguid_yob",
            "chrdemo_sexassigned", "chrguid_sex_chr", "chrguid_sex_hc")
        assist_lifetime = tuple(
            f"chrassist_whoassist_use{number}" for number in range(1, 11))
        discrimination_lifetime = tuple(
            variable for variable in self.catalog.fields
            if (variable.startswith("chrdim_dim_yesno")
                or variable.startswith("chrdlm_dim_yesno")))
        scid_lifetime = tuple(
            variable for variable in SCID_TERMINAL_ENDPOINTS
            if variable in self.catalog.fields
            and "lifetime" in self.catalog.fields[variable].label.lower())

        for position in self._raw.index:
            row = self._row_at(int(position))
            subject = _text(getattr(row, "subjectid", ""))
            if not subject or subject not in self.subject_info:
                continue
            key = (self.network, subject)
            state = self._longitudinal_state.setdefault(key, {
                "values": {}, "form_dates": {}, "assessment_history": [],
                "seen_timepoints": set(), "mri_sessions": set(),
            })
            state.setdefault("seen_timepoints", set()).add(self.timepoint)
            valid_dates = [series.iat[position] for series in parsed_interviews.values()
                           if pd.notna(series.iat[position])]
            row_date = max(valid_dates) if valid_dates else pd.NaT

            self._run_psychs_longitudinal_row(int(position), state)

            for namespace, variable in (
                    ("real", "chrguid_guid"),
                    ("pseudo", "chrguid_pseudoguid")):
                if (variable not in self._raw
                        or not self._observed_mask(variable).iat[position]):
                    continue
                normalized = _text(self._raw[variable].iat[position]).upper()
                registry_key = (self.network, namespace, normalized)
                prior_subject = self._guid_identifiers.get(registry_key)
                if prior_subject is not None and prior_subject != subject:
                    self._emit(
                        int(position), "IDENT-010", [variable],
                        "GUID namespace value is assigned to more than one participant.",
                        forms=self._forms_for([variable]))
                else:
                    self._guid_identifiers[registry_key] = subject

            for number in range(1, 4):
                for aliquot in "ab":
                    variable = f"chrsaliva_id{number}{aliquot}"
                    if (variable not in self._raw
                            or not self._observed_mask(variable).iat[position]):
                        continue
                    normalized = _text(self._raw[variable].iat[position]).upper()
                    registry_key = (self.network, normalized)
                    observation = (subject, self.timepoint, variable)
                    prior_observation = self._saliva_barcodes.get(registry_key)
                    if (prior_observation is not None
                            and prior_observation != observation
                            and prior_observation[:2] != observation[:2]):
                        self._emit(
                            int(position), "SALIVA-LONG-001", [variable],
                            "Saliva barcode is reused by a distinct specimen observation.",
                            forms=self._forms_for([variable]))
                    elif prior_observation is None:
                        self._saliva_barcodes[registry_key] = observation

            # Immutable source fields may be corrected, but the change must be
            # explicit and reviewed. Never include identifier values in output.
            for variable in immutable:
                if variable not in self._raw:
                    continue
                value = self._raw[variable].iat[position]
                if not self._observed_mask(variable).iat[position]:
                    continue
                prior = state["values"].get(variable)
                if prior and _values_match(prior[1], value) is False:
                    prior_date = prior[0]
                    if pd.notna(row_date) and pd.notna(prior_date) and row_date >= prior_date:
                        self._emit(
                            int(position), "IDENT-007", [variable],
                            "Immutable identity attribute changed across visits; verify an approved correction workflow.",
                            forms=self._forms_for([variable]), severity="Warning")
                if (prior is None
                        or (pd.notna(row_date) and pd.isna(prior[0]))
                        or (pd.notna(row_date) and pd.notna(prior[0])
                            and row_date >= prior[0])):
                    state["values"][variable] = (row_date, value)

            for variable in (*assist_lifetime, *discrimination_lifetime):
                if variable not in self._raw or not self._observed_mask(variable).iat[position]:
                    continue
                value = self._raw[variable].iat[position]
                prior = state["values"].get(variable)
                comparable_later = bool(
                    prior is not None
                    and pd.notna(row_date) and pd.notna(prior[0])
                    and row_date >= prior[0])
                replace_state = (
                    prior is None
                    or (pd.notna(row_date) and pd.isna(prior[0]))
                    or (pd.notna(row_date) and pd.notna(prior[0])
                        and row_date >= prior[0]))
                if comparable_later and _is_yes(prior[1]) and _is_no(value):
                    check_id = "ASSIST-LONG-001" if variable.startswith("chrassist_") else "DISCRIM-LONG-001"
                    self._emit(
                        int(position), check_id, [variable],
                        "A lifetime-history response changed from Yes to No at a later visit; review as correction/reinterpretation.",
                        forms=self._forms_for([variable]), severity="Warning")
                if replace_state:
                    state["values"][variable] = (row_date, value)

            for variable in (*scid_lifetime, "chrschizotypal_summ"):
                if variable not in self._raw or not self._observed_mask(variable).iat[position]:
                    continue
                value = self._raw[variable].iat[position]
                prior = state["values"].get(variable)
                comparable_later = bool(
                    prior is not None
                    and pd.notna(row_date) and pd.notna(prior[0])
                    and row_date >= prior[0])
                replace_state = (
                    prior is None
                    or (pd.notna(row_date) and pd.isna(prior[0]))
                    or (pd.notna(row_date) and pd.notna(prior[0])
                        and row_date >= prior[0]))
                was_positive = (
                    self._positive_endpoint(prior[1]) is True if prior and variable != "chrschizotypal_summ"
                    else (prior is not None and _canonical_code(prior[1]) == "1"))
                now_positive = (
                    self._positive_endpoint(value) is True if variable != "chrschizotypal_summ"
                    else _canonical_code(value) == "1")
                if comparable_later and was_positive and not now_positive:
                    self._emit(
                        int(position), "SCID-LONG-001", [variable],
                        "Lifetime SCID/schizotypal endpoint reverses a prior positive state; review clinician reinterpretation/correction.",
                        forms=self._forms_for([variable]), severity="Warning")
                if replace_state:
                    state["values"][variable] = (row_date, value)

            # Cross-administration ordering for paired baseline/follow-up forms.
            form_date_map: dict[str, tuple[str, str]] = {
                "cssrs": ("chrcssrsb_interview_date", "chrcssrsfu_interview_date"),
                "gfr": ("chrgfrs_interview_date", "chrgfrsfu_interview_date"),
                "gfs": ("chrgfss_interview_date", "chrgfssfu_interview_date"),
                "sofas": ("chrsofas_interview_date", "chrsofas_interview_date_fu"),
            }
            for family, (baseline, followup) in form_date_map.items():
                for kind, variable in (("baseline", baseline), ("followup", followup)):
                    if variable not in self._raw:
                        continue
                    current = parsed_interviews.get(variable, self._parsed_series(variable)).iat[position]
                    if pd.isna(current):
                        continue
                    prior_date = state["form_dates"].get(family)
                    if kind == "followup" and prior_date is not None and current <= prior_date:
                        self._emit(
                            int(position), "LONG-DATE-001", [variable],
                            "Follow-up administration does not occur after the prior valid administration.",
                            forms=self._forms_for([variable]))
                    if prior_date is None or current > prior_date:
                        state["form_dates"][family] = current

            # PSYCHS c0 >= d0 becomes deterministic only with >=30 days since
            # the prior valid interview. Apply independently to CHR and HC.
            for population, prefix in (("chr", "chrpsychs_fu_"),
                                       ("hc", "hcpsychs_fu_")):
                date_var = next((variable for variable in interview_fields
                                 if variable.startswith(prefix)
                                 and "interview_date" in variable), None)
                if date_var is None:
                    continue
                current_date = parsed_interviews[date_var].iat[position]
                if pd.isna(current_date):
                    continue
                prior_date = state["form_dates"].get(f"psychs_{population}")
                if prior_date is not None and (current_date - prior_date).days >= 30:
                    for variable in self._raw:
                        match = re.match(fr"^({re.escape(prefix)}.*)c0$", variable)
                        if not match:
                            continue
                        past_month = match.group(1) + "d0"
                        if past_month not in self._raw:
                            continue
                        c_values, c_valid = self._clinical_numeric(variable)
                        d_values, d_valid = self._clinical_numeric(past_month)
                        c_value = c_values.iat[position]
                        d_value = d_values.iat[position]
                        if (c_valid.iat[position] and d_valid.iat[position]
                                and c_value < d_value):
                            self._emit(
                                int(position), "PSYCHS-004", [variable, past_month],
                                "PSYCHS follow-up severity violates since-prior-visit >= past-month with an interval of at least 30 days.",
                                forms=self._forms_for([variable, past_month]))
                if prior_date is None or current_date > prior_date:
                    state["form_dates"][f"psychs_{population}"] = current_date

            # Axivity active-device identity must remain stable unless exchange
            # is documented at the current event.
            device_candidates = [
                ("chrax_device_id", []),
                ("chraxci_device_id", ["chraxci_exchange", "chraxci_exchange1"]),
                ("chraxe_device_id", ["chraxe_dev_exchange", "chraxe_final_exchange"]),
            ]
            for variable, exchange_fields in device_candidates:
                if variable not in self._raw or not self._observed_mask(variable).iat[position]:
                    continue
                device = _text(self._raw[variable].iat[position]).upper()
                prior_device = state.get("axivity_device")
                present_exchange_fields = [
                    field for field in exchange_fields if field in self._raw]
                exchange_known = (
                    all(self._observed_mask(field).iat[position]
                        for field in present_exchange_fields)
                    if present_exchange_fields else not exchange_fields)
                exchanged = any(
                    field in self._raw and _is_yes(self._raw[field].iat[position])
                    for field in exchange_fields)
                if (prior_device and device != prior_device
                        and exchange_known and not exchanged):
                    self._emit(
                        int(position), "AXIVITY-LONG-001", [variable, *exchange_fields],
                        "Axivity device ID changed without a documented exchange.",
                        forms=self._forms_for([variable, *exchange_fields]))
                if prior_device is None or exchanged or device == prior_device:
                    state["axivity_device"] = device

            # MindLAMP anchors: a declared change must actually differ; without
            # a change workflow, anchor values remain stable.
            lamp_pairs = (
                ("lamp_id", "chrdbb_lamp_id", "chrdig_new_lamp_id"),
                ("phone_model", "chrdbb_phone_model", "chrdig_new_phone_mod"),
                ("phone_software", "chrdbb_phone_software", "chrdig_new_phone_soft_version"),
            )
            changed = (
                "chrdig_detail_change" in self._raw
                and _is_yes(self._raw["chrdig_detail_change"].iat[position]))
            change_explicitly_absent = (
                "chrdig_detail_change" in self._raw
                and self._observed_mask("chrdig_detail_change").iat[position]
                and self._code_mask("chrdig_detail_change", [0]).iat[position])
            for state_key, initial, replacement in lamp_pairs:
                current_variable = initial if (
                    initial in self._raw and self._observed_mask(initial).iat[position]
                ) else replacement if (
                    replacement in self._raw and self._observed_mask(replacement).iat[position]
                ) else None
                if current_variable is None:
                    continue
                value = _text(self._raw[current_variable].iat[position]).casefold()
                prior_value = state.get(state_key)
                if current_variable == replacement and changed and prior_value == value:
                    self._emit(
                        int(position), "MINDLAMP-LONG-001", [replacement],
                        "MindLAMP change is documented but the replacement value equals the active prior value.",
                        forms=self._forms_for([replacement]), severity="Warning")
                if (current_variable == initial and prior_value
                        and value != prior_value and change_explicitly_absent):
                    self._emit(
                        int(position), "MINDLAMP-LONG-001", [initial],
                        "MindLAMP onboarding anchor changed without the check-in change workflow.",
                        forms=self._forms_for([initial]))
                if prior_value is None or current_variable == initial or changed:
                    state[state_key] = value

            if "chrdig_monthly_percent" in self._raw:
                raw_percent = self._raw["chrdig_monthly_percent"].iat[position]
                percent = pd.to_numeric(
                    pd.Series([raw_percent]),
                    errors="coerce").iat[0]
                if not _is_missing_value(raw_percent) and pd.notna(percent):
                    state["mindlamp_percent"] = percent
            if "chrdbe_percent_surveys" in self._raw:
                raw_final_percent = self._raw["chrdbe_percent_surveys"].iat[position]
                final_percent = pd.to_numeric(
                    pd.Series([raw_final_percent]),
                    errors="coerce").iat[0]
                prior_percent = state.get("mindlamp_percent")
                if (not _is_missing_value(raw_final_percent)
                        and pd.notna(final_percent) and prior_percent is not None and abs(
                        final_percent - prior_percent) > 0.01):
                    self._emit(
                        int(position), "MINDLAMP-LONG-002",
                        ["chrdbe_percent_surveys", "chrdig_monthly_percent"],
                        "Final MindLAMP survey percentage differs from the last check-in percentage.",
                        forms=["digital_biomarkers_mindlamp_checkin",
                               "digital_biomarkers_mindlamp_end_of_12month_study_p"],
                        severity="Warning")

            # Rater experience must not decrease for the same rater.
            if self._has_columns("chrpred_username", "chrpred_experience"):
                raw_username = self._raw["chrpred_username"].iat[position]
                rater = _text(raw_username).casefold()
                raw_experience = self._raw["chrpred_experience"].iat[position]
                experience = pd.to_numeric(
                    pd.Series([raw_experience]),
                    errors="coerce").iat[0]
                if (rater and not _is_missing_value(raw_username)
                        and not _is_missing_value(raw_experience)
                        and pd.notna(experience)):
                    prior_experience = state.setdefault("rater_experience", {}).get(rater)
                    if prior_experience is not None and experience < prior_experience:
                        self._emit(
                            int(position), "RA-LONG-001",
                            ["chrpred_username", "chrpred_experience"],
                            "Experience months decreased for the same governed rater.",
                            forms=["ra_prediction"])
                    state["rater_experience"][rater] = max(
                        experience, prior_experience if prior_experience is not None else experience)

            # Duplicate MRI session number/date within participant.
            if self._has_columns("chrmri_session_num", "chrmri_session_year",
                                 "chrmri_session_month", "chrmri_session_day"):
                components = [self._raw[field].iat[position] for field in (
                    "chrmri_session_num", "chrmri_session_year",
                    "chrmri_session_month", "chrmri_session_day")]
                if not any(_is_missing_value(value) for value in components):
                    session = tuple(_canonical_code(value) for value in components)
                    if session in state["mri_sessions"]:
                        self._emit(
                            int(position), "MRI-LONG-001",
                            ["chrmri_session_num", "chrmri_session_year",
                             "chrmri_session_month", "chrmri_session_day"],
                            "MRI session number/date duplicates a prior participant session.",
                            forms=["mri_run_sheet"])
                    state["mri_sessions"].add(session)

            # Store dated substantive administrations so a later floating
            # status record can identify assessments during active withdrawal.
            for form, date_variables in interview_fields_by_form.items():
                if not substantive_by_form[form].iat[position]:
                    continue
                current = next((parsed_interviews[variable].iat[position]
                                for variable in date_variables
                                if pd.notna(parsed_interviews[variable].iat[position])), None)
                if current is not None:
                    state["assessment_history"].append((form, current, self.timepoint))

            withdrawal_var = next((field for field in (
                "chr_subject_date_of_withdrawal", "statusform_withdrawal")
                if field in self._raw and self._observed_mask(field).iat[position]), None)
            reentry_var = next((field for field in (
                "chr_statusform_reentry_date", "statusform_reentrydate1")
                if field in self._raw and self._observed_mask(field).iat[position]), None)
            withdrawal = (
                self._parsed_series(withdrawal_var).iat[position] if withdrawal_var else pd.NaT)
            reentry = self._parsed_series(reentry_var).iat[position] if reentry_var else pd.NaT
            if pd.notna(withdrawal) and pd.notna(reentry):
                allowlisted = {
                    "adverse_events", "revised_status_form", "status_form", "missing_data",
                    "gcp_cbc_with_differential", "gcp_current_health_status"}
                for form, assessment_date, timepoint in state["assessment_history"]:
                    if form in allowlisted:
                        continue
                    if assessment_date > withdrawal and assessment_date < reentry:
                        self._emit(
                            int(position), "STATUS-004", [withdrawal_var or "withdrawal"],
                            f"{form} at {timepoint} occurred after withdrawal and before re-entry; review applicability.",
                            forms=["revised_status_form", form], severity="Warning")


def clear_catalog_cache() -> None:
    """Public test/deployment hook for invalidating dictionary/run caches."""

    ProposedChecks.clear_catalog_cache()
    ProposedChecks.reset_run_state()
