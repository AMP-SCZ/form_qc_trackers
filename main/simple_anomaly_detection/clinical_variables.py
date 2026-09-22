"""Fail-closed clinical-variable scoping for the anomaly report.

The study already defines a ``Clinical measures`` missingness domain.  This
module joins that form list to ``grouped_variables.json::var_forms`` and uses
the result as the anomaly detector's production allowlist.  Interview dates
and rater identifiers are retained only as calculation context; they can never
be emitted as anomaly variables.

The REDCap data dictionary is then applied as a second, additive filter: a
variable is reportable only if the name/label rules below *and* the
dictionary's own structured columns both call it a measurement.  Each source
catches what the other cannot.  The dictionary knows that a field is
display-only, PHI-bearing, free text, or a date, none of which is recoverable
from a variable name; the name/label rules know that a ``yesno`` field labelled
"Interviewer has reviewed and approved the automatic calculation" is an
attestation rather than a participant response, which no dictionary column
expresses.  Neither is used alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Dict, Mapping, Sequence, Set, Tuple

import pandas as pd

from .common import ID_COLUMNS, looks_like_date_column
from .selection import variable_components


CLINICAL_DOMAIN_LABEL = "Clinical measures"
_CHECKBOX_SUFFIX_RE = re.compile(r"___\d+$")

# REDCap display, validation, instruction, and companion fields that can sit on
# an otherwise-clinical form.  They are not participant measurements.  These
# patterns intentionally complement (rather than replace) common.py's shared
# exclusions; the production metadata currently contains hundreds of
# ``dateerror`` and ``*_instructions`` fields that the shared rules do not
# catch.
_WORKFLOW_NAME_RE = re.compile(
    r"date_?error|dateerror|_invalid\d*$|_instructions?(?:_|$)"
    r"|_instruct(?:ion|ions)?(?:_|$)|_inst(?:r|\d|_|$)|_err\d*$"
    r"|_errors?\d*$|_table(?:_|$)|_anchors?$|_remind"
    r"|_prompts?[a-z0-9]*(?:_|$)|prompt_"
    r"|_probes?$|_show_|_comments?(?:_|$)|_notes?(?:_|$)"
    r"|_(?:desc|descrip)$|_descriptions?(?:_|$)"
    r"|_overview_version$|_version$|_list$"
    r"|_overview$|(?:score|adult)error$|currscore(?:high|low)$"
    r"|^(?:chr|hc)psychs_.*_pro_|^chrscid_lifechart_check$"
    r"|^chrscid_substance_overview_2$|^chrbprs_bprs_.*_p\d*$"
    r"|^chrcdss_cdss(?:check|interviewer)$|^chrnsipr_.*_p\d+$"
    r"|^chrnsipr_(?:encourage|item7_desc|nsipr\d+_pr\d+|school_work|selfcare_clean)$"
    r"|^(?:chr|hc)psychs_.*_app$",
    re.IGNORECASE,
)
_STRUCTURAL_NAME_RE = re.compile(
    r"(?:^|_)(?:complete|missing)(?:_|$)|redcap|record_id|_ampscz_id(?:_|$)"
    r"|^_tp$",
    re.IGNORECASE,
)
_PLURAL_DATE_RE = re.compile(r"(?:^|_)(?:date|dates)(?:_|$)", re.IGNORECASE)
_WORKFLOW_LABEL_PHRASES = (
    "redcap username",
    "interviewer location",
    "dates of other",
    "show scid overview version",
    "please fix",
    "must be entered",
    "proceed to",
    "skip to",
    "scroll down",
)
_CODED_RESPONSE_LABEL_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?,\s*")
_PSEUDO_VARIABLES = frozenset({"(subject composite)", "(site composite)"})
_PSEUDO_ANOMALY_TYPES = frozenset({
    "whole_subject_outlier", "whole_site_outlier",
})
_CLEAN_VARIABLE_LIST_TYPES = frozenset({
    "correlative_outlier",
    "cluster_3d_outlier",
    "cluster_3d_longitudinal_outlier",
    "date_anomaly_simple",
    "timeline_anomaly",
})

# ---------------------------------------------------------------------------
# REDCap data dictionary contract
# ---------------------------------------------------------------------------

_DD_VARIABLE_COLUMN = "Variable / Field Name"
_DD_FORM_COLUMN = "Form Name"
_DD_TYPE_COLUMN = "Field Type"
_DD_VALIDATION_COLUMN = "Text Validation Type OR Show Slider Number"
_DD_IDENTIFIER_COLUMN = "Identifier?"
_DD_REQUIRED_COLUMNS = (
    _DD_VARIABLE_COLUMN, _DD_FORM_COLUMN, _DD_TYPE_COLUMN,
    _DD_VALIDATION_COLUMN, _DD_IDENTIFIER_COLUMN,
)
# ``descriptive`` fields are rendered HTML that captures nothing; ``file`` is an
# upload slot.  Neither can hold a participant measurement.
_DD_NON_DATA_TYPES = frozenset({"descriptive", "file"})
# Free text is real participant input but is not analyzable by these detectors,
# and on the clinical forms it is overwhelmingly identifier-bearing narrative.
_DD_FREE_TEXT_TYPES = frozenset({"notes"})
# Dates stay context-only, matching the name-based rule below.  The dictionary
# catches the ones whose names never say "date" (``chrmed_cond10_onset``).
_DD_DATE_VALIDATIONS = frozenset({
    "date_ymd", "date_mdy", "date_dmy",
    "datetime_ymd", "datetime_mdy", "datetime_dmy",
    "datetime_seconds_ymd", "datetime_seconds_mdy", "datetime_seconds_dmy",
    "time", "time_mm_ss",
})
# Scored scale items the dictionary mismarks as identifiers: a radio rating of
# recurrent suspiciousness and a dropdown count of head-injury occurrences hold
# no identifying information.  The exemption is deliberately scoped to the
# Identifier? rule alone -- these must still satisfy every other check -- so a
# later dictionary edit that makes one of them display-only still excludes it.
_DD_IDENTIFIER_OVERRIDES = frozenset({
    "chrschizotypal_par00a7",
    "chrtbi_parent_times",
})
# Searched in order when no explicit path is supplied.  The first entry matches
# utils.read_data_dictionary's convention; the second is the copy that ships in
# this repository's root.  Each is relative to the repository root.
_DD_SEARCH_PATHS = (
    ("dependencies", "data_dictionary", "current_data_dictionary.csv"),
    ("data_dict_new.csv",),
)
# Searched, in order and relative to an explicitly supplied dependencies
# folder.  A caller who names their metadata folder has said where to look;
# reaching back out to its parent would search somewhere they did not name.
_DD_DEPENDENCY_SEARCH_PATHS = (
    ("data_dictionary", "current_data_dictionary.csv"),
    ("data_dict_new.csv",),
)


class ClinicalVariableContractError(RuntimeError):
    """Raised when clinical-scope metadata is absent or malformed."""


def _norm(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold()


def _base_variable(value: object) -> str:
    return _CHECKBOX_SUFFIX_RE.sub("", _norm(value))


def _load_json_object(path: str, description: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ClinicalVariableContractError(
            f"Cannot load {description} at {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ClinicalVariableContractError(
            f"{description} at {path} must be a non-empty JSON object")
    return value


def _label_from_translation(variable: str, translation: object) -> str:
    text = str(translation or "")
    prefix = f"{variable} ="
    if text.casefold().startswith(prefix.casefold()):
        return text[len(prefix):].strip()
    return text.strip()


def _is_workflow_field(variable: str, label: str) -> bool:
    if _WORKFLOW_NAME_RE.search(variable):
        return True
    lower_label = label.casefold().strip()
    if lower_label in {"comment", "comments", "note", "notes",
                       "instruction", "instructions"}:
        return True
    # Routing text can also be the LABEL of a real coded SCID response (e.g.
    # ``1, IF ... CHECK HERE AND SKIP``).  Only use these phrases as a display
    # signal when the field is neither a coded response nor a direct question.
    return (not _CODED_RESPONSE_LABEL_RE.match(label)
            and "?" not in label
            and any(phrase in lower_label
                    for phrase in _WORKFLOW_LABEL_PHRASES))


def _resolve_data_dictionary_path(explicit: str | None,
                                  candidates: Sequence[str]) -> str:
    """Find the REDCap data dictionary, or say exactly where it was sought."""
    if explicit:
        if not os.path.isfile(explicit):
            raise ClinicalVariableContractError(
                f"Data dictionary not found at {explicit}")
        return explicit
    searched = []
    for candidate in candidates:
        searched.append(candidate)
        if os.path.isfile(candidate):
            return candidate
        # Accept one unambiguous dated variant beside the canonical name.
        directory = os.path.dirname(candidate)
        stem = os.path.splitext(os.path.basename(candidate))[0]
        if os.path.isdir(directory):
            variants = sorted(
                os.path.join(directory, name)
                for name in os.listdir(directory)
                if stem in name and name.lower().endswith(".csv"))
            if len(variants) == 1:
                return variants[0]
            if len(variants) > 1:
                raise ClinicalVariableContractError(
                    f"Multiple data dictionary files match {stem!r} in "
                    f"{directory}: {[os.path.basename(v) for v in variants]}. "
                    "Keep one, or pass data_dictionary_path explicitly.")
    raise ClinicalVariableContractError(
        "Cannot locate the REDCap data dictionary; it is required to scope "
        "reportable clinical variables. Searched: " + ", ".join(searched)
        + ". Pass data_dictionary_path explicitly, or run with "
        "clinical_only=False to bypass clinical scoping entirely.")


def _load_data_dictionary(path: str) -> Tuple[Dict[str, dict], Set[str]]:
    """Return ``{variable: dictionary metadata}`` and the set of form names."""
    try:
        frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    except (OSError, ValueError) as exc:
        raise ClinicalVariableContractError(
            f"Cannot read data dictionary at {path}: {exc}") from exc
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in _DD_REQUIRED_COLUMNS
               if column not in frame.columns]
    if missing:
        raise ClinicalVariableContractError(
            f"Data dictionary at {path} is missing required column(s): "
            + ", ".join(missing))
    if frame.empty:
        raise ClinicalVariableContractError(
            f"Data dictionary at {path} has no rows")

    entries: Dict[str, dict] = {}
    forms: Set[str] = set()
    for record in frame.itertuples(index=False):
        values = dict(zip(frame.columns, record))
        variable = _norm(values[_DD_VARIABLE_COLUMN])
        form = _norm(values[_DD_FORM_COLUMN])
        if form:
            forms.add(form)
        if not variable:
            continue
        entries[variable] = {
            "form": form,
            "field_type": _norm(values[_DD_TYPE_COLUMN]),
            "validation": _norm(values[_DD_VALIDATION_COLUMN]),
            "identifier": _norm(values[_DD_IDENTIFIER_COLUMN]),
        }
    return entries, forms


def _dictionary_exclusion(variable: str, entry: dict | None) -> str:
    """Reason the dictionary disqualifies a variable, or "" to keep it.

    Silence is deliberately permissive: a variable the dictionary does not
    describe keeps whatever verdict the name/label rules gave it.  That is only
    safe because ``load_clinical_variable_catalog`` refuses to run against a
    dictionary missing any clinical form, so silence cannot mean "this whole
    instrument went unchecked".
    """
    if entry is None:
        return ""
    field_type = entry["field_type"]
    if field_type in _DD_NON_DATA_TYPES:
        return f"data dictionary: '{field_type}' field captures no data"
    if (entry["identifier"] == "y"
            and variable not in _DD_IDENTIFIER_OVERRIDES):
        return "data dictionary: marked Identifier? = y"
    if field_type in _DD_FREE_TEXT_TYPES:
        return f"data dictionary: '{field_type}' field is free text"
    if entry["validation"] in _DD_DATE_VALIDATIONS:
        return (f"data dictionary: '{entry['validation']}' date/time field "
                "retained only as context")
    return ""


@dataclass(frozen=True)
class ClinicalVariableCatalog:
    """Resolved production contract for reportable and support variables."""

    clinical_forms: frozenset
    clinical_variables: frozenset
    auxiliary_variables: frozenset
    variable_forms: Mapping[str, str]
    excluded_reasons: Mapping[str, str]
    domain_map_path: str
    grouped_variables_path: str
    important_form_vars_path: str
    data_dictionary_path: str = ""
    dictionary_excluded_count: int = 0

    def form_for(self, variable: object) -> str:
        return self.variable_forms.get(_base_variable(variable), "")

    def is_clinical(self, variable: object) -> bool:
        return _base_variable(variable) in self.clinical_variables

    def is_auxiliary(self, variable: object) -> bool:
        return _base_variable(variable) in self.auxiliary_variables

    def disposition(self, variable: object) -> Tuple[str, str]:
        """Return ``(disposition, reason)`` for an observed source column."""
        base = _base_variable(variable)
        if base in {_norm(v) for v in ID_COLUMNS}:
            return "retained_identity", "participant identifier required for findings"
        if base in self.clinical_variables:
            return "retained_clinical", "mapped to Clinical measures"
        if base in self.auxiliary_variables:
            return "retained_context", "calculation context only; blocked from findings"
        if base in self.excluded_reasons:
            return "excluded", self.excluded_reasons[base]
        form = self.variable_forms.get(base, "")
        if form:
            return "excluded", f"mapped to nonclinical form: {form}"
        return "excluded", "unmapped variable (fail-closed)"


def load_clinical_variable_catalog(
    domain_map_path: str | None = None,
    grouped_variables_path: str | None = None,
    important_form_vars_path: str | None = None,
    data_dictionary_path: str | None = None,
    dependencies_path: str | None = None,
) -> ClinicalVariableCatalog:
    """Load and validate the repository's clinical-variable contract.

    ``dependencies_path`` relocates the whole metadata folder in one argument;
    the per-file arguments above still win individually where supplied.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if dependencies_path:
        dependencies = os.path.abspath(dependencies_path)
        if not os.path.isdir(dependencies):
            raise ClinicalVariableContractError(
                f"Dependencies folder does not exist: {dependencies}")
        # Search inside the folder the caller named. The folder need not be
        # called "dependencies", so resolving through its parent would look
        # somewhere they never pointed at.
        dictionary_candidates = [os.path.join(dependencies, *parts)
                                 for parts in _DD_DEPENDENCY_SEARCH_PATHS]
    else:
        dependencies = os.path.join(project_root, "dependencies")
        dictionary_candidates = [os.path.join(project_root, *parts)
                                 for parts in _DD_SEARCH_PATHS]
    domain_map_path = domain_map_path or os.path.join(
        dependencies, "missingness_domain_forms.json")
    grouped_variables_path = grouped_variables_path or os.path.join(
        dependencies, "grouped_variables.json")
    important_form_vars_path = important_form_vars_path or os.path.join(
        dependencies, "important_form_vars.json")

    domains = _load_json_object(domain_map_path, "missingness domain map")
    matching_domains = [entry for entry in domains.values()
                        if isinstance(entry, dict)
                        and _norm(entry.get("label")) == _norm(CLINICAL_DOMAIN_LABEL)]
    if len(matching_domains) != 1:
        raise ClinicalVariableContractError(
            f"Expected exactly one {CLINICAL_DOMAIN_LABEL!r} domain in "
            f"{domain_map_path}; found {len(matching_domains)}")
    forms_value = matching_domains[0].get("forms")
    if not isinstance(forms_value, list):
        raise ClinicalVariableContractError(
            f"{CLINICAL_DOMAIN_LABEL!r} forms must be a JSON list in "
            f"{domain_map_path}")
    clinical_forms = frozenset(_norm(form) for form in forms_value if _norm(form))
    if not clinical_forms:
        raise ClinicalVariableContractError(
            f"{CLINICAL_DOMAIN_LABEL!r} has no mapped forms in {domain_map_path}")

    grouped = _load_json_object(grouped_variables_path, "grouped variable map")
    raw_forms = grouped.get("var_forms")
    if not isinstance(raw_forms, dict) or not raw_forms:
        raise ClinicalVariableContractError(
            f"grouped variable map at {grouped_variables_path} has no "
            "non-empty var_forms object")
    raw_translations = grouped.get("var_translations", {})
    if not isinstance(raw_translations, dict):
        raw_translations = {}
    translations = {
        _norm(variable): text for variable, text in raw_translations.items()
        if _norm(variable)
    }
    variable_forms = {
        _norm(variable): _norm(form)
        for variable, form in raw_forms.items()
        if _norm(variable) and _norm(form)
    }
    mapped_forms = frozenset(variable_forms.values())
    missing_variable_maps = sorted(clinical_forms - mapped_forms)
    if missing_variable_maps:
        raise ClinicalVariableContractError(
            "Clinical form(s) have no variable mapping in "
            f"{grouped_variables_path}: {', '.join(missing_variable_maps)}")

    important = _load_json_object(
        important_form_vars_path, "important form-variable map")
    important_forms = frozenset(_norm(form) for form in important)
    missing_form_contracts = sorted(clinical_forms - important_forms)
    if missing_form_contracts:
        raise ClinicalVariableContractError(
            "Clinical form(s) have no important-form metadata in "
            f"{important_form_vars_path}: {', '.join(missing_form_contracts)}")
    admin_roles: Set[str] = set()
    auxiliary: Set[str] = set()
    for form, info in important.items():
        if _norm(form) not in clinical_forms or not isinstance(info, dict):
            continue
        for role in ("missing_var", "completion_var", "missing_spec_var",
                     "interview_date_var", "entry_date_var",
                     "redcap_user_var"):
            value = _norm(info.get(role))
            if value:
                admin_roles.add(value)
        for role in ("interview_date_var", "redcap_user_var"):
            value = _norm(info.get(role))
            if value:
                auxiliary.add(value)

    # Catch legacy/typo rater fields not populated in important_form_vars.
    from .rater_effect import is_rater_variable  # lazy: avoids import cycle

    clinical: Set[str] = set()
    excluded: Dict[str, str] = {}
    empty_sample = pd.Series(dtype=object)
    for variable, form in variable_forms.items():
        if form not in clinical_forms:
            continue
        translation = translations.get(variable, "")
        label = _label_from_translation(variable, translation)
        if variable in auxiliary or is_rater_variable(variable):
            auxiliary.add(variable)
            continue
        if variable in admin_roles:
            excluded[variable] = "clinical-form administration field"
        elif _STRUCTURAL_NAME_RE.search(variable):
            excluded[variable] = "identifier/completion/missingness metadata field"
        elif (looks_like_date_column(variable, empty_sample)
              or _PLURAL_DATE_RE.search(variable)):
            excluded[variable] = "date field retained only when needed as context"
        elif _is_workflow_field(variable, label):
            excluded[variable] = "instruction/display/validation companion field"
        else:
            clinical.add(variable)

    # Second, additive pass: the dictionary's structured columns disqualify
    # display-only, PHI-marked, free-text, and date fields that survived the
    # name/label rules.  It can only narrow the allowlist, never widen it.
    dictionary_path = _resolve_data_dictionary_path(
        data_dictionary_path, dictionary_candidates)
    dictionary, dictionary_forms = _load_data_dictionary(dictionary_path)
    forms_absent_from_dictionary = sorted(clinical_forms - dictionary_forms)
    if forms_absent_from_dictionary:
        # Without this guard a truncated dictionary would leave every variable
        # on the missing instruments unexamined and silently reportable.
        raise ClinicalVariableContractError(
            "Data dictionary at "
            f"{dictionary_path} does not describe clinical form(s): "
            + ", ".join(forms_absent_from_dictionary))
    # The form-level guard alone is not enough. A dictionary whose variable
    # names have drifted from grouped_variables.json -- which is regenerated
    # from a *different* copy of the dictionary -- keeps every form name while
    # describing none of the variables, so every field on that instrument would
    # be silently unexamined and reportable again. Assert the coverage the
    # exclusion rule's docstring actually relies on. This costs nothing today:
    # the dictionary describes all 16,458 clinical-form variables.
    undescribed = sorted(variable for variable in clinical
                         if variable not in dictionary)
    if undescribed:
        preview = ", ".join(undescribed[:10])
        raise ClinicalVariableContractError(
            f"Data dictionary at {dictionary_path} does not describe "
            f"{len(undescribed)} candidate clinical variable(s), so they "
            "cannot be screened for display-only, identifier, free-text, or "
            f"date fields: {preview}"
            + (", ..." if len(undescribed) > 10 else ""))
    dictionary_excluded = 0
    for variable in sorted(clinical):
        reason = _dictionary_exclusion(variable, dictionary.get(variable))
        if reason:
            clinical.discard(variable)
            excluded[variable] = reason
            dictionary_excluded += 1

    if not clinical:
        raise ClinicalVariableContractError(
            "Clinical variable allowlist resolved to zero fields; refusing to "
            "fall back to unrestricted anomaly detection")
    forms_with_data = frozenset(variable_forms[v] for v in clinical)
    empty_clinical_forms = sorted(clinical_forms - forms_with_data)
    if empty_clinical_forms:
        raise ClinicalVariableContractError(
            "Clinical form(s) resolved to no reportable data variables after "
            f"administration/display exclusions: {', '.join(empty_clinical_forms)}")
    return ClinicalVariableCatalog(
        clinical_forms=clinical_forms,
        clinical_variables=frozenset(clinical),
        auxiliary_variables=frozenset(auxiliary),
        variable_forms=variable_forms,
        excluded_reasons=excluded,
        domain_map_path=os.path.abspath(domain_map_path),
        grouped_variables_path=os.path.abspath(grouped_variables_path),
        important_form_vars_path=os.path.abspath(important_form_vars_path),
        data_dictionary_path=os.path.abspath(dictionary_path),
        dictionary_excluded_count=dictionary_excluded,
    )


def scope_slices_to_clinical(
    per_slice: Mapping[Tuple[str, str], pd.DataFrame],
    catalog: ClinicalVariableCatalog,
) -> Tuple[Dict[Tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Return detector inputs limited to clinical fields plus safe context."""
    scoped: Dict[Tuple[str, str], pd.DataFrame] = {}
    observed: Dict[str, dict] = {}
    id_names = {_norm(v) for v in ID_COLUMNS}

    for key, frame in per_slice.items():
        keep = []
        slice_name = f"{key[0]}/{key[1]}"
        for column in frame.columns:
            normalized = _norm(column)
            base = _base_variable(column)
            is_identity = normalized in id_names
            retained = (is_identity or catalog.is_clinical(base)
                        or catalog.is_auxiliary(base))
            if retained:
                keep.append(column)
            disposition, reason = catalog.disposition(base)
            audit = observed.setdefault(normalized, {
                "variable": str(column),
                "base_variable": base,
                "form": catalog.form_for(base),
                "disposition": disposition,
                "reason": reason,
                "slices": set(),
            })
            audit["slices"].add(slice_name)
        scoped[key] = frame.loc[:, keep].copy()

    audit_rows = []
    for item in observed.values():
        slices = sorted(item.pop("slices"))
        audit_rows.append({
            **item,
            "present_slice_count": len(slices),
            "present_slices": "; ".join(slices),
        })
    audit_frame = pd.DataFrame(audit_rows)
    if not audit_frame.empty:
        audit_frame = audit_frame.sort_values(
            ["disposition", "form", "variable"], kind="stable").reset_index(drop=True)
    return scoped, audit_frame


def filter_findings_to_clinical(
    frame: pd.DataFrame,
    catalog: ClinicalVariableCatalog,
) -> Tuple[pd.DataFrame, int]:
    """Drop any finding whose real variable legs are not all clinical."""
    if frame is None or frame.empty:
        return (frame.copy() if isinstance(frame, pd.DataFrame)
                else pd.DataFrame()), 0

    def _clean_components(value: object) -> Tuple[str, ...]:
        """Parse detector-owned comma lists containing variable names only."""
        components = set()
        for raw in str(value or "").split(","):
            leg = raw.strip()
            if not leg or leg.startswith("("):
                continue
            if "::" in leg:
                leg = leg.split("::", 1)[1]
            if "@" in leg:
                leg = leg.split("@", 1)[0]
            if "=" in leg:
                leg = leg.split("=", 1)[0]
            normalized = _norm(leg)
            if normalized and normalized not in {"corr-drift", "(corr-drift)"}:
                components.add(normalized)
        return tuple(sorted(components))

    def _keep(row: pd.Series) -> bool:
        # Quota selection may deliberately set ``primary_variable`` to one
        # dominant leg.  Clinical enforcement is stricter: inspect every leg
        # named by the finding, not only the quota key.  Date fan-out rows still
        # use variables_involved through variable_components' date-specific
        # rule; rater rows use their scored ``variable`` and therefore do not
        # mistake the auxiliary rater id for a reported measurement.
        anomaly_type = _norm(row.get("anomaly_type"))
        clean_list = row.get("variables_involved", "")
        clean_text = str(clean_list or "")
        if (anomaly_type == "correlative_outlier"
                and " (residual target) + " in clean_text):
            # Directional fan-out summaries render their evidence as
            # ``target (residual target) + partner1, partner2``. Recover the
            # target from the structured residual_target field and the clean
            # partner-name tail; the human annotation is not a variable name.
            _, partner_text = clean_text.split(
                " (residual target) + ", 1)
            target = row.get("residual_target", "") or row.get("variable", "")
            components = _clean_components(f"{target}, {partner_text}")
        elif anomaly_type in _CLEAN_VARIABLE_LIST_TYPES and _norm(clean_list):
            # Correlative binary-cell labels embed HUMAN values in ``variable``
            # (values may themselves contain commas or '&').  These detectors
            # also provide a machine-oriented, names-only variables_involved
            # list, which is the only safe component source.
            components = _clean_components(clean_list)
        else:
            guard_row = row.copy()
            guard_row["primary_variable"] = ""
            components = variable_components(guard_row)
        if components:
            return all(catalog.is_clinical(component)
                       for component in components)
        variable = _norm(row.get("variable"))
        return (variable in _PSEUDO_VARIABLES
                and anomaly_type in _PSEUDO_ANOMALY_TYPES)

    mask = frame.apply(_keep, axis=1)
    kept = frame.loc[mask].reset_index(drop=True)
    return kept, int((~mask).sum())
