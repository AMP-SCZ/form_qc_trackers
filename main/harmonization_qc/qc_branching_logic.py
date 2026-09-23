"""Standalone audit of translated branching logic against combined exports.

This utility is intentionally separate from the normal QC run. It compares
whether each observed field is populated with whether its translated REDCap
branching expression is true, aggregating examples without changing source
data. The legacy regex translator and Python-expression evaluator remain in
place; invalid translations are quarantined and diagnosed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import pandas as pd


if __package__ in {None, ""}:
    for parent in Path(__file__).resolve().parents[1:3]:
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))

try:  # Full deployed package layout.
    from main.process_variables.transform_branching_logic import (
        TransformBranchingLogic,
    )
    from main.utils.utils import Utils
except ModuleNotFoundError:  # Standalone/flattened development checkout.
    from process_variables.transform_branching_logic import TransformBranchingLogic
    from utils.utils import Utils

from utils.branching_logic_eval import evaluate_branching_logic


class _BranchingAuditUtils:
    """Small Utils subset that does not read global config during tests."""

    missing_code_list = [
        '-3', '-9', -3, -9, -3.0, -9.0, '-3.0', '-9.0',
        '1909-09-09', '1903-03-03', '1901-01-01', '-99', -99, -99.0,
        '-99.0', 999, 999.0, '999', '999.0',
    ]
    absolute_path = str(Path(__file__).resolve().parents[1])

    @staticmethod
    def can_be_float(value):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def all_dtype(self, values):
        output = []
        for value in values:
            if self.can_be_float(value):
                integer = int(value)
                output.extend([
                    integer, str(integer), float(integer), str(float(integer)),
                ])
        return output

    @staticmethod
    def collect_digit(value):
        result = ''
        for char in value:
            if char.isdigit():
                result += char
            elif result:
                return result
        return result


TIMEPOINTS = (
    "screening", "baseline", *(f"month{x}" for x in range(1, 13)),
    "month18", "month24", "floating", "conversion",
)
TIMEPOINT_ORDER = {value: index for index, value in enumerate(TIMEPOINTS)}
NETWORK_ORDER = {"PRONET": 0, "PRESCIENT": 1}
CANONICAL_FILE_RE = re.compile(
    r"^AMPSCZ-combined-redcap_"
    r"(?P<timepoint>screening|baseline|month_(?:[1-9]|1[0-2]|18|24)|"
    r"floating_forms|conversion)_"
    r"(?P<network>ProNET|PRONET|PRESCIENT)(?:-day1to1)?\.csv$",
    re.IGNORECASE,
)

MISMATCH_COLUMNS = (
    "network", "timepoint", "variable", "form", "variable_value",
    "count", "subjects", "pronet_branching_logic",
    "converted_branching_logic", "BL_cond",
)
EVAL_ERROR_COLUMNS = (
    "network", "timepoint", "variable", "error_type", "error", "count",
)


@dataclass(frozen=True)
class CombinedCSVSpec:
    path: Path
    network: str
    timepoint: str


def _normalize_timepoint(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = normalized.replace("floating_forms", "floating")
    match = re.fullmatch(r"month_?(\d+)", normalized)
    if match:
        return f"month{int(match.group(1))}"
    return normalized


def _configured_path(raw: str, base: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


class TestTransformBranchingLogic:
    """Audit branching-logic behavior across canonical combined CSVs."""

    __test__ = False

    def __init__(
            self, config_path: str | Path | None = None, *,
            config_info: Mapping[str, Any] | None = None,
            data_dictionary_df: pd.DataFrame | None = None):
        self.utils = _BranchingAuditUtils()

        if config_info is not None:
            self.config_info = dict(config_info)
            config_base = Path.cwd()
        else:
            if config_path is None:
                candidates = [
                    Path(__file__).resolve().parents[1] / "config.json",
                    Path(__file__).resolve().parents[2] / "config.json",
                ]
                selected = next((path for path in candidates if path.is_file()), None)
                if selected is None:
                    raise FileNotFoundError(
                        "Could not locate config.json for branching-logic audit")
                config_path = selected
            config_file = Path(config_path).expanduser().resolve()
            with config_file.open("r", encoding="utf-8") as handle:
                self.config_info = json.load(handle)
            config_base = config_file.parent

        paths = self.config_info.get("paths", {})
        self.combined_csv_path = _configured_path(
            paths["combined_csv_path"], config_base)
        self.dependencies_path = _configured_path(
            paths["dependencies_path"], config_base)
        self.output_path = _configured_path(paths["output_path"], config_base)
        if str(self.config_info.get("testing_enabled", "False")).casefold() == "true":
            self.output_path = self.output_path / "testing"
        self.output_path.mkdir(parents=True, exist_ok=True)

        if data_dictionary_df is not None:
            self.data_dictionary_df = data_dictionary_df.copy()
            self.data_dictionary_path = None
            self.data_dictionary_candidates = []
        else:
            self.data_dictionary_df = self._read_data_dictionary(
                paths, config_base)
        transformer = TransformBranchingLogic(
            self.data_dictionary_df, utils=self.utils,
            config_info=self.config_info)
        self.converted_bl = transformer.convert_all_branching_logic()
        self.invalid_conversions = transformer.validate_converted_branching_logic(
            self.converted_bl, write_diagnostics=False)

        self.grouped_variables = self._load_json("grouped_variables.json")
        self.forms_per_var = self.grouped_variables["var_forms"]
        self.important_form_vars = self._load_json("important_form_vars.json")
        self.forms_per_timepoint = self._load_json("forms_per_timepoint.json")
        self.general_check_vars = self._load_json(
            "general_check_vars.json", required=False, default={})
        self.excluded_forms = {
            network: set(self.general_check_vars.get(
                "excluded_forms", {}).get(network, []))
            for network in NETWORK_ORDER
        }
        self.excl_bl = self._load_json(
            "excluded_branching_logic_vars.json", required=False, default={})
        self.subject_info = self._load_json(
            "subject_info.json", required=False, default={})
        self.vars_added_later = self._load_json(
            "variables_added_later.json", required=False, default={})

        self.excl_bl.update({
            row["variable"]: row["original_branching_logic"]
            for row in self.invalid_conversions
        })
        identifier_path = self.dependencies_path / "identifier_effects.csv"
        self.identifier_effects_missing = not identifier_path.is_file()
        if self.identifier_effects_missing:
            self.ident_bl_vars = set()
        else:
            identifier_df = pd.read_csv(identifier_path, keep_default_na=False)
            self.ident_bl_vars = set(identifier_df.loc[
                identifier_df["affected_col"] == "branching_logic", "var"
            ].astype(str))

        self.miss_codes = set(self.utils.missing_code_list)
        self._dtype_one = set(self.utils.all_dtype([1]))
        self._dtype_two = set(self.utils.all_dtype([2]))
        self._dtype_missing_completion = set(self.utils.all_dtype([3, 4]))
        self.eval_errors: dict[tuple[str, str, str, str, str], int] = {}
        self.discovery_diagnostics: dict[str, Any] = {}

    def _read_data_dictionary(self, paths, config_base):
        explicit = paths.get("data_dictionary_path") or self.config_info.get(
            "data_dictionary_path")
        if explicit:
            selected = _configured_path(explicit, config_base)
            matches = [selected]
        else:
            directory = self.dependencies_path / "data_dictionary"
            matches = sorted(
                directory.glob("*current_data_dictionary*.csv"),
                key=lambda path: path.name.casefold())
            if not matches:
                raise FileNotFoundError(
                    "No data dictionary matching 'current_data_dictionary' "
                    f"was found in {directory}")
            exact = directory / "current_data_dictionary.csv"
            if exact.is_file():
                selected = exact
            elif len(matches) == 1:
                selected = matches[0]
            else:
                raise FileNotFoundError(
                    "Multiple current data dictionaries were found; set "
                    "paths.data_dictionary_path explicitly: "
                    + ", ".join(str(path.resolve()) for path in matches))
        if not selected.is_file():
            raise FileNotFoundError(f"Data dictionary does not exist: {selected}")
        self.data_dictionary_path = selected.resolve()
        self.data_dictionary_candidates = [str(path.resolve()) for path in matches]
        return pd.read_csv(selected, keep_default_na=False)

    def _load_json(self, filename, *, required=True, default=None):
        path = self.dependencies_path / filename
        if not path.is_file():
            if required:
                raise FileNotFoundError(
                    f"Required branching-logic dependency is missing: {path}")
            return default
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read dependency {path}: {exc}") from exc
        return value

    def run_script(self):
        return self.compare_bl_to_database()

    def discover_combined_csvs(self) -> list[CombinedCSVSpec]:
        """Return unambiguous canonical combined inputs in stable order."""

        if not self.combined_csv_path.is_dir():
            raise FileNotFoundError(
                f"Combined CSV directory does not exist: {self.combined_csv_path}")
        slots: dict[tuple[str, str], CombinedCSVSpec] = {}
        for path in self.combined_csv_path.iterdir():
            if not path.is_file():
                continue
            match = CANONICAL_FILE_RE.fullmatch(path.name)
            if match is None:
                continue
            network = match.group("network").upper()
            if network == "PRONET":
                network = "PRONET"
            timepoint = _normalize_timepoint(match.group("timepoint"))
            slot = (network, timepoint)
            spec = CombinedCSVSpec(path.resolve(), network, timepoint)
            if slot in slots and slots[slot].path != spec.path:
                raise RuntimeError(
                    f"Ambiguous combined CSVs for {network}/{timepoint}: "
                    f"{slots[slot].path}, {spec.path}")
            slots[slot] = spec
        if not slots:
            raise FileNotFoundError(
                "No canonical AMPSCZ combined CSVs were found in "
                f"{self.combined_csv_path}")
        expected = {
            (network, timepoint)
            for network in NETWORK_ORDER for timepoint in TIMEPOINTS
        }
        missing = sorted(
            expected.difference(slots),
            key=lambda item: (
                NETWORK_ORDER[item[0]], TIMEPOINT_ORDER[item[1]]),
        )
        self.discovery_diagnostics = {
            "input_count": len(slots),
            "missing_canonical_slots": [
                f"{network}/{timepoint}" for network, timepoint in missing
            ],
        }
        return sorted(
            slots.values(),
            key=lambda spec: (
                NETWORK_ORDER[spec.network], TIMEPOINT_ORDER[spec.timepoint],
                spec.path.name.casefold()),
        )

    def _cohort(self, curr_row) -> str:
        subject = str(getattr(curr_row, "subjectid", ""))
        info = self.subject_info.get(subject, {})
        cohort = str(info.get("cohort", "")).strip().casefold()
        if cohort in {"chr", "hc"}:
            return cohort
        raw = getattr(curr_row, "chrcrit_part", "")
        if raw in self.utils.all_dtype([1]):
            return "chr"
        if raw in self.utils.all_dtype([2]):
            return "hc"
        return ""

    @staticmethod
    def _prescient_completion_var(completion_var: str) -> str:
        return (completion_var + "_rpms").replace(
            "_hc", "").replace("onboarding", "checkin")

    def _has_moved_past(self, curr_row, timepoint: str) -> bool:
        if timepoint in {"floating", "conversion"}:
            return False
        status = _normalize_timepoint(getattr(curr_row, "visit_status", ""))
        return (status in TIMEPOINT_ORDER
                and TIMEPOINT_ORDER[status] > TIMEPOINT_ORDER[timepoint])

    def _form_has_evidence(self, curr_row, form_info) -> bool:
        candidates = [
            form_info.get("interview_date_var", ""),
            form_info.get("entry_date_var", ""),
            *form_info.get("non_branch_logic_vars", []),
        ]
        return any(
            variable and hasattr(curr_row, variable)
            and str(getattr(curr_row, variable)).strip() != ""
            for variable in candidates
        )

    def _extra_form_applicable(self, curr_row, form: str) -> bool:
        info = self.subject_info.get(
            str(getattr(curr_row, "subjectid", "")), {})
        if form == "pubertal_developmental_scale":
            age = info.get("age", "")
            return self.utils.can_be_float(age) and float(age) <= 18
        if "axivity" in form:
            return info.get("axivity_opt", "") in self.utils.all_dtype([1])
        if "mindlamp" in form:
            return info.get("mindlamp_opt", "") in self.utils.all_dtype([1, 2])
        return True

    def _form_is_active(
            self, curr_row, form: str, form_info: Mapping[str, Any],
            network: str, timepoint: str) -> bool:
        if timepoint == "floating":
            return True
        completion_var = str(form_info.get("completion_var", ""))
        if not completion_var:
            return self._form_has_evidence(curr_row, form_info)
        if network == "PRESCIENT":
            completion_var = self._prescient_completion_var(completion_var)
        if hasattr(curr_row, completion_var):
            value = getattr(curr_row, completion_var)
            if network == "PRESCIENT" and value in self._dtype_missing_completion:
                return False
            if value in self._dtype_two:
                return True
        if network == "PRESCIENT" and self._has_moved_past(curr_row, timepoint):
            return True
        return False

    def _form_is_missing(
            self, curr_row, form_info: Mapping[str, Any], network: str) -> bool:
        missing_var = str(form_info.get("missing_var", ""))
        if (missing_var and hasattr(curr_row, missing_var)
                and getattr(curr_row, missing_var) in self._dtype_one):
            return True
        completion_var = str(form_info.get("completion_var", ""))
        if network == "PRESCIENT" and completion_var:
            completion_var = self._prescient_completion_var(completion_var)
            if (hasattr(curr_row, completion_var)
                    and getattr(curr_row, completion_var)
                    in self._dtype_missing_completion):
                return True
        return False

    def filter_rows(
            self, curr_row, col: str, network: str, timepoint: str,
            repeat_forms=frozenset()) -> bool:
        """Return whether this row/field is applicable to the BL audit."""

        if col not in self.forms_per_var or col not in self.converted_bl:
            return False
        form = self.forms_per_var[col]
        if form not in self.important_form_vars:
            return False
        if col in self.excl_bl:
            return False
        if form in self.excluded_forms.get(network, set()):
            return False
        if col in self.ident_bl_vars and network == "PRONET":
            return False
        if any(token in col for token in (
                "error", "chrsaliva_flag", "chrchs_flag", "_err",
                "invalid", "notes")):
            return False

        row_repeat = str(
            getattr(curr_row, "redcap_repeat_instrument", "") or ""
        ).strip()
        if row_repeat and row_repeat != form:
            return False
        if not row_repeat and form in repeat_forms:
            return False

        cohort = self._cohort(curr_row)
        if not cohort:
            return False
        scheduled = self.forms_per_timepoint.get(cohort, {}).get(timepoint, [])
        if form not in scheduled:
            return False

        form_info = self.important_form_vars[form]
        if not self._extra_form_applicable(curr_row, form):
            return False
        if self._form_is_missing(curr_row, form_info, network):
            return False
        return self._form_is_active(
            curr_row, form, form_info, network, timepoint)

    def _record_eval_error(self, spec, variable, exc):
        key = (
            spec.network, spec.timepoint, variable,
            type(exc).__name__, str(exc),
        )
        self.eval_errors[key] = self.eval_errors.get(key, 0) + 1

    def _write_csv(self, path: Path, rows, columns):
        frame = pd.DataFrame(rows, columns=columns)
        tmp = path.with_name(f"{path.name}.tmp")
        try:
            frame.to_csv(tmp, index=False)
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def compare_bl_to_database(self):
        mismatch_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        processed_rows = 0
        evaluated_fields = 0
        specs = self.discover_combined_csvs()

        for spec in specs:
            combined_df = pd.read_csv(
                spec.path, keep_default_na=False, low_memory=False)
            repeat_forms = frozenset(
                str(value).strip()
                for value in combined_df.get(
                    "redcap_repeat_instrument", pd.Series(dtype="object"))
                if str(value).strip()
            )
            for curr_row in combined_df.itertuples(index=False):
                processed_rows += 1
                for col in combined_df.columns:
                    if not self.filter_rows(
                            curr_row, col, spec.network, spec.timepoint,
                            repeat_forms):
                        continue
                    expression = self.converted_bl[col][
                        "converted_branching_logic"]
                    if expression == "":
                        continue
                    try:
                        branch_is_true = evaluate_branching_logic(
                            expression, curr_row=curr_row, instance=self)
                    except Exception as exc:  # Structured, never silent.
                        self._record_eval_error(spec, col, exc)
                        continue
                    evaluated_fields += 1
                    value = getattr(curr_row, col)
                    is_blank = (value == "" or (
                        spec.network == "PRESCIENT" and value in self.miss_codes))
                    is_observed = value != "" and value not in self.miss_codes
                    if not ((is_blank and branch_is_true)
                            or (is_observed and not branch_is_true)):
                        continue

                    form = self.forms_per_var[col]
                    key = (
                        spec.network, spec.timepoint, col, str(value),
                        branch_is_true,
                    )
                    if key not in mismatch_groups:
                        mismatch_groups[key] = {
                            "network": spec.network,
                            "timepoint": spec.timepoint,
                            "variable": col,
                            "form": form,
                            "variable_value": value,
                            "count": 0,
                            "subjects": [],
                            "pronet_branching_logic": self.converted_bl[col][
                                "original_branching_logic"],
                            "converted_branching_logic": expression,
                            "BL_cond": branch_is_true,
                        }
                    group = mismatch_groups[key]
                    group["count"] += 1
                    if len(group["subjects"]) < 50:
                        group["subjects"].append(
                            getattr(curr_row, "subjectid", ""))

        mismatch_rows = list(mismatch_groups.values())
        for network in NETWORK_ORDER:
            network_rows = [
                row for row in mismatch_rows if row["network"] == network
            ]
            self._write_csv(
                self.output_path / f"bl_mismatches_{network}.csv",
                network_rows, MISMATCH_COLUMNS)

        error_rows = [
            {
                "network": network,
                "timepoint": timepoint,
                "variable": variable,
                "error_type": error_type,
                "error": error,
                "count": count,
            }
            for (network, timepoint, variable, error_type, error), count
            in sorted(self.eval_errors.items())
        ]
        self._write_csv(
            self.output_path / "branching_logic_eval_errors.csv",
            error_rows, EVAL_ERROR_COLUMNS)
        invalid_columns = (
            "variable", "original_branching_logic",
            "converted_branching_logic", "reason", "error_type", "error",
        )
        self._write_csv(
            self.output_path / "branching_logic_invalid_conversions.csv",
            self.invalid_conversions, invalid_columns)

        diagnostics = {
            **self.discovery_diagnostics,
            "processed_rows": processed_rows,
            "evaluated_fields": evaluated_fields,
            "mismatch_groups": len(mismatch_rows),
            "evaluation_error_groups": len(error_rows),
            "evaluation_error_count": sum(self.eval_errors.values()),
            "invalid_conversion_count": len(self.invalid_conversions),
            "identifier_effects_missing": self.identifier_effects_missing,
            "dependencies_path": str(self.dependencies_path),
            "combined_csv_path": str(self.combined_csv_path),
            "data_dictionary_path": (
                str(self.data_dictionary_path)
                if self.data_dictionary_path is not None else "<provided>"),
            "data_dictionary_candidate_count": len(
                self.data_dictionary_candidates),
        }
        diagnostics_path = self.output_path / "branching_logic_qc_diagnostics.json"
        tmp = diagnostics_path.with_name(f"{diagnostics_path.name}.tmp")
        try:
            tmp.write_text(
                json.dumps(diagnostics, indent=2), encoding="utf-8")
            tmp.replace(diagnostics_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return mismatch_rows


if __name__ == "__main__":
    TestTransformBranchingLogic().run_script()
