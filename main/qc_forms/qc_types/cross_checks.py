"""Audited, vectorized cross-form consistency checks.

The source catalog lives in ``test_cross_checks.py`` for traceability to the
operator-supplied prototype.  This adapter evaluates only the rules approved by
that audit, once per combined dataframe, and emits the canonical FormCheck row
schema used by reconciliation and tracker generation.
"""

from dataclasses import dataclass

import pandas as pd

from utils.branching_logic_eval import evaluate_branching_logic

try:  # Production package layout.
    from qc_forms.form_check import FormCheck
    from qc_forms.test_cross_checks import (
        ALTERNATIVE_EVIDENCE_GROUPS,
        DATE_COLUMNS,
        ENABLED_RULE_NUMBERS,
        RULE_LABELS,
        RULES,
        extract_columns,
        prepare_dataframe,
    )
except ModuleNotFoundError:  # Flat local-test workspace.
    from qc_forms.form_check import FormCheck
    from qc_forms.test_cross_checks import (
        ALTERNATIVE_EVIDENCE_GROUPS,
        DATE_COLUMNS,
        ENABLED_RULE_NUMBERS,
        RULE_LABELS,
        RULES,
        extract_columns,
        prepare_dataframe,
    )


REPORT_NAME = "Cross Checks"
NUMERIC_MISSING_CODES = frozenset({-3.0, -9.0, -99.0, 999.0})


@dataclass(frozen=True)
class CrossCheckRule:
    """One immutable, precompiled operational rule."""

    number: int
    check_id: str
    label: str
    expression: str
    variables: tuple[str, ...]
    alternative_evidence_groups: tuple[tuple[str, ...], ...]
    compiled: object


def _build_catalog() -> tuple[CrossCheckRule, ...]:
    catalog = []
    for number in sorted(ENABLED_RULE_NUMBERS):
        expression = RULES[number - 1]
        if "df.filter" in expression:
            raise ValueError(
                f"Enabled rule {number} uses untracked regex evidence")
        catalog.append(CrossCheckRule(
            number=number,
            check_id=f"CROSS-QC-{number:03d}",
            label=RULE_LABELS[number],
            expression=expression,
            variables=tuple(extract_columns(expression)),
            alternative_evidence_groups=tuple(
                tuple(group)
                for group in ALTERNATIVE_EVIDENCE_GROUPS.get(number, ())),
            compiled=compile(
                expression, f"<cross-check-{number:03d}>", "eval"),
        ))
    return tuple(catalog)


RULE_CATALOG = _build_catalog()


def _casefold_key(value) -> str:
    return "" if value is None else str(value).strip().casefold()


class CrossChecks(FormCheck):
    """Run approved cross-form rules once per network/timepoint dataframe."""

    REPORT_NAME = REPORT_NAME
    RULE_CATALOG = RULE_CATALOG

    def __init__(self, combined_df, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
        self.rule_status = {}
        self._var_forms = self.grouped_vars.get("var_forms", {})
        self._run(combined_df)

    def __call__(self):
        return self.final_output_list

    def _run(self, combined_df):
        if combined_df.empty:
            return
        if "subjectid" not in combined_df.columns:
            raise KeyError(
                "CrossChecks requires a subjectid column in the combined CSV")

        raw = combined_df.reset_index(drop=True)
        applicable = []
        for rule in self.RULE_CATALOG:
            missing = [
                variable for variable in rule.variables
                if variable not in raw.columns]
            if missing:
                self.rule_status[rule.check_id] = (
                    "not_applicable_missing_columns", tuple(missing))
                continue
            applicable.append(rule)

        columns = list(dict.fromkeys(
            variable
            for rule in applicable
            for variable in rule.variables
        ))
        prepared = prepare_dataframe(raw, columns)
        rule_masks = {}
        flagged_positions = set()

        for rule in applicable:
            try:
                mask = eval(
                    rule.compiled,
                    {"pd": pd, "__builtins__": {}},
                    {"df": prepared},
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{rule.check_id} failed during evaluation: {exc}") from exc
            if not isinstance(mask, pd.Series):
                raise TypeError(
                    f"{rule.check_id} did not return a pandas Series")

            mask = mask.fillna(False).astype(bool)
            mask &= self._valid_observation_mask(prepared, rule)
            self.rule_status[rule.check_id] = (
                "evaluated", int(mask.sum()))
            rule_masks[rule.check_id] = mask
            flagged_positions.update(int(value) for value in mask[mask].index)

        if not flagged_positions:
            return

        # Build named rows only for candidates.  Materializing itertuples for
        # every row would duplicate references to tens of thousands of export
        # columns even when only a handful of subjects trigger a cross-check.
        candidate_rows = raw.iloc[sorted(flagged_positions)]
        raw_rows = dict(zip(
            (int(value) for value in candidate_rows.index),
            candidate_rows.itertuples(index=False),
        ))
        seen = set()

        for rule in applicable:
            mask = rule_masks[rule.check_id]
            for position in mask[mask].index:
                row = raw_rows[int(position)]
                if row.subjectid not in self.subject_info:
                    continue

                variables = self._ordered_variables(rule.variables)
                forms = self._forms_for_variables(variables)
                if not forms:
                    raise KeyError(
                        f"{rule.check_id} has no mapped source forms")
                if (not self._row_is_applicable(row, forms)
                        or not self._variables_are_applicable(row, variables)):
                    continue

                identity = (
                    self.network, row.subjectid, self.timepoint,
                    rule.check_id,
                )
                if identity in seen:
                    continue
                seen.add(identity)

                message = (
                    f"[{rule.check_id}] Cross-form review: {rule.label}.")
                output = self.create_row_output(
                    row,
                    forms,
                    variables,
                    message,
                    {
                        "reports": [self.REPORT_NAME],
                        "check_id": rule.check_id,
                        "nda_excluder": False,
                    },
                )
                # Cross-instrument disagreements are reviewer prompts, not
                # hard validation failures.  create_row_output normally marks
                # screening and selected clinical forms as priority; undo that
                # inherited highlighting for this dedicated review surface.
                output["priority"] = False
                output["priority_item"] = False
                self.final_output_list.append(output)

    @staticmethod
    def _valid_observation_mask(prepared, rule):
        """Require real values for mandatory or alternative operands."""
        valid = pd.Series(True, index=prepared.index, dtype=bool)
        alternative_variables = {
            variable
            for group in rule.alternative_evidence_groups
            for variable in group
        }
        for variable in rule.variables:
            if variable in alternative_variables:
                continue
            values = prepared[variable]
            valid &= values.notna()
            if variable not in DATE_COLUMNS:
                valid &= ~values.isin(NUMERIC_MISSING_CODES)
        for group in rule.alternative_evidence_groups:
            any_observed = pd.Series(
                False, index=prepared.index, dtype=bool)
            for variable in group:
                values = prepared[variable]
                observed = values.notna()
                if variable not in DATE_COLUMNS:
                    observed &= ~values.isin(NUMERIC_MISSING_CODES)
                any_observed |= observed
            valid &= any_observed
        return valid

    @staticmethod
    def _ordered_variables(variables):
        """Put actionable answer fields before supporting interview dates."""
        non_dates = [value for value in variables if value not in DATE_COLUMNS]
        dates = [value for value in variables if value in DATE_COLUMNS]
        return list(dict.fromkeys(non_dates + dates))

    def _form_for_variable(self, variable):
        form = self._var_forms.get(variable, "")
        if not form and "___" in variable:
            form = self._var_forms.get(variable.split("___", 1)[0], "")
        return form

    def _forms_for_variables(self, variables):
        mapped = [
            (variable, self._form_for_variable(variable))
            for variable in variables]
        missing = [variable for variable, form in mapped if not form]
        if missing:
            raise KeyError(
                "Cross-check operand(s) have no source-form mapping: "
                f"{missing}")
        return list(dict.fromkeys(form for _, form in mapped))

    def _scheduled_forms(self, row):
        cohort = _casefold_key(
            self.subject_info[row.subjectid].get("cohort", ""))
        timepoint = _casefold_key(self.timepoint)
        for raw_cohort, schedule in self.forms_per_tp.items():
            if _casefold_key(raw_cohort) != cohort:
                continue
            for raw_timepoint, forms in schedule.items():
                if _casefold_key(raw_timepoint) == timepoint:
                    return set(forms)
        return set()

    def _row_is_applicable(self, row, forms):
        scheduled = self._scheduled_forms(row)
        if not all(form in scheduled for form in forms):
            return False
        for form in forms:
            if form not in self.important_form_vars:
                raise KeyError(
                    f"Cross-check form {form!r} is missing from "
                    "important_form_vars")
            if not self.standard_form_filter(row, form):
                return False
        return True

    def _variables_are_applicable(self, row, variables):
        """Require every operand's REDCap branching logic to be active.

        Form-level completion is not enough for a cross-form comparison: a
        completed instrument can contain values in fields whose branches are
        inactive (for example after an upstream answer is corrected).  Those
        stale values must not become cross-check findings.  Checkbox columns
        inherit the base field's branching logic when no expanded entry is
        present in the converted dictionary.
        """
        excluded_branching = getattr(self, "excl_bl", {})
        converted_branching = getattr(self, "conv_bl", {})
        for variable in variables:
            candidates = [variable]
            if "___" in variable:
                candidates.append(variable.split("___", 1)[0])

            if any(
                    candidate in excluded_branching
                    for candidate in candidates):
                # The pipeline explicitly records these fields as having
                # branching logic it cannot evaluate safely.  Skipping is the
                # conservative choice for a reviewer-facing consistency tab.
                return False

            for candidate in dict.fromkeys(candidates):
                metadata = converted_branching.get(candidate)
                if not isinstance(metadata, dict):
                    continue
                expression = metadata.get("converted_branching_logic", "")
                if (expression and not evaluate_branching_logic(
                        expression, curr_row=row, instance=self)):
                    return False
        return True


__all__ = [
    "CrossCheckRule",
    "CrossChecks",
    "REPORT_NAME",
    "RULE_CATALOG",
]
