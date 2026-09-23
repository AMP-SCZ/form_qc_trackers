"""Audited, vectorized cross-form consistency checks.

The source catalog lives in ``test_cross_checks.py`` for traceability to the
operator-supplied prototype.  This adapter evaluates only the rules approved by
that audit, once per combined dataframe, and emits the canonical FormCheck row
schema used by reconciliation and tracker generation.  Cross Checks is an
intentionally unfiltered reviewer surface: a rule hit is emitted independently
of cohort scheduling, form completion/missingness, REDCap branching, special
form eligibility, and the pipeline's normal participant-routing filters.
"""

from dataclasses import dataclass

import pandas as pd

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
    from form_check import FormCheck
    from test_cross_checks import (
        ALTERNATIVE_EVIDENCE_GROUPS,
        DATE_COLUMNS,
        ENABLED_RULE_NUMBERS,
        RULE_LABELS,
        RULES,
        extract_columns,
        prepare_dataframe,
    )


REPORT_NAME = "Cross Checks"
BASE_ROW_COLUMNS = frozenset({
    "visit_status_string",
    "recruitment_status_v2",
})


class CrossCheckConfigurationError(RuntimeError):
    """A dependency snapshot cannot safely support Cross Checks."""


class CrossCheckInputError(RuntimeError):
    """A combined CSV cannot be evaluated or rendered safely."""


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


class CrossChecks(FormCheck):
    """Run approved rules once per dataframe without applicability filters."""

    REPORT_NAME = REPORT_NAME
    RULE_CATALOG = RULE_CATALOG

    def __init__(self, combined_df, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
        self.rule_status = {}
        # Keep legacy suppression counters in the metrics payload for
        # compatibility; under the unfiltered policy they remain zero and the
        # explicit filters_bypassed marker documents why.
        self.rule_metrics = {}
        self._var_forms = self.grouped_vars.get("var_forms", {})
        self._run(combined_df)

    def __call__(self):
        return self.final_output_list

    def _run(self, combined_df):
        if combined_df.empty:
            return
        if "subjectid" not in combined_df.columns:
            raise CrossCheckInputError(
                "CrossChecks requires a subjectid column in the combined CSV")

        # A deep reset_index copies every one of the thousands of combined
        # export columns (hundreds of MiB on baseline files).  We only need a
        # private positional index; a shallow dataframe shell safely shares
        # the immutable source blocks while operand normalization copies the
        # small selected subset below.
        raw = combined_df.copy(deep=False)
        raw.index = pd.RangeIndex(len(raw))
        self._validate_dependency_containers()
        self._validate_rows(raw)
        self._validate_output_columns(set(raw.columns))

        # Scheduling and form state are deliberately irrelevant to this tab.
        # A rule can run whenever its actual operands exist in the combined
        # export.  Missing operands remain visible in metrics, but do not make
        # unrelated Cross Checks fail merely because their forms are absent
        # from this network/timepoint export.
        available_columns = set(raw.columns)
        applicable = []
        for rule in self.RULE_CATALOG:
            missing = tuple(
                variable for variable in rule.variables
                if variable not in available_columns)
            if missing:
                self.rule_status[rule.check_id] = (
                    "unavailable_missing_columns", missing)
                self.rule_metrics[rule.check_id] = {
                    "status": "unavailable_missing_columns",
                    "missing_columns": missing,
                    "raw_candidates": 0,
                    "emitted": 0,
                    "routed": 0,
                    "filters_bypassed": True,
                }
                continue

            # Form mappings are output metadata rather than applicability
            # filters.  Keep them strict so every emitted row remains
            # actionable and round-trippable in the tracker.
            self._forms_for_variables(rule.variables)
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

            # The catalog expression is the complete decision boundary.  Do
            # not add completion, missing-code, schedule, or branching masks.
            mask = mask.fillna(False).astype(bool)
            raw_candidates = int(mask.sum())
            self.rule_status[rule.check_id] = (
                "evaluated", raw_candidates)
            self.rule_metrics[rule.check_id] = {
                "status": "evaluated",
                "raw_candidates": raw_candidates,
                "emitted": 0,
                "routed": 0,
                "suppressed_form_or_schedule": 0,
                "suppressed_branching": 0,
                "suppressed_duplicate_identity": 0,
                "suppressed_by_global_routing": 0,
                "filters_bypassed": True,
            }
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

                variables = self._ordered_variables(rule.variables)
                forms = self._forms_for_variables(variables)
                metrics = self.rule_metrics[rule.check_id]

                identity = (
                    self.network, row.subjectid, self.timepoint,
                    rule.check_id,
                )
                if identity in seen:
                    metrics["suppressed_duplicate_identity"] += 1
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
                        # These two flags prevent the normal withdrawn and
                        # inclusion gates inside create_row_output.  The route
                        # is also restored below to bypass recruited_only.
                        "excluded_enabled": True,
                        "withdrawn_enabled": True,
                    },
                )
                # Cross-instrument disagreements are reviewer prompts, not
                # hard validation failures.  create_row_output normally marks
                # screening and selected clinical forms as priority; undo that
                # inherited highlighting for this dedicated review surface.
                output["priority"] = False
                output["priority_item"] = False
                # create_row_output applies global participant routing after
                # output overrides and may still clear the reports list when
                # recruited_only is enabled.  Cross Checks intentionally
                # bypasses that final filter as well.
                output["reports"] = self.REPORT_NAME
                self.final_output_list.append(output)
                metrics["emitted"] += 1
                metrics["routed"] += 1

    def _validate_dependency_containers(self):
        containers = (
            ("subject_info", self.subject_info),
            ("grouped_variables.var_forms", self._var_forms),
        )
        malformed = [
            name for name, value in containers
            if not isinstance(value, dict)]
        if malformed:
            raise CrossCheckConfigurationError(
                "Cross Checks dependency container(s) must be mappings: "
                f"{malformed!r}")

    def _validate_rows(self, raw):
        """Validate identities and metadata required for canonical output.

        Cross-rule masks are evaluated row-wise.  Allowing two records for one
        subject/timepoint can split operands across rows or make the retained
        finding depend on input order, so duplicates are fatal rather than
        silently deduplicated.  Cohort and routing values are intentionally not
        eligibility checks, but the output builder still needs subject metadata
        for every row.
        """
        subject_ids = raw["subjectid"]
        missing_subject = subject_ids.isna() | subject_ids.map(
            lambda value: not str(value).strip())
        if missing_subject.any():
            rows = list(raw.index[missing_subject][:10])
            raise CrossCheckInputError(
                "CrossChecks found blank subjectid value(s) at dataframe "
                f"row position(s) {rows!r}")

        duplicate_mask = subject_ids.duplicated(keep=False)
        if duplicate_mask.any():
            duplicate_ids = list(dict.fromkeys(
                subject_ids[duplicate_mask].tolist()))[:10]
            raise CrossCheckInputError(
                "CrossChecks requires exactly one row per subject within a "
                f"network/timepoint; duplicate subjectid value(s) at "
                f"{self.network}/{self.timepoint}: {duplicate_ids!r}")

        unknown_subjects = list(dict.fromkeys(
            subject for subject in subject_ids
            if subject not in self.subject_info))
        if unknown_subjects:
            raise CrossCheckInputError(
                "CrossChecks cannot build canonical output for subjectid "
                "value(s) absent from subject_info: "
                f"{unknown_subjects[:10]!r}")

        for subject in subject_ids:
            subject_metadata = self.subject_info[subject]
            if not isinstance(subject_metadata, dict):
                raise CrossCheckConfigurationError(
                    f"subject_info[{subject!r}] must be a mapping")
            malformed_subject_fields = [
                key for key in ("inclusion_status", "visit_status")
                if (not isinstance(subject_metadata.get(key), str)
                    or not subject_metadata[key].strip())]
            if malformed_subject_fields:
                raise CrossCheckConfigurationError(
                    f"subject_info[{subject!r}] has missing or malformed "
                    "output field(s): "
                    f"{malformed_subject_fields!r}")

    @staticmethod
    def _validate_output_columns(available_columns):
        """Require only fields consumed while constructing tracker rows."""
        missing = sorted(BASE_ROW_COLUMNS - available_columns)
        if missing:
            raise CrossCheckInputError(
                "CrossChecks cannot build canonical output because the "
                "combined CSV is missing required output column(s): "
                f"{missing!r}")

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
        if not isinstance(form, str):
            return ""
        return form.strip()

    def _forms_for_variables(self, variables):
        mapped = [
            (variable, self._form_for_variable(variable))
            for variable in variables]
        missing = [variable for variable, form in mapped if not form]
        if missing:
            raise CrossCheckConfigurationError(
                "Cross-check operand(s) have no source-form mapping: "
                f"{missing}")
        return list(dict.fromkeys(form for _, form in mapped))


__all__ = [
    "CrossCheckConfigurationError",
    "CrossCheckInputError",
    "CrossCheckRule",
    "CrossChecks",
    "REPORT_NAME",
    "RULE_CATALOG",
]
