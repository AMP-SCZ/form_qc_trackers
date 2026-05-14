import os
import sys

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck


class ConversionChecks(FormCheck):
    """
    QC checks related to conversion status: forward checks (criteria
    variables that exceed thresholds without a matching `converted` flag)
    and reverse checks (subjects marked converted with no supporting
    criteria across timepoints or at the conversion timepoint).
    """

    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)

        self.gt_var_val_pairs = {
            "chrbprs_bprs_somc": 5,
            "chrbprs_bprs_guil": 5,
            "chrbprs_bprs_gran": 5,
            "chrbprs_bprs_susp": 5,
            "chrbprs_bprs_hall": 5,
            "chrbprs_bprs_unus": 5,
            "chrbprs_bprs_bizb": 5,
            "chrbprs_bprs_conc": 5,
        }

        # Sourced from dependencies/conversion_criteria_thresholds.json
        # so process_variables/collect_subject_info.py can read the
        # same psychs/scid threshold spec to aggregate cross-timepoint
        # `has_psychs_criteria` / `has_scid_criteria` flags.
        criteria = self.utils.load_dependency_json(
            'conversion_criteria_thresholds.json'
        )
        self._eq_psychs_pairs = criteria['psychs']
        self._eq_scid_pairs = criteria['scid']
        self.eq_var_val_pairs = {**self._eq_psychs_pairs, **self._eq_scid_pairs}

        self.call_checks(row)

    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        self.call_conversion_check(row)

    def call_conversion_check(self, row):
        for var, threshold in self.gt_var_val_pairs.items():
            form = self.grouped_vars["var_forms"][var]
            if not hasattr(row, var):
                continue
            var_val = getattr(row, var)
            # Missing codes (e.g. 999, -3) are numeric and would otherwise
            # satisfy `> threshold` here, producing false "conversion criteria"
            # flags for every subject with missing data on a chrbprs_* var.
            if var_val in self.utils.missing_code_set:
                continue
            if self.utils.can_be_float(var_val) and float(var_val) > threshold:
                self.conversion_criteria_check(
                    row,
                    [form],
                    [var],
                    {"reports": ["Conversion Report"]},
                )

        for var, threshold in self.eq_var_val_pairs.items():
            form = self.grouped_vars["var_forms"][var]
            if not hasattr(row, var):
                continue
            var_val = getattr(row, var)
            # Defensive: even though == against thresholds 1/3/6/8 is unlikely
            # to match a missing code, filter for consistency with the gt path.
            if var_val in self.utils.missing_code_set:
                continue
            if self.utils.can_be_float(var_val) and float(var_val) == threshold:
                self.conversion_criteria_check(
                    row,
                    [form],
                    [var],
                    {"reports": ["Conversion Report"]},
                )

        # Reverse of the per-var loops above: catches rows that ARE
        # marked converted but where no criteria variable supports it.
        self.marked_converted_no_criteria_check(row)
        # Companion check: flag converted CHR subjects whose
        # chrconv_method is set to anything other than 0 or 1.
        self.marked_converted_no_conv_tp_criteria_check(row)

    def marked_converted_no_criteria_check(self, row):
        """
        Reverse of the forward conversion check. Only runs on the
        floating row, because `chrconv_method` lives on the conversion
        form. Method semantics:
          0 → conversion determined by psychs (chrpsychs_fu_* vars)
          1 → conversion determined by scid   (chrscid_* vars)

        The supporting psychs/scid values can be filed at any
        timepoint, so cross-tp aggregation happens upstream in
        `process_variables/collect_subject_info.collect_conversion_criteria_info`,
        which stamps `has_psychs_criteria` / `has_scid_criteria` onto
        `subject_info`. This check just consults those booleans.
        """
        if self.timepoint != 'floating':
            return

        sub_info = self.subject_info.get(row.subjectid, {})
        if not sub_info.get('converted', False):
            return
        if sub_info.get("cohort", "").lower() != "chr":
            return

        if not hasattr(row, 'chrconv_method'):
            return
        method_val = getattr(row, 'chrconv_method')
        if method_val in self.utils.missing_code_set:
            return
        if not self.utils.can_be_float(method_val):
            return
        method = float(method_val)
        if method == 0:
            criteria_label = "psychs"
            has_criteria_key = 'has_psychs_criteria'
            relevant_vars = list(self._eq_psychs_pairs.keys())
        elif method == 1:
            criteria_label = "scid"
            has_criteria_key = 'has_scid_criteria'
            relevant_vars = list(self._eq_scid_pairs.keys())
        else:
            return

        # Absence of the key means process_variables hasn't been
        # re-run since collect_conversion_criteria_info was added.
        # Skip rather than default-False, which would falsely flag
        # every method 0/1 converted subject during a partial deploy.
        if has_criteria_key not in sub_info:
            return
        if sub_info[has_criteria_key]:
            return

        affected_vars = ["chrconv_method"] + relevant_vars
        error_message = (
            f"Participant is marked as converted via {criteria_label} "
            f"(chrconv_method = {method_val}), but no {criteria_label} "
            "criteria variable across any timepoint meets its threshold. "
            "Confirm the conversion marking or update the criteria values."
        )
        error_output = self.create_row_output(
            row,
            ["conversion_form"],
            affected_vars,
            error_message,
            {"reports": ["Conversion Report"]},
        )
        self.final_output_list.append(error_output)

    def marked_converted_no_conv_tp_criteria_check(self, row):
        """
        Companion to `marked_converted_no_criteria_check`. Same gating
        (floating row, converted, CHR cohort, chrconv_method present
        and numeric), but only runs when `chrconv_method` is anything
        other than 0 or 1 — i.e., the operator did not pick a
        supported psychs/scid attribution. In that case we look at
        the conversion timepoint specifically and fire only if no
        psychs OR scid criteria variable at that tp meets its
        threshold. Conv-tp-only booleans are aggregated upstream in
        `process_variables/collect_subject_info.collect_conversion_criteria_info`.
        """
        if self.timepoint != 'floating':
            return

        sub_info = self.subject_info.get(row.subjectid, {})
        if not sub_info.get('converted', False):
            return
        if sub_info.get("cohort", "").lower() != "chr":
            return

        if not hasattr(row, 'chrconv_method'):
            return
        method_val = getattr(row, 'chrconv_method')
        if method_val in self.utils.missing_code_set:
            return
        if not self.utils.can_be_float(method_val):
            return
        method = float(method_val)
        if method == 0 or method == 1:
            return

        # Absence of either key means process_variables hasn't been
        # re-run since the conv-tp-only booleans were added. Skip
        # rather than default-False, which would falsely flag every
        # non-0/1 method subject during a partial deploy.
        if ('has_psychs_criteria_at_conversion_tp' not in sub_info
                or 'has_scid_criteria_at_conversion_tp' not in sub_info):
            return
        if (sub_info['has_psychs_criteria_at_conversion_tp']
                or sub_info['has_scid_criteria_at_conversion_tp']):
            return

        criteria_vars = (list(self._eq_psychs_pairs.keys())
                         + list(self._eq_scid_pairs.keys()))
        affected_vars = ["chrconv_method"] + criteria_vars
        error_message = (
            f"Participant is marked as converted with chrconv_method = "
            f"{method_val} (not 0 or 1), and no psychs or scid criteria "
            "variable at the conversion timepoint meets its threshold. "
            "Confirm the conversion marking or update chrconv_method / "
            "the criteria values."
        )
        error_output = self.create_row_output(
            row,
            ["conversion_form"],
            affected_vars,
            error_message,
            {"reports": ["Conversion Report"]},
        )
        self.final_output_list.append(error_output)

    @FormCheck.standard_qc_check_filter
    def conversion_criteria_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
    ):
        # Conversion status is sourced from the conversion form's
        # `chrconv_conv` variable (filed at the floating timepoint), not
        # from `visit_status_string`. The status is collected once into
        # subject_info by process_variables/collect_subject_info.py so
        # it is available at every timepoint. Default False if the key
        # is absent (subject has no floating row, process_variables
        # hasn't been re-run since this change, etc.).
        sub_info = self.subject_info.get(row.subjectid, {})
        if not sub_info.get('converted', False):
            return (
                f"{all_vars[0]} is {getattr(row, all_vars[0])}, "
                "but participant is not marked as converted."
            )
