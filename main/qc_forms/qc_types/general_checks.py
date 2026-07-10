import pandas as pd

import os
import sys
import json
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck
from datetime import datetime
from itertools import combinations
import re

class GeneralChecks(FormCheck):  
    """
    General QC checks that are applied
    to all forms
    """      
    # def __init__(self, row, timepoint, network, form_check_info):
    #     super().__init__(timepoint, network,form_check_info)
    #     self.test_val = 0
    #     self.call_checks(row)

    def __init__(self, row, timepoint, network, form_check_info, run_row_checks=True):
        super().__init__(timepoint, network, form_check_info)
        self.test_val = 0

        if run_row_checks:
            self.call_checks(row)
        
    def __call__(self):

        return self.final_output_list

    def call_checks(self, row):
        self.row = row
        self.check_blank_values(row)
        self.check_missing_code_values(row)
        self.call_spec_val_check(row)
        self.call_chs_checks(row)
        self.check_form_completion(row)
        self.call_range_checks(row)
        for guid_var in ['chrguid_guid','chrguid_pseudoguid']:
            self.guid_format_check(row, ['guid_form'],
            [guid_var],{'reports':['Main Report','Non Team Forms'],
            "withdrawn_enabled" : True},
            bl_filtered_vars=[guid_var],filter_excl_vars=True,
            checked_guid_var=guid_var)
        self.missing_data_form_consistency_check(row)
        self.redcap_username_subjectid_check(row)

    def call_cross_timepoint_checks(self, all_rows):
        """
        Runs QC checks that require comparing multiple rows/timepoints
        for the same subject.
        """

        self.chrchs_height_cross_timepoint_check(all_rows)
        self.chrchs_weight_cross_timepoint_check(all_rows)
        self.chrchs_bmi_cross_timepoint_check(all_rows)



        #self.missing_code_check(row)

    def call_range_checks(self, row):
        for var, range_dict in self.variable_ranges[self.network].items():
            min = range_dict['min']
            max = range_dict['max']
            self.range_check(row, [range_dict['form']],[var],
            {"reports" : ['Main Report']},bl_filtered_vars=[],
            filter_excl_vars=True,range_var = var,
            lower = min,upper = max)

    # Blank-check flags on these forms are surfaced as priority items so they
    # don't get buried — pharm course-level data is hand-entered and the
    # missing-field count tends to be high.
    _PRIORITY_BLANK_FORMS = {
        'current_pharmaceutical_treatment_floating_med_125',
        'current_pharmaceutical_treatment_floating_med_2650',
        'past_pharmaceutical_treatment',
    }

    def check_blank_values(self, row):
        #TODO:optimize performance of this part
        for report in ['Main Report', 'Secondary Report']:
            blank_check_forms = self.general_check_vars["blank_check_vars"][
            self.network][report]
            for form in blank_check_forms:
                if (form in self.general_check_vars["excluded_forms"][
                self.network]):
                    continue
                report_list = [report]
                for team, forms in self.forms_per_report.items():
                    if form in forms:
                        report_list.append(team)
                output_changes = {"reports" : report_list}
                if form in self._PRIORITY_BLANK_FORMS:
                    output_changes["priority_item"] = True
                if self.standard_form_filter(row, form):
                    for var in blank_check_forms[form]:
                        if self.prescient_scid_filter(var, row) == True:
                            continue
                        self.check_if_blank(row, [form], [var],
                        output_changes,[var])

    def check_missing_code_values(self, row):
        #TODO:optimize performance of this part
        for report in ['Main Report']:
            missing_code_forms = self.general_check_vars["missing_code_vars"][
            self.network][report]
            for form in missing_code_forms:
                report_list = [report]
                for team, forms in self.forms_per_report.items():
                    if form in forms:
                        report_list.append(team)
                if self.standard_form_filter(row, form):
                    for var in missing_code_forms[form]:
                        if self.prescient_scid_filter(var, row) == True:
                            continue
                        self.check_if_missing_code(row, [form], [var],
                        {"reports" : report_list},[var])

    def prescient_scid_filter(self, var, row):
        """
        Filters out modules
        not used for Prescient 
        """
        if var in self.module_b_vars or var in self.module_c_vars:
            if self.network == 'PRESCIENT':
                if (hasattr(row,'chrscid_b1') and row.chrscid_b1 in (
                self.utils.missing_code_list + [''])
                or hasattr(row,'chrscid_b16') and row.chrscid_b16 in (
                self.utils.missing_code_list + [''])):
                    return True
        return False

    @FormCheck.standard_qc_check_filter
    def check_if_blank(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):  
        """
        Standard check applied across all
        forms to see if form is blank
        """
        if hasattr(row,all_vars[0]) and getattr(row, all_vars[0]) == '':
            return "Variable is blank."
        return 

    @FormCheck.standard_qc_check_filter
    def check_if_missing_code(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):  
        if  (getattr(row, all_vars[0])
        in self.utils.missing_code_set or
        str(getattr(row, all_vars[0])).replace(' ','') == 'NaN'):
            return "Variable is a missing code."
        return 
    
    def call_spec_val_check(self,row):
        for var, conditions in self.general_check_vars["specific_val_check_vars"].items():
            if var in self.grouped_vars['var_forms'].keys():
                form = self.grouped_vars['var_forms'][var]
                if self.standard_form_filter(row, form):
                    self.specific_value_checks(row,[form],[var],
                    {'reports':conditions['report']},[],True,conditions=conditions)


    def call_chs_checks(self,row):
        report_list = ['Main Report']

        self.chrchs_height_check(
            row,
            ['current_health_status'],
            ['chrchs_heightunits', 'chrchs_height', 'chrchs_heightcm'],
            {"reports": report_list}
        )

        self.chrchs_weight_check(
            row,
            ['current_health_status'],
            ['chrchs_weightunits', 'chrchs_weight', 'chrchs_weightkg'],
            {"reports": report_list}
        )

        self.chrchs_bmi_check(
            row,
            ['current_health_status'],
            ['chrchs_weightkg', 'chrchs_heightcm', 'chrchs_bmi'],
            {"reports": report_list}
        )

        self.chrchs_bodytemp_check(
            row,
            ['current_health_status'],
            ['chrchs_bodytemp', 'chrchs_bodytempf', 'chrchs_bodytempunits'],
            {"reports": report_list}
        )

        self.chrchs_heartrate_check(row)
        

    def call_cross_timepoint_checks(self, all_rows):
        self.chrchs_height_cross_timepoint_check(all_rows)
        self.chrchs_weight_cross_timepoint_check(all_rows)
        self.chrchs_bmi_cross_timepoint_check(all_rows)


   # def call_chs_checks(self,row):
    #    report_list= ['Main Report']
     #   self.chrchs_height_check(row,['current_health_status'], ['height', 'heightcm', 'heightunits'], {"reports" : report_list} )
      #  self.chrchs_weight_check(row,['current_health_status'], ['weight', 'weightkg', 'weightunits'], {"reports" : report_list} )

    @FormCheck.standard_qc_check_filter
    def specific_value_checks(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, conditions={}
    ):
        var = conditions['correlated_variable']
        if hasattr(row, var): 
            if conditions["negative"] == 'False':
                if getattr(row, var) in conditions["checked_value_list"]:
                    return conditions['message']
            elif conditions["negative"] == 'True':
                if getattr(row, var) not in conditions["checked_value_list"]:
                    return conditions['message']


    def _is_official_missing(self, val):
        return val in self.utils.missing_code_list


    def _add_current_health_flag(self, row, vars_to_flag, error_message):
        error_output = self.create_row_output(
            row,
            ["current_health_status"],
            vars_to_flag,
            error_message,
            {"reports": ["Main Report"]}
        )
        self.final_output_list.append(error_output)


    def chrchs_height_check(
        self, row, filtered_forms, all_vars, changed_output_vals,
        bl_filtered_vars=[], filter_excl_vars=True, conditions={}
    ):
        required_vars = ["chrchs_heightunits", "chrchs_height", "chrchs_heightcm"]

        if not all(hasattr(row, var) for var in required_vars):
            return

        units = getattr(row, "chrchs_heightunits")
        height = getattr(row, "chrchs_height")
        heightcm = getattr(row, "chrchs_heightcm")

        if (
            self._is_official_missing(units)
            or self._is_official_missing(height)
            or self._is_official_missing(heightcm)
        ):
            return

        if not (
            self.utils.can_be_float(units)
            and self.utils.can_be_float(height)
            and self.utils.can_be_float(heightcm)
        ):
            return

        units = int(float(units))
        height = float(height)
        heightcm = float(heightcm)

        if height <= 0 or heightcm <= 0:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"Height values must be greater than zero. "
                    f"chrchs_height={height:g}, chrchs_heightcm={heightcm:g}."
                )
            )
            return

        if units == 1:
            if 1.0 <= height <= 2.5:
                msg = "Value suggests meters, but units are recorded as centimeters."
                expected_heightcm = height * 100
            elif 48 <= height <= 96 and abs((height * 2.54) - heightcm) <= 1:
                msg = "Value suggests inches, but units are recorded as centimeters."
                expected_heightcm = height * 2.54
            else:
                msg = None
                expected_heightcm = height

        elif units == 2:
            if 100 <= height <= 230 and abs(height - heightcm) <= 1:
                msg = "Value suggests centimeters, but units are recorded as inches."
                expected_heightcm = height
            elif 4 <= height <= 8:
                msg = "Value suggests feet, but units are recorded as inches."
                expected_heightcm = height * 30.48
            else:
                msg = None
                expected_heightcm = height * 2.54

        else:
            self._add_current_health_flag(
                row,
                required_vars,
                f"Height unit code is invalid. chrchs_heightunits={units}, expected 1=cm or 2=inches."
            )
            return

        if expected_heightcm < 100 or expected_heightcm > 230:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"Height value is implausible. "
                    f"chrchs_heightunits={units}, chrchs_height={height:g}, "
                    f"chrchs_heightcm={heightcm:g}. Expected converted height 100-230 cm."
                )
            )
            return

        if msg is not None:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"Height units may be incorrect. "
                    f"chrchs_heightunits={units}, chrchs_height={height:g}, "
                    f"chrchs_heightcm={heightcm:g}. {msg}"
                )
            )
            return

        if abs(expected_heightcm - heightcm) > 1:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"Height conversion mismatch. "
                    f"chrchs_heightunits={units}, chrchs_height={height:g}, "
                    f"expected chrchs_heightcm={round(expected_heightcm, 1):g}, "
                    f"actual chrchs_heightcm={heightcm:g}."
                )
            )


    def chrchs_weight_check(
        self, row, filtered_forms, all_vars, changed_output_vals,
        bl_filtered_vars=[], filter_excl_vars=True, conditions={}
    ):
        required_vars = ["chrchs_weightunits", "chrchs_weight", "chrchs_weightkg"]

        if not all(hasattr(row, var) for var in required_vars):
            return

        units = getattr(row, "chrchs_weightunits")
        weight = getattr(row, "chrchs_weight")
        weightkg = getattr(row, "chrchs_weightkg")

        if (
            self._is_official_missing(units)
            or self._is_official_missing(weight)
            or self._is_official_missing(weightkg)
        ):
            return

        if not (
            self.utils.can_be_float(units)
            and self.utils.can_be_float(weight)
            and self.utils.can_be_float(weightkg)
        ):
            return

        units = int(float(units))
        weight = float(weight)
        weightkg = float(weightkg)

        if weight <= 0 or weightkg <= 0:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"Weight values must be greater than zero. "
                    f"chrchs_weight={weight:g}, chrchs_weightkg={weightkg:g}."
                )
            )
            return

        if units == 1:
            if 90 <= weight <= 700 and abs((weight * 0.453592) - weightkg) <= 1:
                msg = "Value suggests pounds, but units are recorded as kilograms."
                expected_weightkg = weight * 0.453592
            else:
                msg = None
                expected_weightkg = weight

        elif units == 2:
            if 25 <= weight <= 300 and abs(weight - weightkg) <= 1:
                msg = "Value suggests kilograms, but units are recorded as pounds."
                expected_weightkg = weight
            else:
                msg = None
                expected_weightkg = weight * 0.453592

        else:
            self._add_current_health_flag(
                row,
                required_vars,
                f"Weight unit code is invalid. chrchs_weightunits={units}, expected 1=kg or 2=pounds."
            )
            return

        if expected_weightkg < 25 or expected_weightkg > 300:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"Weight value is implausible. "
                    f"chrchs_weightunits={units}, chrchs_weight={weight:g}, "
                    f"chrchs_weightkg={weightkg:g}. Expected converted weight 25-300 kg."
                )
            )
            return

        if msg is not None:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"Weight units may be incorrect. "
                    f"chrchs_weightunits={units}, chrchs_weight={weight:g}, "
                    f"chrchs_weightkg={weightkg:g}. {msg}"
                )
            )
            return

        if abs(expected_weightkg - weightkg) > 1:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"Weight conversion mismatch. "
                    f"chrchs_weightunits={units}, chrchs_weight={weight:g}, "
                    f"expected chrchs_weightkg={round(expected_weightkg, 1):g}, "
                    f"actual chrchs_weightkg={weightkg:g}."
                )
            )


    def chrchs_bmi_check(
        self, row, filtered_forms,
        all_vars, changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
        conditions={}
    ):
        required_vars = ["chrchs_bmi", "chrchs_weightkg", "chrchs_heightcm"]

        if not all(hasattr(row, var) for var in required_vars):
            return

        bmi = getattr(row, "chrchs_bmi")
        weightkg = getattr(row, "chrchs_weightkg")
        heightcm = getattr(row, "chrchs_heightcm")

        if (
            self._is_official_missing(bmi)
            or self._is_official_missing(weightkg)
            or self._is_official_missing(heightcm)
        ):
            return

        if not (
            self.utils.can_be_float(bmi)
            and self.utils.can_be_float(weightkg)
            and self.utils.can_be_float(heightcm)
        ):
            return

        bmi = float(bmi)
        weightkg = float(weightkg)
        heightcm = float(heightcm)

        if bmi <= 0 or weightkg <= 0 or heightcm <= 0:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"BMI, weight, and height must be greater than zero. "
                    f"chrchs_bmi={bmi:g}, chrchs_weightkg={weightkg:g}, chrchs_heightcm={heightcm:g}."
                )
            )
            return

        calculated_bmi = weightkg / ((heightcm / 100) ** 2)
        calculated_bmi = round(calculated_bmi, 1)
        bmi = round(bmi, 1)

        if bmi < 12 or bmi > 60 or calculated_bmi < 12 or calculated_bmi > 60:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"BMI value is implausible. "
                    f"chrchs_bmi={bmi:g}, calculated_bmi={calculated_bmi:g}, "
                    f"chrchs_weightkg={weightkg:g}, chrchs_heightcm={heightcm:g}. Expected BMI 12-60."
                )
            )
            return

        if abs(calculated_bmi - bmi) > 1:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"BMI calculation mismatch. "
                    f"chrchs_bmi={bmi:g}, calculated_bmi={calculated_bmi:g}, "
                    f"chrchs_weightkg={weightkg:g}, chrchs_heightcm={heightcm:g}."
                )
            )


    # def _parse_chrchs_interview_date(self, val):
    #     if val in self.utils.missing_code_list:
    #         return None

    #     val = str(val).strip()

    #     if val == "":
    #         return None

    #     try:
    #         return datetime.strptime(val, "%Y-%m-%d").date()
    #     except ValueError:
    #         return None


    # def _cross_timepoint_date_check(
    #     self,
    #     all_rows,
    #     value_var,
    #     threshold_per_month,
    #     label,
    #     flag_vars
    # ):
    #     records = {}

    #     for row, tp, network in all_rows:
    #         required_vars = ["subjectid", value_var, "chrchs_interview_date"]

    #         if not all(hasattr(row, v) for v in required_vars):
    #             continue

    #         sid = row.subjectid
    #         val = getattr(row, value_var)
    #         interview_date = self._parse_chrchs_interview_date(
    #             getattr(row, "chrchs_interview_date")
    #         )

    #         if interview_date is None:
    #             continue

    #         if val in self.utils.missing_code_list:
    #             continue

    #         if not self.utils.can_be_float(val):
    #             continue

    #         val = float(val)

    #         if val <= 0:
    #             continue

    #         tp_label = f"{network}_{tp}"

    #         records.setdefault(sid, []).append(
    #             (row, tp_label, val, interview_date)
    #         )

    #     for sid, vals in records.items():
    #         if len(vals) < 2:
    #             continue

    #         for a, b in combinations(vals, 2):
    #             row_a, tp_a, val_a, date_a = a
    #             row_b, tp_b, val_b, date_b = b

    #             gap_days = abs((date_b - date_a).days)
    #             gap_months = max(gap_days / 30.44, 1)

    #             actual_diff = abs(val_b - val_a)
    #             allowed_diff = threshold_per_month * gap_months

    #             if actual_diff > allowed_diff:
    #                 msg = (
    #                     f"{label} differs across visits more than expected "
    #                     f"based on interview dates. "
    #                     f"Difference={actual_diff:.1f}; "
    #                     f"gap={gap_days} days; "
    #                     f"allowed={allowed_diff:.1f}. "
    #                     f"{tp_a}={val_a:g} on {date_a}, "
    #                     f"{tp_b}={val_b:g} on {date_b}."
    #                 )

    #                 self._add_current_health_flag(row_a, flag_vars, msg)
    #                 self._add_current_health_flag(row_b, flag_vars, msg)


    # def chrchs_weight_cross_timepoint_check(self, all_rows):
    #     self._cross_timepoint_date_check(
    #         all_rows=all_rows,
    #         value_var="chrchs_weightkg",
    #         threshold_per_month=10.0,
    #         label="Weight",
    #         flag_vars=["chrchs_weightkg", "chrchs_interview_date"]
    #     )


    # def chrchs_height_cross_timepoint_check(self, all_rows):
    #     self._cross_timepoint_date_check(
    #         all_rows=all_rows,
    #         value_var="chrchs_heightcm",
    #         threshold_per_month=2.0,
    #         label="Height",
    #         flag_vars=["chrchs_heightcm", "chrchs_interview_date"]
    #     )


    # def chrchs_bmi_cross_timepoint_check(self, all_rows):
    #     self._cross_timepoint_date_check(
    #         all_rows=all_rows,
    #         value_var="chrchs_bmi",
    #         threshold_per_month=3.0,
    #         label="BMI",
    #         flag_vars=[
    #             "chrchs_bmi",
    #             "chrchs_heightcm",
    #             "chrchs_weightkg",
    #             "chrchs_interview_date"
    #         ]
    #     )

    def chrchs_height_cross_timepoint_check(self, all_rows, threshold_cm=10.0):
        print("\n=== Running OLD height cross-timepoint check ===")
        print(f"Rows received: {len(all_rows)}")

        records = {}

        for row, tp, network in all_rows:
            tp_label = f"{network}_{tp}"

            required_vars = ["subjectid", "chrchs_heightcm", "chrchs_interview_date"]
            if not all(hasattr(row, v) for v in required_vars):
                continue

            sid = row.subjectid
            val = row.chrchs_heightcm
            interview_date = row.chrchs_interview_date

            if self._is_official_missing(val):
                continue

            if not self.utils.can_be_float(val):
                continue

            val = float(val)

            if val <= 0:
                continue

            records.setdefault(sid, []).append(
                (row, tp_label, val, interview_date)
            )

        print(f"Height subjects grouped: {len(records)}")

        for sid, vals in records.items():
            if len(vals) < 2:
                continue

            low = min(vals, key=lambda x: x[2])
            high = max(vals, key=lambda x: x[2])

            row_low, tp_low, val_low, date_low = low
            row_high, tp_high, val_high, date_high = high

            diff_cm = round(val_high - val_low, 1)

            if diff_cm > threshold_cm:
                msg = (
                    f"Height differs across visits. "
                    f"Difference={diff_cm:g} cm. "
                    f"{tp_low}={val_low:g} cm on {date_low}; "
                    f"{tp_high}={val_high:g} cm on {date_high}."
                )

                print(f"[HEIGHT FLAG] subject={sid} | {msg}")

                for row, tp_label, val, interview_date in vals:
                    self._add_current_health_flag(
                        row,
                        ["chrchs_heightcm", "chrchs_interview_date"],
                        msg
                    )


    def chrchs_weight_cross_timepoint_check(self, all_rows, threshold_kg=10.0):
        print("\n=== Running OLD weight cross-timepoint check ===")
        print(f"Rows received: {len(all_rows)}")

        records = {}

        skipped_missing_cols = 0
        skipped_missing_val = 0
        skipped_non_numeric = 0
        skipped_non_positive = 0
        added_records = 0

        for row, tp, network in all_rows:
            tp_label = f"{network}_{tp}"

            required_vars = ["subjectid", "chrchs_weightkg", "chrchs_interview_date"]
            if not all(hasattr(row, v) for v in required_vars):
                skipped_missing_cols += 1
                continue

            sid = row.subjectid
            val = row.chrchs_weightkg
            interview_date = row.chrchs_interview_date

            if self._is_official_missing(val):
                skipped_missing_val += 1
                continue

            if not self.utils.can_be_float(val):
                skipped_non_numeric += 1
                continue

            val = float(val)

            if val <= 0:
                skipped_non_positive += 1
                continue

            records.setdefault(sid, []).append(
                (row, tp_label, val, interview_date)
            )
            added_records += 1

        print(f"Skipped missing columns: {skipped_missing_cols}")
        print(f"Skipped official missing weight: {skipped_missing_val}")
        print(f"Skipped non-numeric weight: {skipped_non_numeric}")
        print(f"Skipped non-positive weight: {skipped_non_positive}")
        print(f"Added valid weight records: {added_records}")
        print(f"Weight subjects grouped: {len(records)}")

        for sid, vals in records.items():
            if len(vals) < 2:
                continue

            low = min(vals, key=lambda x: x[2])
            high = max(vals, key=lambda x: x[2])

            row_low, tp_low, val_low, date_low = low
            row_high, tp_high, val_high, date_high = high

            diff_kg = round(val_high - val_low, 1)

            if diff_kg > threshold_kg:
                msg = (
                    f"Weight differs across visits. "
                    f"Difference={diff_kg:g} kg. "
                    f"{tp_low}={val_low:g} kg on {date_low}; "
                    f"{tp_high}={val_high:g} kg on {date_high}."
                )

                print(f"[WEIGHT FLAG] subject={sid} | {msg}")

                for row, tp_label, val, interview_date in vals:
                    self._add_current_health_flag(
                        row,
                        ["chrchs_weightkg", "chrchs_interview_date"],
                        msg
                    )


    def chrchs_bmi_cross_timepoint_check(self, all_rows, threshold=4.0):
        print("\n=== Running OLD BMI cross-timepoint check ===")
        print(f"Rows received: {len(all_rows)}")

        records = {}

        for row, tp, network in all_rows:
            tp_label = f"{network}_{tp}"

            required_vars = ["subjectid", "chrchs_bmi", "chrchs_interview_date"]
            if not all(hasattr(row, v) for v in required_vars):
                continue

            sid = row.subjectid
            val = row.chrchs_bmi
            interview_date = row.chrchs_interview_date

            if self._is_official_missing(val):
                continue

            if not self.utils.can_be_float(val):
                continue

            val = float(val)

            if val <= 0:
                continue

            records.setdefault(sid, []).append(
                (row, tp_label, val, interview_date)
            )

        print(f"BMI subjects grouped: {len(records)}")

        for sid, vals in records.items():
            if len(vals) < 2:
                continue

            low = min(vals, key=lambda x: x[2])
            high = max(vals, key=lambda x: x[2])

            row_low, tp_low, val_low, date_low = low
            row_high, tp_high, val_high, date_high = high

            diff = round(val_high - val_low, 1)

            if diff >= threshold:
                msg = (
                    f"BMI differs across visits. "
                    f"Difference={diff:g} units. "
                    f"{tp_low}={val_low:g} on {date_low}; "
                    f"{tp_high}={val_high:g} on {date_high}."
                )

                print(f"[BMI FLAG] subject={sid} | {msg}")

                for row, tp_label, val, interview_date in vals:
                    self._add_current_health_flag(
                        row,
                        [
                            "chrchs_bmi",
                            "chrchs_heightcm",
                            "chrchs_weightkg",
                            "chrchs_interview_date"
                        ],
                        msg
                    )



    def chrchs_bodytemp_check(
            self, row, filtered_forms,
            all_vars, changed_output_vals,
            bl_filtered_vars=[],
            filter_excl_vars=True,
            conditions={}
        ):

            def extract_first_number(value):
                match = re.search(r"-?\d+(\.\d+)?", str(value))
                if match:
                    return float(match.group())
                return None

            required_vars = [
                "chrchs_bodytemp",
                "chrchs_bodytempf",
                "chrchs_bodytempunits"
            ]

            if not all(hasattr(row, var) for var in required_vars):
                return

            bodytemp = getattr(row, "chrchs_bodytemp")
            bodytempf = getattr(row, "chrchs_bodytempf")
            units = getattr(row, "chrchs_bodytempunits")

            form = "current_health_status"
            output_changes = {"reports": ["Main Report"]}

            if (
                bodytemp in self.utils.missing_code_list
                or units in self.utils.missing_code_list
            ):
                return

            bodytempf_num = None

            if bodytempf not in self.utils.missing_code_list:
                bodytempf_num = extract_first_number(bodytempf)

                if bodytempf_num is not None:
                    if bodytempf_num < 86 or bodytempf_num > 113:
                        error_message = (
                            f"Body temperature Fahrenheit value appears outside a valid range. "
                            f"chrchs_bodytempf = {bodytempf}."
                        )

                        error_output = self.create_row_output(
                            row, [form], ["chrchs_bodytempf"],
                            error_message, output_changes
                        )
                        self.final_output_list.append(error_output)

            bodytemp_num = extract_first_number(bodytemp)
            units_num = extract_first_number(units)

            if bodytemp_num is None or units_num is None:
                return

            bodytemp = bodytemp_num
            units = int(units_num)

            if units == 1:
                if bodytemp < 30 or bodytemp > 45:
                    error_message = (
                        f"Body temperature appears outside a valid Celsius range. "
                        f"chrchs_bodytemp = {bodytemp:g}°C."
                    )

                    error_output = self.create_row_output(
                        row, [form], ["chrchs_bodytemp", "chrchs_bodytempunits"],
                        error_message, output_changes
                    )
                    self.final_output_list.append(error_output)

            elif units == 2:
                if bodytemp < 86 or bodytemp > 113:
                    error_message = (
                        f"Body temperature appears outside a valid Fahrenheit range. "
                        f"chrchs_bodytemp = {bodytemp:g}°F."
                    )

                    error_output = self.create_row_output(
                        row, [form], ["chrchs_bodytemp", "chrchs_bodytempunits"],
                        error_message, output_changes
                    )
                    self.final_output_list.append(error_output)

                if bodytempf_num is not None:
                    if abs(bodytemp - bodytempf_num) > 1:
                        error_message = (
                            f"Body temperature values do not match. "
                            f"chrchs_bodytemp = {bodytemp:g}, "
                            f"chrchs_bodytempf = {bodytempf}."
                        )

                        error_output = self.create_row_output(
                            row, [form], ["chrchs_bodytemp", "chrchs_bodytempf"],
                            error_message, output_changes
                        )
                        self.final_output_list.append(error_output)

            else:
                error_message = (
                    f"Body temperature unit code is invalid. "
                    f"chrchs_bodytempunits = {units}, expected 1=Celsius or 2=Fahrenheit."
                )

                error_output = self.create_row_output(
                    row, [form], ["chrchs_bodytempunits"],
                    error_message, output_changes
                )
                self.final_output_list.append(error_output)

    def chrchs_heartrate_check(self, row):
        required_vars = ["chrchs_hr"]

        if not all(hasattr(row, var) for var in required_vars):
            return

        heartrate = getattr(row, "chrchs_hr")

        if self._is_official_missing(heartrate):
            return

        if not self.utils.can_be_float(heartrate):
            return

        heartrate = float(heartrate)

        if heartrate < 40 or heartrate > 200:
            self._add_current_health_flag(
                row,
                required_vars,
                (
                    f"Heart rate ({heartrate}) is outside the plausible "
                    f"range (40–200 bpm)."
                )
            )



    def redcap_username_subjectid_check(self, row):

        redcap_user_vars = [
        'chrguid_redcap_user', 'chrcrit_redcap_user', 'chric_redcap_user',
        'chric_reconsent_redcap_user', 'chrpharm_redcap_user',
        'chrrecruit_redcap_user', 'chrdemo_redcap_user',
        'chrhealth_redcap_user', 'chrschizotypal_redcap_user',
        'chrscid_redcap_user',
    
        'chrtbi_redcap_user', 'chrfigs_redcap_user',
        'chrsofas_redcap_users', 'chrsofas_redcap_users_fu',
        'chrpsychs_scr_redcap_user', 'chrpsychs_scr_redcap_user_2',
        'chrpsychs_fu_redcap_user', 'chrpsychs_fu_redcap_user_2',
        'hcpsychs_fu_redcap_user', 'hcpsychs_fu_redcap_user_2',
    
        'chrpsychs_av_redcap_user', 'chrpas_redcap_user',
        'chrpenn_redcap_user', 'chriq_redcap_user',
        'chrpreiq_redcap_user', 'chreeg_redcap_user',
        'chrmri_redcap_user', 'chrif_redcap_user',
        'chrax_redcap_user', 'chraxci_redcap_user',
    
        'chraxe_redcap_user', 'chrdbb_redcap_user',
        'chrdig_redcap_user', 'chrdbe_redcap_user',
        'chrchs_redcap_user', 'chrblood_redcap_user',
        'chrsaliva_redcap_user', 'chrcbc_redcap_user',
        'chrspeech_redcap_user', 'chrnsipr_redcap_user',
    
        'chrcdss_redcap_user', 'chrassist_redcap_user',
        'chrcssrsb_redcap_user', 'chrcssrsfu_redacp_user',
        'chrgfss_redcap_user', 'chrgfssfu_redcap_user',
        'chrgfrs_redcap_user', 'chrgfrsfu_redcap_user',
        'chrdim_redcap_user', 'chrpds_redcap_user',
    
        'chroasis_redcap', 'chrpromis_redcap_user',
        'chrpss_redcap_user', 'chrpgi_redcap_user',
        'chrpps_redcao_user', 'chrbprs_redcap_user',
        'chrpred_username'
    ]

        all_subject_ids = set(
            str(subject_id).strip().lower()
            for subject_id in self.subject_info.keys()
            if str(subject_id).strip() != ''
        )

        for user_var in redcap_user_vars:

            if not hasattr(row, user_var):
                continue

            username = str(getattr(row, user_var)).strip().lower()

            if username == '':
                continue

            if username in all_subject_ids:

                error_message = (
                    f"REDCap user field should not match any participant "
                    f"subject ID. {user_var} = {username}."
                )

                form = self.grouped_vars['var_forms'].get(user_var, 'redcap_usernames')
                report_list = ['Main Report']
                output_changes = {'reports': report_list}

                error_output = self.create_row_output(
                    row,
                    [form],
                    [user_var, 'subjectid'],
                    error_message,
                    output_changes
                )

                self.final_output_list.append(error_output)


    def check_form_completion(self,row):
        cohort = self.subject_info[row.subjectid]['cohort']
        if cohort.lower() not in ["hc", "chr"]:
            return
        curr_tp_forms = self.forms_per_tp[cohort][self.timepoint]
        if (self.check_if_next_tp(row) == True ):
            for form in curr_tp_forms:
                if (form in self.prescient_forms_no_compl_status
                or form in self.general_check_vars["excluded_forms"][
                self.network]):
                    continue
                compl_var = self.important_form_vars[form]['completion_var']
                if self.network == 'PRESCIENT':
                    compl_var += '_rpms'
                    # rpms compl variables same for both cohorts
                    compl_var = compl_var.replace('_hc','')
                    if 'informed_consent' in compl_var:
                        # Skip ONLY this form, not the whole loop —
                        # `return` here previously aborted check_form_completion
                        # for every form ordered after informed_consent in
                        # `curr_tp_forms`, silently dropping completion checks.
                        continue
                if (hasattr(row, compl_var) and 
                getattr(row, compl_var) not in self.utils.all_dtype([2,3,4])):
                    error_message = (f"{form} not marked as complete, but"
                    " subject has started the next timepoint")
                    if self.network == 'PRONET':
                        report_list =  ['Incomplete Forms']
                    else:
                        report_list =  ['Main Report', 'Incomplete Forms']

                    for team, forms in self.forms_per_report.items():
                        if form in forms:
                            report_list.append(team)

                    output_changes = {'reports' : report_list}
                    error_output = self.create_row_output(
                    row,[form],[compl_var], error_message, output_changes)
                    self.final_output_list.append(error_output)

    def guid_format_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, checked_guid_var = 'chrguid_guid'
    ):
        """Checks if GUID is in proper format"""
        if not hasattr(row,checked_guid_var):
            return
        guid = str(getattr(row,checked_guid_var))
        if guid == '' or guid in self.utils.missing_code_set:
            return
        if not re.search(r"^NDAR[A-Z0-9]+$", guid):
            error_message = f"GUID in incorrect format. GUID was reported to be {guid}."
            error_output = self.create_row_output(
            row,filtered_forms,[checked_guid_var], error_message,
            changed_output_vals)
            self.final_output_list.append(error_output)

    def range_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, range_var = '', lower = 0, upper = 100
    ):  
        if not hasattr(row, range_var):
            return
        var_val = getattr(row, range_var)
        if (self.utils.can_be_float(var_val) and
        var_val not in self.utils.missing_code_set):
            if float(var_val) < lower or float(var_val) > upper:
                error_message = f'{range_var} value ({var_val}) is out of range'
                error_output = self.create_row_output(
                row, filtered_forms, [range_var], error_message, 
                changed_output_vals)
                self.final_output_list.append(error_output)

    def age_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        """
        Checks all ages to make sure
        they are in the correct range.
        """
        age = self.subject_info[row.subjectid]["age"]
        if age == "unknown":
            return "Age is Unknown."
        elif self.utils.can_be_float(age) and age < 12 or age > 30:
            return f"Age ({age}) is out of range."

    # missing_data form trigger variables (per missing_data_form_info.csv).
    # All four are radio fields with single choice value 1; set to 1 means
    # missingness of that scope has been recorded on the missing_data form.
    # Time/withdrawn/discon imply the WHOLE TP (and beyond, for the latter
    # two) is missing. domain implies only specific assessment domains
    # listed in chrmiss_domain_type___1..7 are missing.
    _WHOLE_TP_TRIGGER_VARS = (
        'chrmiss_time', 'chrmiss_withdrawn', 'chrmiss_discon',
    )
    _DOMAIN_TRIGGER_VAR = 'chrmiss_domain'
    _MISSING_DATA_TRIGGER_VARS = (
        _WHOLE_TP_TRIGGER_VARS + (_DOMAIN_TRIGGER_VAR,)
    )

    def missing_data_form_consistency_check(self, row):
        """
        Flags contradictions between the missing_data form (chrmiss_*
        triggers) and individual forms' missing-data buttons
        (chr{form}_missing).

        Three flag types:
          A) Per form. chr{form}_missing=1 but none of the missing_data
             triggers are set — the RA marked the form missing without
             recording the reason on the missing_data form.
          B-strict) Per form. A whole-TP trigger (chrmiss_time /
             chrmiss_withdrawn / chrmiss_discon) is set, but the form's
             missing button is not checked. When the whole TP/participant
             is missing, every applicable form is expected to have its
             missing button set.
          B-coarse) Single summary. chrmiss_domain=1 (with no whole-TP
             trigger) but no individual form has its missing button
             checked. Domain-aware verification (matching
             chrmiss_domain_type___N to forms in that domain) requires a
             domain→form mapping not yet available; coarse "at least one"
             check is the conservative substitute.

        Gate: runs when the subject has moved past this TP OR any
        chrmiss_* trigger is set. The OR is required because withdrawn /
        discontinued participants will not progress past the withdrawal
        TP, so check_if_next_tp alone would never evaluate exactly the
        rows we most need to check.

        Form applicability filters (mirrored from check_form_completion /
        extra_form_conditions): forms in excluded_forms[network] and forms
        whose extra conditions don't apply (axivity/mindlamp opt-out,
        pubertal scale age cutoff) are excluded from Direction B-strict
        to avoid flooding withdrawal rows with irrelevant flags.
        """
        cohort = self.subject_info[row.subjectid]['cohort']
        if cohort.lower() not in ['hc', 'chr']:
            return
        # Skip if the missing_data form columns aren't present on this row
        # (TP that doesn't include the missing_data form).
        if not any(hasattr(row, v) for v in self._MISSING_DATA_TRIGGER_VARS):
            return

        triggered_chrmiss_vars = [
            v for v in self._MISSING_DATA_TRIGGER_VARS
            if hasattr(row, v)
            and getattr(row, v) in self.utils.all_dtype([1])
        ]
        any_trigger_set = len(triggered_chrmiss_vars) > 0
        whole_tp_triggered = [
            v for v in triggered_chrmiss_vars
            if v in self._WHOLE_TP_TRIGGER_VARS
        ]
        has_whole_tp_trigger = len(whole_tp_triggered) > 0
        has_domain_trigger = self._DOMAIN_TRIGGER_VAR in triggered_chrmiss_vars

        if self.check_if_next_tp(row) != True and not any_trigger_set:
            return

        excluded_forms = self.general_check_vars['excluded_forms'][self.network]
        applicable_forms = []   # (form, missing_var) pairs eligible for B-strict
        forms_marked_missing = []   # subset of applicable_forms with _missing=1
        for form, vars in self.important_form_vars.items():
            if form == 'missing_data':
                continue
            missing_var = vars.get('missing_var', '')
            if missing_var == '':
                continue
            if not hasattr(row, missing_var):
                continue
            if form in excluded_forms:
                continue
            if self.extra_form_conditions(row, form) == False:
                continue
            applicable_forms.append((form, missing_var))
            if getattr(row, missing_var) in self.utils.all_dtype([1]):
                forms_marked_missing.append((form, missing_var))

        # Direction A: per-form, _missing=1 but no triggers on missing_data.
        if not any_trigger_set and forms_marked_missing:
            for form, missing_var in forms_marked_missing:
                error_message = (
                    f"{form} marked as missing on the form, but the"
                    " missing_data form has no missing-data triggers set"
                    " for this timepoint."
                )
                error_output = self.create_row_output(
                row, [form, 'missing_data'],
                [missing_var] + list(self._MISSING_DATA_TRIGGER_VARS),
                error_message, {'reports': ['Missingness Report']})
                self.final_output_list.append(error_output)

        # Direction B-strict: whole-TP trigger set but a form's missing
        # button is not checked. One flag per applicable form.
        if has_whole_tp_trigger:
            marked_set = {f for f, _ in forms_marked_missing}
            for form, missing_var in applicable_forms:
                if form in marked_set:
                    continue
                error_message = (
                    "missing_data form indicates whole-timepoint"
                    f" missingness ({', '.join(whole_tp_triggered)} set),"
                    f" but {form} does not have its missing-data button"
                    " checked."
                )
                error_output = self.create_row_output(
                row, [form, 'missing_data'],
                [missing_var] + whole_tp_triggered,
                error_message, {'reports': ['Missingness Report']})
                self.final_output_list.append(error_output)

        # Direction B-coarse: domain-only trigger, but no form is marked
        # missing anywhere. Single summary flag attributed to missing_data.
        if (has_domain_trigger and not has_whole_tp_trigger
        and not forms_marked_missing):
            error_message = (
                "missing_data form indicates a domain is missing"
                " (chrmiss_domain set), but no individual form has its"
                " missing-data button checked for this timepoint."
            )
            error_output = self.create_row_output(
            row, ['missing_data'], [self._DOMAIN_TRIGGER_VAR],
            error_message, {'reports': ['Missingness Report']})
            self.final_output_list.append(error_output)

    def missing_code_check(self, row):
        for form, vars in self.important_form_vars.items():
            if 'missing_spec_var' in list(vars.keys()) and vars['missing_spec_var'] != '':                
                if (hasattr(row, vars['missing_spec_var'])
                and getattr(row, vars['missing_var']) in self.utils.all_dtype([1])
                and getattr(row, vars['missing_spec_var']) == ''):
                    form = self.grouped_vars['var_forms'][vars['missing_spec_var']]
                    error_message = f'Missing data button clicked, but reason not specified.'
                    error_output = self.create_row_output(
                    row, [form], [vars['missing_spec_var'],vars['missing_var']], error_message, 
                     {"reports" : ['Main Report']})
                    self.final_output_list.append(error_output)
    
    def date_inconsistency_check(self, row):
        pass