import pandas as pd
import os
import sys
import json
from datetime import datetime, timedelta

parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-4])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck
from qc_forms.qc_types.clinical_checks.scid_checks import ScidChecks
from qc_forms.qc_types.clinical_checks.pharm_checks import PharmChecks
from qc_forms.qc_types.clinical_checks.conversion_checks import ConversionChecks

class ClinicalChecksMain(FormCheck):
    """
    QC Checks related to clinical forms
    """

    ONGOING_OFFSET_CODES = {"1901-01-01", "1909-03-03", "1909-09-09"}

    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
        self.test_val = 0

        self.gf_score_check_vars = {
            "global_functioning_role_scale": [
                "chrgfr_gf_role_low",
                "chrgfr_gf_role_scole",
                "chrgfr_gf_role_high",
            ],
            "global_functioning_social_scale": [
                "chrgfs_gf_social_low",
                "chrgfs_gf_social_scale",
                "chrgfs_gf_social_high",
            ],
        }

        self.score_range_check_vars = {
            "sofas_screening": {
                "vars": [
                    "chrsofas_premorbid",
                    "chrsofas_currscore12mo",
                    "chrsofas_currscore",
                    "chrsofas_lowscore",
                ],
                "min": 0,
                "max": 100,
            },
            "sofas_followup": {
                "vars": [
                    "chrsofas_currscore_fu",
                    "chrsofas_currscore12mo_fu",
                ],
                "min": 0,
                "max": 100,
            },
            "global_functioning_role_scale": {
                "vars": [
                    "chrgfr_gf_role_low",
                    "chrgfr_gf_role_scole",
                    "chrgfr_gf_role_high",
                ],
                "min": 0,
                "max": 10,
            },
            "global_functioning_role_scale_followup": {
                "vars": [
                    "chrgfrfu_gf_role_scole",
                ],
                "min": 0,
                "max": 10,
            },
            "global_functioning_social_scale": {
                "vars": [
                    "chrgfs_gf_social_low",
                    "chrgfs_gf_social_scale",
                    "chrgfs_gf_social_high",
                ],
                "min": 0,
                "max": 10,
            },
            "global_functioning_social_scale_followup": {
                "vars": [
                    "chrgfsfu_gf_social_scale",
                ],
                "min": 0,
                "max": 10,
            },
        }

        self.excluded_21_day_forms = [
            "cbc_with_differential",
            "gcp_cbc_with_differential",
            "gcp_current_health_status",
            "psychs_p1p8_fu",
            "psychs_p1p8_fu_hc",
            "chrpred_interview_date",
            "sociodemographics",
        ]

        scid_checks = ScidChecks(row, timepoint, network, form_check_info)
        self.final_output_list = scid_checks()
        med_checks = PharmChecks(row, timepoint, network, form_check_info)
        self.final_output_list.extend(med_checks())
        conv_checks = ConversionChecks(row, timepoint, network, form_check_info)
        self.final_output_list.extend(conv_checks())

        self.call_checks(row)

    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        self.call_global_function_checks(row)
        self.call_score_range_checks(row)
        self.call_oasis_checks(row)
        self.call_cssrs_checks(row)
        self.call_twenty_one_day_check(row)
        self.call_tbi_checks(row)
        self.call_bprs_checks(row)
        self.call_premorbid_adjustment_checks(row)
        self.call_age_comparisons(row)
        self.call_pps_checks(row)

    def call_score_range_checks(self, row):
        report_list = ["Main Report", "Non Team Forms"]
        for form, range_info in self.score_range_check_vars.items():
            for var in range_info["vars"]:
                self.score_range_check(
                    row,
                    [form],
                    [var],
                    {"reports": report_list},
                    bl_filtered_vars=[],
                    filter_excl_vars=True,
                    score_min=range_info["min"],
                    score_max=range_info["max"],
                )

    @FormCheck.standard_qc_check_filter
    def score_range_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
        score_min=0,
        score_max=0,
    ):
        var = all_vars[0]
        val = getattr(row, var)
        # missing codes not allowed for now
        #if val in self.utils.missing_code_set or val == "":
        #    return
        if not self.utils.can_be_float(val) or val in self.utils.all_dtype([-3,-9]):
            return
        fval = float(val)
        if fval < score_min or fval > score_max:
            return (
                f"{var} ({val}) is outside the valid range "
                f"[{score_min}, {score_max}]."
            )

    def call_pps_checks(self, row):
        forms = ["psychosis_polyrisk_score"]
        reports = {"reports": ["Main Report", "Non Team Forms"]}
        for dob_var, pps_age_var in {
            "chrpps_fdob": "chrpps_fage",
            "chrpps_mdob": "chrpps_mage",
        }.items():
            self.pps_dob_age_range_check(
                row, forms, ["chrpps_interview_date", dob_var], reports
            )
            for age_var in [pps_age_var]:
                self.pps_age_comp(
                    row,
                    forms,
                    ["chrpps_interview_date", dob_var, age_var],
                    bl_filtered_vars=[],
                    filter_excl_vars=False,
                    age_var=age_var,
                    pps_dob_var=dob_var,
                )

    def call_age_comparisons(self, row):
        for var in [
            "chrpps_fage",
            "chrpps_mage",
            "chrfigs_mother_age",
            "chrfigs_father_age",
        ]:
            forms = [self.grouped_vars["var_forms"][var]]
            reports = {"reports": ["Main Report", "Non Team Forms"]}
            self.age_comparison(
                row,
                forms,
                [var],
                reports,
                bl_filtered_vars=[],
                filter_excl_vars=True,
                compared_age_var=var,
                diff_min=10,
                diff_max=85,
            )

        """for var in [
            "chrscid_c56_c65",
            "chrscid_a52",
            "chrscid_a109",
            "chrscid_c58_c66",
            "chrscid_d45_d49_d54_d56",
            "chrscid_d61_d65",
            "chrscid_e19_37",
            "chrscid_e164_302",
            "chrscid_e168_306",
            "chrscid_e172_310",
            "chrscid_e176_314",
            "chrscid_e180_318",
            "chrscid_e184_322",
            "chrscid_e188_326",
            "chrscid_e192_330",
        ]:
            forms = [self.grouped_vars["var_forms"][var]]
            reports = {"reports": ["Main Report", "Non Team Forms"]}
            self.age_comparison(
                row,
                forms,
                [var],
                reports,
                bl_filtered_vars=[],
                filter_excl_vars=True,
                compared_age_var=var,
                diff_min=0,
                diff_max=0,
            )"""

    def call_premorbid_adjustment_checks(self, row):
        forms = ["premorbid_adjustment_scale"]
        reports = {"reports": ["Main Report", "Non Team Forms"]}
        self.pas_marriage_check(
            row,
            forms,
            ["chrpas_pmod_adult3v3", "chrpas_pmod_adult3v1"],
            reports,
        )

    def call_bprs_checks(self, row):
        changed_output = {"reports": ["Main Report"]}
        form = "bprs"
        initial_vars = [
            "chrbprs_bprs_somc",
            "chrbprs_bprs_guil",
            "chrbprs_bprs_gran",
        ]

        var_val_pairs = []
        for var in initial_vars:
            val_checks = {}
            val_checks[var] = [6, 7]
            val_checks["chrbprs_bprs_unus"] = [1, 2, 3]
            var_val_pairs.append(val_checks)

        var_val_pairs.append(
            {"chrbprs_bprs_susp": [6, 7], "chrbprs_bprs_unus": [1]}
        )

        for pair in var_val_pairs:
            vars_ = list(pair.keys())
            self.bprs_val_comparisons(
                row,
                [form],
                all_vars=vars_,
                changed_output_vals=changed_output,
                bl_filtered_vars=[],
                filter_excl_vars=False,
                var_comps=pair,
            )

    @FormCheck.standard_qc_check_filter
    def bprs_val_comparisons(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=False,
        var_comps={},
    ):
        all_vars = list(var_comps.keys())
        if all(self.utils.can_be_float(getattr(row, var)) for var in all_vars):
            if all(float(getattr(row, var)) in var_comps[var] for var in all_vars):
                return (
                    f"{all_vars[0]} is {getattr(row, all_vars[0])}, but "
                    f"{all_vars[1]} is {getattr(row, all_vars[1])}."
                )

    def call_global_function_checks(self, row):
        report_list = ["Main Report", "Non Team Forms"]
        for form, score_vars in self.gf_score_check_vars.items():
            self.functioning_score_check(
                row,
                [form],
                score_vars,
                {"reports": report_list},
                bl_filtered_vars=[],
                filter_excl_vars=True,
            )

    def call_oasis_checks(self, row):
        forms = ["oasis"]
        report_list = ["Secondary Report"]
        for x in range(2, 6):
            oasis_anx_var = f"chroasis_oasis_{x}"
            self.oasis_anxiety_check(
                row,
                forms,
                ["chroasis_oasis_1", oasis_anx_var],
                {"reports": report_list},
                bl_filtered_vars=[],
                filter_excl_vars=True,
                anx_var=oasis_anx_var,
            )
        oasis_lifestyle_vars = {"chroasis_oasis_4": 1, "chroasis_oasis_5": 0}
        for oasis_lifestyle_var, cutoff_val in oasis_lifestyle_vars.items():
            self.oasis_lifestyle_check(
                row,
                forms,
                ["chroasis_oasis_1", oasis_lifestyle_var],
                {"reports": report_list},
                bl_filtered_vars=[],
                filter_excl_vars=True,
                lifestyle_var=oasis_lifestyle_var,
                cutoff=cutoff_val,
            )

    def call_cssrs_checks(self, row):
        forms = ["cssrs_baseline"]
        report_list = ["Main Report", "Non Team Forms"]
        cssrs_unequal_vals_dict = {
            "chrcssrsb_cssrs_actual": "chrcssrsb_sb1l",
            "chrcssrsb_cssrs_nssi": "chrcssrsb_sbnssibl",
            "chrcssrsb_cssrs_yrs_ia": "chrcssrsb_sb3l",
            "chrcssrsb_cssrs_yrs_aa": "chrcssrsb_sb4l",
            "chrcssrsb_cssrs_yrs_pab": "chrcssrsb_sb5l",
            "chrcssrsb_cssrs_yrs_sb": "chrcssrsb_sb6l",
        }
        cssr_greater_vals_dict = {
            "chrcssrsb_idintsvl": "chrcssrsb_css_sipmms",
            "chrcssrsb_snmacatl": "chrcssrsb_cssrs_num_attempt",
            "chrcssrsb_nminatl": "chrcssrsb_cssrs_yrs_nia",
            "chrcssrsb_nmabatl": "chrcssrsb_cssrs_yrs_naa",
        }

        for x in range(1, 6):
            cssrs_unequal_vals_dict[f"chrcssrsb_si{x}l"] = f"chrcssrsb_css_sim{x}"

        for cssrs_dict, method in [
            (cssrs_unequal_vals_dict, self.cssrs_unequal_vals_check),
            (cssr_greater_vals_dict, self.cssrs_greater_vals_check),
        ]:
            for lifetime_cssrs_var, recent_cssrs_var in cssrs_dict.items():
                method(
                    row,
                    forms,
                    [lifetime_cssrs_var, recent_cssrs_var],
                    {"reports": report_list},
                    [],
                    True,
                    lifetime_var=lifetime_cssrs_var,
                    recent_var=recent_cssrs_var,
                )

    def call_tbi_checks(self, row):
        forms = ["traumatic_brain_injury_screen"]
        self.tbi_inj_mismatch_check(
            row,
            forms,
            [
                "chrtbi_parent_headinjury",
                "chrtbi_subject_head_injury",
                "chrtbi_sourceinfo",
            ],
            {"reports": ["Main Report", "Non Team Forms"]},
        )

        tbi_info_source_reports = {
            "PRONET": {"reports": ["Secondary Report"]},
            "PRESCIENT": {
                "reports": ["Main Report", "Non Team Forms", "Secondary Report"]
            },
        }
        self.tbi_info_source_check(
            row,
            forms,
            [
                "chrtbi_parent_headinjury",
                "chrtbi_subject_head_injury",
                "chrtbi_sourceinfo",
            ],
            tbi_info_source_reports[self.network],
        )

    @FormCheck.standard_qc_check_filter
    def tbi_inj_mismatch_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
    ):
        sub_inj = row.chrtbi_subject_head_injury
        par_inj = row.chrtbi_parent_headinjury
        if row.chrtbi_sourceinfo in self.utils.all_dtype([3]):
            if (
                self.utils.can_be_float(sub_inj)
                and self.utils.can_be_float(par_inj)
                and not any(val in self.utils.missing_code_set for val in [sub_inj, par_inj])
            ):
                if float(sub_inj) != float(par_inj):
                    return (
                        "Subject and parent answered differently to whether "
                        "or not the subject has ever had a head injury."
                    )

    @FormCheck.standard_qc_check_filter
    def tbi_info_source_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
    ):
        if row.chrtbi_sourceinfo in self.utils.all_dtype([1, 2]):
            if (
                row.chrtbi_subject_head_injury in self.utils.all_dtype([1, 0])
                and row.chrtbi_parent_headinjury in self.utils.all_dtype([1, 0])
            ):
                return (
                    "Subject and parent not both selected as source of information, "
                    "but answers appear to be provided by both the subject and parent."
                )

    @FormCheck.standard_qc_check_filter
    def cssrs_unequal_vals_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
        lifetime_var="",
        recent_var="",
    ):
        lifetime_var_val = getattr(row, lifetime_var)
        recent_var_val = getattr(row, recent_var)
        for var_val in [lifetime_var_val, recent_var_val]:
            if var_val in self.utils.missing_code_set or not self.utils.can_be_float(var_val):
                return

        if recent_var_val in self.utils.all_dtype([2]) and lifetime_var_val in self.utils.all_dtype([1]):
            return (
                f"Recent variable ({recent_var}) was answered as yes, "
                f"but lifetime variable ({lifetime_var}) was answered as no."
            )

    @FormCheck.standard_qc_check_filter
    def cssrs_greater_vals_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
        lifetime_var="",
        recent_var="",
    ):
        lifetime_var_val = getattr(row, lifetime_var)
        recent_var_val = getattr(row, recent_var)
        for var_val in [lifetime_var_val, recent_var_val]:
            if var_val in self.utils.missing_code_set or not self.utils.can_be_float(var_val):
                return

        if float(lifetime_var_val) < float(recent_var_val):
            return (
                f"Lifetime variable ({lifetime_var} = {lifetime_var_val}) has lower intensity "
                f"than recent variable ({recent_var} = {recent_var_val})."
            )

    @FormCheck.standard_qc_check_filter
    def oasis_anxiety_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
        anx_var="",
    ):
        if row.chroasis_oasis_1 in self.utils.all_dtype([0]):
            anx_var_val = getattr(row, anx_var)
            if (
                anx_var_val not in (self.utils.missing_code_list + [""])
                and self.utils.can_be_float(anx_var_val)
                and float(anx_var_val) > 0
            ):
                return (
                    f"Marked as having no anxiety in chroasis_oasis_1, "
                    f"but {anx_var} is equal to {anx_var_val}."
                )

    @FormCheck.standard_qc_check_filter
    def oasis_lifestyle_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
        lifestyle_var="",
        cutoff=0,
    ):
        compared_oasis_val = getattr(row, "chroasis_oasis_3")
        lifestyle_var_val = getattr(row, lifestyle_var)
        for var_val in [compared_oasis_val, lifestyle_var_val]:
            if var_val in self.utils.missing_code_set or not self.utils.can_be_float(var_val):
                return

        if float(compared_oasis_val) < 2 and float(lifestyle_var_val) > cutoff:
            return (
                f"chroasis_oasis_3 states that lifestyle was not affected, "
                f"but {lifestyle_var} is greater than {cutoff}."
            )

    @FormCheck.standard_qc_check_filter
    def functioning_score_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
    ):
        for score_ind in range(1, len(all_vars)):
            gt_var = all_vars[score_ind]
            lt_var = all_vars[score_ind - 1]
            if all(
                var_val not in (self.utils.missing_code_list + [""])
                and self.utils.can_be_float(var_val)
                for var_val in [getattr(row, gt_var), getattr(row, lt_var)]
            ):
                if float(getattr(row, gt_var)) < float(getattr(row, lt_var)):
                    return (
                        f"{gt_var} ({getattr(row, gt_var)}) is less than "
                        f"{lt_var} ({getattr(row, lt_var)})."
                    )

    def call_twenty_one_day_check(self, row):
        if self.timepoint != "baseline":
            return
        cohort = self.subject_info[row.subjectid]["cohort"]
        if cohort.lower() not in ["hc", "chr"]:
            return

        curr_tp_forms = self.forms_per_tp[cohort][self.timepoint]
        missing_spec_vars = {
            "psychs_p1p8_fu": "chrpsychs_fu_missing_spec_fu",
            "psychs_p1p8_fu_hc": "hcpsychs_fu_missing_spec_fu",
        }
        scr_int_date = str(self.subject_info[row.subjectid]["psychs_screen_date"])
        if (not self.utils.check_if_val_date_format(scr_int_date)
        or scr_int_date in self.utils.missing_code_set):
            return

        curr_psychs_form = None
        for form in curr_tp_forms:
            if form in ["psychs_p1p8_fu_hc", "psychs_p1p8_fu"]:
                curr_psychs_form = form

        if curr_psychs_form is None:
            return

        missing_spec_var = missing_spec_vars[curr_psychs_form]
        self.check_if_over_21_days(row, missing_spec_var,
        scr_int_date, curr_tp_forms, curr_psychs_form)

    def check_if_over_21_days(self, row, missing_spec_var,
    scr_int_date, curr_tp_forms, curr_psychs_form):
        reports = ["Main Report", "Non Team Forms"]
        if hasattr(row, missing_spec_var) and str(getattr(row, missing_spec_var)) == "M6":
            flagged_forms = []
            for form in curr_tp_forms:
                if form in self.excluded_21_day_forms:
                    continue
                if self.check_if_missing(row, form) is True:
                    continue

                int_date_var = self.important_form_vars[form]["interview_date_var"]
                if hasattr(row, int_date_var):
                    int_date = str(getattr(row, int_date_var))
                    if not self.utils.check_if_val_date_format(int_date) or int_date in self.utils.missing_code_set:
                        continue
                    days_between = self.utils.find_days_between(int_date, scr_int_date)
                    if days_between > 21:
                        flagged_forms.append(f"{form} (date = {int_date})")

            if flagged_forms:
                error_message = (
                    "M6 clicked on baseline psychs, but there are more than 21 days between "
                    f"Screening Psychs (date = {scr_int_date}) and the following forms: {flagged_forms}"
                )
                error_output = self.create_row_output(
                    row,
                    [curr_psychs_form],
                    [missing_spec_var],
                    error_message,
                    {"reports": reports},
                )
                self.final_output_list.append(error_output)

    @FormCheck.standard_qc_check_filter
    def pas_marriage_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=False,
    ):
        if (
            getattr(row, "chrpas_pmod_adult3v1") in self.utils.all_dtype([0, 1, 2, 3])
            and getattr(row, "chrpas_pmod_adult3v3") in self.utils.all_dtype([0, 1, 2, 3, 4, 5, 6])
        ):
            return (
                f"chrpas_pmod_adult3v1 is equal to {getattr(row, 'chrpas_pmod_adult3v1')}, "
                f"but chrpas_pmod_adult3v3 is equal to {getattr(row, 'chrpas_pmod_adult3v3')}."
            )

    @FormCheck.standard_qc_check_filter
    def pps_dob_age_range_check(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
    ):
        if any(
            getattr(row, var) in self.utils.missing_code_set
            or not self.utils.check_if_val_date_format(str(getattr(row, var)).split(" ")[0])
            for var in all_vars
        ):
            return
        pps_int_date = str(getattr(row, all_vars[0])).split(" ")[0]
        pps_date = str(getattr(row, all_vars[1])).split(" ")[0]
        days_between = self.utils.find_days_between(pps_int_date, pps_date)
        yrs_between = days_between / 365
        if yrs_between < 10 or yrs_between > 85:
            return f"Difference between {all_vars[0]} and {all_vars[1]} is {yrs_between}."

    @FormCheck.standard_qc_check_filter
    def pps_age_comp(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
        age_var="",
        pps_dob_var="",
    ):
        if any(
            getattr(row, var) in self.utils.missing_code_set
            or not self.utils.check_if_val_date_format(str(getattr(row, var)).split(" ")[0])
            for var in [pps_dob_var, "chrpps_interview_date"]
        ):
            return

        if age_var == "demographics_age":
            age = self.subject_info[row.subjectid]["age"]
        else:
            age = getattr(row, age_var)

        if age in self.utils.missing_code_set or not self.utils.can_be_float(age):
            return

        pps_int_date = str(getattr(row, "chrpps_interview_date")).split(" ")[0]
        pps_dob = str(getattr(row, pps_dob_var)).split(" ")[0]
        days_between = self.utils.find_days_between(pps_int_date, pps_dob)
        yrs_between = days_between / 365
        age_diffs = yrs_between - float(age)
        if age_diffs >= 2:
            return f"Age ({age_var} = {age}) does not align with date of birth ({pps_dob})."

    @FormCheck.standard_qc_check_filter
    def age_comparison(
        self,
        row,
        filtered_forms,
        all_vars,
        changed_output_vals,
        bl_filtered_vars=[],
        filter_excl_vars=True,
        compared_age_var="",
        diff_min=0,
        diff_max=0,
    ):
        if any(getattr(row, var) in self.utils.missing_code_set for var in all_vars):
            return
        if "age" in self.subject_info[row.subjectid].keys():
            age = self.subject_info[row.subjectid]["age"]
            compared_age = getattr(row, compared_age_var)
            if self.utils.can_be_float(age) and self.utils.can_be_float(compared_age):
                diff = round(abs(float(age) - float(compared_age)), 2)
                if diff < diff_min or diff > diff_max:
                    return (
                        f"Difference between demographics age ({age}) and "
                        f"{compared_age_var} ({compared_age}) is {diff}."
                    )

