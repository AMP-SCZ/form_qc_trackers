import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime

# Try to locate the project root more robustly than slicing __file__ paths.
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT_CANDIDATES = [_CURRENT_FILE.parent] + list(_CURRENT_FILE.parents)

for _candidate in _PROJECT_ROOT_CANDIDATES:
    if (_candidate / "utils").exists() and (_candidate / "qc_forms").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

try:
    from utils.utils import Utils
    from qc_forms.form_check import FormCheck
except ImportError as e:
    raise ImportError(
        "Could not import project dependencies. Make sure this file is inside your "
        "project tree (with 'utils' and 'qc_forms' packages available), or add the "
        "project root to PYTHONPATH."
    ) from e


class PharmChecks(FormCheck):
    """
    Rule checks for current and past pharmaceutical treatment instruments.
    """

    ACUTE_INJECTION_CODES = {
        724, 725, 257, 727, 123, 703, 11, 726, 271, 76, 245, 13,
    }

    # Offset sentinel meaning "ongoing" for medication courses.
    ONGOING_OFFSET_CODES = frozenset({"1901-01-01"})

    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)

        self.ONGOING_OFFSET_CODES = set(type(self).ONGOING_OFFSET_CODES)

        self.missing_code_list = [
            "-3",
            "-9",
            -3,
            -9,
            -3.0,
            -9.0,
            "-3.0",
            "-9.0",
            "1909-09-09",
            "1903-03-03",
            "-99",
            -99,
            -99.0,
            "-99.0",
            999,
            999.0,
            "999",
            "999.0",
        ]

        self.call_checks(row)

    def call_checks(self, row):
        self.call_pharm_checks(row)

    def call_pharm_checks(self, row):
        curr_forms = [
            "current_pharmaceutical_treatment_floating_med_125",
            "current_pharmaceutical_treatment_floating_med_2650",
        ]

        past_forms = ["past_pharmaceutical_treatment"]
        self.med_name_missing_check(row, past=False, forms=curr_forms)
        self.med_name_missing_check(row, past=True, forms=past_forms)
        self.det_med_name_improper_value(row, past=False, forms=curr_forms)
        self.det_med_name_improper_value(row, past=True, forms=past_forms)

        self.pharm_firstdose_check(row, curr_forms)

        if self.timepoint == "screening":
            self.check_pharm_date_chronologies(row, curr_forms + past_forms)

        # MED-QC-01: modification date after assessments
        self.check_pharm_interview_dates(
            row,
            curr_forms,
            {"reports": ["Main Report", "Non Team Forms"]},
        )

        self.check_current_med_dates_not_after_mod_date(row, forms=curr_forms)

        self.check_onset_after_offset(row, past=False, forms=curr_forms)
        self.check_onset_after_offset(row, past=True, forms=past_forms)

        self.check_duplicate_medication_courses(row, past=False, forms=curr_forms)
        self.check_duplicate_medication_courses(row, past=True, forms=past_forms)

        self.check_no_medication_overlap(row, past=False, forms=curr_forms)
        self.check_no_medication_overlap(row, past=True, forms=past_forms)

        self.check_no_medication_missing_offset(row, past=False, forms=curr_forms)
        self.check_no_medication_missing_offset(row, past=True, forms=past_forms)

        self.check_no_ongoing_med_in_past_form(row, forms=past_forms)

        self.check_current_med_not_ongoing_but_has_offset(row, forms=curr_forms)

        self.check_current_med_not_before_past_form(row, forms=curr_forms)

        self.check_no_medication_code_with_info(row, past=False, forms=curr_forms)
        self.check_no_medication_code_with_info(row, past=True, forms=past_forms)

        self.check_compliance_range(row, past=False, forms=curr_forms)
        self.check_compliance_range(row, past=True, forms=past_forms)

        self.check_frequency_missing_code(row, past=False, forms=curr_forms)
        self.check_frequency_missing_code(row, past=True, forms=past_forms)

        self.check_acute_injection_longer_than_two_weeks(
            row, past=False, forms=curr_forms
        )
        self.check_acute_injection_longer_than_two_weeks(
            row, past=True, forms=past_forms
        )

    def __call__(self):
        return self.final_output_list

    def med_name_missing_check(self, row, past=False, forms=None):
        # MED-QC-11
        if forms is None:
            forms = []

        for med_num in self._iter_med_nums(row, past=past):
            name_var = self._med_var(med_num, "name", past=past)
            if not hasattr(row, name_var):
                continue

            name_val = getattr(row, name_var)
            if str(name_val).strip() not in [str(v) for v in self.utils.all_dtype([888])]:
                continue

            vars_to_check = [
                self._med_use_var(med_num, past=past),
                self._med_var(med_num, "dosage", past=past),
                self._med_var(med_num, "dosage_2", past=past),
                self._med_var(med_num, "frequency", past=past),
                self._med_var(med_num, "comp", past=past),
                self._med_var(med_num, "indication", past=past),
            ]

            provided_vars = self._has_nonmissing_info(row, vars_to_check)
            if provided_vars:
                self._append_qc(
                    row,
                    forms,
                    [name_var] + provided_vars,
                    f"{name_var} is coded as 888 (no information), but other course-level fields "
                    f"contain data ({', '.join(provided_vars)}). Please check whether 777 would be more appropriate.",
                )

    def det_med_name_improper_value(self, row, past=False, forms=None):
        # MED-QC-13
        if forms is None:
            forms = []

        blinded_codes = {str(v) for v in self.utils.all_dtype([573, 542, 538, 539])}
        for med_num in self._iter_med_nums(row, past=past):
            name_var = self._med_var(med_num, "name", past=past)
            if not hasattr(row, name_var):
                continue
            name_val = str(getattr(row, name_var)).strip()
            if name_val in blinded_codes:
                self._append_qc(
                    row,
                    forms,
                    [name_var],
                    f"{name_var} is coded as blinded medication ({name_val}). "
                    "Please check the medication name and update it if the medication is known now.",
                )

    def check_pharm_interview_dates(self, row, forms, changed_output_vals):
        # MED-QC-01
        if self.network == "PRESCIENT":
            return

        excluded_interview_date_vars = {
            "chrmiss_interview_date",
        }

        mod_vars = ["chrpharm_date_mod", "chrpharm_date_mod_2"]
        mod_dates = []
        for mod_var in mod_vars:
            if hasattr(row, mod_var) and self._valid_date(getattr(row, mod_var)):
                mod_dates.append((mod_var, self._parse_date(getattr(row, mod_var))))

        if not mod_dates:
            return

        latest_assessment_var = None
        latest_assessment_dt = None
        for form, form_info in self.important_form_vars.items():
            int_date_var = form_info.get("interview_date_var", "")
            if not int_date_var or not hasattr(row, int_date_var):
                continue
            if int_date_var in excluded_interview_date_vars:
                continue
            int_date_val = getattr(row, int_date_var)
            if not self._valid_date(int_date_val):
                continue
            parsed = self._parse_date(int_date_val)
            if latest_assessment_dt is None or parsed > latest_assessment_dt:
                latest_assessment_dt = parsed
                latest_assessment_var = int_date_var

        if latest_assessment_dt is None:
            return

        latest_mod_var, latest_mod_dt = max(mod_dates, key=lambda x: x[1])
        if latest_assessment_dt > latest_mod_dt:
            self._append_qc(
                row,
                forms,
                [latest_assessment_var, latest_mod_var],
                f"Latest modification date ({latest_mod_var} = {latest_mod_dt.strftime('%Y-%m-%d')}) "
                f"is before the latest assessment date ({latest_assessment_var} = {latest_assessment_dt.strftime('%Y-%m-%d')}). "
                "Please update the pharmaceutical treatment modification date to reflect the most recent assessment.",
            )

    def check_current_med_dates_not_after_mod_date(self, row, forms=None):
        # MED-QC-03
        if self.network == "PRESCIENT":
            return

        if forms is None:
            forms = []

        mod_candidates = []
        for mod_var in ["chrpharm_date_mod", "chrpharm_date_mod_2"]:
            if hasattr(row, mod_var) and self._valid_date(getattr(row, mod_var)):
                mod_candidates.append((mod_var, self._parse_date(getattr(row, mod_var))))

        if not mod_candidates:
            return

        mod_var, mod_dt = max(mod_candidates, key=lambda x: x[1])

        for med_num in self._iter_med_nums(row, past=False):
            for date_type in ["onset", "offset"]:
                med_date_var = self._med_var(med_num, date_type, past=False)
                if not hasattr(row, med_date_var):
                    continue

                med_date_val = getattr(row, med_date_var)
                if not self._valid_date(med_date_val):
                    continue

                if self._is_ongoing_offset_code(med_date_val):
                    continue

                med_dt = self._parse_date(med_date_val)
                if med_dt > mod_dt:
                    self._append_qc(
                        row,
                        forms,
                        [med_date_var, mod_var],
                        f"{med_date_var} ({self._date_to_str(med_date_val)}) is later than "
                        f"{mod_var} ({mod_dt.strftime('%Y-%m-%d')}). Medication onset/offset dates "
                        "cannot occur after the form modification date.",
                    )

    def check_onset_after_offset(self, row, past=False, forms=None):
        # MED-QC-02
        if forms is None:
            forms = []

        for med_num in self._iter_med_nums(row, past=past):
            onset_var = self._med_var(med_num, "onset", past=past)
            offset_var = self._med_var(med_num, "offset", past=past)
            if not hasattr(row, onset_var) or not hasattr(row, offset_var):
                continue

            onset_val = getattr(row, onset_var)
            offset_val = getattr(row, offset_var)

            if not self._valid_date(onset_val) or not self._valid_date(offset_val):
                continue

            if self._is_ongoing_offset_code(offset_val):
                continue

            onset_dt = self._parse_date(onset_val)
            offset_dt = self._parse_date(offset_val)
            if onset_dt > offset_dt:
                self._append_qc(
                    row,
                    forms,
                    [onset_var, offset_var],
                    f"{onset_var} ({self._date_to_str(onset_val)}) is later than "
                    f"{offset_var} ({self._date_to_str(offset_val)}). A medication course cannot "
                    "end before it starts.",
                )

    def check_duplicate_medication_courses(self, row, past=False, forms=None):
        # MED-QC-05
        if forms is None:
            forms = []

        med_nums = self._iter_med_nums(row, past=past)
        for i, med_num_1 in enumerate(med_nums):
            for med_num_2 in med_nums[i + 1 :]:
                name_var_1 = self._med_var(med_num_1, "name", past=past)
                name_var_2 = self._med_var(med_num_2, "name", past=past)
                onset_var_1 = self._med_var(med_num_1, "onset", past=past)
                onset_var_2 = self._med_var(med_num_2, "onset", past=past)
                offset_var_1 = self._med_var(med_num_1, "offset", past=past)
                offset_var_2 = self._med_var(med_num_2, "offset", past=past)
                use_var_1 = self._med_use_var(med_num_1, past=past)
                use_var_2 = self._med_use_var(med_num_2, past=past)

                needed = [
                    name_var_1,
                    name_var_2,
                    onset_var_1,
                    onset_var_2,
                    offset_var_1,
                    offset_var_2,
                    use_var_1,
                    use_var_2,
                ]
                if not all(hasattr(row, v) for v in needed):
                    continue

                name_1 = getattr(row, name_var_1)
                name_2 = getattr(row, name_var_2)
                if self._is_missing_val(name_1) or self._is_missing_val(name_2):
                    continue
                if str(name_1).strip() != str(name_2).strip():
                    continue

                use_1 = getattr(row, use_var_1)
                use_2 = getattr(row, use_var_2)
                if not (str(use_1).strip() == "1" and str(use_2).strip() == "1"):
                    continue

                if not all(
                    self._valid_date(getattr(row, v))
                    for v in [onset_var_1, onset_var_2, offset_var_1, offset_var_2]
                ):
                    continue

                onset_1_str = self._date_to_str(getattr(row, onset_var_1))
                offset_1_str = self._date_to_str(getattr(row, offset_var_1))
                onset_2_str = self._date_to_str(getattr(row, onset_var_2))
                offset_2_str = self._date_to_str(getattr(row, offset_var_2))

                if self._date_ranges_overlap_strict(
                    getattr(row, onset_var_1),
                    getattr(row, offset_var_1),
                    getattr(row, onset_var_2),
                    getattr(row, offset_var_2),
                ):
                    self._append_qc(
                        row,
                        forms,
                        [
                            name_var_1,
                            onset_var_1,
                            offset_var_1,
                            use_var_1,
                            name_var_2,
                            onset_var_2,
                            offset_var_2,
                            use_var_2,
                        ],
                        f"{name_var_1} ({onset_1_str} to {offset_1_str}) and "
                        f"{name_var_2} ({onset_2_str} to {offset_2_str}) have the same medication name "
                        f"({str(name_1).strip()}), overlapping date ranges, and both courses are marked as simultaneously used "
                        f"({use_var_1} = 1 and {use_var_2} = 1). Please check for a duplicate medication course.",
                    )

    def check_no_medication_overlap(self, row, past=False, forms=None):
        # MED-QC-06
        if forms is None:
            forms = []

        courses = []
        for med_num in self._iter_med_nums(row, past=past):
            name_var = self._med_var(med_num, "name", past=past)
            onset_var = self._med_var(med_num, "onset", past=past)
            offset_var = self._med_var(med_num, "offset", past=past)

            if not all(hasattr(row, v) for v in [name_var, onset_var, offset_var]):
                continue

            name_val = getattr(row, name_var)
            onset_val = getattr(row, onset_var)
            offset_val = getattr(row, offset_var)

            name_is_no_medication = self._is_no_medication(name_val)
            if not name_is_no_medication and self._is_missing_med_name(name_val):
                continue

            if not self._valid_date(onset_val) or not self._valid_date(offset_val):
                continue

            if self._is_ongoing_offset_code(offset_val):
                continue

            courses.append(
                {
                    "name_var": name_var,
                    "name_val": self._normalize_med_code(name_val),
                    "is_no_medication": name_is_no_medication,
                    "onset_var": onset_var,
                    "offset_var": offset_var,
                    "onset_val": onset_val,
                    "offset_val": offset_val,
                }
            )

        no_med_courses = [c for c in courses if c["is_no_medication"]]
        other_med_courses = [c for c in courses if not c["is_no_medication"]]

        for no_med in no_med_courses:
            for other_med in other_med_courses:
                if self._date_ranges_overlap_strict(
                    no_med["onset_val"],
                    no_med["offset_val"],
                    other_med["onset_val"],
                    other_med["offset_val"],
                ):
                    no_med_onset_str = self._date_to_str(no_med["onset_val"])
                    no_med_offset_str = self._date_to_str(no_med["offset_val"])
                    other_onset_str = self._date_to_str(other_med["onset_val"])
                    other_offset_str = self._date_to_str(other_med["offset_val"])
                    self._append_qc(
                        row,
                        forms,
                        [
                            no_med["name_var"],
                            no_med["onset_var"],
                            no_med["offset_var"],
                            other_med["name_var"],
                            other_med["onset_var"],
                            other_med["offset_var"],
                        ],
                        f"{no_med['name_var']} contains 999 (no medication) from "
                        f"{no_med_onset_str} to {no_med_offset_str}, but its date range "
                        f"strictly overlaps with another medication course "
                        f"({other_med['name_var']}: {other_onset_str} to {other_offset_str}). "
                        "A no-medication period can touch another course on the same start/end date, "
                        "but it should not overlap beyond that boundary.",
                    )

    def check_no_medication_missing_offset(self, row, past=False, forms=None):
        # MED-QC-07
        if forms is None:
            forms = []

        med_nums = self._iter_med_nums(row, past=past)

        for med_num in med_nums:
            name_var = self._med_var(med_num, "name", past=past)
            onset_var = self._med_var(med_num, "onset", past=past)
            offset_var = self._med_var(med_num, "offset", past=past)

            if not all(hasattr(row, v) for v in [name_var, onset_var, offset_var]):
                continue

            name_val = getattr(row, name_var)
            onset_val = getattr(row, onset_var)
            offset_val = getattr(row, offset_var)

            if not self._is_no_medication(name_val):
                continue

            if not self._valid_date(onset_val):
                continue

            offset_missing_or_ongoing = self._is_missing_val(offset_val) or self._is_ongoing_offset_code(offset_val)
            if not offset_missing_or_ongoing:
                continue

            no_med_onset_dt = self._parse_date(onset_val)

            for other_num in med_nums:
                if other_num == med_num:
                    continue
                other_name_var = self._med_var(other_num, "name", past=past)
                other_onset_var = self._med_var(other_num, "onset", past=past)

                if not all(hasattr(row, v) for v in [other_name_var, other_onset_var]):
                    continue

                other_name_val = getattr(row, other_name_var)
                other_onset_val = getattr(row, other_onset_var)

                if self._is_missing_med_name(other_name_val) or self._is_no_medication(other_name_val):
                    continue

                if not self._valid_date(other_onset_val):
                    continue

                if self._parse_date(other_onset_val) > no_med_onset_dt:
                    self._append_qc(
                        row,
                        forms,
                        [name_var, onset_var, offset_var, other_name_var, other_onset_var],
                        f"{name_var} contains 999 (no medication) starting {self._date_to_str(onset_val)}, "
                        f"but {offset_var} is missing or coded as ongoing "
                        f"while another medication course ({other_name_var}) starts later on "
                        f"{self._date_to_str(other_onset_val)}. Please end the no-medication period or review the later medication course.",
                    )
                    break

    def check_no_ongoing_med_in_past_form(self, row, forms=None):
        # MED-QC-08
        if forms is None:
            forms = []

        for med_num in self._iter_med_nums(row, past=True):
            offset_var = self._med_var(med_num, "offset", past=True)
            if not hasattr(row, offset_var):
                continue

            offset_val = getattr(row, offset_var)
            if self._is_ongoing_offset_code(offset_val):
                self._append_qc(
                    row,
                    forms,
                    [offset_var],
                    f"{offset_var} is coded as ongoing (1901-01-01), but medication courses in the past "
                    "pharmaceutical treatment form cannot be ongoing. Please add this medication in the current pharmaceutical treatment form.",
                )

    def check_current_med_not_ongoing_but_has_offset(self, row, forms=None):
        # MED-QC-09
        if forms is None:
            forms = []

        for med_num in self._iter_med_nums(row, past=False):
            ongoing_var = self._course_ongoing_var(med_num)
            if not hasattr(row, ongoing_var):
                continue

            offset_var = self._med_var(med_num, "offset", past=False)
            if not hasattr(row, offset_var):
                continue

            ongoing_val = getattr(row, ongoing_var)
            offset_val = getattr(row, offset_var)

            if str(ongoing_val).strip() == "2":
                if not self._is_missing_val(offset_val) and not self._is_ongoing_offset_code(offset_val):
                    if self._valid_date(offset_val):
                        self._append_qc(
                            row,
                            forms,
                            [ongoing_var, offset_var],
                            f"{ongoing_var} is coded as 2, but {offset_var} is also populated with "
                            f"{self._date_to_str(offset_val)} instead of the ongoing code. Please confirm "
                            "whether this course is ongoing and reconcile the ongoing/intermittent flag with the offset date.",
                        )

    def check_current_med_not_before_past_form(self, row, forms=None):
        # MED-QC-10
        if forms is None:
            forms = []

        past_form_date_var = "chrpharm_interview_date"
        if not hasattr(row, past_form_date_var):
            return

        past_form_date_val = getattr(row, past_form_date_var)
        if not self._valid_date(past_form_date_val):
            return
        past_form_dt = self._parse_date(past_form_date_val)

        for med_num in self._iter_med_nums(row, past=False):
            offset_var = self._med_var(med_num, "offset", past=False)
            if not hasattr(row, offset_var):
                continue
            offset_val = getattr(row, offset_var)

            if not self._valid_date(offset_val):
                continue
            if self._is_ongoing_offset_code(offset_val):
                continue

            offset_dt = self._parse_date(offset_val)
            if offset_dt < past_form_dt:
                self._append_qc(
                    row,
                    forms,
                    [offset_var, past_form_date_var],
                    f"{offset_var} ({self._date_to_str(offset_val)}) is before "
                    f"{past_form_date_var} ({self._date_to_str(past_form_date_val)}). "
                    "A current-form medication course should not end before the past pharmaceutical treatment assessment date.",
                )

    def check_no_medication_code_with_info(self, row, past=False, forms=None):
        # MED-QC-12
        if forms is None:
            forms = []

        for med_num in self._iter_med_nums(row, past=past):
            name_var = self._med_var(med_num, "name", past=past)
            if not hasattr(row, name_var):
                continue

            name_val = getattr(row, name_var)
            if not self._is_no_medication(name_val):
                continue

            vars_to_check = [
                self._med_var(med_num, "indication", past=past),
                self._med_var(med_num, "dosage", past=past),
                self._med_var(med_num, "dosage_2", past=past),
                self._med_var(med_num, "frequency", past=past),
                self._med_var(med_num, "comp", past=past),
            ]

            provided_vars = self._has_nonmissing_info(row, vars_to_check)
            if provided_vars:
                self._append_qc(
                    row,
                    forms,
                    [name_var] + provided_vars,
                    f"{name_var} is coded as 999 (no medication), but other course-level fields "
                    f"contain data ({', '.join(provided_vars)}). Please check whether this period was really no medication.",
                )

    def pharm_firstdose_check(
        self,
        row,
        filtered_forms,
        all_vars=None,
        changed_output_vals=None,
        bl_filtered_vars=None,
        filter_excl_vars=True,
    ):
        # MED-QC-04 — med_info: chrpharm_med{n}_onset vs chrpharm_firstdose_med{n}.
        if bl_filtered_vars is None:
            bl_filtered_vars = []

        for med_num in range(1, 61):
            onset_var = self._med_var(med_num, "onset", past=False)
            firstdose_var = f"chrpharm_firstdose_med{med_num}"
            if not hasattr(row, onset_var) or not hasattr(row, firstdose_var):
                continue

            onset_val = getattr(row, onset_var)
            firstdose_val = getattr(row, firstdose_var)
            if self._is_missing_val(onset_val) or self._is_missing_val(firstdose_val):
                continue

            onset_date = self._date_to_str(onset_val)
            firstdose_date = self._date_to_str(firstdose_val)

            if not onset_date or not firstdose_date:
                continue

            if onset_date != firstdose_date:
                self._append_qc(
                    row,
                    filtered_forms,
                    [onset_var, firstdose_var],
                    f"{onset_var} ({onset_date}) does not match the date portion of "
                    f"{firstdose_var} ({firstdose_date}). The first-dose datetime should "
                    "have the same calendar date as the medication onset date.",
                )

    def check_pharm_date_chronologies(self, row, forms):
        subject = row.subjectid
        if not all(
            key in self.subject_info[subject].keys()
            for key in ["past_pharm_date", "curr_pharm_date"]
        ):
            return

        past_pharm_date = self._date_to_str(self.subject_info[subject]["past_pharm_date"])
        curr_pharm_date = self._date_to_str(self.subject_info[subject]["curr_pharm_date"])

        if not past_pharm_date or not curr_pharm_date:
            return

        days_diff = (
            datetime.strptime(past_pharm_date, "%Y-%m-%d")
            - datetime.strptime(curr_pharm_date, "%Y-%m-%d")
        ).days
        if days_diff > 20:
            self._append_qc(
                row,
                forms,
                ["chrpharm_interview_date", "chrpharm_date_first"],
                f"Past pharmaceutical treatment date ({past_pharm_date}) is more than 20 days later than "
                f"current pharmaceutical treatment date ({curr_pharm_date}).",
            )

    def check_compliance_range(self, row, past=False, forms=None):
        # MED-QC-14
        if forms is None:
            forms = []

        for med_num in self._iter_med_nums(row, past=past):
            comp_var = self._med_var(med_num, "comp", past=past)
            if not hasattr(row, comp_var):
                continue

            comp_val = getattr(row, comp_var)
            if not self._valid_number(comp_val):
                continue

            if float(comp_val) < 0 or float(comp_val) > 100:
                self._append_qc(
                    row,
                    forms,
                    [comp_var],
                    f"{comp_var} is {comp_val}, but compliance must be between 0 and 100 inclusive.",
                )

    def check_frequency_missing_code(self, row, past=False, forms=None):
        # MED-QC-15
        if forms is None:
            forms = []

        for med_num in self._iter_med_nums(row, past=past):
            freq_var = self._med_var(med_num, "frequency", past=past)
            if not hasattr(row, freq_var):
                continue

            freq_val = getattr(row, freq_var)
            if not self._valid_number(freq_val):
                continue

            if float(freq_val) < 0 and float(freq_val) != -3:
                self._append_qc(
                    row,
                    forms,
                    [freq_var],
                    f"{freq_var} is {freq_val}. Values below 0 are only allowed when coded as -3 for missingness.",
                )

    def check_acute_injection_longer_than_two_weeks(self, row, past=False, forms=None):
        # MED-QC-16
        if forms is None:
            forms = []

        acute_codes = {str(v) for v in self.utils.all_dtype(list(self.ACUTE_INJECTION_CODES))}

        for med_num in self._iter_med_nums(row, past=past):
            name_var = self._med_var(med_num, "name", past=past)
            onset_var = self._med_var(med_num, "onset", past=past)
            offset_var = self._med_var(med_num, "offset", past=past)

            if not all(hasattr(row, v) for v in [name_var, onset_var, offset_var]):
                continue

            name_val = getattr(row, name_var)
            if self._is_missing_val(name_val):
                continue
            if str(name_val).strip() not in acute_codes:
                continue

            onset_val = getattr(row, onset_var)
            offset_val = getattr(row, offset_var)

            if not self._valid_date(onset_val) or not self._valid_date(offset_val):
                continue
            if self._is_ongoing_offset_code(offset_val):
                continue

            onset_dt = self._parse_date(onset_val)
            offset_dt = self._parse_date(offset_val)
            if offset_dt < onset_dt:
                continue

            duration_days = (offset_dt - onset_dt).days
            if duration_days > 14:
                self._append_qc(
                    row,
                    forms,
                    [name_var, onset_var, offset_var],
                    f"{name_var} ({str(name_val).strip()}) is coded as an acute injection medication, "
                    f"but the course lasts {duration_days} days "
                    f"({onset_var} = {self._date_to_str(onset_val)}, {offset_var} = {self._date_to_str(offset_val)}), "
                    "which is longer than two weeks. Please review whether this course is truly an acute injection.",
                )

    # ---------------------------------------------------------------------
    # Medication QC helpers
    # ---------------------------------------------------------------------

    def _past_pharm_marked_complete(self, row):
        past_form = "past_pharmaceutical_treatment"
        compl_var = self.important_form_vars[past_form]["completion_var"]
        if self.network == "PRESCIENT":
            compl_var += "_rpms"
            compl_var = compl_var.replace("_hc", "").replace("onboarding", "checkin")
            if self.timepoint == "floating":
                return True
            if self.check_if_next_tp(row) is True:
                return True
        if hasattr(row, compl_var) and getattr(row, compl_var) in self.utils.all_dtype([2]):
            return True
        return False

    def _is_missing_val(self, val):
        if pd.isna(val):
            return True

        val_str = str(val).strip()
        val_str_lower = val_str.lower()
        missing_strings = {"", "nan", "nat", "none", "null"}

        if val_str_lower in missing_strings:
            return True

        # Always use the local class missing-code list first so all flag filtering
        # consistently excludes study missing codes.
        if val in self.missing_code_list or val_str in {str(v) for v in self.missing_code_list}:
            return True

        # Fallback to utils list if present.
        utils_missing = getattr(self.utils, "missing_code_list", [])
        if val in utils_missing or val_str in {str(v) for v in utils_missing}:
            return True

        return False

    def _normalize_med_code(self, val):
        val_str = str(val).strip()
        if self.utils.can_be_float(val_str):
            as_float = float(val_str)
            if as_float.is_integer():
                return str(int(as_float))
        return val_str

    def _is_no_medication(self, val):
        """True when the medication *name* field is explicitly no-medication (999), incl. 999.0 / '999'."""
        if pd.isna(val):
            return False
        return self._normalize_med_code(val) == "999"

    def _is_missing_med_name(self, val):
        """True for blank/missing name codes; false for explicit no-medication (999)."""
        return self._is_missing_val(val) and not self._is_no_medication(val)

    def _valid_number(self, val):
        return (not self._is_missing_val(val)) and self.utils.can_be_float(val)

    def _normalize_date_str(self, val):
        """
        Return a clean YYYY-MM-DD string or None.

        This helper makes medication date checks robust to blank strings,
        NaN/NaT, None, timestamps with time components, and malformed values.
        """
        if self._is_missing_val(val):
            return None

        val_str = str(val).strip()
        if not val_str:
            return None

        date_str = val_str.split(" ")[0].strip()
        if date_str.lower() in {"nan", "nat", "none", "null", ""}:
            return None

        if not self.utils.check_if_val_date_format(date_str, date_format="%Y-%m-%d"):
            return None

        # Missing-code dates (e.g. 1903-03-03) can pass format checks when the raw
        # value was a Timestamp — treat as non-dates for medication logic.
        if date_str in {str(v) for v in self.missing_code_list}:
            return None

        return date_str

    def _valid_date(self, val):
        return self._normalize_date_str(val) is not None

    def _parse_date(self, val):
        date_str = self._normalize_date_str(val)
        if date_str is None:
            return None
        return datetime.strptime(date_str, "%Y-%m-%d")

    def _date_to_str(self, val):
        date_str = self._normalize_date_str(val)
        return date_str if date_str is not None else ""

    def _is_ongoing_offset_code(self, val):
        raw_date_str = str(val).strip().split(" ")[0].strip()
        if raw_date_str in self.ONGOING_OFFSET_CODES:
            return True
        date_str = self._date_to_str(val)
        return date_str in self.ONGOING_OFFSET_CODES

    def _med_var(self, med_num, suffix, past=False):
        base = f"chrpharm_med{med_num}_{suffix}"
        return f"{base}_past" if past else base

    def _med_use_var(self, med_num, past=False):
        return f"chrpharm_med{med_num}_use_past" if past else f"chrpharm_med{med_num}_use"

    def _course_ongoing_var(self, med_num):
        # REDCap: first intermittent flag is chrpharm_interm_meds; each further
        # course uses chrpharm_interm_meds_{k-1} (see grouped_variables.json).
        if med_num <= 1:
            return "chrpharm_interm_meds"
        return f"chrpharm_interm_meds_{med_num - 1}"

    def _iter_med_nums(self, row, past=False, max_meds=60):
        med_nums = []
        for med_num in range(1, max_meds + 1):
            candidate_vars = [
                self._med_var(med_num, "name", past=past),
                self._med_var(med_num, "onset", past=past),
                self._med_var(med_num, "offset", past=past),
                self._med_use_var(med_num, past=past),
            ]
            if any(hasattr(row, v) for v in candidate_vars):
                med_nums.append(med_num)
        return med_nums

    def _append_qc(self, row, forms, vars_list, message):
        # For flags pertaining only to the past pharmaceutical treatment form,
        # require only that the form is marked complete. The full
        # standard_form_filter also runs check_if_missing, which for this form
        # counts non_branch_logic_vars (124 entries dominated by 60 unused
        # past-med dose-limit slots) and treats almost every real subject as
        # "missing", silently swallowing legitimate flags.
        past_form = "past_pharmaceutical_treatment"
        if forms == [past_form] and past_form in self.important_form_vars:
            if not self._past_pharm_marked_complete(row):
                return

        filtered_vars = []
        for var in vars_list:
            if not hasattr(row, var):
                continue

            val = getattr(row, var)

            # Exclude vars whose values are missing codes, including:
            # -3, -9, -99, 999, 1909-09-09, 1903-03-03, etc.
            if self._is_missing_val(val) and not (
                (self._is_med_name_var(var) and self._is_no_medication(val))
                or (self._is_med_offset_var(var) and self._is_ongoing_offset_code(val))
            ):
                continue

            filtered_vars.append(var)

        # Do not create a QC flag at all if every flagged field was a missing code.
        if not filtered_vars:
            return

        error_output = self.create_row_output(
            row,
            forms,
            filtered_vars,
            message,
            {"reports": ["Main Report", "Non Team Forms"]},
        )
        self.final_output_list.append(error_output)

    def _is_med_name_var(self, var):
        return (
            var.startswith("chrpharm_med")
            and (var.endswith("_name") or var.endswith("_name_past"))
        )

    def _is_med_offset_var(self, var):
        return var.startswith("chrpharm_med") and (
            var.endswith("_offset") or var.endswith("_offset_past")
        )

    def _date_ranges_overlap_inclusive(self, start1, end1, start2, end2):
        d1s = self._parse_date(start1)
        d1e = self._parse_date(end1)
        d2s = self._parse_date(start2)
        d2e = self._parse_date(end2)
        if None in [d1s, d1e, d2s, d2e]:
            return False
        return max(d1s, d2s) <= min(d1e, d2e)

    def _date_ranges_overlap_strict(self, start1, end1, start2, end2):
        d1s = self._parse_date(start1)
        d1e = self._parse_date(end1)
        d2s = self._parse_date(start2)
        d2e = self._parse_date(end2)
        if None in [d1s, d1e, d2s, d2e]:
            return False
        return max(d1s, d2s) < min(d1e, d2e)

    def _has_nonmissing_info(self, row, var_names):
        provided = []
        for var in var_names:
            if hasattr(row, var) and not self._is_missing_val(getattr(row, var)):
                provided.append(var)
        return provided
