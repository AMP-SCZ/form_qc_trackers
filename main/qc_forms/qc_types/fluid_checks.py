import pandas as pd

import os
import sys
import json
import re
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-3])
sys.path.insert(1, parent_dir)

from utils.utils import Utils
from qc_forms.form_check import FormCheck
from datetime import datetime

class FluidChecks(FormCheck):
    """
    QC Checks related to
    fluid biomarker forms
    """

    # Class-level registries persist across per-row instances so that
    # duplicate blood IDs / barcodes can be detected across subjects.
    # Maps value -> list of (subject, var, timepoint).
    _seen_blood_id_vals = {}
    _seen_blood_barcode_vals = {}

    def __init__(self, row, timepoint, network, form_check_info):
        super().__init__(timepoint, network, form_check_info)
        self.test_val = 0
        self.call_checks(row)
               
    def __call__(self):
        return self.final_output_list

    def call_checks(self, row):
        self.call_blood_checks(row)
        self.chrsaliva_id_check(row)
        self.call_blood_pressure_check(row)

    def call_blood_pressure_check(self, row):
        report_list = ["Fluids Report", "Main Report"]

        self.blood_pressure_check(
            row,
            ["current_health_status"],
            ["chrchs_systolic", "chrchs_diastolic"],
            {"reports": report_list},
        )

    # def call_chrsaliva_id_check(self, row):

    #     report_list = ["Fluids Report"]

    #     self.chrsaliva_id_check(
    #         row,
    #         ["daily_activity_and_saliva_sample_collection"],
    #         ["chrsaliva_id1a", "chrsaliva_id1b", "chrsaliva_id2a", "chrsaliva_id2b", "chrsaliva_id3a", "chrsaliva_id3b"],
    #         {"reports": report_list},
    #     )

        
    def call_blood_checks(self,row):
        form = 'blood_sample_preanalytic_quality_assurance'
        blood_reports =  ['Main Report','Blood Report','Fluids Report']
        all_vol_vars = self.grouped_vars['blood_vars']['volume_variables']
        for vol_var in all_vol_vars:
            self.blood_vol_check(row,[form],[vol_var],{"reports":blood_reports})
        proc_time_vars = ['chrblood_wholeblood_freeze','chrblood_serum_freeze',
        'chrblood_plasma_freeze','chrblood_buffy_freeze']
        for proc_time_var in proc_time_vars:
            self.check_blood_freeze(row, [form], [proc_time_var],{"reports":blood_reports})
        self.cbc_differential_check(row)
        self.check_blood_date(row,[form],
        ['chrblood_drawdate','chrblood_labdate'], {"reports":blood_reports})
        self.barcode_format_check(row)
        # PRESCIENT disabled per operator request 2026-04-30. PRONET
        # rows still run the cross-subject duplicate detector. To
        # re-enable PRESCIENT, remove the network guard.
        if self.network != 'PRESCIENT':
            self.cross_subject_blood_duplicate_check(row)

    def cbc_differential_check(self,row):
        """Checks for additional errors in
        CBC with differntial form"""
        forms = ['cbc_with_differential']
        blood_reports =  ['Main Report','Blood Report','Fluids Report']

        self.white_bc_check(row,forms,['chrcbc_wbcinrange',
        'chrcbc_wbcsum','chrcbc_wbc'],{"reports":blood_reports})
        self.cbc_compl_check(row,['cbc_with_differential'],
        ['cbc_with_differential_complete','chrblood_cbc'],
        {"reports":blood_reports,"affected_forms":
        ['cbc_with_differential','blood_sample_preanalytic_quality_assurance']})

        if self.network != 'PRESCIENT':
            self.cbc_unit_checks(row)

    def cbc_unit_checks(self, row):
        forms = ['cbc_with_differential']
        reports =  ['Main Report','Blood Report','Fluids Report']
        gt_unit_ranges = {'chrcbc_wbc':30,'chrcbc_neut':20,'chrcbc_rbc':12,\
        'chrcbc_lymph':20,'chrcbc_eos':5,'chrcbc_monos':5,'chrcbc_baso':5}
        lt_unit_ranges = {'chrcbc_monos':0.1,'chrcbc_hct':1}

        for var, value_range in gt_unit_ranges.items():
            self.cbc_range_check(row,forms, [var],
            {'reports':reports},[],True,gt = True, threshold=value_range)
        for var, value_range in lt_unit_ranges.items():
            self.cbc_range_check(row,forms, [var],
             {'reports':reports},[],True, gt = False, threshold=value_range)
    
    @FormCheck.standard_qc_check_filter 
    def check_blood_date(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):
        """
        Function to check if the blood
        draw date is later than the date
        sent to lab.
        """

        drawdate = row.chrblood_drawdate
        labdate = row.chrblood_labdate
        if any((date in (self.missing_code_list +['']) or
        not self.utils.check_if_val_date_format(date, "%Y-%m-%d %H:%M")) for\
        date in [drawdate,labdate]):
            return
        if (datetime.strptime(drawdate,"%Y-%m-%d %H:%M") > 
        datetime.strptime(labdate,"%Y-%m-%d %H:%M")):
            return  f"Blood draw date ({drawdate}) is later than date sent to lab ({labdate})."

    @FormCheck.standard_qc_check_filter    
    def cbc_range_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True, gt = True, threshold = 0
    ):
        cbc_var = all_vars[0]
        cbc_val = getattr(row, cbc_var)
        if (self.utils.can_be_float(cbc_val)
        and cbc_val not in self.utils.missing_code_set):            
            if ((gt == True and float(cbc_val) > threshold) or 
            (gt == False and float(cbc_val) < threshold)):
                return f'Incorrect units used ({cbc_var} = {cbc_val})'
        
    @FormCheck.standard_qc_check_filter    
    def blood_vol_check(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):  
        vol_var = all_vars[0]
        if hasattr(row ,vol_var):
            vol = getattr(row,vol_var)
            if vol not in (self.missing_code_list+['']) and\
            self.utils.can_be_float(vol):
                if float(vol) <= 0:
                    return  f'Volume ({vol_var} = {vol}) is less than or equal to 0'
                elif float(vol) > 1.1:
                    return  f'Volume ({vol_var} = {vol}) is greater than 1.1'
            
    @FormCheck.standard_qc_check_filter 
    def check_blood_freeze(self, row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ):    
        proc_time_var = all_vars[0]
        proc_time = getattr(row, proc_time_var)
        if (proc_time not in (self.missing_code_list+[''])
        and self.utils.can_be_float(proc_time) and float(proc_time) > 480):
            return  f'Processing time ({proc_time_var} = {proc_time}) is greater than 480'

    @FormCheck.standard_qc_check_filter 
    def white_bc_check(self,row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ): 
        if not any(x in (self.missing_code_list+[''])
        for x in [row.chrcbc_wbcsum,row.chrcbc_wbc]):
            if (self.utils.can_be_float(row.chrcbc_wbcsum)
            and self.utils.can_be_float(row.chrcbc_wbc)):
                wbc_sum = float(row.chrcbc_wbcsum)
                wbc = float(row.chrcbc_wbc)
                if (wbc_sum < (wbc-(0.1*wbc))) or wbc_sum > (wbc+(0.1*wbc)):
                    return (f'Sum of absolute neutrophils, lymphocytes,'
                        f' monocytes, basophils, and eosinophils'
                        f' ({row.chrcbc_wbcsum}) is not within 10% of '
                        f' WBC count ({row.chrcbc_wbc}).')

    @FormCheck.standard_qc_check_filter
    def cbc_compl_check(self,row, filtered_forms,
        all_vars, changed_output_vals, bl_filtered_vars=[],
        filter_excl_vars=True
    ): 
        if getattr(row, 'cbc_with_differential_complete') not in self.utils.all_dtype([2]):
            if (row.chrblood_cbc in self.utils.all_dtype([1]) and
            row.chrblood_interview_date not in (self.missing_code_list+[''])):
                time_since_blood = self.utils.find_days_between(str(row.chrblood_interview_date),
                str(datetime.today()))
                if time_since_blood > 5:
                    return ('Blood form indicates EDTA tube was sent to lab for CBC'
                    f', but CBC form has not been completed.')         
        elif getattr(row, 'cbc_with_differential_complete') not in self.utils.all_dtype([2]):
            if self.check_if_missing(row,'cbc_with_differential') != True:
                if row.chrblood_cbc not in self.utils.all_dtype([1]):
                    return ('Blood form indicates EDTA tube was not sent to lab for CBC'
                    f', but CBC form has been completed and not marked as missing.')
                
    def barcode_format_check(self, row):
        """
        Makes sure blood barcodes are in
        proper format
        """

        forms = ['blood_sample_preanalytic_quality_assurance']
        reports = ['Main Report', 'Blood Report', 'Fluids Report']
        barcode_vars = self.grouped_vars['blood_vars']['barcode_variables']
        output_changes = {'reports' : reports}

        for barcode_var in barcode_vars:
            if (hasattr(row, barcode_var) 
            and getattr(row, barcode_var) not in
            (self.utils.missing_code_list + ['','NA'])):
                barcode_val = str(getattr(row,barcode_var))
                # blood team said to ignore these 
                if 'pronet' in barcode_val.lower():
                    continue    
                if len(barcode_val) != 10:
                    error_message = f"Barcode ({barcode_val}) length is not 10 characters."
                    error_output = self.create_row_output(
                    row,forms,[barcode_var], error_message, output_changes)
                    self.final_output_list.append(error_output)
                if any(not char.isdigit() for char in barcode_val):
                    error_message = f"Barcode ({barcode_val}) contains non-numeric characters."
                    error_output = self.create_row_output(
                    row,forms,[barcode_var], error_message, output_changes)
                    self.final_output_list.append(error_output)

    # Storage-location identifiers — these are SHARED across many subjects
    # by design (a single freezer / rack / box holds many subjects' samples),
    # so they will always produce false-positive cross-subject "duplicates"
    # if checked. Confirmed directly from grouped_variables.json: this list
    # is what was causing the check to flag essentially every subject.
    _SHARED_LOCATION_VARS = frozenset({
        'chrblood_freezerid',
        'chrblood_rack_barcode',
        'chrblood_bc1box',
    })

    # Hot-path constants for cross_subject_blood_duplicate_check. Both were
    # being rebuilt from scratch on every per-row call (~280K calls × 2
    # networks). Cache both at class level. `_PER_SUBJECT_VARS_CACHE` is
    # populated lazily on first call; the key is `id(grouped_vars)` so a
    # different grouped_vars dict (e.g., test harness) recomputes correctly.
    _EXTRA_SKIP_STRINGS = frozenset({'', 'na', 'n/a', 'nan', 'nat', 'none', 'null'})
    _PER_SUBJECT_VARS_CACHE = None
    _PER_SUBJECT_VARS_KEY = None

    def cross_subject_blood_duplicate_check(self, row):
        """
        Flags per-vial blood ID / barcode values that appear on more than
        one subject. Uses class-level registries (`_seen_blood_id_vals`,
        `_seen_blood_barcode_vals`) that accumulate across per-row
        FluidChecks instances so a value on the current row can be compared
        against values already seen on prior subjects. Storage-location IDs
        (freezer / rack / box) are excluded — those are intentionally shared
        across many subjects' samples and would all collide.
        """
        forms = ['blood_sample_preanalytic_quality_assurance']
        output_changes = {"reports":
            ['Main Report', 'Blood Report', 'Fluids Report']}

        # `id_variables` and `barcode_variables` overlap heavily (most
        # per-vial vars appear in BOTH). Without dedup, every cross-subject
        # collision produces TWO error rows (one labeled "blood ID", one
        # "blood barcode") with different error_message text — which then
        # become two separate merge keys in calculate_resolved_errors, so
        # reviewer comments split across them. Take the union and use a
        # single label. Cached per process — see class-level
        # `_PER_SUBJECT_VARS_CACHE` comment for cache invalidation rules.
        cls = FluidChecks
        if cls._PER_SUBJECT_VARS_KEY != id(self.grouped_vars):
            blood_var_groups = self.grouped_vars['blood_vars']
            id_vars = blood_var_groups.get('id_variables', [])
            barcode_vars = blood_var_groups.get('barcode_variables', [])
            cls._PER_SUBJECT_VARS_CACHE = tuple(sorted(
                (set(id_vars) | set(barcode_vars)) - cls._SHARED_LOCATION_VARS
            ))
            cls._PER_SUBJECT_VARS_KEY = id(self.grouped_vars)
        per_subject_vars = cls._PER_SUBJECT_VARS_CACHE

        # Single registry for the unioned set (was two; the split was
        # producing the double-flag issue described above). The legacy
        # `_seen_blood_barcode_vals` class attribute is left in place but
        # unused so any external introspection code doesn't break.
        registry = cls._seen_blood_id_vals
        label = 'blood ID/barcode'

        # Class-level frozenset (was rebuilt on every call).
        extra_skip_strings = cls._EXTRA_SKIP_STRINGS

        for var in per_subject_vars:
            if not hasattr(row, var):
                continue
            val = getattr(row, var)
            if val in self.utils.missing_code_set:
                continue
            # Normalize to a canonical key so 12345 / "12345" / "12345.0"
            # / " 12345 " all hash to the same registry slot. Without
            # this, the same blood ID exported as int in one row and as
            # string in another would not be detected as a duplicate.
            val_str = str(val).strip()
            if val_str.lower() in extra_skip_strings:
                continue
            if self.utils.can_be_float(val_str):
                # Integer-shaped IDs: collapse "12345.0" -> "12345".
                try:
                    as_float = float(val_str)
                    if as_float.is_integer():
                        val_str = str(int(as_float))
                except (ValueError, OverflowError):
                    pass
            # Blood team treats PRONET-prefixed values as non-IDs (see
            # barcode_format_check).
            if 'pronet' in val_str.lower():
                continue

            prior_entries = registry.get(val_str, [])
            conflicts = [(s, v, tp) for (s, v, tp) in prior_entries
                         if s != row.subjectid]
            if conflicts:
                # Sort for determinism — registry insertion order
                # depends on row processing order, which depends on
                # network ordering and CSV row order. Without sorting,
                # the same conflict produces a different error_message
                # across runs, breaking the merge-key match in
                # calculate_resolved_errors and losing reviewer comments.
                conflicts_sorted = sorted(set(conflicts))
                conflict_str = ', '.join(
                    f"{s} ({v} at {tp})" for (s, v, tp) in conflicts_sorted)
                error_message = (
                    f"Duplicate {label} value ({var} = {val_str})"
                    f" also found on other subject(s): {conflict_str}.")
                error_output = self.create_row_output(
                    row, forms, [var], error_message, output_changes)
                self.final_output_list.append(error_output)

            # ALWAYS register (outside the conflicts branch), so the first
            # occurrence of any value is recorded for subsequent rows to
            # match against. Previously this was nested inside `if conflicts`
            # after a refactor, which silently broke the entire check by
            # never recording first occurrences.
            registry.setdefault(val_str, []).append(
                (row.subjectid, var, self.timepoint))

    def height_weight_unit_checks(self):
        height_val = getattr(row, 'chrchs_height') 
        height_units = getattr(row,'chrchs_height_units')
        if all(self.utils.can_be_float(var_val) and var_val not
        in self.utils.missing_code_set for
        var_val in [height_val, height_units]):
            if float(height_val) < 10 and float(height_units) in [1,2]:
                print('error')

    def height_bmi_check(self):
        bmi_var = 'chrchs_bmi' 


    @FormCheck.standard_qc_check_filter
    def blood_pressure_check(
        self, row, filtered_forms, all_vars,changed_output_vals,
        bl_filtered_vars=[], filter_excl_vars=True, conditions={}
    ):

        if all(
            hasattr(row, var)
            for var in ["chrchs_systolic", "chrchs_diastolic"]
        ):

            systolic = getattr(row, "chrchs_systolic")
            diastolic = getattr(row, "chrchs_diastolic")

            if(
                systolic in self.utils.missing_code_list
                or diastolic in self.utils.missing_code_list
            ):
                return

            if (
                self.utils.can_be_float(systolic)
                and self.utils.can_be_float(diastolic)
            ):

                systolic = float(systolic)
                diastolic = float(diastolic)

                if systolic <= diastolic:
                    return (
                        f"Systolic blood pressure ({systolic}) should be greater "
                        f"than diastolic blood pressure ({diastolic})."
                    )
        

    def chrsaliva_id_check(self, row):
        forms = ["daily_activity_and_saliva_sample_collection"]
        reports = ["Fluids Report", "Main Report"]

        id_vars = [
            "chrsaliva_id1a",
            "chrsaliva_id1b",
            "chrsaliva_id2a",
            "chrsaliva_id2b",
            "chrsaliva_id3a",
            "chrsaliva_id3b",
        ]

        output_changes = {"reports": reports}
        pattern = re.compile(r"^[A-Za-z0-9]{10,13}$")

        for var in id_vars:
            if not hasattr(row, var):
                continue

            value = getattr(row, var)

            if (
                value in self.utils.missing_code_list
                or str(value).strip().upper() == "N/A"
                or str(value).strip() == ""
                ):
                continue

            value = str(value).strip()

            if not pattern.fullmatch(value):
                error_message = (
                    f"{var} has an invalid value ({value}). "
                    f"It must contain 10-13 alphanumeric characters."
                    f"Special characters and spaces are not allowed."
                )

                error_output = self.create_row_output(
                    row, forms,[var], error_message, output_changes,
                )

                self.final_output_list.append(error_output)

                