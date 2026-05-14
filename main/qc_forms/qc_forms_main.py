import pandas as pd
import os
import sys
import json
import shutil
import traceback
import time
import random
parent_dir = "/".join(os.path.realpath(__file__).split("/")[0:-2])
sys.path.insert(1, parent_dir)
from utils.utils import Utils
from qc_forms.qc_types.general_checks import GeneralChecks
from qc_forms.qc_types.fluid_checks import FluidChecks
from qc_forms.qc_types.clinical_checks.clinical_checks_main import ClinicalChecksMain
from qc_forms.qc_types.cognition_checks import CognitionChecks
from qc_forms.qc_types.SOP_checks import SOPChecks
from qc_forms.qc_types.multi_tp_checks import MultiTPChecks

class QCFormsMain():
    def __init__(self):
        self.utils = Utils()
        self.absolute_path = self.utils.absolute_path
        with open(f'{self.absolute_path}/config.json','r') as file:
            self.config_info = json.load(file)
        self.comb_csv_path = self.config_info['paths']['combined_csv_path']
        self.depen_path = self.config_info['paths']['dependencies_path']
        with open(f'{self.depen_path}converted_branching_logic.json','r') as file:
            self.conv_bl = json.load(file)

        self.output_path = self.config_info['paths']['output_path']
        if self.config_info['testing_enabled'] == "True":
            self.output_path += "testing/"

        self.final_output_list = []

        self.combined_flags_path = f'{self.output_path}combined_outputs/'

        self.form_check_info = {'cognition_csvs':{}}

        for filename in ['subject_info', 'general_check_vars',
        'important_form_vars', 'forms_per_timepoint',
        'converted_branching_logic', 'excluded_branching_logic_vars',
        'team_report_forms', 'grouped_variables', 'variables_added_later',
        'raw_csv_conversions', 'variable_ranges', 'earliest_latest_dates_per_tp']:
            self.form_check_info[filename] = self.utils.load_dependency_json(f"{filename}.json")

        for iq_type in ['wais','wasi']:
            for conv_type in ['iq_raw','fsiq']:
                self.form_check_info['cognition_csvs'][
                f'{conv_type}_conversion_{iq_type}'] = pd.read_csv(
                f'{self.depen_path}cognition/{conv_type}_conversion_{iq_type}.csv',
                keep_default_na = False) 

        self.scid_subs = []
        self.scid_subs_df = []

    def run_script(self):
        self.move_previous_output()
        self.iterate_combined_dfs()

    def move_previous_output(self):
        for path in [f"{self.combined_flags_path}new_output",
        f"{self.combined_flags_path}old_output",self.combined_flags_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        new_path =  f'{self.combined_flags_path}new_output/combined_qc_flags.parquet'
        old_path = new_path.replace('new_output','old_output')
        # Direct file copy: avoids a pandas read+write of a potentially huge
        # CSV, and preserves the file byte-for-byte (no quoting/dtype drift).
        if os.path.exists(new_path) and os.path.getsize(new_path) > 0:
            shutil.copyfile(new_path, old_path)

    def iterate_combined_dfs(self):
        # TODO: split checks by ones that will only be checked 
        # if a form in compl and no
        # t missing and ones 
        # that will be checked regardless
        final_output = []
        # DEBUG_PERF: when the env var is set, record per-(network, tp)
        # wall time and row counts plus a final blood-duplicate-registry
        # summary. Zero overhead when unset; no effect on QC output.
        debug_perf = bool(os.environ.get('DEBUG_PERF'))
        tp_durations = []
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        for network in ['PRESCIENT', 'PRONET']:
            multi_tp_path = f"{self.depen_path}multi_tp_{network}_combined.csv"
            """
            multi_tp_df = pd.read_csv(multi_tp_path,
            keep_default_na = False)
            for row in multi_tp_df.itertuples():
                multi_tp_vars = multi_tp_df.columns
                multi_tp_checks = MultiTPChecks(row,
                'multiple_timepoints', network, 
                self.form_check_info,multi_tp_vars)
                final_output.extend(multi_tp_checks())
            """
            for tp in tp_list:
                tp_start = time.perf_counter() if debug_perf else None
                print(tp)
                print(tp_list)
                csv_path = (f'{self.comb_csv_path}AMPSCZ-combined-redcap_'
                f'{tp.replace("month","month_").replace("floating","floating_forms")}'
                f'_{network.replace("PRONET","ProNET")}-day1to1.csv')
                # Fail loudly on a malformed CSV instead of silently skipping
                # rows. Skipping rows here causes a CRITICAL data-loss cascade:
                # the dropped subject vanishes from `new_output`, then
                # calculate_resolved_errors marks every previously-open flag
                # for that subject as "resolved" with today's date — destroying
                # QC history. Better to halt and let an operator fix the CSV.
                try:
                    combined_df = pd.read_csv(
                        csv_path,
                        keep_default_na=False,
                        encoding='utf-8',
                        on_bad_lines='error',
                    )
                except FileNotFoundError:
                    print(f"[qc_forms_main] FATAL: missing combined CSV: {csv_path}")
                    raise
                except (pd.errors.ParserError, UnicodeDecodeError) as e:
                    print(
                        f"[qc_forms_main] FATAL: cannot parse {csv_path}: {e}. "
                        f"Refusing to continue — silently skipping rows here "
                        f"would falsely mark every previously-open flag for "
                        f"the affected subject(s) as resolved on the next run. "
                        f"Fix the CSV and re-run."
                    )
                    raise
                # PARITY ASSERTION (TEMPORARY — remove after one full run with
                # no failures; see removal procedure in project notes).
                # Validates that the proposed vectorized prefilter
                # `combined_df['subjectid'].isin(subject_info)` matches the
                # original in-loop guard exactly on real data, BEFORE we
                # rely on it. The two failure directions are NOT symmetric:
                #
                #   over_filter  = guard would KEEP, isin() DROPS
                #     -> rows never reach the checkers; QC FLAGS LOST.
                #     -> DANGEROUS — this is the failure mode we are
                #        guarding against.
                #
                #   under_filter = guard would DROP, isin() KEEPS
                #     -> extra rows enter the loop, but the in-loop guard
                #        below still drops them. No flags lost; just a
                #        small amount of wasted work.
                #     -> Unexpected but not dangerous.
                #
                # The in-loop `if row.subjectid not in subject_info: continue`
                # guard a few lines below is intentionally left in place as
                # defense-in-depth and is NOT modified by this patch.
                if 'subjectid' in combined_df.columns:
                    subj_info = self.form_check_info['subject_info']
                    _guard_kept = combined_df['subjectid'].apply(
                        lambda s: s in subj_info)
                    _isin_kept = combined_df['subjectid'].isin(subj_info)
                    if not _guard_kept.equals(_isin_kept):
                        over_filter_mask = _guard_kept & ~_isin_kept
                        under_filter_mask = ~_guard_kept & _isin_kept
                        over_ids = (
                            combined_df.loc[over_filter_mask, 'subjectid']
                            .drop_duplicates().head(10).tolist())
                        under_ids = (
                            combined_df.loc[under_filter_mask, 'subjectid']
                            .drop_duplicates().head(10).tolist())
                        raise AssertionError(
                            f"[qc_forms_main] PARITY FAILURE for "
                            f"{network}/{tp}: vectorized prefilter .isin() "
                            f"disagrees with in-loop guard. "
                            f"over_filter={int(over_filter_mask.sum())} "
                            f"(LOSES QC FLAGS — guard would KEEP these, "
                            f"isin() would DROP them; sample: {over_ids}). "
                            f"under_filter={int(under_filter_mask.sum())} "
                            f"(harmless — isin() keeps these but the "
                            f"in-loop guard drops them; sample: {under_ids}). "
                            f"Refusing to apply vectorized prefilter."
                        )
                    # Parity verified for this (network, tp). Apply
                    # prefilter. No reset_index — preserves original row
                    # order for the itertuples() loop below.
                    combined_df = combined_df[_isin_kept]
                #combined_df = combined_df.iloc[80:120]
                #combined_df = combined_df.sample(n=20)
                #combined_df = combined_df.sample(n=100, random_state=42)
                for row in combined_df.itertuples():
                    #print(row.Index)
                    #TODO: Add tracker for all subjects not existing here 
                    if (row.subjectid not
                    in self.form_check_info['subject_info']):
                        continue
                    #print(row.Index)
                    gen_checks = GeneralChecks(row, tp,
                    network, self.form_check_info)
                    fluid_checks = FluidChecks(row, tp,
                    network, self.form_check_info)
                    clinical_checks = ClinicalChecksMain(row,
                    tp, network, self.form_check_info)
                    cognition_checks = CognitionChecks(row,
                    tp, network, self.form_check_info)
                    sop_checks = SOPChecks(row,
                    tp, network, self.form_check_info)
                    final_output.extend(gen_checks())
                    final_output.extend(fluid_checks())
                    final_output.extend(clinical_checks())
                    final_output.extend(sop_checks())
                    final_output.extend(cognition_checks())
                if debug_perf:
                    tp_durations.append((
                        network, tp,
                        time.perf_counter() - tp_start,
                        int(len(combined_df)),
                    ))

        if debug_perf:
            print("[DEBUG_PERF] per-timepoint timings (network, tp, seconds, rows):")
            for entry in tp_durations:
                print(f"  {entry}")
            try:
                reg = FluidChecks._seen_blood_id_vals
                top = sorted(
                    ((len(v), k) for k, v in reg.items()),
                    reverse=True)[:5]
                print(
                    f"[DEBUG_PERF] blood-dup registry: "
                    f"{len(reg)} unique values, "
                    f"top-5 by collision count: {top}")
            except Exception as e:
                print(f"[DEBUG_PERF] registry introspection failed: {e}")

        # Write the combined output once after all networks/timepoints have
        # been processed. Previously this lived inside the inner loop and
        # rewrote the same file (with an ever-growing dataframe) on every
        # iteration — O(N^2) writes for N timepoints * 2 networks.
        if len(final_output) == 0:
            # Zero flags across all networks/timepoints almost certainly
            # means QC checks did not actually run (input CSV missing,
            # subject_info empty so every row fails the in-loop guard,
            # all checker classes raised during __call__, etc.). If we
            # don't write here, the prior run's new_output stays on
            # disk and calculate_resolved_errors merges it as if it
            # belonged to this run — corrupting detection-date history.
            # Halt instead so the operator investigates before any
            # merge runs.
            print(
                "[qc_forms_main] FATAL: zero QC flags produced across "
                "all networks/timepoints. Refusing to skip the write — "
                "that would let calculate_resolved_errors pick up the "
                "previous run's new_output snapshot as if it belonged "
                "to this run. Investigate the input CSVs / checker "
                "config and re-run."
            )
            sys.exit(1)
        combined_output_df = pd.DataFrame(final_output)
        if combined_output_df.shape[0] > 3000000:
            print(
                f"[qc_forms_main] FATAL: output rows "
                f"{combined_output_df.shape[0]} exceeds 3000000 row "
                f"safety ceiling. Refusing to write combined_qc_flags."
                f"parquet. Exiting nonzero."
            )
            sys.exit(1)
        combined_flags_path = f'{self.output_path}combined_outputs'
        new_out_path = f'{combined_flags_path}/new_output/'
        os.makedirs(new_out_path, exist_ok=True)
        tmp_path = None
        try:
            # Parquet handoff: smaller file, preserves dtypes (avoids
            # bool/int round-tripping as strings), faster read/write.
            # Requires `pyarrow` to be installed.
            #
            # Atomic write: write to a PID-stamped .tmp first, then
            # os.replace into the canonical filename. If the process
            # is killed mid-write, the canonical .parquet is
            # unchanged. PID-stamping prevents two concurrent runs
            # from colliding on the same .tmp path (the second's
            # to_parquet would otherwise truncate the first's
            # in-flight write, or the second os.replace would publish
            # the wrong run's data into the canonical filename).
            final_path = f'{new_out_path}combined_qc_flags.parquet'
            tmp_path = f'{final_path}.{os.getpid()}.tmp'
            combined_output_df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, final_path)
        except Exception as e:
            # Best-effort cleanup of the partial .tmp so a re-run
            # starts clean. Swallow OSError here — failure to clean
            # up shouldn't mask the original exception that's about
            # to terminate the process.
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(e)
            traceback.print_exc()
            sys.exit(1)
