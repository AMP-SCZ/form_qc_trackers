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
from qc_forms.qc_types.date_checks import DateChecks
from qc_forms.qc_types.cross_checks import CrossChecks
from qc_forms.qc_types.proposed_checks import ProposedChecks

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
        self.cross_check_metrics = []

        self.combined_flags_path = f'{self.output_path}combined_outputs/'

        self.form_check_info = {'cognition_csvs':{}}
        # Pass the configuration snapshot loaded for THIS QC run into every
        # FormCheck. FormCheck previously maintained a separate process cache,
        # so changing recruited_only between two runs in one scheduler/notebook
        # process could leave checker routing on the prior value.
        self.form_check_info['config_info'] = self.config_info

        for filename in ['subject_info', 'general_check_vars',
        'important_form_vars', 'forms_per_timepoint',
        'converted_branching_logic', 'excluded_branching_logic_vars',
        'team_report_forms', 'grouped_variables', 'variables_added_later',
        'raw_csv_conversions', 'variable_ranges', 'earliest_latest_dates_per_tp',
        'missingness_domain_forms']:
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
        # Reset FluidChecks class-level registries at the start of a
        # full QC run. The cross-subject blood-duplicate detector
        # accumulates `(value -> [(subject, var, tp), ...])` entries
        # in a class attribute so that PRONET's network/tp loop can
        # detect duplicates across rows. A single `python run_qc.py`
        # invocation does not need this reset (the class attribute is
        # initialized to {} at import time), but a long-running
        # scheduler / notebook that calls `QCFormsMain().run_script()`
        # twice in the same process would carry run N's registry into
        # run N+1, producing spurious conflicts the moment any value
        # that legitimately moved between subjects (data correction)
        # appears. .clear() rather than reassigning to a new dict so
        # the `[DEBUG_PERF]` introspection at the end of this method
        # still references the live registry.
        FluidChecks._seen_blood_id_vals.clear()
        # `_PER_SUBJECT_VARS_CACHE` is keyed by `id(grouped_vars)`. In
        # the same-process re-run case the new run typically reuses the
        # same `form_check_info` dict (same id), so the cache key would
        # match and the prior cache would be reused — harmless today,
        # but invalidating it defensively keeps the contract simple.
        FluidChecks._PER_SUBJECT_VARS_CACHE = None
        FluidChecks._PER_SUBJECT_VARS_KEY = None
        # ProposedChecks keeps only run-scoped longitudinal/configuration
        # registries.  Clear them once for a complete pipeline run, never once
        # per network/timepoint, so legitimate cross-timepoint comparisons can
        # accumulate while repeated scheduler/notebook invocations remain
        # isolated from one another.
        ProposedChecks.reset_run_state()
        final_output = []
        # Date findings are derived from every interview-date field on a row.
        # Exact duplicate source rows must not multiply identical report rows,
        # while conflicting duplicate rows with different dates remain visible.
        date_output_seen = set()
        date_output_counts = {
            network: {'generated': 0, 'routed': 0}
            for network in self.utils.pipeline_networks
        }
        missingness_output_seen = set()
        # DEBUG_PERF: when the env var is set, record per-(network, tp)
        # wall time and row counts plus a final blood-duplicate-registry
        # summary. Zero overhead when unset; no effect on QC output.
        debug_perf = bool(os.environ.get('DEBUG_PERF'))
        tp_durations = []
        tp_list = self.utils.create_timepoint_list()
        tp_list.extend(['floating','conversion'])
        # Loop-coverage guard: every (network, timepoint) pair must complete
        # the per-row loop below. A leftover test restriction in a deployed
        # copy of this loop (`if tp not in ['screen','screening','floating']:
        # continue`, committed 2026-08-07) silently limited production QC to
        # screening+floating for weeks — reconciliation then marked every
        # other timepoint's open flags resolved. Refuse to hand off a parquet
        # produced by a partial sweep.
        expected_tp_pairs = {
            (network, tp)
            for network in self.utils.pipeline_networks
            for tp in tp_list
        }
        processed_tp_pairs = set()
        for network in self.utils.pipeline_networks:
            print('------------')
            print(network)
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
                        # Inspect each column as a whole before inferring its
                        # dtype. Combined REDCap columns legitimately mix
                        # numeric codes and strings; chunk-wise inference emits
                        # a noisy DtypeWarning and can choose inconsistently.
                        low_memory=False,
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
                # Every expected combined export contains one row per REDCap
                # record even when no forms were completed at this timepoint.
                # A header-only file therefore indicates an interrupted or
                # truncated upstream merge. Continuing would emit no new
                # findings for this network/timepoint and falsely resolve its
                # entire prior QC history during reconciliation.
                if combined_df.empty:
                    raise RuntimeError(
                        f"FATAL: combined CSV contains zero participant rows: "
                        f"{csv_path}. Refusing to continue because an empty "
                        "expected export would make prior flags appear "
                        "resolved. Regenerate the combined CSV and re-run.")
                # CrossChecks sees the complete dataframe before the legacy
                # per-row guard below. Unknown subjects must fail loudly here;
                # silently dropping them would make prior flags look resolved.
                # Cross-form rules are vectorized and must run once per
                # network/timepoint dataframe. Running them inside the row loop
                # would evaluate the same catalog hundreds of times per file.
                cross_checks = CrossChecks(
                    combined_df, tp, network, self.form_check_info)
                final_output.extend(cross_checks())
                self.cross_check_metrics.append({
                    "network": network,
                    "timepoint": tp,
                    # ``rule_metrics`` was added to CrossChecks after the
                    # checker itself entered production.  Some deployments
                    # still use the earlier implementation, which returns QC
                    # rows correctly but exposes no metrics attribute.  Metrics
                    # are diagnostic only, so treat an absent attribute as an
                    # empty snapshot rather than aborting the QC run.
                    "rules": getattr(cross_checks, "rule_metrics", {}),
                })
                # Dictionary-derived additions are dataframe-wide: running the
                # engine here avoids reparsing the complete dictionary and
                # reevaluating schema/cross-row rules once per participant.
                #proposed_checks = ProposedChecks(
                #    combined_df, tp, network, self.form_check_info)
                #final_output.extend(proposed_checks())
                # Snapshot taken after the dataframe-wide engines above so the
                # delta below counts only per-row checker output for this
                # (network, timepoint).
                row_flags_before = len(final_output)
                rows_iterated = 0
                for row in combined_df.itertuples():
                    rows_iterated += 1
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
                    date_checks = DateChecks(row,
                    tp, network, self.form_check_info)
                    for general_output in gen_checks():
                        if 'Missingness Report' in str(
                                general_output.get('reports', '')):
                            missingness_key = (
                                general_output.get('network', network),
                                general_output.get('subject', row.subjectid),
                                general_output.get('displayed_timepoint', tp),
                                general_output.get('check_id', ''),
                                general_output.get('displayed_form', ''),
                                general_output.get('error_message', ''),
                            )
                            if missingness_key in missingness_output_seen:
                                continue
                            missingness_output_seen.add(missingness_key)
                        final_output.append(general_output)
                    final_output.extend(fluid_checks())
                    final_output.extend(clinical_checks())
                    final_output.extend(sop_checks())
                    final_output.extend(cognition_checks())
                    for date_output in date_checks():
                        date_network = date_output.get('network', network)
                        date_key = (
                            date_network,
                            date_output.get('subject', row.subjectid),
                            date_output.get('displayed_timepoint', tp),
                            date_output.get('displayed_variable', ''),
                            date_output.get('error_message', ''),
                        )
                        if date_key in date_output_seen:
                            continue
                        date_output_seen.add(date_key)
                        date_output_counts.setdefault(
                            date_network, {'generated': 0, 'routed': 0})
                        date_output_counts[date_network]['generated'] += 1
                        if 'Date Report' in str(
                                date_output.get('reports', '')):
                            date_output_counts[date_network]['routed'] += 1
                        final_output.append(date_output)
                # A partially iterated dataframe (e.g. a resurrected
                # `.sample(n=...)` / `.iloc[...]` test slice) is the same
                # data-loss cascade as a malformed CSV: dropped subjects'
                # open flags would be marked resolved on the next merge.
                if rows_iterated != len(combined_df):
                    print(
                        f"[qc_forms_main] FATAL: iterated {rows_iterated} of "
                        f"{len(combined_df)} rows for {network}/{tp}. A "
                        f"partial sweep would falsely resolve open flags for "
                        f"the skipped subjects. Refusing to continue.")
                    sys.exit(1)
                if len(final_output) == row_flags_before:
                    # Not fatal: a sparse timepoint can legitimately be quiet,
                    # but zero per-row flags from a nonempty CSV is the exact
                    # signature of every checker being gated off — surface it.
                    print(
                        f"[qc_forms_main] ERROR: STRUCTURAL ZERO — "
                        f"{network}/{tp}: {len(combined_df)} rows produced "
                        f"zero per-row QC flags. Check completion/cohort "
                        f"gates and stage-1 dependencies.")
                processed_tp_pairs.add((network, tp))
                if debug_perf:
                    tp_durations.append((
                        network, tp,
                        time.perf_counter() - tp_start,
                        int(len(combined_df)),
                    ))

        skipped_tp_pairs = expected_tp_pairs - processed_tp_pairs
        if skipped_tp_pairs:
            print(
                f"[qc_forms_main] FATAL: {len(skipped_tp_pairs)} "
                f"(network, timepoint) pair(s) never completed the per-row "
                f"QC loop: {sorted(skipped_tp_pairs)}. A partial sweep would "
                f"mark every skipped timepoint's open flags resolved during "
                f"reconciliation. Refusing to write combined_qc_flags.")
            sys.exit(1)

        print(
            "[qc_forms_main] Date Report rows "
            "(unique generated / routed to tracker): "
            + ", ".join(
                f"{network}={counts['generated']}/{counts['routed']}"
                for network, counts in date_output_counts.items())
        )

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
