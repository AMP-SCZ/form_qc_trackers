# Modifying the Form QC Pipeline 

A short reference for adding checks, adding `process_variables` stages, and the gotchas that bite first-time editors.

## Pipeline

1. **`process_variables_main.py`** — builds various files needed to run the code in `dependencies/` (subject info, grouped vars, ranges, branching logic, etc.).

2. **`qc_forms_main.py`** — iterates each combined CSV (per network, per timepoint), runs every `FormCheck` subclass per row, writes the canonical output `combined_outputs/new_output/combined_qc_flags.parquet` (note: parquet, not CSV).

Stage 2 reads what stage 1 wrote. If you add a new dependency JSON, save it in stage 1 and load it in `qc_forms_main.QCFormsMain.__init__` (`form_check_info` dict, line 40 area).

## Adding a new QC check

Almost every check is a method on a `FormCheck` subclass in `qc_types/`. Pattern (mirroring `GeneralChecks` / `FluidChecks`):

1. **Pick (or create) the right class of the check.** `general_checks.py`, `fluid_checks.py`, `cognition_checks.py`, `SOP_checks.py`, `multi_tp_checks.py`, and `clinical_checks/*` already exist. For a wholly new domain, create `qc_types/your_checks.py` modeled on `qc_types/general_checks.py:13–25`:
   ```python
   class YourChecks(FormCheck):
       def __init__(self, row, timepoint, network, form_check_info):
           super().__init__(timepoint, network, form_check_info)
           self.call_checks(row)        # runs everything; populates self.final_output_list
       def __call__(self):
           return self.final_output_list
       def call_checks(self, row):
           self.your_check_method(row, [form], [var], {"reports": [...]})
   ```
2. **Wire it into `qc_forms_main.py`.** Import it at the top (around line 17), then inside the row loop (`qc_forms_main.py:206–227`) instantiate it per row: `your_checks = YourChecks(row, tp, network, self.form_check_info)` — a fresh object per `(row, tp, network)`. Then `final_output.extend(your_checks())`.
3. **Write each check method.** Decorate with `@FormCheck.standard_qc_check_filter` (`form_check.py:90`) if you would like to apply the standard filter that applies to most forms. However, do not use it if you would like to bypass some of these filters and only implement certain filters in the function itself. The decorator (verified at `form_check.py:90–141`) gates the call on, in order: cohort ∈ {hc, chr}; every form in `filtered_forms` belongs to this timepoint's `forms_per_tp[cohort]` (skipped for `timepoint == 'multiple_timepoints'`); `standard_form_filter` passes for each form (completion / missing / age / opt-in conditions); every var in `all_vars` exists on the row; no var is in `general_check_vars['excluded_vars'][network]` (unless `filter_excl_vars=False`). Only then is the underlying function invoked. After it returns, if `bl_filtered_vars` is non-empty, each branching-logic expression is run for all variables in that list. A `False` result for any of the branching logic conditions drops the flag.
4. **Inside the method**, return a string error message to raise a flag, or fall off the end / return `None` for no flag. Excluded-form filtering and report routing are NOT handled by the decorator — your `call_checks` body must loop over forms and consult `self.general_check_vars["excluded_forms"][self.network]` itself (see `general_checks.py:68–69` for the canonical pattern).
5. **Set the reports list** via the `changed_output_vals` kwarg, e.g. `{"reports": ["Main Report", "Non Team Forms"]}`. Common values: `"Main Report"`, `"Secondary Report"`, `"Non Team Forms"`, `"Incomplete Forms"`, `"Blood Report"`, `"Fluids Report"`. Reports drive which Excel tracker the flag lands in (`generate_reports/create_trackers.py`). `create_row_output` (`form_check.py:280`) writes the standard flag dict and merges `changed_output_vals` into it.

### Network-aware checks

If the check should behave differently on PRONET vs PRESCIENT (see `pronet_prescient_differences.md`), branch on `self.network` (`"PRONET"` / `"PRESCIENT"` — uppercase). For PRESCIENT completion variables append `_rpms`, strip `_hc`, and rewrite `onboarding` → `checkin`. The blood-team's "pronet" placeholder filter (uniform code, PRONET-side effect) lives in `fluid_checks.py` — match its style if you need a similarazz artifact filter.

### Cross-timepoint or cross-network data

Per-row checks that need data from another tp or another subject's row should **collect that data in `process_variables/collect_subject_info.py`** and read it from `self.subject_info[subjectid][...]` at check time. Do NOT gate the check on a specific tp to peek at other tps — that pattern has bitten the pipeline before. See `collect_subject_info.py:127` for the floating-tp harvest as a template.

## Adding a new `process_variables` stage

Pattern:

1. Create `process_variables/your_module.py` exporting a class with `__init__`, `__call__` (or `run_script`), and any helpers.
2. Read inputs via `self.utils.load_dependency_json('name.json')` — never raw `open()` on dependency files (the helper caches and raises on corrupt JSON).
3. Write output via `self.utils.save_dependency_json(your_dict, 'your_artifact.json')`.
4. Instantiate inside `ProcessVariables.run_script` in `process_variables_main.py`. **Ordering matters:** `MultiTPDataCollector` and `DuplicateFinder` must come after their upstream dependencies (`organize_reports`, `collect_subject_info`, etc.). `RangeDefiner` runs last because it consumes earlier outputs (`process_variables_main.py:74–81`).
5. If `qc_forms_main` should consume the artifact, add its filename to the `form_check_info` loader loop (`qc_forms_main.py:40`).


- **Canonical output is parquet, not CSV.** Stage handoff is `combined_qc_flags.parquet`. Boolean-as-string round-trip bugs that used to surface with CSV no longer apply.
- **Network filename casing:** in-code identifiers are `"PRONET"` / `"PRESCIENT"`, but the combined-CSV filenames embed `ProNET` for PRONET. Build paths with `network.replace("PRONET","ProNET")` — see `qc_forms_main.py:126`.
- **`config.json` is cached process-wide** (`form_check.py:34`). Re-running in the same process won't pick up edits unless you clear `_CONFIG_CACHE`. Same applies to `_BL_COMPILED_CACHE` for branching logic.
- **`FluidChecks` carries class-level state** (`_seen_blood_id_vals`) for cross-subject duplicate detection. If you call `QCFormsMain().run_script()` more than once per process (notebook / scheduler), the class state needs the `.clear()` already in `iterate_combined_dfs` — keep that reset if you copy the loop.
- **Branching logic is `eval()`'d at check time.** Variables referenced must exist as attributes on the row tuple. If your new check creates derived variables, document them in `grouped_variables.json` so they survive a stage-1 rebuild.
- **PHI safety:** never open `AMPSCZ-combined-*` CSVs, per-subject CSVs, or `Prescient_bloods_combined.xlsx` directly when investigating — reason from code or ask the operator to inspect.
- **`med_info.txt` and JSON pharm rules can disagree.** If you find a conflict between `dependencies/med_info.txt` and `ap_med_mappings.json` / `pharm_checks.py` logic, don't reconcile — flag it to the operator. The spec and the code are intentionally allowed to drift in places.
- **Editing style:** keep imports and method signatures intact, even when refactoring. Stage code is dropped back into the full pipeline as-is, so surgical edits are safer than wider cleanups. Flag judgment calls in comments instead of silently changing behavior.
