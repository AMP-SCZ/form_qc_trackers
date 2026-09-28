# AMPSCZ form QC pipeline


## Quick start

1. Configure `config.json` at the project root (`paths.*`, `testing_enabled`,
   `pipeline_networks`). Set `pipeline_networks` to `["PRONET", "PRESCIENT"]`
   for both networks, or a one-item list for a single network. This setting is
   required unless the `QC_NETWORKS` environment variable supplies a one-off
   override (for example, `QC_NETWORKS=PRESCIENT`).
2. Ensure combined day1 CSVs exist under `paths.combined_csv_path` and
   dependencies under `paths.dependencies_path`.
3. Run the pipeline:

```bash
python run_qc.py
```

The runner performs QC, generates and uploads the reports, then generates and
uploads the error graphs. Dependency preparation through `ProcessVariables`
is currently commented out in the runner, so its outputs must already exist.

## Code guide

The three main workflow folders below list each class, including internal
helpers and unfinished placeholders. File paths in their tables are relative
to the named folder. The remaining folders are summarized at folder level.

### `main/qc_forms`

Runs form checks and builds the combined QC findings. It also contains
separate discovery tools, shared rule helpers, and older exploratory checks.

#### Core and clinical checks

| Class | File | Summary |
| --- | --- | --- |
| `QCFormsMain` | `qc_forms_main.py` | Loads dependencies, applies QC across the selected networks and timepoints, and writes the combined flag output. |
| `FormCheck` | `form_check.py` | Provides shared applicability filters, branching-logic evaluation, dependency access, and standardized QC output. |
| `GeneralChecks` | `qc_types/general_checks.py` | Checks blanks, missing codes, required values, form completion, ranges, GUID formats, and missing-data consistency. |
| `DateChecks` | `qc_types/date_checks.py` | Detects interview dates that precede the same field at an earlier timepoint and sends findings to the Date Report. |
| `CognitionChecks` | `qc_types/cognition_checks.py` | Verifies eligible IQ totals and standardized scores against conversion tables using estimated assessment age. |
| `FluidChecks` | `qc_types/fluid_checks.py` | Checks blood volumes, processing times, dates, CBC consistency, barcode formats, and duplicate identifiers. |
| `SOPChecks` | `qc_types/SOP_checks.py` | Flags subjects identified as converted whose conversion form lacks an interview date. |
| `ClinicalChecksMain` | `qc_types/clinical_checks/clinical_checks_main.py` | Runs clinical score, age, timing, and response-consistency checks and combines SCID, medication, and conversion findings. |
| `ConversionChecks` | `qc_types/clinical_checks/conversion_checks.py` | Checks conversion status against supporting PSYCHS and SCID criteria across visits and at conversion. |
| `PharmChecks` | `qc_types/clinical_checks/pharm_checks.py` | Checks medication names, dates, overlapping courses, ongoing status, compliance, frequency, and antipsychotic-history consistency. |
| `ScidChecks` | `qc_types/clinical_checks/scid_checks.py` | Checks SCID diagnoses and mood-episode classifications against supporting criteria and responses. |

#### Cross Checks and Proposed Checks

Cross Checks runs during the current pipeline. The `ProposedChecks` invocation
is currently commented out in `QCFormsMain`; its supporting classes are listed
here for reference.

| Class | File | Summary |
| --- | --- | --- |
| `CrossChecks` | `qc_types/cross_checks.py` | Evaluates approved cross-form rules once per export, independently of the standard form-applicability filters. |
| `CrossCheckRule` | `qc_types/cross_checks.py` | Stores a compiled rule with its identifier, expression, operands, and alternative evidence groups. |
| `CrossCheckConfigurationError` | `qc_types/cross_checks.py` | Reports dependency configurations that cannot support Cross Checks. |
| `CrossCheckInputError` | `qc_types/cross_checks.py` | Reports combined-export inputs that Cross Checks cannot evaluate or render. |
| `ProposedChecks` | `qc_types/proposed_checks.py` | Evaluates dictionary-derived validation and curated consistency rules for the Proposed Checks report. |
| `FieldSpec` | `qc_types/proposed_checks.py` | Describes a dictionary field's choices, validation, branching logic, and calculation metadata. |
| `DictionaryCatalog` | `qc_types/proposed_checks.py` | Compiles dictionary fields, form mappings, calculation dependencies, and configuration diagnostics. |
| `_OverlayRow` | `qc_types/proposed_checks.py` | Overlays recalculated values on a source row for use by dependent calculations. |
| `ProposedCheckConfigurationError` | `qc_types/proposed_checks.py` | Reports dictionary or dependency configurations that cannot support Proposed Checks. |
| `ProposedCheckInputError` | `qc_types/proposed_checks.py` | Reports combined exports that violate the Proposed Checks input requirements. |

#### Discovery and longitudinal analysis

These configurable detectors produce separate review artifacts. The
`run_discovery` subfolder contains command-line runners for the discovery checks.

| Class | File | Summary |
| --- | --- | --- |
| `LongitudinalAnomalyChecks` | `qc_types/longitudinal_anomaly_checks.py` | Flags numeric observations that deviate substantially from the same subject's longitudinal median. |
| `CohortTrajectoryDeviationChecks` | `qc_types/discovery/cohort_trajectory_deviation_checks.py` | Flags observations that deviate from the corresponding cohort and timepoint median trajectory. |
| `CopyForwardChecks` | `qc_types/discovery/copy_forward_checks.py` | Finds consecutive visits with unusually many unchanged form values, accounting for cohort variability. |
| `DateAnomalyChecks` | `qc_types/discovery/date_anomaly_checks.py` | Finds backward interview dates and unusually short or long intervals between visits. |
| `DuplicateRecordChecks` | `qc_types/discovery/duplicate_record_checks.py` | Finds matching records across subjects and near-duplicate forms across a subject's visits. |
| `InternalConsistencyChecks` | `qc_types/discovery/internal_consistency_checks.py` | Applies configured BMI, numeric-range, sum, and ratio rules to find inconsistent measurements. |
| `IsolationForestChecks` | `qc_types/discovery/isolation_forest_checks.py` | Uses Isolation Forest models to identify unusual combinations of numeric form values. |
| `LocalOutlierChecks` | `qc_types/discovery/local_outlier_checks.py` | Uses nearest-neighbor distances to find multivariate form observations in unusually sparse regions. |
| `LongitudinalDeltaChecks` | `qc_types/discovery/longitudinal_delta_checks.py` | Flags unusually large numeric changes between adjacent visits. |
| `MahalanobisChecks` | `qc_types/discovery/mahalanobis_checks.py` | Uses covariance-adjusted distances to find unusual combinations of correlated form variables. |
| `MissingnessSummaryChecks` | `qc_types/discovery/missingness_summary_checks.py` | Finds unusually high missing-code rates for variables and sites within each network and timepoint. |
| `MultiTPConsistencyChecks` | `qc_types/discovery/multi_tp_consistency_checks.py` | Finds repeated blood identifiers across visits and inconsistent FIGS/PPS parental ages. |
| `PairwiseRelationshipChecks` | `qc_types/discovery/pairwise_relationship_checks.py` | Flags observations that depart from strong within-form linear relationships between variable pairs. |
| `PCAReconstructionChecks` | `qc_types/discovery/pca_reconstruction_checks.py` | Finds multivariate observations poorly reconstructed by the cohort's principal components. |
| `PointSpikeChecks` | `qc_types/discovery/point_spike_checks.py` | Detects isolated extreme observations surrounded by comparatively ordinary visits for the same subject. |
| `SiteCorrelationDriftChecks` | `qc_types/discovery/site_correlation_drift_checks.py` | Finds sites whose configured scale relationships differ from other sites using Fisher-transformed correlations. |
| `SiteDistributionDriftChecks` | `qc_types/discovery/site_distribution_drift_checks.py` | Compares site medians, spread, and floor/ceiling rates within matched cohort and timepoint groups. |
| `SiteTrajectorySlopeChecks` | `qc_types/discovery/site_trajectory_slope_checks.py` | Uses mixed-effects models to identify sites with unusual longitudinal slopes. |
| `SubjectSummary` | `qc_types/discovery/subject_summary.py` | Combines discovery findings into subject-level counts and severity summaries across detectors. |
| `TrajectoryShapeChecks` | `qc_types/discovery/trajectory_shape_checks.py` | Finds unusual longitudinal shapes using variability in residuals from the cohort median trajectory. |

#### Standalone, legacy, and unfinished checks

| Class | File | Summary |
| --- | --- | --- |
| `MultiTPChecks` | `qc_types/multi_tp_checks.py` | Checks cross-timepoint blood IDs and FIGS/PPS ages; its dispatch in `QCFormsMain` is currently commented out. |
| `ExcludedChecks` | `manual_checks/excluded_beyond_screening.py` | Reports excluded subjects recorded beyond screening, including their latest form dates. |
| `ClusterAnalysis` | `anomaly_detection/cluster_analysis.py` | Provides exploratory numeric-outlier, variable-pair, and rater analyses; its current entry point produces baseline numeric-outlier scores. |
| `HarmonizationQC` | `anomaly_detection/network_harmonization_qc.py` | Compares PRONET and PRESCIENT numeric medians and percentage differences by variable and timepoint. |
| `NumericalOutliers` | `anomaly_detection/detect_numerical_outliers.py` | Unimplemented placeholder whose empty class body currently prevents the module from importing. |

### `main/generate_reports`

Reconciles QC findings with earlier results and reviewer edits, then builds
the formatted tracker workbooks.

| Class | File | Summary |
| --- | --- | --- |
| `GenerateReports` | `generate_reports_main.py` | Coordinates flag reconciliation and tracker generation using the configured report column names. |
| `CalculateResolvedErrors` | `calculate_resolved_errors.py` | Tracks new, reopened, and resolved flags by comparing QC runs, and imports reviewer comments and manual resolutions from Dropbox trackers. |
| `CreateTrackers` | `create_trackers.py` | Formats V2 Excel trackers for selected networks, sites, and Melbourne research assistants, clears stale report data, and uploads the workbooks to Dropbox. |

### `main/process_variables`

Prepares the data-dictionary mappings, subject metadata, and other dependencies
used by QC checks and reporting.

#### Preprocessing workflow

| Class | File | Summary |
| --- | --- | --- |
| `ProcessVariables` | `process_variables_main.py` | Coordinates preprocessing of the REDCap dictionary and study CSVs into QC and reporting dependencies. |
| `DefineEssentialFormVars` | `define_important_variables.py` | Identifies each form's date, completion, missingness, user, and unconditional fields. |
| `CollectMiscVariables` | `define_important_variables.py` | Groups blood, SCID, and pharmaceutical variables and builds field-to-form and normalized-name lookups. |
| `AnalyzeIdentifiers` | `analyze_identifier_effects.py` | Audits identifier references in dictionary calculations and branching logic. |
| `TransformBranchingLogic` | `transform_branching_logic.py` | Converts REDCap branching logic into validated Python conditions, applies study-specific corrections, and records excluded conversions. |
| `CollectSubjectInfo` | `collect_subject_info.py` | Collects subject demographics, participation status, conversion criteria, medication dates, and blood-visit dates. |
| `OrganizeReports` | `organize_reports.py` | Defines network-specific QC variable groups, report assignments, exclusions, team reports, and later-added variables. |
| `RaSubjects` | `collect_ra_subjects.py` | Maps research assistants to their assigned subjects using PRESCIENT raw client-list CSVs. |
| `RawCSVCollector` | `collect_raw_csv_info.py` | Extracts and normalizes PRESCIENT transition/conversion dates from raw client-status CSVs. |
| `PlusDosageMedCollector` | `collect_plus_dosage_meds.py` | Finds medication courses with `+` dosage expressions and translates medication codes into names. |
| `MultiTPDataCollector` | `collect_multi_timepoint_data.py` | Combines selected fields across timepoints and produces interview-date summaries, variable introduction dates, and value-type distributions. |
| `DuplicateFinder` | `find_duplicates.py` | Scans PRESCIENT multi-timepoint blood-ID columns and prints counts of columns with repeated nonmissing IDs across subjects. |
| `RangeDefiner` | `define_ranges.py` | Combines dictionary limits with study and network overrides to define valid numeric ranges. |

#### Additional tools

These classes are not invoked by `ProcessVariables.run_script()`.

| Class | File | Summary |
| --- | --- | --- |
| `DataMerger` | `create_merged_csv.py` | Concatenates selected calculated, radio, and yes/no fields with subject/event metadata across networks and timepoints. |
| `APMedMapper` | `map_antipsychotic_meds.py` | Maps lifetime antipsychotic medication names to pharmaceutical dropdown codes and exposure fields. |
| `PrepareDateChecks` | `prepare_date_checks.py` | Unfinished date-chronology scaffold with an empty chronology method and an undefined entry-point method call. |
| `MultiTPLongFormatProducer` | `produce_multi_tp_long.py` | Produces per-network long-format Parquet datasets for eligible numeric variables, retaining missingness metadata. |
| `MultiTPDatesLongFormatProducer` | `produce_multi_tp_dates_long.py` | Produces per-network long-format Parquet datasets for eligible date variables after metadata and distinct-date filtering. |

#### Calculated-field translation

All classes below are in `transform_calculated_fields.py`; the helper classes
support the translator rather than serving as separate programs.

| Class | Summary |
| --- | --- |
| `TransformCalculatedFields` | Converts calculated fields into validated Python expressions, analyzes their dependencies, and records conversion results and exclusions. |
| `CalculationTranslationError` | Reports an invalid calculation translation, optionally identifying the source character position. |
| `UnsupportedCalculationError` | Reports well-formed REDCap calculation syntax that the translator does not support. |
| `CalculationSchemaError` | Reports a data dictionary that fails the translator's input requirements. |
| `Token` | Stores a calculation token's type, value, source position, and original text. |
| `LiteralNode` | Represents a literal value in a parsed calculation. |
| `NumberNode` | Preserves a numeric literal's original spelling for accurate translation. |
| `FieldNode` | Represents a field reference with optional event, checkbox-choice, and repeating-instance qualifiers. |
| `UnaryNode` | Represents a unary operator and its operand. |
| `BinaryNode` | Represents a binary operator and its left and right operands. |
| `CallNode` | Represents a calculation function call and its arguments. |
| `CalculationTokenizer` | Converts a REDCap calculation into a validated token stream. |
| `CalculationParser` | Builds a syntax tree while preserving operator precedence, nested calls, and qualified field references. |
| `CalculationEvaluation` | Stores an evaluated calculation's value, status, and diagnostic text. |

### Other folders

| Folder | General contents |
| --- | --- |
| `main/analyze_dataset` | Study-data exploration, summaries, variable mappings, cohort and form-completion investigations, and targeted extracts or plots. |
| `main/analyze_flags` | Tracker-history collection, manual review mappings, flag-resolution and value-change analysis, QC visualizations, and outcome calculation and comparison tools. |
| `main/clinical_trial_testing` | Experimental clinical-trial branching-logic checks and an unfinished synthetic-data generation scaffold. |
| `main/discover_errors` | Standalone investigations of candidate QC issues using statistical, machine-learning, longitudinal, duplicate-record, calculated-field, and domain-specific checks. |
| `main/harmonization_qc` | Audits of translated REDCap branching logic and comparisons between REDCap exports and combined study datasets. |
| `main/interactive_graphs` | Interactive Dash/Plotly QC dashboards, data preparation and caching, and HTML export support. |
| `main/qc_pipeline` | Legacy monitoring and notification utilities for data volume, output freshness, and dependencies, plus incomplete pipeline scaffolding. |
| `main/simple_anomaly_detection` | A standalone clinical-measures anomaly-detection suite that combines multiple detectors into ranked Excel reports and CSV outputs. |
| `main/utils` | Shared configuration, dependency loading, study/site metadata, missing-value handling, data access, Dropbox, and branching-expression evaluation helpers. |
| `main/visualize_data` | Static QC graphs and data-change visualizations, plus a nested copy of the interactive dashboard tooling. |
| `docs` | Design and review notes, detector/output guides, and example configuration files. |
| `tests` | Unit, regression, synthetic-data, clinical, and integration checks for the pipeline and analysis tools. |
