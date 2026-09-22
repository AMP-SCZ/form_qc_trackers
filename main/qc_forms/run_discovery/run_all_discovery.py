"""
Run all discovery jobs that are enabled in config.json (Layer 2).

Order is fixed: missingness summary (uses long parquet only), then
copy-forward, longitudinal delta, point_spike, internal_consistency,
pairwise_relationship, mahalanobis, pca_reconstruction, local_outlier,
trajectory_shape, isolation_forest, date_anomaly, duplicate_record,
and finally subject_summary which aggregates all per-detector
parquets into a cross-detector ranking.

Each submodule is default-off via its own discovery.<name>.enabled flag.
Failures are ISOLATED per detector — one crash logs a warning and the
chain continues with the rest. subject_summary still runs and the
workbook still builds from whatever parquets did write.

Does not run produce_multi_tp_long — ensure long parquets exist before
copy-forward / delta / missingness.

Does not touch combined_qc_flags or raw data.
"""

import os
import sys
import json
import traceback

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _enabled(node, key):
    """
    Sprint 1 P0-2: strict enabled-flag parser. Wraps
    `qc_types.discovery._common.is_truthy_enabled` so this module
    doesn't need a heavy import at the top — the function is
    imported lazily on first call to avoid a chicken/egg with the
    sys.path setup below.
    """
    # Local import: sys.path is wired up below the _HERE constants;
    # _common can't be imported until after that block runs. Wrap
    # in a try so the parser still works in the unlikely event
    # _common is unavailable (degrade to the old loose semantics).
    try:
        from qc_types.discovery._common import is_truthy_enabled
        return is_truthy_enabled(
            node.get(key, {}).get('enabled', False))
    except ImportError:
        return bool(node.get(key, {}).get('enabled', False))


def _enabled_default_true(node, key):
    """
    Same as _enabled but the omitted-key default is True. Used
    only for `subject_summary` — the aggregator runs unless an
    operator explicitly turns it off. Strict parsing still applies
    when the `enabled` key IS present, so a typo'd
    `"enabled": "false"` is rejected the same way.
    """
    cfg = node.get(key, {})
    if 'enabled' not in cfg:
        return True
    try:
        from qc_types.discovery._common import is_truthy_enabled
        return is_truthy_enabled(cfg['enabled'])
    except ImportError:
        return bool(cfg['enabled'])


def _find_ancestor_with(*relpaths):
    """Walk up from _HERE; return the first ancestor that contains
    every entry in relpaths (file or dir). Skips shadows that lack the
    expected subtree."""
    candidate = _HERE
    for _ in range(6):
        if all(os.path.exists(os.path.join(candidate, r))
               for r in relpaths):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return None
        candidate = parent
    return None


# Add to sys.path: the dir containing qc_types/discovery/ AND the dir
# containing utils/utils.py. In production these are different
# directories (main/qc_forms/ vs main/); locally they're the same.
_QC_TYPES_PARENT = _find_ancestor_with(os.path.join("qc_types", "discovery"))
_UTILS_PARENT = _find_ancestor_with(os.path.join("utils", "utils.py"))
for _p in (_UTILS_PARENT, _QC_TYPES_PARENT):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

_CONFIG_PATH = None
_cfg_anc = _find_ancestor_with("config.json")
if _cfg_anc is not None:
    _CONFIG_PATH = os.path.join(_cfg_anc, "config.json")


def _run_detector(name, run_callable, ran, failed):
    """
    Invoke one detector inside a try/except so a single crash doesn't
    kill the rest of the chain. Records success in `ran`, failure
    (with type + message) in `failed`. Prints the traceback so the
    operator can diagnose without losing the rest of the run.
    """
    print(f"[run_all_discovery] running {name} ...")
    try:
        run_callable()
        ran.append(name)
        return True
    except Exception as e:
        print(
            f"[run_all_discovery] WARNING: detector {name!r} failed "
            f"with {type(e).__name__}: {e}. The pipeline will "
            f"CONTINUE with the remaining detectors; the workbook "
            f"will reuse this detector's parquet from a prior run "
            f"(if any) or omit its sheet."
        )
        traceback.print_exc()
        failed.append((name, type(e).__name__, str(e)))
        return False


def main():
    if _CONFIG_PATH is None:
        print(
            "[run_all_discovery] FATAL: config.json not found in any "
            f"ancestor of {_HERE}"
        )
        return 1
    config_path = _CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    disc = cfg.get("discovery", {})
    ran = []
    failed = []

    if _enabled(disc, "missingness_summary"):
        from qc_types.discovery.missingness_summary_checks import (
            MissingnessSummaryChecks,
        )
        _run_detector(
            "missingness_summary",
            lambda: MissingnessSummaryChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "copy_forward"):
        from qc_types.discovery.copy_forward_checks import CopyForwardChecks
        _run_detector(
            "copy_forward",
            lambda: CopyForwardChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "longitudinal_delta"):
        from qc_types.discovery.longitudinal_delta_checks import (
            LongitudinalDeltaChecks,
        )
        _run_detector(
            "longitudinal_delta",
            lambda: LongitudinalDeltaChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "point_spike"):
        from qc_types.discovery.point_spike_checks import (
            PointSpikeChecks,
        )
        _run_detector(
            "point_spike",
            lambda: PointSpikeChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "internal_consistency"):
        from qc_types.discovery.internal_consistency_checks import (
            InternalConsistencyChecks,
        )
        _run_detector(
            "internal_consistency",
            lambda: InternalConsistencyChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "pairwise_relationship"):
        from qc_types.discovery.pairwise_relationship_checks import (
            PairwiseRelationshipChecks,
        )
        _run_detector(
            "pairwise_relationship",
            lambda: PairwiseRelationshipChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "mahalanobis"):
        from qc_types.discovery.mahalanobis_checks import (
            MahalanobisChecks,
        )
        _run_detector(
            "mahalanobis",
            lambda: MahalanobisChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "pca_reconstruction"):
        from qc_types.discovery.pca_reconstruction_checks import (
            PCAReconstructionChecks,
        )
        _run_detector(
            "pca_reconstruction",
            lambda: PCAReconstructionChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "local_outlier"):
        from qc_types.discovery.local_outlier_checks import (
            LocalOutlierChecks,
        )
        _run_detector(
            "local_outlier",
            lambda: LocalOutlierChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "trajectory_shape"):
        from qc_types.discovery.trajectory_shape_checks import (
            TrajectoryShapeChecks,
        )
        _run_detector(
            "trajectory_shape",
            lambda: TrajectoryShapeChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "isolation_forest"):
        from qc_types.discovery.isolation_forest_checks import (
            IsolationForestChecks,
        )
        _run_detector(
            "isolation_forest",
            lambda: IsolationForestChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "date_anomaly"):
        from qc_types.discovery.date_anomaly_checks import (
            DateAnomalyChecks,
        )
        _run_detector(
            "date_anomaly",
            lambda: DateAnomalyChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "duplicate_record"):
        from qc_types.discovery.duplicate_record_checks import (
            DuplicateRecordChecks,
        )
        _run_detector(
            "duplicate_record",
            lambda: DuplicateRecordChecks(config_dict=cfg)(),
            ran, failed,
        )

    # Sprint 1 P0-1: register detectors added after the original
    # chain was built. Each runs only when its own enabled flag is
    # true; otherwise the chain skips it like every other detector.
    if _enabled(disc, "multi_tp_consistency"):
        from qc_types.discovery.multi_tp_consistency_checks import (
            MultiTPConsistencyChecks,
        )
        _run_detector(
            "multi_tp_consistency",
            lambda: MultiTPConsistencyChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "cohort_trajectory_deviation"):
        from qc_types.discovery.cohort_trajectory_deviation_checks import (
            CohortTrajectoryDeviationChecks,
        )
        _run_detector(
            "cohort_trajectory_deviation",
            lambda: CohortTrajectoryDeviationChecks(
                config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "site_distribution_drift"):
        from qc_types.discovery.site_distribution_drift_checks import (
            SiteDistributionDriftChecks,
        )
        _run_detector(
            "site_distribution_drift",
            lambda: SiteDistributionDriftChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "site_correlation_drift"):
        from qc_types.discovery.site_correlation_drift_checks import (
            SiteCorrelationDriftChecks,
        )
        _run_detector(
            "site_correlation_drift",
            lambda: SiteCorrelationDriftChecks(config_dict=cfg)(),
            ran, failed,
        )

    if _enabled(disc, "site_trajectory_slope"):
        from qc_types.discovery.site_trajectory_slope_checks import (
            SiteTrajectorySlopeChecks,
        )
        _run_detector(
            "site_trajectory_slope",
            lambda: SiteTrajectorySlopeChecks(config_dict=cfg)(),
            ran, failed,
        )

    # Subject-level cross-detector aggregator runs LAST so it can
    # read every other detector's output, even those that crashed
    # mid-chain (their prior-run parquets, if any, are still on disk).
    if _enabled_default_true(disc, "subject_summary"):
        from qc_types.discovery.subject_summary import SubjectSummary
        _run_detector(
            "subject_summary",
            lambda: SubjectSummary(config_dict=cfg)(),
            ran, failed,
        )

    if not ran and not failed:
        print(
            "[run_all_discovery] no discovery.*.enabled flags are true — "
            "nothing to do. See docs/config.discovery.example.json."
        )
    else:
        print(f"[run_all_discovery] completed: {', '.join(ran)}")
        if failed:
            print(
                f"[run_all_discovery] FAILED ({len(failed)}): " +
                ", ".join(f"{n} ({t})" for n, t, _ in failed)
            )

    # Build a single Excel workbook from whatever discovery parquets are on
    # disk. Mirrors the path resolution used by each detector class.
    output_path = cfg['paths']['output_path']
    if str(cfg.get('testing_enabled', '')) == "True":
        output_path = output_path + "testing/"
    disc_dir = os.path.join(output_path, 'combined_outputs', 'discovery')

    detector_files = [
        # Subject summary first so triage starts here.
        ('subject_summary', 'subject_anomaly_summary.parquet'),
        ('missingness_summary', 'missingness_anomaly_summary.parquet'),
        ('copy_forward', 'copy_forward_candidates.parquet'),
        ('longitudinal_delta', 'longitudinal_delta_candidates.parquet'),
        ('point_spike', 'point_spike_candidates.parquet'),
        ('internal_consistency',
         'internal_consistency_candidates.parquet'),
        ('pairwise_relationship',
         'pairwise_relationship_candidates.parquet'),
        ('mahalanobis', 'mahalanobis_candidates.parquet'),
        ('pca_reconstruction',
         'pca_reconstruction_candidates.parquet'),
        ('local_outlier', 'local_outlier_candidates.parquet'),
        ('trajectory_shape',
         'trajectory_shape_candidates.parquet'),
        ('isolation_forest',
         'isolation_forest_candidates.parquet'),
        ('date_anomaly', 'date_anomaly_candidates.parquet'),
        ('duplicate_record', 'duplicate_record_candidates.parquet'),
        # Sprint 1 P0-1: detectors added after the original
        # workbook list was wired.
        ('multi_tp_consistency',
         'multi_tp_consistency_candidates.parquet'),
        ('cohort_trajectory_deviation',
         'cohort_trajectory_deviation.parquet'),
        ('site_distribution_drift',
         'site_distribution_drift.parquet'),
        ('site_correlation_drift',
         'site_correlation_drift.parquet'),
        ('site_trajectory_slope',
         'site_trajectory_slope.parquet'),
    ]
    present = []
    for key, fname in detector_files:
        p = os.path.join(disc_dir, fname)
        if os.path.isfile(p):
            try:
                df = pd.read_parquet(p)
            except Exception as e:
                print(
                    f"[run_all_discovery] WARNING: could not read "
                    f"{p}: {e}; skipping its sheet."
                )
                continue
            if 'severity_score' in df.columns:
                df = df.sort_values(
                    'severity_score',
                    ascending=False,
                    na_position='last',
                )
            df = df.head(2000)
            present.append((key, df))

    if not present:
        print(
            "[run_all_discovery] no discovery parquets found in "
            f"{disc_dir} — skipping workbook."
        )
        return 0

    final_xlsx = os.path.join(disc_dir, 'discovery_anomaly_summary.xlsx')
    tmp_xlsx = f"{final_xlsx}.{os.getpid()}.tmp.xlsx"
    sheet_failures = []
    try:
        with pd.ExcelWriter(tmp_xlsx, engine='openpyxl') as xw:
            for key, df in present:
                # Per-sheet isolation: one bad sheet shouldn't kill
                # the whole workbook write.
                try:
                    df.to_excel(xw, sheet_name=key, index=False)
                except Exception as e:
                    print(
                        f"[run_all_discovery] WARNING: failed to "
                        f"write sheet {key!r}: {e}; skipping that "
                        f"sheet."
                    )
                    sheet_failures.append((key, str(e)))
        os.replace(tmp_xlsx, final_xlsx)
    except (ImportError, ModuleNotFoundError) as e:
        # openpyxl missing on the server. The per-detector parquets
        # are already written; only the convenience xlsx is missing.
        print(
            f"[run_all_discovery] WARNING: cannot build xlsx "
            f"workbook because openpyxl is not installed "
            f"({e}). The per-detector parquets are at {disc_dir}; "
            f"install openpyxl to enable the workbook build "
            f"(`pip install openpyxl`)."
        )
        if os.path.exists(tmp_xlsx):
            try:
                os.remove(tmp_xlsx)
            except OSError:
                pass
        return 0
    except Exception:
        if os.path.exists(tmp_xlsx):
            try:
                os.remove(tmp_xlsx)
            except OSError:
                pass
        raise

    counts = ', '.join(f"{k}={len(df)}" for k, df in present)
    print(
        f"[run_all_discovery] wrote {final_xlsx} ({counts})"
    )
    if sheet_failures:
        print(
            f"[run_all_discovery] WARNING: {len(sheet_failures)} "
            f"sheet(s) could not be written: "
            f"{', '.join(k for k, _ in sheet_failures)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
