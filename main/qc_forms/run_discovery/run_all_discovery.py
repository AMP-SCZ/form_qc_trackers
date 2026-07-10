"""
Run all discovery jobs that are enabled in config.json (Layer 2).

Order is fixed: missingness summary (uses long parquet only), then
copy-forward, longitudinal delta, date anomaly, duplicate-record
(I/O heavy).

Each submodule is default-off via its own discovery.<name>.enabled flag.
Failures stop the batch (first exception propagates).

Does not run produce_multi_tp_long — ensure long parquets exist before
copy-forward / delta / missingness.

Does not touch combined_qc_flags or raw data.
"""

import os
import sys
import json

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


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

    if bool(disc.get("missingness_summary", {}).get("enabled", False)):
        from qc_types.discovery.missingness_summary_checks import (
            MissingnessSummaryChecks,
        )
        print("[run_all_discovery] running missingness_summary ...")
        MissingnessSummaryChecks(config_dict=cfg)()
        ran.append("missingness_summary")

    if bool(disc.get("copy_forward", {}).get("enabled", False)):
        from qc_types.discovery.copy_forward_checks import CopyForwardChecks
        print("[run_all_discovery] running copy_forward ...")
        CopyForwardChecks(config_dict=cfg)()
        ran.append("copy_forward")

    if bool(disc.get("longitudinal_delta", {}).get("enabled", False)):
        from qc_types.discovery.longitudinal_delta_checks import (
            LongitudinalDeltaChecks,
        )
        print("[run_all_discovery] running longitudinal_delta ...")
        LongitudinalDeltaChecks(config_dict=cfg)()
        ran.append("longitudinal_delta")

    if bool(disc.get("point_spike", {}).get("enabled", False)):
        from qc_types.discovery.point_spike_checks import (
            PointSpikeChecks,
        )
        print("[run_all_discovery] running point_spike ...")
        PointSpikeChecks(config_dict=cfg)()
        ran.append("point_spike")

    if bool(disc.get("internal_consistency", {}).get(
            "enabled", False)):
        from qc_types.discovery.internal_consistency_checks import (
            InternalConsistencyChecks,
        )
        print("[run_all_discovery] running internal_consistency ...")
        InternalConsistencyChecks(config_dict=cfg)()
        ran.append("internal_consistency")

    if bool(disc.get("pairwise_relationship", {}).get(
            "enabled", False)):
        from qc_types.discovery.pairwise_relationship_checks import (
            PairwiseRelationshipChecks,
        )
        print("[run_all_discovery] running pairwise_relationship ...")
        PairwiseRelationshipChecks(config_dict=cfg)()
        ran.append("pairwise_relationship")

    if bool(disc.get("mahalanobis", {}).get("enabled", False)):
        from qc_types.discovery.mahalanobis_checks import (
            MahalanobisChecks,
        )
        print("[run_all_discovery] running mahalanobis ...")
        MahalanobisChecks(config_dict=cfg)()
        ran.append("mahalanobis")

    if bool(disc.get("pca_reconstruction", {}).get(
            "enabled", False)):
        from qc_types.discovery.pca_reconstruction_checks import (
            PCAReconstructionChecks,
        )
        print("[run_all_discovery] running pca_reconstruction ...")
        PCAReconstructionChecks(config_dict=cfg)()
        ran.append("pca_reconstruction")

    if bool(disc.get("local_outlier", {}).get("enabled", False)):
        from qc_types.discovery.local_outlier_checks import (
            LocalOutlierChecks,
        )
        print("[run_all_discovery] running local_outlier ...")
        LocalOutlierChecks(config_dict=cfg)()
        ran.append("local_outlier")

    if bool(disc.get("trajectory_shape", {}).get(
            "enabled", False)):
        from qc_types.discovery.trajectory_shape_checks import (
            TrajectoryShapeChecks,
        )
        print("[run_all_discovery] running trajectory_shape ...")
        TrajectoryShapeChecks(config_dict=cfg)()
        ran.append("trajectory_shape")

    if bool(disc.get("isolation_forest", {}).get(
            "enabled", False)):
        from qc_types.discovery.isolation_forest_checks import (
            IsolationForestChecks,
        )
        print("[run_all_discovery] running isolation_forest ...")
        IsolationForestChecks(config_dict=cfg)()
        ran.append("isolation_forest")

    if bool(disc.get("date_anomaly", {}).get("enabled", False)):
        from qc_types.discovery.date_anomaly_checks import (
            DateAnomalyChecks,
        )
        print("[run_all_discovery] running date_anomaly ...")
        DateAnomalyChecks(config_dict=cfg)()
        ran.append("date_anomaly")

    if bool(disc.get("duplicate_record", {}).get("enabled", False)):
        from qc_types.discovery.duplicate_record_checks import (
            DuplicateRecordChecks,
        )
        print("[run_all_discovery] running duplicate_record ...")
        DuplicateRecordChecks(config_dict=cfg)()
        ran.append("duplicate_record")

    # Subject-level cross-detector aggregator runs LAST so it can
    # read every other detector's output. Default-on if any other
    # detector ran (controlled by its own enabled flag).
    if bool(disc.get("subject_summary", {}).get("enabled", True)):
        from qc_types.discovery.subject_summary import SubjectSummary
        print("[run_all_discovery] running subject_summary ...")
        SubjectSummary(config_dict=cfg)()
        ran.append("subject_summary")

    if not ran:
        print(
            "[run_all_discovery] no discovery.*.enabled flags are true — "
            "nothing to do. See docs/config.discovery.example.json."
        )
    else:
        print(f"[run_all_discovery] completed: {', '.join(ran)}")

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
    ]
    present = []
    for key, fname in detector_files:
        p = os.path.join(disc_dir, fname)
        if os.path.isfile(p):
            df = pd.read_parquet(p)
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
    try:
        with pd.ExcelWriter(tmp_xlsx, engine='openpyxl') as xw:
            for key, df in present:
                df.to_excel(xw, sheet_name=key, index=False)
        os.replace(tmp_xlsx, final_xlsx)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
