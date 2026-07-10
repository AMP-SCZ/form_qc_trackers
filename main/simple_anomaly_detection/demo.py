"""Tiny synthetic data generator for smoke-testing the seven detectors.

Generates a small set of AMPSCZ-shaped CSVs in a temp folder with seeded
anomalies of every type. Used when ``python -m simple_anomaly_detection
--demo`` is run.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd


def build_demo_dataset(out_dir: str, n_subjects: int = 150, seed: int = 7) -> Tuple[str, list, list]:
    """Write AMPSCZ-shaped CSVs into ``out_dir``.

    Returns ``(out_dir, written_files, seeded_anomalies)`` where
    ``seeded_anomalies`` is the deterministic oracle: a list of dicts each
    describing one injected anomaly and the detector/``anomaly_type`` that
    should surface it. Used by the regression test
    (tests/synthetic/test_simple_anomaly_demo.py) to assert the framework
    actually catches what it is fed. (Earlier callers ignored the return
    value entirely; the extra element is additive and backward-compatible.)
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    networks = {
        "PRONET": ["KC", "BI", "SD", "NL", "OR", "CA", "IR", "MU", "YA", "HA"],
        "PRESCIENT": ["BM", "CG", "CP", "GW", "HK", "JE"],
    }
    tps = ["baseline", "month_1", "month_2"]

    subjects_by_network = {}
    for net, sites in networks.items():
        ids = []
        for s in sites:
            for i in range(1, n_subjects // len(sites) + 2):
                ids.append(f"{s}{i:05d}")
        subjects_by_network[net] = ids[:n_subjects]

    written = []
    for net, subs in subjects_by_network.items():
        # Per-subject baseline values so trajectories make sense.
        baseline = {
            "age":           rng.normal(24, 5, len(subs)).round(0),
            "weight_kg":     rng.normal(72, 12, len(subs)).round(1),
            "height_cm":     rng.normal(170, 10, len(subs)).round(1),
            "score_total":   rng.normal(40, 8, len(subs)).round(1),
            "score_anxiety": rng.normal(15, 4, len(subs)).round(1),
            "smoker":        rng.choice([0, 1], len(subs), p=[0.75, 0.25]),
            "education":     rng.choice([1, 2, 3, 4], len(subs)),
        }
        # Three mutually-correlated variables driven by one latent factor ->
        # a single tight, elongated cloud along the diagonal in 3D, for the
        # cluster_3d detector. The factor is unimodal so each variable is
        # marginally normal (no 1D outliers); the seeded anomaly below is
        # OFF that diagonal -- a 3D-only deviation.
        _factor = rng.normal(0, 1, len(subs))
        baseline["cluster_a"] = (50 + 12 * _factor + rng.normal(0, 1, len(subs))).round(1)
        baseline["cluster_b"] = (60 + 12 * _factor + rng.normal(0, 1, len(subs))).round(1)
        baseline["cluster_c"] = (40 + 12 * _factor + rng.normal(0, 1, len(subs))).round(1)
        baseline_date = pd.Series(
            pd.to_datetime("2026-01-15") + pd.to_timedelta(rng.integers(0, 60, len(subs)), unit="D")
        )

        for ti, tp in enumerate(tps):
            df = pd.DataFrame({"subjectid": subs})
            # Numeric drift per timepoint
            df["age"]           = baseline["age"] + ti
            df["weight_kg"]     = baseline["weight_kg"] + rng.normal(0, 0.5, len(subs)).round(1)
            df["height_cm"]     = baseline["height_cm"] + rng.normal(0, 0.1, len(subs)).round(1)
            df["score_total"]   = baseline["score_total"] - 0.5 * ti + rng.normal(0, 1.5, len(subs)).round(1)
            df["score_anxiety"] = baseline["score_anxiety"] + rng.normal(0, 0.5, len(subs)).round(1)
            # Score_anxiety correlated with score_total (binary-numeric for variety)
            df["score_anxiety"] = (0.3 * df["score_total"] + rng.normal(0, 2, len(subs))).round(1)
            df["smoker"]        = baseline["smoker"]
            df["education"]     = baseline["education"]
            df["cluster_a"]     = (baseline["cluster_a"] + rng.normal(0, 0.3, len(subs))).round(1)
            df["cluster_b"]     = (baseline["cluster_b"] + rng.normal(0, 0.3, len(subs))).round(1)
            df["cluster_c"]     = (baseline["cluster_c"] + rng.normal(0, 0.3, len(subs))).round(1)

            # Site code IDs.
            df["site_visit_id"] = [f"{s[:2]}-V{ti+1}-{i:04d}" for i, s in enumerate(subs)]

            # Dates: baseline + tp offset, +/- small noise.
            offset = {"baseline": 0, "month_1": 30, "month_2": 60}[tp]
            df["visit_date"] = (
                baseline_date + pd.to_timedelta(offset + rng.integers(-3, 3, len(subs)), unit="D")
            ).dt.strftime("%Y-%m-%d").values
            df["consent_date"] = (
                baseline_date - pd.to_timedelta(rng.integers(0, 10, len(subs)), unit="D")
            ).dt.strftime("%Y-%m-%d").values

            # Seed anomalies (deterministic indexes).
            if ti == 0:
                df.loc[3, "weight_kg"] = 320.0                         # standard outlier
                df.loc[7, "score_total"] = -50.0                      # extreme low
                df.loc[11, "site_visit_id"] = "weirdformat!!!"        # format deviation
                df.loc[15, "consent_date"] = (pd.to_datetime("2030-12-31")
                                              ).strftime("%Y-%m-%d")   # date order reversal
                # whole-subject outlier seed (row 19): high z everywhere
                df.loc[19, ["weight_kg", "height_cm", "score_total", "score_anxiety"]] = [
                    180.0, 240.0, 250.0, 95.0,
                ]
                # 3D cluster-deviation seed (row 40): each value is marginally
                # normal (~+1sd on a and b, ~-1sd on c), so no 1D outlier -- but
                # a/b high while c is low breaks the a~b~c diagonal, putting the
                # point off the correlated cloud in 3D (a 3D-only anomaly).
                df.loc[40, "cluster_a"] = 62.0
                df.loc[40, "cluster_b"] = 72.0
                df.loc[40, "cluster_c"] = 28.0
                # cross-timepoint seed (row 45): cluster_a jumps across visits
                # (62 at baseline -> 38 at month1 -> 62 at month2). Each value
                # is marginally normal (~+-1sd), but the baseline/month1/month2
                # trajectory is off the cross-timepoint diagonal (a@b ~ a@m1 ~
                # a@m2 for everyone else) -> a longitudinal-only anomaly.
                df.loc[45, "cluster_a"] = 62.0
                # Site-wide spike: site BI all get same odd height
                if net == "PRONET":
                    df.loc[df.subjectid.str.startswith("BI"), "height_cm"] = 220.0
            if ti == 1:
                # Time-series outlier: subject 23 jumps weight by 100 kg
                df.loc[23, "weight_kg"] = float(df.loc[23, "weight_kg"]) + 100.0
                # Date gap anomaly: subject 30 visit_date is 500 days from baseline
                df.loc[30, "visit_date"] = "2027-08-01"
                # Correlative violation: subject 35's anxiety low while total very high
                df.loc[35, "score_total"] = 95.0
                df.loc[35, "score_anxiety"] = 1.0
                # cross-timepoint seed (row 45) continued: dip at month1.
                df.loc[45, "cluster_a"] = 38.0
            if ti == 2:
                # cross-timepoint seed (row 45) continued: back up at month2,
                # completing the off-diagonal 62 -> 38 -> 62 trajectory.
                df.loc[45, "cluster_a"] = 62.0

            fname = f"AMPSCZ-combined-redcap_{tp}_{net}.csv"
            path = os.path.join(out_dir, fname)
            df.to_csv(path, index=False)
            written.append(path)

    # Deterministic oracle: which seeded anomaly each detector should catch.
    # Row indices below match the df.loc[...] seeds above; subjectids are
    # resolved from PRONET's subject order (seeds are injected into every
    # network's frame, PRONET is sufficient for the regression assertions).
    ps = subjects_by_network["PRONET"]
    seeded_anomalies = [
        {"subjectid": ps[3],  "network": "PRONET", "anomaly_type": "standard_outlier",     "kind": "weight_kg=320"},
        {"subjectid": ps[7],  "network": "PRONET", "anomaly_type": "standard_outlier",     "kind": "score_total=-50"},
        {"subjectid": ps[11], "network": "PRONET", "anomaly_type": "format_deviation",     "kind": "site_visit_id weirdformat"},
        {"subjectid": ps[15], "network": "PRONET", "anomaly_type": "date_anomaly_simple",  "kind": "consent_date future / order"},
        {"subjectid": ps[19], "network": "PRONET", "anomaly_type": "whole_subject_outlier", "kind": "high-z on every variable"},
        {"subjectid": ps[23], "network": "PRONET", "anomaly_type": "time_series_outlier",  "kind": "weight +100kg at month1"},
        {"subjectid": ps[30], "network": "PRONET", "anomaly_type": "date_anomaly_simple",  "kind": "visit_date ~500-day gap"},
        {"subjectid": ps[35], "network": "PRONET", "anomaly_type": "correlative_outlier",  "kind": "score_total high / anxiety low"},
        {"subjectid": ps[40], "network": "PRONET", "anomaly_type": "cluster_3d_outlier",   "kind": "off-diagonal in (cluster_a,b,c) 3D space"},
        {"subjectid": ps[45], "network": "PRONET", "anomaly_type": "cluster_3d_longitudinal_outlier", "kind": "cluster_a jumpy 62->38->62 across timepoints"},
        {"site_id": "BI", "network": "PRONET", "anomaly_type": "site_network_outlier",
         "variable_contains": "height", "kind": "BI site-wide height=220"},
    ]
    return out_dir, written, seeded_anomalies
