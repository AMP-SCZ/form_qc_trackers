"""Focused regressions for conservative 3D-cluster eligibility."""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simple_anomaly_detection import cluster_3d as C  # noqa: E402
from simple_anomaly_detection import cluster_3d_longitudinal as L  # noqa: E402
from simple_anomaly_detection.common import pairwise_nan_spearman  # noqa: E402


_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def test_joint_complete_correlation_rejects_pairwise_missingness_artifact():
    """Strong pairs on three different row sets do not make a valid triple."""
    rng = np.random.default_rng(21)
    n_joint = 60
    n_pair = 80
    total = n_joint + 3 * n_pair
    data = np.full((total, 3), np.nan)

    # The only rows complete on all axes have unrelated values.
    data[:n_joint] = rng.normal(size=(n_joint, 3))

    # Each pair gets a separate, high-valued diagonal block.  Pairwise deletion
    # sees three strong relationships, but none is supported by the exact rows
    # used for 3D clustering.
    start = n_joint
    for left, right in ((0, 1), (0, 2), (1, 2)):
        latent = 20 + np.linspace(-2, 2, n_pair)
        sl = slice(start, start + n_pair)
        data[sl, left] = latent + rng.normal(0, 0.02, n_pair)
        data[sl, right] = latent + rng.normal(0, 0.02, n_pair)
        start += n_pair

    pair_corr, _ = pairwise_nan_spearman(data, min_joint_n=C.MIN_POINTS)
    pair_weakest = np.abs(pair_corr[np.triu_indices(3, 1)]).min()
    assert pair_weakest >= C.MIN_TRIPLE_CORR
    assert C._weakest_joint_spearman(data) < C.MIN_TRIPLE_CORR


def test_within_slice_gate_rejects_correlation_created_by_visit_shifts():
    """A pooled location shift cannot qualify unrelated within-visit axes."""
    rng = np.random.default_rng(9)
    base = rng.normal(size=(90, 3))
    later = rng.normal(size=(90, 3)) + 20.0
    pooled = np.vstack([base, later])

    assert C._weakest_joint_spearman(pooled) >= C.MIN_TRIPLE_CORR
    assert C._weakest_joint_spearman(base) < C.MIN_TRIPLE_CORR
    assert C._weakest_joint_spearman(later) < C.MIN_TRIPLE_CORR

    class MustNotRunDBSCAN:
        def __init__(self, **_):
            raise AssertionError("DBSCAN ran before the within-slice relationship gate")

    subjects = np.array([f"S{i:03d}" for i in range(len(base))])
    assert C._cluster_triple(
        "PRONET", "baseline", ("a", "b", "c"),
        [base[:, 0], base[:, 1], base[:, 2]], subjects, MustNotRunDBSCAN,
    ) == []


def test_subminimum_dbscan_component_is_scored_as_noise(monkeypatch):
    """A labelled 5-row island is not silently omitted from both populations."""
    rng = np.random.default_rng(4)
    t = np.linspace(-3, 3, 60)
    big = np.column_stack([
        t + rng.normal(0, 0.02, len(t)),
        t + rng.normal(0, 0.02, len(t)),
        t + rng.normal(0, 0.02, len(t)),
    ])
    island_t = np.linspace(-0.2, 0.2, 5)
    island = np.column_stack([island_t, island_t + 3.0, island_t - 3.0])
    data = np.vstack([big, island])
    subjects = np.array([f"S{i:03d}" for i in range(len(data))])

    class FixedDBSCAN:
        def __init__(self, **_):
            pass

        def fit_predict(self, _):
            labels = np.zeros(len(data), dtype=int)
            labels[-len(island):] = 7  # labelled, but < MIN_CLUSTER_SIZE
            return labels

    monkeypatch.setattr(C, "_adaptive_eps", lambda *_: 0.5)
    rows = C._cluster_triple(
        "PRONET", "baseline", ("a", "b", "c"),
        [data[:, 0], data[:, 1], data[:, 2]], subjects, FixedDBSCAN,
    )
    flagged = {row["subjectid"] for row in rows}
    assert flagged
    assert flagged <= set(subjects[-len(island):])
    assert all(row["n_clusters"] == 1 for row in rows)


def test_longitudinal_pivot_rejects_blank_and_conflicting_duplicate_ids():
    df = pd.DataFrame({
        "subjectid": ["", " S1 ", "S1", "S2", "S2", "S3", None],
        "a": [99, 1, 1, 2, 9, 3, 88],
        "b": [99, 2, 2, 3, 3, 4, 88],
        "c": [99, 3, 3, 4, 4, 5, 88],
    })
    cats = {"numeric": ["c", "a", "b"]}
    wide = L._build_wide([("baseline", df, cats)])

    # Whitespace-normalized exact S1 duplicates safely collapse; conflicting S2
    # rows and missing/blank ids are excluded rather than arbitrarily selected.
    assert list(wide.index) == ["S1", "S3"]
    assert wide.index.is_unique
    assert list(wide.columns) == ["a@baseline", "b@baseline", "c@baseline"]
    assert float(wide.loc["S1", "a@baseline"]) == 1.0


@pytest.mark.skipif(not _HAS_SKLEARN, reason="cluster_3d needs scikit-learn")
def test_completeness_ties_choose_variables_by_stable_name(monkeypatch):
    n = 80
    latent = np.linspace(-2, 2, n)
    df = pd.DataFrame({
        "subjectid": [f"S{i:03d}" for i in range(n)],
        "z": latent + 0.03,
        "c": latent + 0.02,
        "b": latent + 0.01,
        "a": latent,
    })
    seen = []

    def record(_network, _tp, cols, *_args, **_kwargs):
        seen.append(tuple(cols))
        return []

    monkeypatch.setattr(C, "N_NUMERIC", 3)
    monkeypatch.setattr(C, "_cluster_triple", record)
    cats = {"numeric": ["z", "c", "b", "a"]}
    C._detect_network("PRONET", [("baseline", df, cats)])
    assert seen == [("a", "b", "c")]

