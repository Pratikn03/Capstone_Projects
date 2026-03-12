from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gridpulse.forecasting.uncertainty.reliability_mondrian import (
    ReliabilityMondrian,
    ReliabilityMondrianConfig,
)
from scripts.compute_reliability_group_coverage import build_summary


def test_reliability_mondrian_fit_predict_and_group_coverage() -> None:
    y_pred = np.zeros(200, dtype=float)
    reliability = np.repeat([0.2, 0.8], 100)
    noise = np.concatenate([np.full(100, 2.0), np.full(100, 0.5)])
    y_true = y_pred + noise

    model = ReliabilityMondrian(ReliabilityMondrianConfig(alpha=0.10, n_bins=2, min_bin_size=10))
    model.fit(y_true=y_true, y_pred=y_pred, reliability=reliability)
    lower, upper = model.predict_interval(y_pred=y_pred, reliability=reliability)
    rows = model.group_coverage(y_true=y_true, lower=lower, upper=upper, reliability=reliability)

    assert len(rows) == 2
    assert rows[0]["qhat"] >= rows[1]["qhat"]
    assert np.all(upper >= lower)


def test_reliability_mondrian_min_bin_size_falls_back_to_global() -> None:
    y_pred = np.zeros(12, dtype=float)
    reliability = np.linspace(0.05, 0.95, 12)
    y_true = np.linspace(-1.0, 1.0, 12)

    model = ReliabilityMondrian(ReliabilityMondrianConfig(alpha=0.10, n_bins=4, min_bin_size=50))
    model.fit(y_true=y_true, y_pred=y_pred, reliability=reliability)

    assert model.global_q_ is not None
    assert all(np.isclose(qhat, model.global_q_) for qhat in model.q_by_bin_.values())


def test_compute_reliability_group_coverage_builds_stable_artifacts(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "y_true": np.linspace(0.0, 9.0, 20),
            "y_pred": np.linspace(0.5, 9.5, 20),
            "reliability_w": np.linspace(0.1, 0.9, 20),
        }
    )
    rows, summary = build_summary(
        df,
        y_true_col="y_true",
        y_pred_col="y_pred",
        reliability_col="reliability_w",
        alpha=0.10,
        n_bins=4,
        min_bin_size=2,
    )
    assert set(rows.columns) >= {"bin_id", "picp", "mean_interval_width", "qhat"}
    assert summary["n_samples"] == 20
    assert summary["n_bins"] >= 2
    json.loads(json.dumps(summary))
