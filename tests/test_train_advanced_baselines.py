"""Functional tests for the unified advanced-baseline trainer.

These exercise the full plumbing on a synthetic dataset that doesn't require
external data downloads. Prophet, Darts, and FLAML branches are skipped if the
corresponding optional package is not installed; NGBoost is in the core lock
file and is therefore always exercised.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orius.forecasting.train_advanced import (
    AdvancedTrainerConfig,
    SplitPaths,
    _conformalize,
    _safe_metrics,
    run_advanced_baselines,
)


def _make_synthetic_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    hour = ts.hour.to_numpy(dtype=float)
    dow = ts.dayofweek.to_numpy(dtype=float)
    temperature = 10 + 8 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 1, n)
    base = 1000 + 200 * np.sin(2 * np.pi * hour / 24) + 80 * np.sin(2 * np.pi * dow / 7)
    load = base + 6 * temperature + rng.normal(0, 25, n)
    wind = 400 + 120 * np.cos(2 * np.pi * np.arange(n) / 168) + rng.normal(0, 30, n)
    solar = np.maximum(0.0, 600 * np.sin(2 * np.pi * (hour - 6) / 24)) + rng.normal(0, 20, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "hour": hour,
            "day_of_week": dow,
            "temperature": temperature,
            "load_mw": load,
            "wind_mw": wind,
            "solar_mw": solar,
        }
    )


@pytest.fixture()
def synthetic_splits(tmp_path: Path) -> tuple[SplitPaths, Path]:
    df = _make_synthetic_frame(n=24 * 30, seed=42)
    n = len(df)
    train = df.iloc[: int(n * 0.7)].reset_index(drop=True)
    cal = df.iloc[int(n * 0.7) : int(n * 0.85)].reset_index(drop=True)
    test = df.iloc[int(n * 0.85) :].reset_index(drop=True)
    train_path = tmp_path / "train.parquet"
    cal_path = tmp_path / "cal.parquet"
    test_path = tmp_path / "test.parquet"
    train.to_parquet(train_path)
    cal.to_parquet(cal_path)
    test.to_parquet(test_path)
    return SplitPaths(train=train_path, calibration=cal_path, test=test_path), tmp_path


def test_safe_metrics_align_lengths() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 100)
    p = y + rng.normal(0, 0.1, 100)
    metrics = _safe_metrics(y, p[:90])
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["r2"] <= 1.0


def test_conformalize_meets_target_coverage() -> None:
    rng = np.random.default_rng(1)
    n_cal, n_test = 600, 600
    y_cal = rng.normal(0, 1, n_cal)
    p_cal = y_cal + rng.normal(0, 0.5, n_cal)
    y_test = rng.normal(0, 1, n_test)
    p_test = y_test + rng.normal(0, 0.5, n_test)
    _, lo, hi, metrics = _conformalize(y_cal, p_cal, y_test, p_test, alpha=0.10)
    assert 0.85 <= metrics["picp_90"] <= 0.97
    assert metrics["mean_interval_width"] > 0.0
    assert np.all(hi >= lo)


def test_ngboost_pipeline_writes_metrics_and_conformal(
    synthetic_splits: tuple[SplitPaths, Path],
) -> None:
    if importlib.util.find_spec("ngboost") is None:
        pytest.skip("ngboost not installed in this environment")
    splits, root = synthetic_splits
    metrics_path = root / "week2_metrics.json"
    conformal_dir = root / "uncertainty"
    cfg = AdvancedTrainerConfig(
        region="DE",
        release_id="TEST_R1",
        splits=splits,
        out_root=root,
        targets=("load_mw",),
        seeds=(42, 123),
        horizon=24,
        lookback=72,
        alpha=0.10,
        holiday_country=None,
        metrics_json=metrics_path,
        conformal_dir=conformal_dir,
        enabled_models=("ngboost",),
        flaml_time_budget=30,
    )
    summary = run_advanced_baselines(cfg)
    assert summary["models"]["ngboost"]["load_mw"]["status"] == "ok"

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    block = payload["targets"]["load_mw"]["ngboost"]
    assert block["model"] == "ngboost"
    assert block["uncertainty"]["picp_90"] is not None
    assert block["uncertainty"]["mean_interval_width"] > 0.0
    assert "uncertainty_native" in block

    conformal_path = conformal_dir / "ngboost_load_mw_conformal.json"
    assert conformal_path.exists()
    sidecar = json.loads(conformal_path.read_text(encoding="utf-8"))
    assert sidecar["meta"]["model"] == "ngboost"
    assert sidecar["meta"]["region"] == "DE"
    assert 0.0 < sidecar["meta"]["picp_90"] <= 1.0


def test_per_seed_artifacts_emitted(synthetic_splits: tuple[SplitPaths, Path]) -> None:
    if importlib.util.find_spec("ngboost") is None:
        pytest.skip("ngboost not installed in this environment")
    splits, root = synthetic_splits
    cfg = AdvancedTrainerConfig(
        region="DE",
        release_id="TEST_R2",
        splits=splits,
        out_root=root,
        targets=("load_mw",),
        seeds=(42,),
        horizon=24,
        lookback=72,
        alpha=0.10,
        holiday_country=None,
        metrics_json=root / "week2_metrics.json",
        conformal_dir=root / "uncertainty",
        enabled_models=("ngboost",),
    )
    run_advanced_baselines(cfg)
    runs_dir = root / "artifacts" / "runs" / "de" / "TEST_R2" / "advanced_baselines"
    seed_files = list(runs_dir.glob("ngboost_load_mw_seed*.json"))
    assert len(seed_files) == 1
    payload = json.loads(seed_files[0].read_text(encoding="utf-8"))
    assert payload["model"] == "ngboost"
    assert payload["region"] == "DE"
    assert payload["metrics"]["rmse"] >= 0.0
    assert "uncertainty_conformal" in payload
