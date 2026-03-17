import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import yaml

from orius.forecasting.train import main as train_main
import orius.pipeline.run as pr

class MockGBM(dict):
    def predict(self, X):
        return np.ones(len(X))

def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

def test_boost_coverage_train_and_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path
    
    # 1. Setup minimal directory structure
    (repo / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (repo / "data" / "processed" / "splits").mkdir(parents=True, exist_ok=True)
    (repo / "reports").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "models").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "scalers").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "uncertainty").mkdir(parents=True, exist_ok=True)
    (repo / "artifacts" / "backtests").mkdir(parents=True, exist_ok=True)
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    
    # 2. Setup tiny mock data
    n_rows = 60
    ts = pd.date_range("2024-01-01", periods=n_rows, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "load_mw": np.linspace(100.0, 120.0, n_rows),
        "wind_mw": np.full(n_rows, 25.0),
        "solar_mw": np.full(n_rows, 15.0),
        "price_eur_mwh": np.full(n_rows, 50.0),
        "carbon_kg_per_mwh": np.full(n_rows, 400.0),
    })
    for i in range(1, 4):
        df[f"load_mw_lag_{i}"] = df["load_mw"].shift(i).bfill()
        df[f"wind_mw_lag_{i}"] = df["wind_mw"].shift(i).bfill()
        df[f"solar_mw_lag_{i}"] = df["solar_mw"].shift(i).bfill()
    
    df.to_parquet(repo / "data" / "processed" / "features.parquet")
    df.iloc[:30].to_parquet(repo / "data" / "processed" / "splits" / "train.parquet")
    df.iloc[30:45].to_parquet(repo / "data" / "processed" / "splits" / "val.parquet")
    df.iloc[45:].to_parquet(repo / "data" / "processed" / "splits" / "test.parquet")
    
    # 3. Setup configs for training
    _write_yaml(repo / "configs" / "train_forecast.yaml", {
        "features": {
            "target": ["load_mw", "wind_mw", "solar_mw"],
            "categorical": [],
            "drop": ["timestamp"]
        },
        "data": {
            "processed_path": str(repo / "data" / "processed" / "features.parquet"),
            "timestamp_col": "timestamp"
        },
        "cross_validation": {"enabled": False},
        "task": {"horizon_hours": 4},
        "tuning": {"enabled": False},
        "artifacts": {
            "out_dir": str(repo / "artifacts" / "models"),
            "scaler_dir": str(repo / "artifacts" / "scalers")
        },
        "reports": {
            "out_dir": str(repo / "reports")
        },
        "models": {
            "baseline_gbm": {"enabled": True, "params": {"n_estimators": 2, "max_depth": 2}},
            "quantile_gbm": {"enabled": False},
            "dl_lstm": {"enabled": False},
            "dl_tcn": {"enabled": False}
        },
        "splits": {"train_ratio": 0.5, "val_ratio": 0.25}
    })
    
    _write_yaml(repo / "configs" / "uncertainty.yaml", {
        "enabled": True,
        "mode": "conformal",
        "conformal": {"alpha": 0.10, "eps": 1e-6},
        "artifacts_dir": str(repo / "artifacts" / "uncertainty")
    })
    
    # 4. Mock subprocess.run and execute train.py main
    import subprocess
    
    from orius.forecasting import train as train_module
    
    # Mock hardcoded config loader in train.py so it reads from our tmp repo
    def mock_load_uncertainty_cfg(path: str = "configs/uncertainty.yaml") -> dict:
        mock_path = repo / "configs" / "uncertainty.yaml"
        if not mock_path.exists():
            return {"enabled": False}
        return yaml.safe_load(mock_path.read_text(encoding="utf-8")) or {"enabled": False}

    monkeypatch.setattr(train_module, "_load_uncertainty_cfg", mock_load_uncertainty_cfg)
    
    # Patch subprocess to not spawn anything real
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr=""))
    
    # Execute train.py with absolute path to config
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(repo / "configs" / "train_forecast.yaml")])
    train_main()
    
    assert (repo / "artifacts" / "models" / "gbm_lightgbm_load_mw.pkl").exists()

    # Ensure dummy models exist for the pipeline to load
    import joblib
    for t in ["load_mw", "wind_mw", "solar_mw"]:
        bundle = {
            "model": MockGBM(),
            "target": t,
            "model_type": "gbm",
            "feature_cols": ["load_mw", "wind_mw", "solar_mw", "price_eur_mwh", "carbon_kg_per_mwh"]
        }
        joblib.dump(bundle, repo / "artifacts" / "models" / f"gbm_lightgbm_{t}.pkl")

    # 5. Setup configs for pipeline.run
    _write_yaml(repo / "configs" / "forecast.yaml", {
        "models": {
            "load_mw": str(repo / "artifacts/models/gbm_lightgbm_load_mw.pkl"),
            "wind_mw": str(repo / "artifacts/models/gbm_lightgbm_wind_mw.pkl"),
            "solar_mw": str(repo / "artifacts/models/gbm_lightgbm_solar_mw.pkl"),
        },
        "fallback_order": ["gbm"]
    })
    _write_yaml(repo / "configs" / "optimization.yaml", {
        "battery": {"capacity_mwh": 100.0, "max_power_mw": 50.0, "efficiency": 0.95, "min_soc_mwh": 0.0, "initial_soc_mwh": 50.0},
        "grid": {"max_import_mw": 1500.0, "price_per_mwh": 50.0},
        "robust": {"risk_weight_worst_case": 0.5},
        "solver_name": "appsi_highs"
    })
    _write_yaml(repo / "configs" / "data.yaml", {})
    
    # We patch _repo_root because pipeline expects to be at project root
    monkeypatch.setattr(pr, "_repo_root", lambda: repo)
    monkeypatch.setattr(sys, "argv", ["run.py", "--steps", "research", "--run-id", "test-cov-boost", "--research-datasets", "de"])
    
    pr.main()
    
    print("Listing files in:", repo)
    for p in repo.rglob("*"):
        print(" -", p)
    
    assert (repo / "reports" / "research_metrics_de.csv").exists()
