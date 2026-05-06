#!/usr/bin/env python3
"""Quick verification that training pipelines work for all regions."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REGIONS = {
    "DE": {
        "config": "configs/train_forecast.yaml",
        "splits": "data/processed/splits",
        "targets": ["load_mw", "wind_mw", "solar_mw"],
        "exclude_cols": ["timestamp", "load_mw", "wind_mw", "solar_mw", "price_eur_mwh"],
    },
    "US_MISO": {
        "config": "configs/train_forecast_eia930.yaml",
        "splits": "data/processed/us_eia930/splits",
        "targets": ["load_mw", "wind_mw", "solar_mw"],
        "exclude_cols": ["timestamp", "load_mw", "wind_mw", "solar_mw"],
    },
    "US_ERCOT": {
        "config": "configs/train_forecast_eia930_ercot.yaml",
        "splits": "data/processed/us_eia930_ercot/splits",
        "targets": ["load_mw", "wind_mw", "solar_mw"],
        "exclude_cols": ["timestamp", "load_mw", "wind_mw", "solar_mw"],
    },
    "US_PJM": {
        "config": "configs/train_forecast_eia930_pjm.yaml",
        "splits": "data/processed/us_eia930_pjm/splits",
        "targets": ["load_mw", "wind_mw", "solar_mw"],
        "exclude_cols": ["timestamp", "load_mw", "wind_mw", "solar_mw"],
    },
}


def verify_region(name, info):
    splits_dir = Path(info["splits"])
    if not splits_dir.exists():
        print(f"  ⚠️  Splits not found at {splits_dir}")
        return False

    train = pd.read_parquet(splits_dir / "train.parquet")
    val = pd.read_parquet(splits_dir / "val.parquet")
    test = pd.read_parquet(splits_dir / "test.parquet")
    print(f"  Data: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    feature_cols = [
        c
        for c in train.columns
        if c not in info["exclude_cols"] and train[c].dtype in ["float64", "float32", "int64", "int32"]
    ]
    print(f"  Features: {len(feature_cols)}")

    for target in info["targets"]:
        if target not in train.columns:
            print(f"  ⚠️  Target {target} not in data")
            continue

        X_tr = train[feature_cols].values
        y_tr = train[target].values
        X_te = test[feature_cols].values
        y_te = test[target].values

        model = lgb.LGBMRegressor(
            n_estimators=30,
            learning_rate=0.1,
            max_depth=4,
            num_leaves=16,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        mae = mean_absolute_error(y_te, preds)
        denom = np.clip(np.abs(y_te), 1.0, None)
        mape = np.mean(np.abs((y_te - preds) / denom)) * 100
        print(f"  {target}: RMSE={rmse:.1f}  MAE={mae:.1f}  MAPE={mape:.1f}%")

    return True


def main():
    print("=" * 60)
    print("  QUICK TRAINING PIPELINE VERIFICATION")
    print("=" * 60)

    results = {}
    for name, info in REGIONS.items():
        print(f"\n--- {name} ---")
        try:
            ok = verify_region(name, info)
            results[name] = "✅" if ok else "⚠️"
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[name] = "❌"

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, status in results.items():
        print(f"  {status} {name}")

    # Also check existing model artifacts
    print("\n--- Existing Model Artifacts ---")
    for d in [
        "artifacts/models",
        "artifacts/models_eia930",
        "artifacts/models_eia930_ercot",
        "artifacts/models_eia930_pjm",
    ]:
        p = Path(d)
        if p.exists():
            models = sorted(p.glob("*.pkl")) + sorted(p.glob("*.pt"))
            print(f"  {d}: {len(models)} models")
            for m in models:
                print(f"    {m.name} ({m.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  {d}: NOT FOUND")

    # Check CPSBench runner
    print("\n--- CPSBench Quick Check ---")
    try:
        import orius.cpsbench_iot.runner as runner

        if not hasattr(runner, "run_single"):
            raise RuntimeError("orius.cpsbench_iot.runner.run_single is missing")
        print("  ✅ CPSBench runner importable")
    except Exception as e:
        print(f"  ❌ CPSBench import error: {e}")

    # Check table builder
    print("\n--- Table Builder Check ---")
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("builder", "scripts/build_camera_ready_tables.py")
        if spec is None:
            raise RuntimeError("could not load build_camera_ready_tables.py spec")
        print("  ✅ build_camera_ready_tables.py found")
    except Exception as e:
        print(f"  ❌ Table builder error: {e}")

    # Check paper table CSVs
    print("\n--- Paper Table CSVs ---")
    csv_dir = Path("paper/assets/tables")
    for csv in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv)
        print(f"  {csv.name}: {len(df)} rows, {len(df.columns)} cols")

    all_ok = all(s == "✅" for s in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
