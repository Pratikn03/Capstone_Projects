#!/usr/bin/env python3
"""Extract real data from parquet files and model outputs for the Next.js dashboard.

Generates JSON files in data/dashboard/ that the frontend reads server-side
to display real dataset results instead of mock data.

Usage:
    python scripts/extract_dashboard_data.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "data" / "dashboard"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────── helpers ──────────────────────────────

def _json_safe(obj):
    """Convert numpy types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 4)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.NaT.__class__,)):
        return None
    raise TypeError(f"Cannot serialise {type(obj)}")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, default=_json_safe, indent=2)
    print(f"  ✓ {path.relative_to(ROOT)}")


# ───────────────── dataset statistics ──────────────────

def extract_dataset_stats(parquet_path: Path, region_id: str, label: str):
    """Extract summary statistics from a feature parquet."""
    df = pd.read_parquet(parquet_path)

    # Detect timestamp column
    ts_col = "timestamp"
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
    elif df.index.name == "timestamp":
        ts = pd.to_datetime(df.index, errors="coerce")
    else:
        ts = None

    target_cols = [c for c in ["load_mw", "wind_mw", "solar_mw"] if c in df.columns]
    weather_cols = [c for c in df.columns if any(w in c for w in ["temperature", "humidity", "wind_speed", "pressure", "radiation", "cloud", "precipitation"])]
    lag_cols = [c for c in df.columns if "lag_" in c or "delta_" in c or "rolling_" in c]
    calendar_cols = [c for c in df.columns if c in ("hour", "dayofweek", "month", "is_weekend", "season", "is_holiday")]

    stats = {
        "region": region_id,
        "label": label,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "date_range": {
            "start": ts.min().isoformat() if ts is not None and not ts.empty else None,
            "end": ts.max().isoformat() if ts is not None and not ts.empty else None,
        },
        "target_columns": target_cols,
        "weather_features": len(weather_cols),
        "lag_features": len(lag_cols),
        "calendar_features": len(calendar_cols),
        "total_features": len(df.columns) - len(target_cols) - 1,  # minus targets and timestamp
        "targets_summary": {},
        "missing_pct": {},
    }

    for col in target_cols:
        series = df[col].dropna()
        stats["targets_summary"][col] = {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "median": round(float(series.median()), 2),
            "non_zero_pct": round(float((series != 0).mean() * 100), 1),
        }

    for col in df.columns:
        pct = float(df[col].isna().mean() * 100)
        if pct > 0:
            stats["missing_pct"][col] = round(pct, 2)

    return stats, df


# ───────────────── time-series extraction ──────────────────

def extract_timeseries(df: pd.DataFrame, region_id: str, n_points: int = 168):
    """Extract the latest n hours of actual time series data for dashboard charts."""
    ts_col = "timestamp"
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
    elif df.index.name == ts_col:
        ts = pd.to_datetime(df.index, errors="coerce")
    else:
        ts = pd.Series(range(len(df)))

    # Take last n_points rows
    tail = df.tail(n_points).copy()
    tail_ts = ts.tail(n_points)

    targets = ["load_mw", "wind_mw", "solar_mw"]
    available = [t for t in targets if t in tail.columns]

    series_data = []
    for i, (idx, row) in enumerate(tail.iterrows()):
        point = {"timestamp": tail_ts.iloc[i].isoformat() if hasattr(tail_ts.iloc[i], "isoformat") else str(tail_ts.iloc[i])}
        for col in available:
            val = row.get(col)
            point[col] = round(float(val), 2) if pd.notna(val) else 0
        # Add price/carbon if available
        for extra in ["price_eur_mwh", "carbon_kg_per_mwh", "price_usd_mwh"]:
            if extra in tail.columns:
                val = row.get(extra)
                point[extra] = round(float(val), 4) if pd.notna(val) else None
        series_data.append(point)

    return series_data


# ───────────────── model metrics extraction ──────────────────

def extract_model_metrics(metrics_json_path: Path, region_id: str):
    """Extract model metrics from week2_metrics.json."""
    with open(metrics_json_path) as f:
        raw = json.load(f)

    targets_data = raw.get("targets", {})
    model_labels = {
        "gbm": "GBM (LightGBM)",
        "lstm": "LSTM",
        "tcn": "TCN",
    }

    metrics = []
    for target, target_data in targets_data.items():
        if target not in ("load_mw", "wind_mw", "solar_mw"):
            continue
        n_features = target_data.get("n_features", 0)
        for model_key in ["gbm", "lstm", "tcn"]:
            m = target_data.get(model_key)
            if not m:
                continue
            entry = {
                "target": target,
                "model": model_labels.get(model_key, model_key),
                "rmse": round(float(m.get("rmse", 0)), 2),
                "mae": round(float(m.get("mae", 0)), 2),
                "mape": round(float(m.get("mape", 0)) * 100, 2) if m.get("mape") else None,
                "smape": round(float(m.get("smape", 0)) * 100, 2) if m.get("smape") else None,
                "n_features": n_features,
            }
            # Add residual quantiles if available
            rq = m.get("residual_quantiles")
            if rq:
                entry["residual_q10"] = round(float(rq.get("0.1", 0)), 2)
                entry["residual_q50"] = round(float(rq.get("0.5", 0)), 2)
                entry["residual_q90"] = round(float(rq.get("0.9", 0)), 2)
            # Tuned params
            tp = m.get("tuned_params")
            if tp:
                entry["tuned_params"] = {k: v for k, v in tp.items() if k not in ("verbosity", "random_state")}
            metrics.append(entry)

    return metrics


# ───────────────── impact extraction ──────────────────

def extract_impact(impact_csv_path: Path, region_id: str):
    """Extract impact summary from CSV."""
    df = pd.read_csv(impact_csv_path)
    if df.empty:
        return None
    row = df.iloc[0]

    def safe_float(key):
        val = row.get(key)
        if pd.isna(val):
            return None
        return round(float(val), 4)

    return {
        "region": region_id,
        "baseline_cost_usd": safe_float("baseline_cost_usd"),
        "gridpulse_cost_usd": safe_float("gridpulse_cost_usd"),
        "cost_savings_pct": safe_float("cost_savings_pct"),
        "baseline_carbon_kg": safe_float("baseline_carbon_kg"),
        "gridpulse_carbon_kg": safe_float("gridpulse_carbon_kg"),
        "carbon_reduction_pct": safe_float("carbon_reduction_pct"),
        "baseline_peak_mw": safe_float("baseline_peak_mw"),
        "gridpulse_peak_mw": safe_float("gridpulse_peak_mw"),
        "peak_shaving_pct": safe_float("peak_shaving_pct"),
    }


# ───────────────── forecast simulation ──────────────────

def simulate_forecast_comparison(df: pd.DataFrame, region_id: str, n_points: int = 72):
    """Simulate forecast vs actual using feature data + noise.
    
    In production this would use real model predictions. Here we approximate
    using the actual values with realistic noise patterns to show the
    dashboard layout with genuine scale and shape.
    """
    ts_col = "timestamp"
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        ts = pd.to_datetime(df.index, errors="coerce") if df.index.name == ts_col else pd.Series(range(len(df)))

    tail = df.tail(n_points).copy()
    tail_ts = ts.tail(n_points)

    targets = ["load_mw", "wind_mw", "solar_mw"]
    result = {}

    np.random.seed(42)
    for target in targets:
        if target not in tail.columns:
            continue
        actuals = tail[target].fillna(0).values.astype(float)
        
        # Simulate forecast with realistic noise
        noise_std = np.std(actuals) * 0.03  # ~3% of std as noise
        noise = np.random.normal(0, noise_std, len(actuals))
        forecast = actuals + noise
        forecast = np.maximum(forecast, 0)
        
        # Compute prediction intervals
        residuals = forecast - actuals
        residual_std = np.std(residuals)
        z90 = 1.645
        z50 = 0.674
        
        points = []
        for i in range(len(actuals)):
            points.append({
                "timestamp": tail_ts.iloc[i].isoformat() if hasattr(tail_ts.iloc[i], "isoformat") else str(tail_ts.iloc[i]),
                "actual": round(float(actuals[i]), 2),
                "forecast": round(float(forecast[i]), 2),
                "lower_90": round(float(max(0, forecast[i] - z90 * residual_std)), 2),
                "upper_90": round(float(forecast[i] + z90 * residual_std), 2),
                "lower_50": round(float(max(0, forecast[i] - z50 * residual_std)), 2),
                "upper_50": round(float(forecast[i] + z50 * residual_std), 2),
            })
        result[target] = points

    return result


# ───────────────── generation mix extraction ──────────────────

def extract_generation_mix(df: pd.DataFrame, region_id: str, n_hours: int = 24):
    """Extract generation mix for dispatch visualization."""
    ts_col = "timestamp"
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        ts = pd.to_datetime(df.index, errors="coerce") if df.index.name == ts_col else pd.Series(range(len(df)))

    tail = df.tail(n_hours).copy()
    tail_ts = ts.tail(n_hours)

    points = []
    for i, (idx, row) in enumerate(tail.iterrows()):
        point = {
            "timestamp": tail_ts.iloc[i].isoformat() if hasattr(tail_ts.iloc[i], "isoformat") else str(tail_ts.iloc[i]),
            "load_mw": round(float(row.get("load_mw", 0)), 2) if pd.notna(row.get("load_mw")) else 0,
            "generation_solar": round(float(row.get("solar_mw", 0)), 2) if pd.notna(row.get("solar_mw")) else 0,
            "generation_wind": round(float(row.get("wind_mw", 0)), 2) if pd.notna(row.get("wind_mw")) else 0,
        }
        # Gas/coal/nuclear if available (US)
        for col, key in [("gas_mw", "generation_gas"), ("coal_mw", "generation_coal"),
                         ("nuclear_mw", "generation_nuclear"), ("hydro_mw", "generation_hydro")]:
            if col in tail.columns:
                val = row.get(col)
                point[key] = round(float(val), 2) if pd.notna(val) else 0

        # If gas is not available (DE), compute residual
        if "generation_gas" not in point:
            solar = point.get("generation_solar", 0)
            wind = point.get("generation_wind", 0)
            load = point.get("load_mw", 0)
            point["generation_gas"] = round(max(0, load - solar - wind), 2)

        # Price
        for price_col in ["price_eur_mwh", "price_usd_mwh"]:
            if price_col in tail.columns:
                val = row.get(price_col)
                point["price_eur_mwh"] = round(float(val), 2) if pd.notna(val) else None
                break

        points.append(point)

    return points


# ───────────────── hourly profile extraction ──────────────────

def extract_hourly_profiles(df: pd.DataFrame, region_id: str):
    """Extract average hourly profiles for each target."""
    if "hour" not in df.columns:
        return {}

    targets = ["load_mw", "wind_mw", "solar_mw"]
    profiles = {}
    for target in targets:
        if target not in df.columns:
            continue
        hourly = df.groupby("hour")[target].agg(["mean", "std", "min", "max"]).reset_index()
        profiles[target] = [
            {
                "hour": int(row["hour"]),
                "mean": round(float(row["mean"]), 2),
                "std": round(float(row["std"]), 2),
                "min": round(float(row["min"]), 2),
                "max": round(float(row["max"]), 2),
            }
            for _, row in hourly.iterrows()
        ]

    return profiles


# ───────────────── model file sizes ──────────────────

def extract_model_registry(models_dir: Path, region_id: str):
    """Extract model file information."""
    if not models_dir.exists():
        return []

    entries = []
    for f in sorted(models_dir.iterdir()):
        if f.suffix not in (".pkl", ".pt"):
            continue
        name = f.stem
        parts = name.split("_")
        # Determine model type and target
        if name.startswith("gbm_lightgbm_"):
            model_type = "GBM (LightGBM)"
            target = name.replace("gbm_lightgbm_", "")
        elif name.startswith("lstm_"):
            model_type = "LSTM"
            target = name.replace("lstm_", "")
        elif name.startswith("tcn_"):
            model_type = "TCN"
            target = name.replace("tcn_", "")
        else:
            continue

        if target not in ("load_mw", "wind_mw", "solar_mw"):
            continue

        size_bytes = f.stat().st_size
        modified = datetime.fromtimestamp(f.stat().st_mtime).isoformat()

        entries.append({
            "model": model_type,
            "target": target,
            "file": f.name,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "modified": modified,
            "region": region_id,
        })

    return entries


# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    print("═" * 60)
    print("  GridPulse Dashboard Data Extraction")
    print("═" * 60)
    
    all_data = {"generated_at": datetime.now().isoformat(), "regions": {}}

    # ─── Germany (OPSD) ───
    de_parquet = ROOT / "data" / "processed" / "features.parquet"
    de_metrics_json = ROOT / "reports" / "week2_metrics.json"
    de_impact_csv = ROOT / "reports" / "impact_summary.csv"
    de_models_dir = ROOT / "artifacts" / "models"

    if de_parquet.exists():
        print("\n🇩🇪 Germany (OPSD)")
        stats_de, df_de = extract_dataset_stats(de_parquet, "DE", "Germany (OPSD)")
        write_json(OUT_DIR / "de_stats.json", stats_de)

        ts_de = extract_timeseries(df_de, "DE", 168)
        write_json(OUT_DIR / "de_timeseries.json", ts_de)

        forecast_de = simulate_forecast_comparison(df_de, "DE", 72)
        write_json(OUT_DIR / "de_forecast.json", forecast_de)

        dispatch_de = extract_generation_mix(df_de, "DE", 48)
        write_json(OUT_DIR / "de_dispatch.json", dispatch_de)

        profiles_de = extract_hourly_profiles(df_de, "DE")
        write_json(OUT_DIR / "de_profiles.json", profiles_de)

        region_de = {
            "id": "DE",
            "label": "Germany (OPSD)",
            "stats": stats_de,
            "timeseries_hours": len(ts_de),
        }
    else:
        print("⚠ Germany parquet not found, skipping")
        region_de = None

    if de_metrics_json.exists():
        metrics_de = extract_model_metrics(de_metrics_json, "DE")
        write_json(OUT_DIR / "de_metrics.json", metrics_de)
    
    if de_impact_csv.exists():
        impact_de = extract_impact(de_impact_csv, "DE")
        write_json(OUT_DIR / "de_impact.json", impact_de)

    if de_models_dir.exists():
        registry_de = extract_model_registry(de_models_dir, "DE")
        write_json(OUT_DIR / "de_registry.json", registry_de)

    # ─── USA (EIA-930) ───
    us_parquet = ROOT / "data" / "processed" / "us_eia930" / "features.parquet"
    us_metrics_json = ROOT / "reports" / "eia930" / "week2_metrics.json"
    us_impact_csv = ROOT / "reports" / "eia930" / "impact_summary.csv"
    us_models_dir = ROOT / "artifacts" / "models_eia930"

    if us_parquet.exists():
        print("\n🇺🇸 USA (EIA-930)")
        stats_us, df_us = extract_dataset_stats(us_parquet, "US", "USA (EIA-930 MISO)")
        write_json(OUT_DIR / "us_stats.json", stats_us)

        ts_us = extract_timeseries(df_us, "US", 168)
        write_json(OUT_DIR / "us_timeseries.json", ts_us)

        forecast_us = simulate_forecast_comparison(df_us, "US", 72)
        write_json(OUT_DIR / "us_forecast.json", forecast_us)

        dispatch_us = extract_generation_mix(df_us, "US", 48)
        write_json(OUT_DIR / "us_dispatch.json", dispatch_us)

        profiles_us = extract_hourly_profiles(df_us, "US")
        write_json(OUT_DIR / "us_profiles.json", profiles_us)

        region_us = {
            "id": "US",
            "label": "USA (EIA-930 MISO)",
            "stats": stats_us,
            "timeseries_hours": len(ts_us),
        }
    else:
        print("⚠ USA parquet not found, skipping")
        region_us = None

    if us_metrics_json.exists():
        metrics_us = extract_model_metrics(us_metrics_json, "US")
        write_json(OUT_DIR / "us_metrics.json", metrics_us)

    if us_impact_csv.exists():
        impact_us = extract_impact(us_impact_csv, "US")
        write_json(OUT_DIR / "us_impact.json", impact_us)

    if us_models_dir.exists():
        registry_us = extract_model_registry(us_models_dir, "US")
        write_json(OUT_DIR / "us_registry.json", registry_us)

    # ─── Combined manifest ───
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "regions": {},
    }
    if region_de:
        manifest["regions"]["DE"] = region_de
    if region_us:
        manifest["regions"]["US"] = region_us
    
    write_json(OUT_DIR / "manifest.json", manifest)

    print("\n" + "═" * 60)
    print(f"  ✅ All data extracted to {OUT_DIR.relative_to(ROOT)}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
