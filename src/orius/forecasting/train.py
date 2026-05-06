"""
Forecasting: Model Training Pipeline.

This is the main training orchestration module for ORIUS forecasting models.
It handles the complete training workflow:

1. **Data Loading**: Load preprocessed features from parquet files
2. **Model Training**: Train LightGBM (fast) and optional deep learning models
3. **Hyperparameter Tuning**: Optuna-based hyperparameter optimization (optional)
4. **Evaluation**: Compute metrics on validation and test sets
5. **Artifact Persistence**: Save models, scalers, and evaluation reports
6. **Uncertainty Quantification**: Conformal prediction intervals (optional)

Supported Models:
    - LightGBM: Gradient boosting (fast, accurate, handles missing data)
    - LSTM: Long Short-Term Memory (captures long-range dependencies)
    - TCN: Temporal Convolutional Network (parallel training, no vanishing gradients)

Training Modes:
    - Quick: LightGBM only with default hyperparameters (~2 minutes)
    - Full: All models with hyperparameter tuning (~30 minutes)
    - Deep Learning: LSTM/TCN with early stopping (~15 minutes)

Usage:
    # Train with default config
    python -m orius.forecasting.train --config configs/train_forecast.yaml

    # Train US EIA-930 data
    python -m orius.forecasting.train --config configs/train_forecast_eia930.yaml

    # Train with Optuna tuning
    python -m orius.forecasting.train --config configs/train_forecast.yaml --tune

Configuration:
    See configs/train_forecast.yaml for full configuration options including:
    - Target variables (load_mw, solar_mw, wind_mw, price_eur_mwh)
    - Lag features and rolling windows
    - Model hyperparameters
    - Evaluation settings

Outputs:
    - artifacts/models/*.pkl: Trained model files
    - artifacts/scalers/*.pkl: Feature scalers
    - reports/*.md: Evaluation reports
    - artifacts/backtests/*.npz: Backtest results for uncertainty quantification

See Also:
    - configs/train_forecast.yaml: Configuration reference
    - forecasting/ml_gbm.py: GBM model implementation
    - forecasting/dl_lstm.py: LSTM model implementation
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from orius.forecasting.backtest import walk_forward_horizon_metrics
from orius.forecasting.evaluate import time_series_cv_score
from orius.forecasting.ml_gbm import predict_gbm, train_gbm
from orius.forecasting.train_config import (
    resolve_gap_hours as _resolve_gap_hours,
)
from orius.forecasting.train_config import (
    resolve_split_cfg as _resolve_split_cfg,
)
from orius.forecasting.train_config import (
    resolve_task_horizon as _resolve_task_horizon,
)
from orius.forecasting.train_config import (
    resolve_task_lookback as _resolve_task_lookback,
)
from orius.forecasting.uncertainty.conformal import ConformalConfig, ConformalInterval, save_conformal
from orius.utils.manifest import create_run_manifest, print_manifest_summary
from orius.utils.metrics import daylight_mape, mae, mape, r2_score, rmse, smape
from orius.utils.scaler import StandardScaler
from orius.utils.seed import set_seed

# Lazy imports for torch-dependent modules (allows GBM-only training without torch)
torch = None  # type: ignore[assignment]
DataLoader = None  # type: ignore[assignment,misc]
SeqConfig = None  # type: ignore[assignment]
TimeSeriesWindowDataset = None  # type: ignore[assignment]
LSTMForecaster = None  # type: ignore[assignment]
NBEATSForecaster = None  # type: ignore[assignment]
PatchTSTForecaster = None  # type: ignore[assignment]
TFTForecaster = None  # type: ignore[assignment]
TCNForecaster = None  # type: ignore[assignment]


def _ensure_torch():
    """Import torch and DL modules on first use."""
    global torch, DataLoader, SeqConfig, TimeSeriesWindowDataset
    global LSTMForecaster, NBEATSForecaster, PatchTSTForecaster, TFTForecaster, TCNForecaster
    if torch is not None:
        return
    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader

    from orius.forecasting.datasets import SeqConfig as _SeqConfig
    from orius.forecasting.datasets import TimeSeriesWindowDataset as _TSWD
    from orius.forecasting.dl_lstm import LSTMForecaster as _LSTM
    from orius.forecasting.dl_nbeats import NBEATSForecaster as _NBEATS
    from orius.forecasting.dl_patchtst import PatchTSTForecaster as _PatchTST
    from orius.forecasting.dl_tcn import TCNForecaster as _TCN
    from orius.forecasting.dl_tft import TFTForecaster as _TFT

    torch = _torch
    DataLoader = _DataLoader
    SeqConfig = _SeqConfig
    TimeSeriesWindowDataset = _TSWD
    LSTMForecaster = _LSTM
    NBEATSForecaster = _NBEATS
    PatchTSTForecaster = _PatchTST
    TFTForecaster = _TFT
    TCNForecaster = _TCN


def _select_torch_device() -> str:
    """Pick the fastest available PyTorch device for local training."""
    requested = os.environ.get("ORIUS_TORCH_DEVICE", "auto").strip().lower()

    try:
        _ensure_torch()
    except ImportError:
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        warnings.warn(
            "ORIUS_TORCH_DEVICE=cuda requested but CUDA is unavailable; falling back to CPU.",
            stacklevel=2,
        )
        return "cpu"
    if requested == "mps":
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        warnings.warn(
            "ORIUS_TORCH_DEVICE=mps requested but MPS is unavailable; falling back to CPU.",
            stacklevel=2,
        )
        return "cpu"
    if requested not in {"", "auto"}:
        warnings.warn(
            f"Unrecognised ORIUS_TORCH_DEVICE={requested!r}; using automatic device selection.",
            stacklevel=2,
        )

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _apply_gbm_thread_limits(params: dict[str, Any], threads: int | None) -> dict[str, Any]:
    """Apply a per-model LightGBM thread cap without mutating caller params."""
    updated = dict(params)
    if threads is not None and int(threads) > 0:
        updated["n_jobs"] = int(threads)
    return updated


def _apply_deep_training_overrides(
    cfg: dict[str, Any],
    *,
    max_epochs: int | None,
    patience: int | None,
    warmup_epochs: int | None,
) -> None:
    """Cap deep-model budgets for smoke/fast profiles without editing YAML files."""
    models_cfg = cfg.get("models", {})
    if not isinstance(models_cfg, dict):
        return
    for model_key in ("dl_lstm", "dl_tcn", "dl_nbeats", "dl_tft", "dl_patchtst"):
        model_cfg = models_cfg.get(model_key)
        if not isinstance(model_cfg, dict):
            continue
        params = model_cfg.get("params")
        if not isinstance(params, dict):
            continue
        if max_epochs is not None and int(max_epochs) > 0:
            params["epochs"] = min(int(params.get("epochs", max_epochs)), int(max_epochs))
        if warmup_epochs is not None and int(warmup_epochs) > 0:
            params["warmup_epochs"] = min(
                int(params.get("warmup_epochs", warmup_epochs)),
                int(warmup_epochs),
            )
        if patience is not None and int(patience) > 0:
            early_stopping = params.get("early_stopping")
            if not isinstance(early_stopping, dict):
                early_stopping = {}
                params["early_stopping"] = early_stopping
            early_stopping["patience"] = min(
                int(early_stopping.get("patience", patience)),
                int(patience),
            )


def _load_reusable_gbm_params(path: str | None) -> dict[str, dict[str, Any]]:
    """Load target-specific tuned GBM params from a previous metrics JSON."""
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    targets = payload.get("targets", {}) if isinstance(payload, dict) else {}
    if not isinstance(targets, dict):
        return {}
    reusable: dict[str, dict[str, Any]] = {}
    for target, target_payload in targets.items():
        if not isinstance(target_payload, dict):
            continue
        gbm = target_payload.get("gbm", {})
        if not isinstance(gbm, dict):
            continue
        tuned = gbm.get("tuned_params")
        if isinstance(tuned, dict):
            reusable[str(target)] = dict(tuned)
    return reusable


# ============================================================================
# OPTIONAL IMPORTS (graceful degradation if not installed)
# ============================================================================


def _try_optuna(*, required: bool = False):
    """
    Import Optuna for hyperparameter optimization.

    Optuna is optional - if not installed, tuning will be skipped
    and default hyperparameters will be used.
    """
    try:
        import optuna  # type: ignore

        return optuna
    except Exception as exc:
        if required:
            raise RuntimeError(
                "Optuna is required for tuning but is not installed. Install dependencies from requirements.lock.txt."
            ) from exc
        return None


def _try_lightgbm(*, required: bool = False):
    """Import LightGBM if available (needed for quantile GBM and tuning)."""
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception as exc:
        if required:
            raise RuntimeError(
                "LightGBM is required for this configuration but is not installed. Install dependencies from requirements.lock.txt."
            ) from exc
        return None


def _load_uncertainty_cfg(path: str = "configs/uncertainty.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {"enabled": False}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {"enabled": False}


def _merge_uncertainty_cfg(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Merge dataset uncertainty overrides into the global uncertainty config."""
    merged = dict(base or {})
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_uncertainty_cfg(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


_CONFORMAL_CONFIG_KEYS = {field.name for field in fields(ConformalConfig)}


def _conformal_config_for_target(uncertainty_cfg: dict[str, Any], target: str) -> ConformalConfig:
    """Resolve global conformal settings plus optional target-specific safety overrides."""
    base = (
        dict(uncertainty_cfg.get("conformal", {}))
        if isinstance(uncertainty_cfg.get("conformal"), dict)
        else {}
    )
    overrides = uncertainty_cfg.get("target_overrides", {})
    target_override = overrides.get(target, {}) if isinstance(overrides, dict) else {}
    if isinstance(target_override, dict):
        nested = target_override.get("conformal")
        if isinstance(nested, dict):
            base.update(nested)
        else:
            base.update(
                {key: value for key, value in target_override.items() if key in _CONFORMAL_CONFIG_KEYS}
            )
    base = {key: value for key, value in base.items() if key in _CONFORMAL_CONFIG_KEYS}
    return ConformalConfig(**base)


def _conformal_method_for_target(uncertainty_cfg: dict[str, Any], target: str) -> str:
    return str(_conformal_config_for_target(uncertainty_cfg, target).method).lower()


def _parse_uncertainty_targets(cfg: dict, default_targets: list[str]) -> list[str]:
    targets = cfg.get("targets")
    if targets:
        if isinstance(targets, str):
            if targets.strip().lower() == "all":
                return default_targets
            return [t.strip() for t in targets.split(",") if t.strip()]
        if isinstance(targets, list):
            return [str(t).strip() for t in targets if str(t).strip()]
    target = cfg.get("target", "load_mw")
    return [str(target).strip()]


def _format_uncertainty_path(template: str, target: str, suffix: str, multi: bool) -> Path:
    if "{target}" in template:
        return Path(template.format(target=target))
    if multi:
        return Path(f"artifacts/backtests/{target}_{suffix}.npz")
    return Path(template)


def _sha256_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _enforce_data_contract(
    features_path: Path,
    *,
    validation_report_path: Path,
    data_manifest_output: Path,
    required_cols: list[str] | None = None,
    timestamp_col: str | None = "timestamp",
    order_cols: list[str] | None = None,
    skip_data_manifest: bool = False,
) -> dict[str, object]:
    """Validate feature schema and refresh dataset identity manifest before training."""
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run build_features first.")

    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    validate_cmd = [
        sys.executable,
        "-m",
        "orius.data_pipeline.validate_schema",
        "--in",
        str(features_path),
        "--report",
        str(validation_report_path),
    ]
    if required_cols:
        validate_cmd.extend(["--required-cols", ",".join(required_cols)])
    if timestamp_col:
        validate_cmd.extend(["--timestamp-col", timestamp_col])
    if order_cols:
        validate_cmd.extend(["--order-cols", ",".join(order_cols)])
    subprocess.run(validate_cmd, check=True)

    if skip_data_manifest:
        return {
            "path": data_manifest_output,
            "sha256": None,
            "schema_hash": None,
            "split_boundaries": {},
        }

    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "build_data_manifest.py"
    data_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dataset",
            "ALL",
            "--output",
            str(data_manifest_output),
        ],
        cwd=str(repo_root),
        check=True,
    )

    payload: dict[str, object] = {}
    if data_manifest_output.exists():
        try:
            payload = json.loads(data_manifest_output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    return {
        "path": data_manifest_output,
        "sha256": _sha256_file(data_manifest_output) if data_manifest_output.exists() else None,
        "schema_hash": payload.get("schema_hash"),
        "split_boundaries": payload.get("split_boundaries", {}),
    }


def _reshape_horizon(
    y_true: np.ndarray, y_pred: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray] | None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = min(len(y_true), len(y_pred))
    if n < horizon:
        return None
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    step_true = []
    step_pred = []
    min_len = None
    for h in range(horizon):
        idx = np.arange(h, n, horizon)
        yt = y_true[idx]
        yp = y_pred[idx]
        if len(yt) == 0:
            continue
        min_len = len(yt) if min_len is None else min(min_len, len(yt))
        step_true.append(yt)
        step_pred.append(yp)

    if min_len is None or min_len == 0 or len(step_true) != horizon:
        return None

    y_true_mat = np.column_stack([arr[:min_len] for arr in step_true])
    y_pred_mat = np.column_stack([arr[:min_len] for arr in step_pred])
    return y_true_mat, y_pred_mat


def _sample_param(trial, spec: dict):
    """Sample one hyperparameter from an Optuna trial spec."""
    kind = spec.get("type")
    if kind == "int":
        return trial.suggest_int(spec["name"], int(spec["low"]), int(spec["high"]))
    if kind == "float":
        return trial.suggest_float(
            spec["name"], float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
        )
    if kind == "categorical":
        return trial.suggest_categorical(spec["name"], spec.get("choices", []))
    return None


def _aggregate_top_trial_params(trials, param_specs: list[dict], top_pct: float, min_top_trials: int) -> dict:
    """Aggregate top trial params using median (numeric) / mode (categorical)."""
    if not trials:
        return {}

    k = max(min_top_trials, int(np.ceil(len(trials) * top_pct)))
    k = min(k, len(trials))
    top_trials = trials[:k]

    agg: dict[str, object] = {}
    for spec in param_specs:
        name = str(spec.get("name"))
        kind = str(spec.get("type", ""))
        values = [t.params[name] for t in top_trials if name in t.params]
        if not values:
            continue

        if kind == "int":
            agg[name] = int(round(float(np.median([float(v) for v in values]))))
        elif kind == "float":
            agg[name] = float(np.median([float(v) for v in values]))
        else:
            # categorical and unknown fallback to mode
            agg[name] = Counter(values).most_common(1)[0][0]
    return agg


def tune_gbm_params(
    X_train,
    y_train,
    X_val,
    y_val,
    base_params: dict,
    tuning_cfg: dict,
    seed: int,
):
    """Tune GBM hyperparameters on the validation split."""
    optuna = _try_optuna(required=True)
    lgb = _try_lightgbm(required=True)

    params_cfg = tuning_cfg.get("params", {}).get("baseline_gbm", {})
    n_trials = int(tuning_cfg.get("n_trials", 20))
    metric = tuning_cfg.get("metric", "val_loss")
    selection_mode = str(tuning_cfg.get("selection_mode", "top_pct_median")).lower()
    top_pct = float(tuning_cfg.get("select_top_pct", 0.30))
    min_top_trials = int(tuning_cfg.get("min_top_trials", 5))
    n_jobs = max(1, int(tuning_cfg.get("n_jobs", 1) or 1))
    if not (0 < top_pct <= 1.0):
        raise ValueError(f"tuning.select_top_pct must be in (0, 1], got {top_pct}")

    # Normalize params spec to include name for sampling.
    param_specs = []
    for name, spec in params_cfg.items():
        if isinstance(spec, dict):
            spec = {**spec, "name": name}
            param_specs.append(spec)

    def objective(trial):
        trial_params = {}
        for spec in param_specs:
            trial_params[spec["name"]] = _sample_param(trial, spec)

        params = {**base_params, **trial_params, "random_state": seed}
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        if metric == "mape":
            return float(mape(y_val, pred))
        return float(rmse(y_val, pred))

    study = optuna.create_study(direction="minimize")
    seed_param_payload = tuning_cfg.get("seed_params")
    seed_param_rows = seed_param_payload if isinstance(seed_param_payload, list) else [seed_param_payload]
    enqueued = 0
    for row in seed_param_rows:
        if not isinstance(row, dict):
            continue
        trial_seed = {spec["name"]: row[spec["name"]] for spec in param_specs if spec["name"] in row}
        if trial_seed:
            study.enqueue_trial(trial_seed)
            enqueued += 1
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=n_jobs)
    complete = [t for t in study.trials if t.value is not None]
    complete = sorted(complete, key=lambda t: float(t.value))

    selected = study.best_params
    if selection_mode == "top_pct_median" and complete:
        selected = _aggregate_top_trial_params(
            complete, param_specs, top_pct=top_pct, min_top_trials=min_top_trials
        )

    tuning_meta = {
        "enabled": True,
        "n_trials": int(n_trials),
        "metric": metric,
        "selection_mode": selection_mode,
        "select_top_pct": top_pct,
        "min_top_trials": min_top_trials,
        "n_jobs": n_jobs,
        "enqueued_seed_trials": enqueued,
        "n_complete_trials": len(complete),
        "best_objective": float(study.best_value) if study.best_value is not None else None,
        "selected_params": selected,
    }
    return {**base_params, **selected}, tuning_meta


def _configured_order_columns(data_cfg: dict[str, Any], df: pd.DataFrame) -> list[str]:
    """Resolve configured ordering columns for time series or replay tables."""
    raw_order_cols = data_cfg.get("order_cols")
    if isinstance(raw_order_cols, list) and raw_order_cols:
        cols = [str(col).strip() for col in raw_order_cols if str(col).strip()]
    elif raw_order_cols:
        cols = [str(raw_order_cols).strip()]
    else:
        timestamp_col = data_cfg.get("timestamp_col", "timestamp")
        cols = [str(timestamp_col).strip()] if timestamp_col else []
    if not cols:
        raise ValueError("Training data must configure data.timestamp_col or data.order_cols.")
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Configured order columns missing from data: {missing}")
    return cols


def _sort_training_frame(df: pd.DataFrame, data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Sort training rows by configured temporal or replay order."""
    return df.sort_values(_configured_order_columns(data_cfg, df)).reset_index(drop=True)


def _configured_non_feature_columns(data_cfg: dict[str, Any]) -> set[str]:
    """Resolve configured non-feature columns that must not enter models."""
    cols: set[str] = {"timestamp", "ts_utc"}
    raw_order_cols = data_cfg.get("order_cols")
    if isinstance(raw_order_cols, list):
        cols.update(str(col).strip() for col in raw_order_cols if str(col).strip())
    elif raw_order_cols:
        cols.add(str(raw_order_cols).strip())
    timestamp_col = data_cfg.get("timestamp_col")
    if timestamp_col:
        cols.add(str(timestamp_col).strip())
    raw_non_feature_cols = data_cfg.get("non_feature_cols")
    if isinstance(raw_non_feature_cols, list):
        cols.update(str(col).strip() for col in raw_non_feature_cols if str(col).strip())
    elif raw_non_feature_cols:
        cols.add(str(raw_non_feature_cols).strip())
    return cols


def make_xy(
    df: pd.DataFrame,
    target: str,
    targets: list[str],
    *,
    non_feature_cols: set[str] | None = None,
):
    """Split a dataframe into features (X) and a single target (y)."""
    # Drop all targets and non-feature columns from features to prevent leakage.
    drop = {"timestamp", "ts_utc", *targets, *(non_feature_cols or set())}
    feat_cols = [c for c in df.columns if c not in drop]
    # Exclude non-numeric columns (object, datetime, etc.) so models receive valid inputs.
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    feature_frame = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    X = feature_frame.to_numpy()
    y = pd.to_numeric(df[target], errors="coerce").astype(np.float32).to_numpy()
    return X, y, feat_cols


def _drop_invalid_target_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where the supervised target is non-finite."""
    valid = np.isfinite(y)
    if valid.all():
        return X, y
    return X[valid], y[valid]


def _fit_feature_fill_values(X_train: np.ndarray) -> np.ndarray:
    """Fit train-split feature fill values for missing/non-finite entries."""
    arr = np.asarray(X_train, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {arr.shape}")
    safe = np.where(np.isfinite(arr), arr, np.nan)
    with np.errstate(all="ignore"):
        fill_values = np.nanmedian(safe, axis=0)
    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0).astype(np.float32)
    return fill_values


def _apply_feature_fill_values(X: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    """Apply train-derived feature fill values to any missing/non-finite cells."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {arr.shape}")
    if fill_values.shape[0] != arr.shape[1]:
        raise ValueError(f"Feature fill vector has length {fill_values.shape[0]}, expected {arr.shape[1]}")
    invalid = ~np.isfinite(arr)
    if not invalid.any():
        return arr
    arr = arr.copy()
    _, col_idx = np.where(invalid)
    arr[invalid] = fill_values[col_idx]
    return arr


def residual_quantiles(y_true: np.ndarray, y_pred: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    """Compute residual quantiles for simple prediction intervals."""
    resid = np.asarray(y_true) - np.asarray(y_pred)
    return {str(q): float(np.quantile(resid, q)) for q in quantiles}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    """Compute standard regression metrics (plus daylight MAPE for solar)."""
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    if target in ("solar_mw", "wind_mw"):
        # Solar & wind have near-zero values that inflate MAPE to billions.
        # Daylight MAPE (non-zero only) is the usable metric.
        out["daylight_mape"] = daylight_mape(y_true, y_pred)
    return out


def _generalization_summary(
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, float | None]:
    """Summarize train/validation/test RMSE gaps for overfit and drift gates."""

    def finite(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    def ratio(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or abs(denominator) < 1.0e-12:
            return None
        return round(float(numerator / denominator), 6)

    train_rmse = finite(train_metrics.get("rmse"))
    val_rmse = finite(validation_metrics.get("rmse"))
    test_rmse = finite(test_metrics.get("rmse"))
    return {
        "train_rmse": train_rmse,
        "validation_rmse": val_rmse,
        "test_rmse": test_rmse,
        "train_validation_rmse_ratio": ratio(val_rmse, train_rmse),
        "validation_test_rmse_ratio": ratio(test_rmse, val_rmse),
        "train_test_rmse_ratio": ratio(test_rmse, train_rmse),
    }


def _prediction_latency_summary(
    predict_fn,
    X: np.ndarray,
    *,
    repeats: int = 5,
    max_rows: int = 2048,
) -> dict[str, float | int]:
    """Measure lightweight batch and per-sample prediction latency."""
    arr = np.asarray(X)
    if arr.ndim == 0:
        raise ValueError("latency input must contain at least one sample")
    sample = arr[: max(1, min(len(arr), int(max_rows)))]
    durations_ms: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        predict_fn(sample)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    batch = np.asarray(durations_ms, dtype=float)
    per_sample = batch / max(1, len(sample))
    return {
        "sample_rows": int(len(sample)),
        "repeats": int(max(1, repeats)),
        "mean_batch_ms": float(np.mean(batch)),
        "p95_batch_ms": float(np.percentile(batch, 95)),
        "mean_per_sample_ms": float(np.mean(per_sample)),
        "p95_per_sample_ms": float(np.percentile(per_sample, 95)),
    }


def _training_history_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize DL loss and gradient behavior for stability gates."""

    def finite_values(key: str) -> list[float]:
        values: list[float] = []
        for row in history:
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return values

    train_losses = finite_values("train_loss")
    val_losses = finite_values("val_loss")
    max_grad_norms = finite_values("max_grad_norm")
    clipped_batches = sum(int(row.get("clipped_batches", 0) or 0) for row in history)
    total_batches = sum(int(row.get("num_batches", 0) or 0) for row in history)
    non_finite_loss = any(
        not np.isfinite(float(row.get(key)))
        for row in history
        for key in ("train_loss", "val_loss")
        if row.get(key) is not None
    )
    return {
        "epochs_ran": int(len(history)),
        "best_val_loss": float(min(val_losses)) if val_losses else None,
        "last_train_loss": float(train_losses[-1]) if train_losses else None,
        "last_val_loss": float(val_losses[-1]) if val_losses else None,
        "non_finite_loss": bool(non_finite_loss),
        "gradient_clipped_fraction": round(float(clipped_batches / total_batches), 6)
        if total_batches
        else None,
        "max_grad_norm": float(max(max_grad_norms)) if max_grad_norms else None,
    }


def _normalize_quantile_levels(quantiles: list[float] | None) -> list[float]:
    return [float(q) for q in (quantiles or [])]


def _median_quantile_index(quantile_levels: list[float]) -> int:
    if not quantile_levels:
        return 0
    arr = np.asarray(quantile_levels, dtype=float)
    return int(np.argmin(np.abs(arr - 0.5)))


def _model_artifact_key(model_type: str) -> str:
    return "gbm" if str(model_type).startswith("gbm") else str(model_type)


def _extract_tensor_forecast(
    pred_seq: torch.Tensor, horizon: int, quantile_levels: list[float] | None = None
) -> torch.Tensor:
    levels = _normalize_quantile_levels(quantile_levels)
    n_quantiles = len(levels)
    if n_quantiles > 1:
        if pred_seq.dim() == 2:
            if pred_seq.shape[1] % n_quantiles != 0:
                raise ValueError(
                    f"Cannot reshape prediction tensor with shape {tuple(pred_seq.shape)} into quantiles"
                )
            pred_seq = pred_seq.view(pred_seq.shape[0], -1, n_quantiles)
        elif pred_seq.dim() != 3:
            raise ValueError(f"Expected 2D/3D quantile prediction tensor, got {pred_seq.dim()}D")
        if pred_seq.shape[-1] != n_quantiles:
            if pred_seq.shape[1] == n_quantiles:
                pred_seq = pred_seq.transpose(1, 2)
            else:
                raise ValueError(
                    f"Quantile prediction tensor shape {tuple(pred_seq.shape)} does not match {n_quantiles} quantiles"
                )
        pred_hz = pred_seq[:, -horizon:, :]
        pred_hz, _ = torch.sort(pred_hz, dim=-1)
        return pred_hz
    if pred_seq.dim() == 3 and pred_seq.shape[-1] == 1:
        pred_seq = pred_seq[..., 0]
    return pred_seq[:, -horizon:]


def _extract_numpy_forecast(
    pred_seq: np.ndarray,
    horizon: int,
    quantile_levels: list[float] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    levels = _normalize_quantile_levels(quantile_levels)
    n_quantiles = len(levels)
    arr = np.asarray(pred_seq, dtype=float)
    if n_quantiles > 1:
        if arr.ndim == 2:
            if arr.shape[1] % n_quantiles != 0:
                raise ValueError(f"Cannot reshape prediction array with shape {arr.shape} into quantiles")
            arr = arr.reshape(arr.shape[0], -1, n_quantiles)
        elif arr.ndim != 3:
            raise ValueError(f"Expected 2D/3D quantile prediction array, got {arr.ndim}D")
        if arr.shape[-1] != n_quantiles:
            if arr.shape[1] == n_quantiles:
                arr = np.transpose(arr, (0, 2, 1))
            else:
                raise ValueError(
                    f"Quantile prediction array shape {arr.shape} does not match {n_quantiles} quantiles"
                )
        arr = np.sort(arr[:, -horizon:, :], axis=-1)
        median_idx = _median_quantile_index(levels)
        point = arr[:, :, median_idx]
        quantile_map = {str(q): arr[:, :, idx] for idx, q in enumerate(levels)}
        return point, quantile_map
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    point = arr[:, -horizon:]
    return point, {}


def _pinball_loss(pred_hz: torch.Tensor, y_true: torch.Tensor, quantile_levels: list[float]) -> torch.Tensor:
    q = torch.as_tensor(quantile_levels, dtype=pred_hz.dtype, device=pred_hz.device).view(1, 1, -1)
    target = y_true.unsqueeze(-1)
    err = target - pred_hz
    return torch.maximum((q - 1.0) * err, q * err).mean()


def _build_conformal_payload(
    *,
    target: str,
    model_key: str,
    calibration_source: str,
    split_boundaries: dict[str, Any],
    method: str,
    y_true_cal: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_cal: np.ndarray | None = None,
    y_pred_test: np.ndarray | None = None,
    q_lo_cal: np.ndarray | None = None,
    q_hi_cal: np.ndarray | None = None,
    q_lo_test: np.ndarray | None = None,
    q_hi_test: np.ndarray | None = None,
    quantile_lo_key: str | None = None,
    quantile_hi_key: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "target": target,
        "model": model_key,
        "artifact_model_key": _model_artifact_key(model_key),
        "calibration_source": calibration_source,
        "split_boundaries": split_boundaries,
        "method": method,
        "y_true_cal": y_true_cal,
        "y_true_test": y_true_test,
    }
    if method == "cqr":
        payload.update(
            {
                "quantile_lo_key": quantile_lo_key,
                "quantile_hi_key": quantile_hi_key,
                "q_lo_cal": q_lo_cal,
                "q_hi_cal": q_hi_cal,
                "q_lo_test": q_lo_test,
                "q_hi_test": q_hi_test,
            }
        )
    else:
        payload.update(
            {
                "y_pred_cal": y_pred_cal,
                "y_pred_test": y_pred_test,
            }
        )
    return payload


def fit_sequence_model(
    model,
    train_dl,
    val_dl,
    epochs: int,
    lr: float,
    horizon: int,
    device: str,
    weight_decay: float = 1e-5,
    patience: int = 15,
    warmup_epochs: int = 5,
    grad_clip: float = 1.0,
    quantile_levels: list[float] | None = None,
):
    """Training loop for sequence models with early stopping and warmup.

    Args:
        model: PyTorch model
        train_dl: Training DataLoader
        val_dl: Validation DataLoader
        epochs: Maximum number of epochs
        lr: Learning rate
        horizon: Forecast horizon
        device: Device to train on
        weight_decay: AdamW weight decay
        patience: Early stopping patience
        warmup_epochs: Number of warmup epochs
        grad_clip: Gradient clipping max norm
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warmup
    def warmup_cosine_schedule(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine_schedule)
    levels = _normalize_quantile_levels(quantile_levels)
    use_quantile_loss = len(levels) > 1
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    no_improve_count = 0
    history: list[dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        grad_norms = []
        clipped_batches = 0
        num_batches = 0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            # Model outputs a full sequence; evaluate only forecast horizon.
            pred_seq = model(xb)
            pred_hz = _extract_tensor_forecast(pred_seq, horizon, levels if use_quantile_loss else None)
            loss = _pinball_loss(pred_hz, yb, levels) if use_quantile_loss else loss_fn(pred_hz, yb)
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            grad_norm_value = float(
                grad_norm.detach().cpu().item() if hasattr(grad_norm, "detach") else grad_norm
            )
            grad_norms.append(grad_norm_value)
            clipped_batches += int(grad_norm_value > grad_clip)
            num_batches += 1
            opt.step()
            tr_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                pred_seq = model(xb)
                pred_hz = _extract_tensor_forecast(pred_seq, horizon, levels if use_quantile_loss else None)
                if use_quantile_loss:
                    val_losses.append(_pinball_loss(pred_hz, yb, levels).item())
                else:
                    val_losses.append(loss_fn(pred_hz, yb).item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        train_loss = float(np.mean(tr_losses)) if tr_losses else float("inf")
        history.append(
            {
                "epoch": int(ep),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
                "max_grad_norm": float(np.max(grad_norms)) if grad_norms else None,
                "clipped_batches": int(clipped_batches),
                "num_batches": int(num_batches),
            }
        )

        # Early stopping check
        if val_loss < best_val - 0.0001:  # min_delta
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {ep}/{epochs} train_mse={train_loss:.4f} val_mse={val_loss:.4f} lr={cur_lr:.6f} best={best_val:.4f} patience={no_improve_count}/{patience}"
        )

        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {ep} — no improvement for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model._orius_training_history = history
    model._orius_training_summary = _training_history_summary(history)
    print(f"Training complete — best val_mse={best_val:.4f}")
    return model


def train_lstm_model(X_train, y_train, X_val, y_val, cfg: dict, device: str = "cpu"):
    """Train an LSTM sequence model with advanced settings."""
    lookback = int(cfg.get("lookback", 168))
    horizon = int(cfg.get("horizon", 48))
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 100))
    hidden_size = int(cfg.get("hidden_size", 128))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.3))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 1e-3)))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    grad_clip = float(cfg.get("gradient_clip", 1.0))
    quantile_levels = _normalize_quantile_levels(cfg.get("quantile_levels"))

    # Enhanced LSTM options
    bidirectional = bool(cfg.get("bidirectional", False))
    attention = bool(cfg.get("attention", False))
    residual = bool(cfg.get("residual", False))

    # Early stopping config
    es_cfg = cfg.get("early_stopping", {})
    patience = int(es_cfg.get("patience", 15)) if isinstance(es_cfg, dict) else 15
    warmup_epochs = int(cfg.get("warmup_epochs", 5))

    scfg = SeqConfig(lookback=lookback, horizon=horizon)
    # Window the time series into (lookback -> horizon) pairs.
    train_ds = TimeSeriesWindowDataset(X_train, y_train, scfg)
    val_ds = TimeSeriesWindowDataset(X_val, y_val, scfg)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMForecaster(
        n_features=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
        bidirectional=bidirectional,
        attention=attention,
        residual=residual,
        n_quantiles=max(1, len(quantile_levels)),
    ).to(device)
    model = fit_sequence_model(
        model,
        train_dl,
        val_dl,
        epochs=epochs,
        lr=lr,
        horizon=horizon,
        device=device,
        weight_decay=weight_decay,
        patience=patience,
        warmup_epochs=warmup_epochs,
        grad_clip=grad_clip,
        quantile_levels=quantile_levels,
    )
    return model


def train_tcn_model(X_train, y_train, X_val, y_val, cfg: dict, device: str = "cpu"):
    """Train a TCN sequence model with advanced settings."""
    lookback = int(cfg.get("lookback", 168))
    horizon = int(cfg.get("horizon", 48))
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 100))
    num_channels = cfg.get("num_channels", [128, 128, 128, 64])
    kernel_size = int(cfg.get("kernel_size", 5))
    dropout = float(cfg.get("dropout", 0.3))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 1e-3)))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    grad_clip = float(cfg.get("gradient_clip", 1.0))
    quantile_levels = _normalize_quantile_levels(cfg.get("quantile_levels"))

    # Enhanced TCN options
    dilation_base = int(cfg.get("dilation_base", 2))
    use_skip_connections = bool(cfg.get("use_skip_connections", False))

    # Early stopping config
    es_cfg = cfg.get("early_stopping", {})
    patience = int(es_cfg.get("patience", 15)) if isinstance(es_cfg, dict) else 15
    warmup_epochs = int(cfg.get("warmup_epochs", 5))

    scfg = SeqConfig(lookback=lookback, horizon=horizon)
    # Same windowing logic as LSTM for fair comparison.
    train_ds = TimeSeriesWindowDataset(X_train, y_train, scfg)
    val_ds = TimeSeriesWindowDataset(X_val, y_val, scfg)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TCNForecaster(
        n_features=X_train.shape[1],
        num_channels=list(num_channels),
        kernel_size=kernel_size,
        dropout=dropout,
        dilation_base=dilation_base,
        use_skip_connections=use_skip_connections,
        horizon=horizon,
        n_quantiles=max(1, len(quantile_levels)),
    ).to(device)
    model = fit_sequence_model(
        model,
        train_dl,
        val_dl,
        epochs=epochs,
        lr=lr,
        horizon=horizon,
        device=device,
        weight_decay=weight_decay,
        patience=patience,
        warmup_epochs=warmup_epochs,
        grad_clip=grad_clip,
        quantile_levels=quantile_levels,
    )
    return model


def train_nbeats_model(X_train, y_train, X_val, y_val, cfg: dict, device: str = "cpu"):
    """Train an N-BEATS style sequence model."""
    lookback = int(cfg.get("lookback", 168))
    horizon = int(cfg.get("horizon", 48))
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 100))
    hidden_dim = int(cfg.get("hidden_dim", 512))
    num_blocks = int(cfg.get("num_blocks", 4))
    dropout = float(cfg.get("dropout", 0.1))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 1e-3)))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    grad_clip = float(cfg.get("gradient_clip", 1.0))
    quantile_levels = _normalize_quantile_levels(cfg.get("quantile_levels"))

    es_cfg = cfg.get("early_stopping", {})
    patience = int(es_cfg.get("patience", 15)) if isinstance(es_cfg, dict) else 15
    warmup_epochs = int(cfg.get("warmup_epochs", 5))

    scfg = SeqConfig(lookback=lookback, horizon=horizon)
    train_ds = TimeSeriesWindowDataset(X_train, y_train, scfg)
    val_ds = TimeSeriesWindowDataset(X_val, y_val, scfg)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = NBEATSForecaster(
        n_features=X_train.shape[1],
        lookback=lookback,
        horizon=horizon,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        n_quantiles=max(1, len(quantile_levels)),
    ).to(device)
    model = fit_sequence_model(
        model,
        train_dl,
        val_dl,
        epochs=epochs,
        lr=lr,
        horizon=horizon,
        device=device,
        weight_decay=weight_decay,
        patience=patience,
        warmup_epochs=warmup_epochs,
        grad_clip=grad_clip,
        quantile_levels=quantile_levels,
    )
    return model


def train_tft_model(X_train, y_train, X_val, y_val, cfg: dict, device: str = "cpu"):
    """Train a lightweight TFT-style transformer baseline."""
    lookback = int(cfg.get("lookback", 168))
    horizon = int(cfg.get("horizon", 48))
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 100))
    d_model = int(cfg.get("d_model", 128))
    n_heads = int(cfg.get("n_heads", 4))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.1))
    dim_feedforward = int(cfg.get("dim_feedforward", 256))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 1e-3)))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    grad_clip = float(cfg.get("gradient_clip", 1.0))
    quantile_levels = _normalize_quantile_levels(cfg.get("quantile_levels"))

    es_cfg = cfg.get("early_stopping", {})
    patience = int(es_cfg.get("patience", 15)) if isinstance(es_cfg, dict) else 15
    warmup_epochs = int(cfg.get("warmup_epochs", 5))

    scfg = SeqConfig(lookback=lookback, horizon=horizon)
    train_ds = TimeSeriesWindowDataset(X_train, y_train, scfg)
    val_ds = TimeSeriesWindowDataset(X_val, y_val, scfg)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TFTForecaster(
        n_features=X_train.shape[1],
        horizon=horizon,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
        n_quantiles=max(1, len(quantile_levels)),
    ).to(device)
    model = fit_sequence_model(
        model,
        train_dl,
        val_dl,
        epochs=epochs,
        lr=lr,
        horizon=horizon,
        device=device,
        weight_decay=weight_decay,
        patience=patience,
        warmup_epochs=warmup_epochs,
        grad_clip=grad_clip,
        quantile_levels=quantile_levels,
    )
    return model


def train_patchtst_model(X_train, y_train, X_val, y_val, cfg: dict, device: str = "cpu"):
    """Train a PatchTST-style transformer baseline."""
    lookback = int(cfg.get("lookback", 168))
    horizon = int(cfg.get("horizon", 48))
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 100))
    patch_len = int(cfg.get("patch_len", 16))
    stride = int(cfg.get("stride", 8))
    d_model = int(cfg.get("d_model", 128))
    n_heads = int(cfg.get("n_heads", 4))
    num_layers = int(cfg.get("num_layers", 3))
    dropout = float(cfg.get("dropout", 0.1))
    dim_feedforward = int(cfg.get("dim_feedforward", 256))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 1e-3)))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    grad_clip = float(cfg.get("gradient_clip", 1.0))
    quantile_levels = _normalize_quantile_levels(cfg.get("quantile_levels"))

    es_cfg = cfg.get("early_stopping", {})
    patience = int(es_cfg.get("patience", 15)) if isinstance(es_cfg, dict) else 15
    warmup_epochs = int(cfg.get("warmup_epochs", 5))

    scfg = SeqConfig(lookback=lookback, horizon=horizon)
    train_ds = TimeSeriesWindowDataset(X_train, y_train, scfg)
    val_ds = TimeSeriesWindowDataset(X_val, y_val, scfg)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PatchTSTForecaster(
        n_features=X_train.shape[1],
        lookback=lookback,
        horizon=horizon,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
        n_quantiles=max(1, len(quantile_levels)),
    ).to(device)
    model = fit_sequence_model(
        model,
        train_dl,
        val_dl,
        epochs=epochs,
        lr=lr,
        horizon=horizon,
        device=device,
        weight_decay=weight_decay,
        patience=patience,
        warmup_epochs=warmup_epochs,
        grad_clip=grad_clip,
        quantile_levels=quantile_levels,
    )
    return model


def collect_seq_output_bundle(
    model,
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int,
    device: str,
    batch_size: int,
    quantile_levels: list[float] | None = None,
) -> dict[str, Any] | None:
    """Collect sequential point and optional quantile predictions for evaluation."""
    ds = TimeSeriesWindowDataset(X, y, SeqConfig(lookback=lookback, horizon=horizon))
    if len(ds) == 0:
        return None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    quantile_preds: dict[str, list[np.ndarray]] = {
        str(q): [] for q in _normalize_quantile_levels(quantile_levels)
    }
    trues = []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            pred_seq = model(xb).cpu().numpy()
            point_pred, quantiles = _extract_numpy_forecast(pred_seq, horizon, quantile_levels)
            preds.append(point_pred.reshape(-1))
            for q_key, q_values in quantiles.items():
                quantile_preds.setdefault(q_key, []).append(q_values.reshape(-1))
            trues.append(yb.numpy().reshape(-1))
    output: dict[str, Any] = {
        "y_true": np.concatenate(trues),
        "point_pred": np.concatenate(preds),
        "quantiles": {},
    }
    for q_key, parts in quantile_preds.items():
        if parts:
            output["quantiles"][q_key] = np.concatenate(parts)
    return output


def _sequence_architecture_summary(
    model_key: str,
    params: dict[str, Any],
    *,
    lookback: int,
    horizon: int,
    n_features: int,
) -> dict[str, Any]:
    es_cfg = params.get("early_stopping", {}) if isinstance(params.get("early_stopping"), dict) else {}
    return {
        "model_type": model_key,
        "n_features": int(n_features),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "epochs": int(params.get("epochs", 100)),
        "batch_size": int(params.get("batch_size", 256)),
        "learning_rate": float(params.get("learning_rate", params.get("lr", 1.0e-3))),
        "weight_decay": float(params.get("weight_decay", 1.0e-5)),
        "dropout": float(params.get("dropout", 0.0)),
        "gradient_clip": float(params.get("gradient_clip", 1.0)),
        "early_stopping_patience": int(es_cfg.get("patience", 15)),
        "warmup_epochs": int(params.get("warmup_epochs", 5)),
    }


def _sequence_prediction_latency_summary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    device: str,
    batch_size: int,
    quantile_levels: list[float] | None,
    repeats: int = 3,
    max_rows: int = 2048,
) -> dict[str, Any]:
    sample_rows = min(len(X), max(int(max_rows), lookback + horizon))
    if sample_rows < lookback + horizon:
        return {
            "sample_rows": int(len(X)),
            "repeats": 0,
            "p95_per_sample_ms": None,
            "note": "insufficient sequence window for latency benchmark",
        }
    X_sample = X[:sample_rows]
    y_sample = y[:sample_rows]
    windows = len(TimeSeriesWindowDataset(X_sample, y_sample, SeqConfig(lookback=lookback, horizon=horizon)))
    durations_ms: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        collect_seq_output_bundle(
            model,
            X_sample,
            y_sample,
            lookback,
            horizon,
            device,
            batch_size,
            quantile_levels=quantile_levels,
        )
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    batch = np.asarray(durations_ms, dtype=float)
    per_window = batch / max(1, windows)
    return {
        "sample_rows": int(sample_rows),
        "forecast_windows": int(windows),
        "repeats": int(max(1, repeats)),
        "mean_batch_ms": float(np.mean(batch)),
        "p95_batch_ms": float(np.percentile(batch, 95)),
        "mean_per_sample_ms": float(np.mean(per_window)),
        "p95_per_sample_ms": float(np.percentile(per_window, 95)),
    }


def _sequence_model_quality_fields(
    *,
    model_key: str,
    model,
    params: dict[str, Any],
    target: str,
    X_train_s: np.ndarray,
    y_train_s: np.ndarray,
    X_test_s: np.ndarray,
    y_test_s: np.ndarray,
    val_outputs_s: dict[str, Any] | None,
    test_outputs_s: dict[str, Any] | None,
    y_scaler: StandardScaler,
    lookback: int,
    horizon: int,
    n_features: int,
    device: str,
    batch_size: int,
    quantile_levels: list[float] | None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "model_architecture": _sequence_architecture_summary(
            model_key,
            params,
            lookback=lookback,
            horizon=horizon,
            n_features=n_features,
        ),
        "training_summary": getattr(model, "_orius_training_summary", {}),
        "latency": _sequence_prediction_latency_summary(
            model,
            X_test_s,
            y_test_s,
            lookback=lookback,
            horizon=horizon,
            device=device,
            batch_size=batch_size,
            quantile_levels=quantile_levels,
        ),
    }
    train_outputs_s = collect_seq_output_bundle(
        model,
        X_train_s,
        y_train_s,
        lookback,
        horizon,
        device,
        batch_size,
        quantile_levels=quantile_levels,
    )
    output_by_split = {
        "train": train_outputs_s,
        "validation": val_outputs_s,
        "test": test_outputs_s,
    }
    split_metrics: dict[str, dict[str, Any]] = {}
    for split, payload in output_by_split.items():
        if payload is None:
            continue
        y_true = _inverse_scaled_array(payload["y_true"], y_scaler)
        y_pred = _inverse_scaled_array(payload["point_pred"], y_scaler)
        split_metrics[split] = compute_metrics(y_true, y_pred, target)
    if {"train", "validation", "test"} <= set(split_metrics):
        fields["split_metrics"] = split_metrics
        fields["generalization"] = _generalization_summary(
            split_metrics["train"],
            split_metrics["validation"],
            split_metrics["test"],
        )
    return fields


def collect_seq_preds(
    model,
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int,
    device: str,
    batch_size: int,
    quantile_levels: list[float] | None = None,
):
    """Collect sequential point predictions for evaluation."""
    payload = collect_seq_output_bundle(
        model,
        X,
        y,
        lookback,
        horizon,
        device,
        batch_size,
        quantile_levels=quantile_levels,
    )
    if payload is None:
        return None, None
    return payload["y_true"], payload["point_pred"]


def evaluate_seq_model(
    model, X, y, lookback, horizon, device, batch_size, target: str, y_scaler: StandardScaler | None = None
):
    """Evaluate a sequence model and return metrics on original scale."""
    y_true, y_pred = collect_seq_preds(model, X, y, lookback, horizon, device, batch_size)
    if y_true is None:
        return {"rmse": None, "mae": None, "mape": None, "smape": None, "note": "insufficient window"}
    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    return compute_metrics(y_true, y_pred, target)


def _resolve_conformal_split_arrays(
    *,
    y_val: np.ndarray,
    y_test: np.ndarray,
    y_cal: np.ndarray | None,
    uncertainty_cfg: dict[str, Any],
) -> tuple[str, np.ndarray, np.ndarray, str, str]:
    cal_split = str(uncertainty_cfg.get("calibration_split", "val")).lower()
    calibration_source = "val"
    calibration_label = "val"
    test_label = "test"
    cal_true = y_val
    test_true = y_test
    if cal_split == "test":
        cal_true = y_test
        test_true = y_val
        calibration_source = "test"
        calibration_label = "test"
        test_label = "val"
    elif cal_split == "calibration":
        if y_cal is not None and len(y_cal) > 0:
            cal_true = y_cal
            calibration_source = "calibration"
            calibration_label = "calibration"
        else:
            calibration_source = "val_fallback_missing_calibration"
            calibration_label = "val"
    return calibration_source, cal_true, test_true, calibration_label, test_label


def _inverse_scaled_array(values: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if scaler is None:
        return arr
    return scaler.inverse_transform(arr.reshape(-1, 1)).reshape(-1)


def _inverse_scaled_quantiles(
    quantiles: dict[str, np.ndarray],
    scaler: StandardScaler | None,
) -> dict[str, np.ndarray]:
    return {str(q): _inverse_scaled_array(values, scaler) for q, values in quantiles.items()}


def _append_conformal_payload_from_outputs(
    *,
    conformal_payloads: list[dict[str, Any]],
    target: str,
    model_key: str,
    horizon: int,
    split_boundaries: dict[str, Any],
    uncertainty_cfg: dict[str, Any],
    quantile_levels: list[float],
    val_outputs: dict[str, Any],
    test_outputs: dict[str, Any],
    cal_outputs: dict[str, Any] | None = None,
) -> None:
    conf_method = _conformal_method_for_target(uncertainty_cfg, target)
    calibration_source, _cal_true_unused, _test_true_unused, calibration_label, test_label = (
        _resolve_conformal_split_arrays(
            y_val=np.asarray(val_outputs["y_true"], dtype=float),
            y_test=np.asarray(test_outputs["y_true"], dtype=float),
            y_cal=np.asarray(cal_outputs["y_true"], dtype=float) if cal_outputs is not None else None,
            uncertainty_cfg=uncertainty_cfg,
        )
    )
    bundle_map = {
        "val": val_outputs,
        "test": test_outputs,
        "calibration": cal_outputs,
    }
    cal_bundle = bundle_map.get(calibration_label)
    test_bundle = bundle_map.get(test_label)
    if cal_bundle is None or test_bundle is None:
        return
    cal_true = np.asarray(cal_bundle["y_true"], dtype=float)
    test_true = np.asarray(test_bundle["y_true"], dtype=float)
    cal_pred = np.asarray(cal_bundle["point_pred"], dtype=float)
    test_pred = np.asarray(test_bundle["point_pred"], dtype=float)
    cal = _reshape_horizon(cal_true, cal_pred, horizon)
    test = _reshape_horizon(test_true, test_pred, horizon)
    if cal is None or test is None:
        return
    if conf_method == "cqr":
        q_keys = sorted({str(float(q)) for q in quantile_levels}, key=lambda x: float(x))
        if len(q_keys) < 2:
            raise RuntimeError(
                f"CQR configured for target={target} but quantile outputs are unavailable for model={model_key}."
            )
        lo_key, hi_key = q_keys[0], q_keys[-1]
        raw_quantiles = (
            cal_bundle.get("quantiles", {}).get(lo_key),
            cal_bundle.get("quantiles", {}).get(hi_key),
            test_bundle.get("quantiles", {}).get(lo_key),
            test_bundle.get("quantiles", {}).get(hi_key),
        )
        if any(value is None for value in raw_quantiles):
            raise RuntimeError(
                f"CQR configured for target={target} but missing quantile outputs for model={model_key} "
                f"calibration_source={calibration_source}."
            )
        cal_q_lo_arr = np.asarray(raw_quantiles[0], dtype=float)
        cal_q_hi_arr = np.asarray(raw_quantiles[1], dtype=float)
        test_q_lo_arr = np.asarray(raw_quantiles[2], dtype=float)
        test_q_hi_arr = np.asarray(raw_quantiles[3], dtype=float)
        cal_q_lo = _reshape_horizon(cal_true, cal_q_lo_arr, horizon)
        cal_q_hi = _reshape_horizon(cal_true, cal_q_hi_arr, horizon)
        test_q_lo = _reshape_horizon(test_true, test_q_lo_arr, horizon)
        test_q_hi = _reshape_horizon(test_true, test_q_hi_arr, horizon)
        if cal_q_lo is None or cal_q_hi is None or test_q_lo is None or test_q_hi is None:
            raise RuntimeError(
                f"CQR reshape failed for target={target}, model={model_key} due horizon alignment."
            )
        conformal_payloads.append(
            _build_conformal_payload(
                target=target,
                model_key=model_key,
                calibration_source=calibration_source,
                split_boundaries=split_boundaries,
                method="cqr",
                y_true_cal=cal[0],
                y_true_test=test[0],
                q_lo_cal=cal_q_lo[1],
                q_hi_cal=cal_q_hi[1],
                q_lo_test=test_q_lo[1],
                q_hi_test=test_q_hi[1],
                quantile_lo_key=lo_key,
                quantile_hi_key=hi_key,
            )
        )
    else:
        conformal_payloads.append(
            _build_conformal_payload(
                target=target,
                model_key=model_key,
                calibration_source=calibration_source,
                split_boundaries=split_boundaries,
                method="residual",
                y_true_cal=cal[0],
                y_true_test=test[0],
                y_pred_cal=cal[1],
                y_pred_test=test[1],
            )
        )


def main():
    """CLI entrypoint to train GBM/LSTM/TCN models and write reports."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_forecast.yaml")
    p.add_argument("--targets", default=None, help="Comma-separated targets to train (override config)")
    p.add_argument(
        "--models", default=None, help="Comma-separated models to train: gbm,lstm,tcn,nbeats,tft,patchtst"
    )
    p.add_argument(
        "--skip-existing", action="store_true", help="Skip training if model artifact already exists"
    )
    p.add_argument(
        "--enable-cv",
        action="store_true",
        help="Enable time-series cross-validation (fold count from config)",
    )
    p.add_argument(
        "--no-cv", action="store_true", help="Disable time-series cross-validation even if config enables it"
    )
    p.add_argument(
        "--use-pipeline",
        action="store_true",
        help="Use sklearn Pipeline for GBM (leakage-safe preprocessing)",
    )
    p.add_argument("--tune", action="store_true", help="Force-enable hyperparameter tuning")
    p.add_argument(
        "--no-tune", action="store_true", help="Disable hyperparameter tuning even if enabled in config"
    )
    p.add_argument("--ensemble", action="store_true", help="Train GBM seed ensemble (uses config.seeds)")
    p.add_argument("--max-seeds", type=int, default=None, help="Optional cap on ensemble seed count")
    p.add_argument("--n-trials", type=int, default=None, help="Override Optuna trial count for this run")
    p.add_argument(
        "--top-pct", type=float, default=None, help="Override top-percent trial selection, e.g. 0.30"
    )
    p.add_argument("--tuning-n-jobs", type=int, default=None, help="Parallel Optuna trials for GBM tuning")
    p.add_argument("--gbm-threads", type=int, default=None, help="Threads per LightGBM model/trial")
    p.add_argument("--reuse-best-gbm-from", default=None, help="Seed GBM tuning from a previous metrics JSON")
    p.add_argument("--max-deep-epochs", type=int, default=None, help="Cap epochs for deep model families")
    p.add_argument("--deep-patience", type=int, default=None, help="Cap deep model early-stopping patience")
    p.add_argument("--deep-warmup-epochs", type=int, default=None, help="Cap deep model warmup epochs")
    p.add_argument("--artifacts-dir", default=None, help="Override trained-model artifact directory")
    p.add_argument("--reports-dir", default=None, help="Override report output directory")
    p.add_argument("--uncertainty-artifacts-dir", default=None, help="Override conformal artifact directory")
    p.add_argument("--backtests-dir", default=None, help="Override calibration/test NPZ directory")
    p.add_argument("--walk-forward-report", default=None, help="Override walk-forward backtest report path")
    p.add_argument("--validation-report", default=None, help="Override schema validation report path")
    p.add_argument("--data-manifest-output", default=None, help="Override refreshed data-manifest path")
    args = p.parse_args()

    # Config-driven training keeps runs reproducible across datasets.
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _apply_deep_training_overrides(
        cfg,
        max_epochs=args.max_deep_epochs,
        patience=args.deep_patience,
        warmup_epochs=args.deep_warmup_epochs,
    )
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
    )

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # Create run manifest for reproducibility.
    art_dir = Path(args.artifacts_dir or cfg["artifacts"]["out_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    cv_cfg = cfg.get("cross_validation", {}) if isinstance(cfg.get("cross_validation"), dict) else {}
    cv_enabled = bool((args.enable_cv or cv_cfg.get("enabled", False)) and not args.no_cv)

    # Load prepared features (Parquet) for training.
    features_path = Path(cfg["data"]["processed_path"])
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run build_features first.")
    validation_report_path = Path(args.validation_report or "reports/data_quality_report_features.md")
    data_manifest_output = Path(args.data_manifest_output or "paper/assets/data/data_manifest.json")
    task_cfg = cfg.get("task", {}) or {}
    targets = task_cfg.get("targets", [])
    targets = [str(t).strip() for t in targets] if isinstance(targets, list) else []
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    timestamp_col = data_cfg.get("timestamp_col", "timestamp")
    order_cols_raw = data_cfg.get("order_cols")
    order_cols = (
        [str(col).strip() for col in order_cols_raw if str(col).strip()]
        if isinstance(order_cols_raw, list)
        else None
    )
    dataset_cfg = cfg.get("dataset", {}) or {}
    region_group = str(dataset_cfg.get("region_group", "") or "")
    is_multi_domain = region_group == "multi-domain"
    required_cols = targets if is_multi_domain and targets else None
    skip_manifest = is_multi_domain
    data_contract = _enforce_data_contract(
        features_path,
        validation_report_path=validation_report_path,
        data_manifest_output=data_manifest_output,
        required_cols=required_cols,
        timestamp_col=str(timestamp_col).strip() if timestamp_col else None,
        order_cols=order_cols,
        skip_data_manifest=skip_manifest,
    )

    print("Creating run manifest for reproducibility...")
    manifest = create_run_manifest(
        config_path=args.config,
        output_dir=art_dir,
        extra_metadata={
            "enable_cv": cv_enabled,
            "cv_folds": int(cv_cfg.get("n_folds", 5)),
            "use_pipeline": args.use_pipeline,
            "seed": seed,
            "ensemble": bool(args.ensemble),
            "deep_training_overrides": {
                "max_deep_epochs": args.max_deep_epochs,
                "deep_patience": args.deep_patience,
                "deep_warmup_epochs": args.deep_warmup_epochs,
            },
        },
        data_manifest_path=data_contract.get("path"),
        data_manifest_sha256=data_contract.get("sha256"),
        split_boundaries=data_contract.get("split_boundaries"),
        schema_hash=str(data_contract.get("schema_hash"))
        if data_contract.get("schema_hash") is not None
        else None,
        config_snapshot_text=yaml.safe_dump(cfg, sort_keys=False),
    )
    print_manifest_summary(manifest)

    # Time-ordered split to prevent leakage.
    df = _sort_training_frame(pd.read_parquet(features_path), data_cfg)
    n = len(df)
    train_ratio, val_ratio, test_ratio, calibration_ratio = _resolve_split_cfg(cfg)
    gap_rows = _resolve_gap_hours(cfg)
    train_end = int(n * train_ratio)
    calibration_start = min(n, train_end + gap_rows)
    calibration_end = min(n, calibration_start + int(n * calibration_ratio))
    val_start = min(n, calibration_end + gap_rows)
    val_end = min(n, val_start + int(n * val_ratio))
    test_start = min(n, val_end + gap_rows)

    train_df = df.iloc[:train_end]
    calibration_df = df.iloc[calibration_start:calibration_end]
    val_df = df.iloc[val_start:val_end]
    test_df = df.iloc[test_start:]
    if len(test_df) == 0:
        # Keep pipeline operable for 70/30 train/val configurations.
        print("Warning: test split is empty; reusing validation split as evaluation holdout.")
        test_df = val_df.copy()
    split_boundaries = {
        "n_rows": int(n),
        "train_end": int(train_end),
        "calibration_start": int(calibration_start),
        "calibration_end": int(calibration_end),
        "val_start": int(val_start),
        "val_end": int(val_end),
        "test_start": int(test_start),
        "gap_hours": int(gap_rows),
    }

    if calibration_ratio > 0.0:
        print(
            f"Data split: train={len(train_df)} ({train_ratio:.2%}), "
            f"calibration={len(calibration_df)} ({calibration_ratio:.2%}), "
            f"val={len(val_df)} ({val_ratio:.2%}), test={len(test_df)} ({max(test_ratio, 0.0):.2%})"
        )
    else:
        print(
            f"Data split: train={len(train_df)} ({train_ratio:.2%}), "
            f"val={len(val_df)} ({val_ratio:.2%}), test={len(test_df)} ({max(test_ratio, 0.0):.2%})"
        )

    # Persist updated split boundaries into the manifest artifact.
    manifest["split_boundaries"] = split_boundaries
    manifest_path_raw = manifest.get("manifest_path")
    if isinstance(manifest_path_raw, str) and manifest_path_raw:
        manifest_path = Path(manifest_path_raw)
        if manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Choose GPU if available, otherwise CPU.
    device = _select_torch_device()
    quantiles = cfg["task"].get("quantiles", [0.1, 0.5, 0.9])
    requested_targets = cfg["task"].get("targets", ["load_mw"])
    if args.targets:
        requested_targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    targets = [t for t in requested_targets if t in df.columns]
    missing_targets = [t for t in requested_targets if t not in df.columns]
    if missing_targets:
        print(f"Warning: missing targets in data and will be skipped: {missing_targets}")
    if not targets:
        raise ValueError(f"No valid targets found in data. Available columns: {list(df.columns)}")
    task_cfg = cfg.get("task", {}) if isinstance(cfg.get("task"), dict) else {}
    horizon = _resolve_task_horizon(task_cfg)
    lookback_default = _resolve_task_lookback(task_cfg)

    rep_dir = Path(args.reports_dir or cfg["reports"]["out_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "device": device,
        "quantiles": quantiles,
        "targets": {},
        "manifest_id": manifest.get("run_id"),
    }
    backtest_cfg = cfg.get("backtest", {})
    backtest_enabled = bool(backtest_cfg.get("enabled", False))
    backtest_payload = {"targets": {}} if backtest_enabled else None
    tuning_cfg = dict(cfg.get("tuning", {}) or {})
    if args.n_trials is not None and args.n_trials > 0:
        tuning_cfg["n_trials"] = int(args.n_trials)
    if args.top_pct is not None:
        tuning_cfg["select_top_pct"] = float(args.top_pct)
    if args.tuning_n_jobs is not None and args.tuning_n_jobs > 0:
        tuning_cfg["n_jobs"] = int(args.tuning_n_jobs)
    tuning_enabled = bool((tuning_cfg.get("enabled", False) or args.tune) and not args.no_tune)
    non_feature_cols = _configured_non_feature_columns(data_cfg)
    reusable_gbm_params = _load_reusable_gbm_params(args.reuse_best_gbm_from)

    ensemble_seeds = [int(seed)]
    cfg_seeds = cfg.get("seeds", [])
    if isinstance(cfg_seeds, list):
        for s in cfg_seeds:
            try:
                ensemble_seeds.append(int(s))
            except Exception:
                continue
    # De-duplicate, preserve order.
    seen = set()
    ensemble_seeds = [s for s in ensemble_seeds if not (s in seen or seen.add(s))]
    if args.max_seeds is not None and args.max_seeds > 0:
        ensemble_seeds = ensemble_seeds[: args.max_seeds]
    uncertainty_cfg = _merge_uncertainty_cfg(_load_uncertainty_cfg(), cfg.get("uncertainty", {}))
    uncertainty_enabled = bool(uncertainty_cfg.get("enabled", False))
    dataset_cfg = cfg.get("dataset", {}) or {}
    is_multi_domain = str(dataset_cfg.get("region_group", "") or "") == "multi-domain"
    # Multi-domain: use dataset targets for conformal; energy: use uncertainty.yaml targets
    uncertainty_targets = targets if is_multi_domain else _parse_uncertainty_targets(uncertainty_cfg, targets)
    conformal_payloads: list[dict[str, Any]] = []

    # Optional model/target filters to avoid retraining everything.
    model_filter = None
    if args.models:
        model_filter = {m.strip().lower() for m in args.models.split(",") if m.strip()}

    def has_gbm_artifact(target_name: str) -> bool:
        return any(art_dir.glob(f"gbm_*_{target_name}.pkl"))

    for target in targets:
        # 1) Build feature/target matrices for the current target.
        X_train, y_train, feat_cols = make_xy(train_df, target, targets, non_feature_cols=non_feature_cols)
        X_val, y_val, _ = make_xy(val_df, target, targets, non_feature_cols=non_feature_cols)
        X_test, y_test, _ = make_xy(test_df, target, targets, non_feature_cols=non_feature_cols)
        X_cal = None
        y_cal = None
        if not calibration_df.empty:
            X_cal, y_cal, _ = make_xy(calibration_df, target, targets, non_feature_cols=non_feature_cols)

        X_train, y_train = _drop_invalid_target_rows(X_train, y_train)
        X_val, y_val = _drop_invalid_target_rows(X_val, y_val)
        X_test, y_test = _drop_invalid_target_rows(X_test, y_test)
        if X_cal is not None and y_cal is not None:
            X_cal, y_cal = _drop_invalid_target_rows(X_cal, y_cal)

        feature_fill_values = _fit_feature_fill_values(X_train)
        feature_fill_values_list = feature_fill_values.astype(float).tolist()
        X_train = _apply_feature_fill_values(X_train, feature_fill_values)
        X_val = _apply_feature_fill_values(X_val, feature_fill_values)
        X_test = _apply_feature_fill_values(X_test, feature_fill_values)
        if X_cal is not None:
            X_cal = _apply_feature_fill_values(X_cal, feature_fill_values)

        target_res = {"n_features": len(feat_cols)}

        # ---- GBM (strong tabular baseline) ----
        gbm_cfg = cfg["models"].get("baseline_gbm", {})
        if gbm_cfg.get("enabled", True) and (model_filter is None or "gbm" in model_filter):
            if args.skip_existing and has_gbm_artifact(target):
                print(f"Skipping GBM for {target} (artifact exists)")
            else:
                gbm_params = _apply_gbm_thread_limits(dict(gbm_cfg.get("params", {})), args.gbm_threads)
                gbm_params.setdefault("random_state", int(cfg.get("seed", 42)))
                tuning_meta = None
                if tuning_enabled:
                    target_tuning_cfg = dict(tuning_cfg)
                    if target in reusable_gbm_params:
                        target_tuning_cfg["seed_params"] = reusable_gbm_params[target]
                    gbm_params, tuning_meta = tune_gbm_params(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        gbm_params,
                        target_tuning_cfg,
                        int(cfg.get("seed", 42)),
                    )
                seed_list = ensemble_seeds if args.ensemble else [int(cfg.get("seed", 42))]
                model_kind = "gbm"
                gbm_members: list[dict] = []
                train_preds = []
                val_preds = []
                test_preds = []

                # Train one GBM per seed (ensemble) or a single model.
                for seed_i in seed_list:
                    params_i = {**gbm_params, "random_state": int(seed_i)}
                    kind_i, gbm_i = train_gbm(
                        X_train,
                        y_train,
                        params_i,
                        use_pipeline=args.use_pipeline,
                        preprocessing="standard" if args.use_pipeline else None,
                    )
                    model_kind = kind_i
                    pred_train_i = predict_gbm(gbm_i, X_train)
                    pred_val_i = predict_gbm(gbm_i, X_val)
                    pred_test_i = predict_gbm(gbm_i, X_test)
                    gbm_members.append(
                        {
                            "seed": int(seed_i),
                            "model": gbm_i,
                            "val_rmse": float(rmse(y_val, pred_val_i)),
                            "val_mae": float(mae(y_val, pred_val_i)),
                        }
                    )
                    train_preds.append(pred_train_i)
                    val_preds.append(pred_val_i)
                    test_preds.append(pred_test_i)

                gbm_pred_train = np.mean(np.vstack(train_preds), axis=0)
                gbm_pred_val = np.mean(np.vstack(val_preds), axis=0)
                gbm_pred_test = np.mean(np.vstack(test_preds), axis=0)
                gbm_pred_cal = None
                if X_cal is not None and len(X_cal) > 0:
                    cal_preds = [predict_gbm(member["model"], X_cal) for member in gbm_members]
                    gbm_pred_cal = np.mean(np.vstack(cal_preds), axis=0)
                gbm_train_metrics = compute_metrics(y_train, gbm_pred_train, target)
                gbm_val_metrics = compute_metrics(y_val, gbm_pred_val, target)
                gbm_test_metrics = compute_metrics(y_test, gbm_pred_test, target)
                latency_members = list(gbm_members)
                gbm_latency = _prediction_latency_summary(
                    lambda batch, members=latency_members: np.mean(
                        np.vstack([predict_gbm(member["model"], batch) for member in members]),
                        axis=0,
                    ),
                    X_test,
                )
                gbm_metrics = {
                    **gbm_test_metrics,
                    "model": f"gbm_{model_kind}",
                    "ensemble_members": len(gbm_members),
                    "split_metrics": {
                        "train": gbm_train_metrics,
                        "validation": gbm_val_metrics,
                        "test": gbm_test_metrics,
                    },
                    "generalization": _generalization_summary(
                        gbm_train_metrics, gbm_val_metrics, gbm_test_metrics
                    ),
                    "latency": gbm_latency,
                }

                # Optional: time-series cross-validation
                if cv_enabled:
                    cv_folds = int(cv_cfg.get("n_folds", 5))
                    print(f"Running {cv_folds}-fold CV for GBM on {target}...")
                    # CV on combined train+val for robust evaluation
                    X_trainval = np.vstack([X_train, X_val])
                    y_trainval = np.concatenate([y_train, y_val])

                    cv_params = dict(gbm_params)
                    cv_results = time_series_cv_score(
                        X=X_trainval,
                        y=y_trainval,
                        train_fn=lambda X, y, params=cv_params: train_gbm(
                            X,
                            y,
                            params,
                            use_pipeline=args.use_pipeline,
                            preprocessing="standard" if args.use_pipeline else None,
                        )[1],
                        predict_fn=predict_gbm,
                        n_splits=cv_folds,
                        target=target,
                    )

                    # Save CV results
                    cv_path = rep_dir / f"{target}_gbm_cv_results.json"
                    cv_path.write_text(json.dumps(cv_results, indent=2), encoding="utf-8")
                    print(
                        f"  CV RMSE: {cv_results['aggregated']['rmse_mean']:.2f} ± {cv_results['aggregated']['rmse_std']:.2f}"
                    )
                    gbm_metrics["cv_results"] = cv_results["aggregated"]

                gbm_q = residual_quantiles(y_val, gbm_pred_val, quantiles)

                # Optional: train quantile GBM models for calibrated intervals.
                quantile_models = {}
                quant_cfg = cfg["models"].get("quantile_gbm", {})
                quantile_predictions: dict[str, dict[str, np.ndarray]] = {}
                if quant_cfg.get("enabled", False):
                    lgb = _try_lightgbm(required=True)
                    q_params_base = _apply_gbm_thread_limits(
                        dict(quant_cfg.get("params", {})), args.gbm_threads
                    )
                    for q in quantiles:
                        params = {**q_params_base, "objective": "quantile", "alpha": float(q)}
                        params.setdefault("random_state", int(cfg.get("seed", 42)))
                        q_model = lgb.LGBMRegressor(**params)
                        q_model.fit(X_train, y_train)
                        quantile_models[str(q)] = q_model
                        quantile_predictions[str(q)] = {
                            "val": np.asarray(q_model.predict(X_val), dtype=float),
                            "test": np.asarray(q_model.predict(X_test), dtype=float),
                        }
                        if X_cal is not None and len(X_cal) > 0:
                            quantile_predictions[str(q)]["calibration"] = np.asarray(
                                q_model.predict(X_cal), dtype=float
                            )

                if uncertainty_enabled and target in uncertainty_targets:
                    conf_method = _conformal_method_for_target(uncertainty_cfg, target)
                    calibration_source, cal_true, test_true, calibration_label, test_label = (
                        _resolve_conformal_split_arrays(
                            y_val=y_val,
                            y_test=y_test,
                            y_cal=y_cal,
                            uncertainty_cfg=uncertainty_cfg,
                        )
                    )
                    if calibration_label == "calibration":
                        if gbm_pred_cal is None:
                            raise RuntimeError(
                                f"CQR configured for target={target} but calibration predictions are unavailable."
                            )
                        cal_pred = gbm_pred_cal
                    else:
                        cal_pred = gbm_pred_val if calibration_label == "val" else gbm_pred_test
                    test_pred = gbm_pred_test if test_label == "test" else gbm_pred_val

                    cal = _reshape_horizon(cal_true, cal_pred, horizon)
                    test = _reshape_horizon(test_true, test_pred, horizon)
                    if cal is None or test is None:
                        pass
                    elif conf_method == "cqr":
                        q_keys = sorted(quantile_predictions.keys(), key=lambda x: float(x))
                        if len(q_keys) < 2:
                            raise RuntimeError(
                                f"CQR configured for target={target} but quantile GBM predictions are unavailable. "
                                "Enable models.quantile_gbm in training config."
                            )
                        lo_key, hi_key = q_keys[0], q_keys[-1]
                        cal_q_lo_src = quantile_predictions[lo_key].get(calibration_label)
                        cal_q_hi_src = quantile_predictions[hi_key].get(calibration_label)
                        test_q_lo_src = quantile_predictions[lo_key].get(test_label)
                        test_q_hi_src = quantile_predictions[hi_key].get(test_label)

                        if (
                            cal_q_lo_src is None
                            or cal_q_hi_src is None
                            or test_q_lo_src is None
                            or test_q_hi_src is None
                        ):
                            raise RuntimeError(
                                f"CQR configured for target={target} but missing quantile predictions for "
                                f"calibration_source={calibration_source}."
                            )
                        cal_q_lo = _reshape_horizon(cal_true, np.asarray(cal_q_lo_src, dtype=float), horizon)
                        cal_q_hi = _reshape_horizon(cal_true, np.asarray(cal_q_hi_src, dtype=float), horizon)
                        test_q_lo = _reshape_horizon(
                            test_true, np.asarray(test_q_lo_src, dtype=float), horizon
                        )
                        test_q_hi = _reshape_horizon(
                            test_true, np.asarray(test_q_hi_src, dtype=float), horizon
                        )
                        if cal_q_lo is None or cal_q_hi is None or test_q_lo is None or test_q_hi is None:
                            raise RuntimeError(
                                f"CQR reshape failed for target={target} due horizon alignment."
                            )
                        conformal_payloads.append(
                            _build_conformal_payload(
                                target=target,
                                model_key=f"gbm_{model_kind}",
                                calibration_source=calibration_source,
                                split_boundaries=split_boundaries,
                                method="cqr",
                                y_true_cal=cal[0],
                                y_true_test=test[0],
                                q_lo_cal=cal_q_lo[1],
                                q_hi_cal=cal_q_hi[1],
                                q_lo_test=test_q_lo[1],
                                q_hi_test=test_q_hi[1],
                                quantile_lo_key=lo_key,
                                quantile_hi_key=hi_key,
                            )
                        )
                    else:
                        conformal_payloads.append(
                            _build_conformal_payload(
                                target=target,
                                model_key=f"gbm_{model_kind}",
                                calibration_source=calibration_source,
                                split_boundaries=split_boundaries,
                                method="residual",
                                y_true_cal=cal[0],
                                y_true_test=test[0],
                                y_pred_cal=cal[1],
                                y_pred_test=test[1],
                            )
                        )

                import pickle

                with open(art_dir / f"{gbm_metrics['model']}_{target}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "model_type": "gbm",
                            # Keep single-model key for compatibility; use ensemble list when present.
                            "model": gbm_members[0]["model"],
                            "ensemble_models": [m["model"] for m in gbm_members]
                            if len(gbm_members) > 1
                            else None,
                            "ensemble_seeds": [m["seed"] for m in gbm_members],
                            "feature_cols": feat_cols,
                            "feature_fill_values": feature_fill_values_list,
                            "target": target,
                            "quantiles": quantiles,
                            "residual_quantiles": gbm_q,
                            "quantile_models": quantile_models if quantile_models else None,
                            "tuned_params": gbm_params if tuning_enabled else None,
                            "tuning_meta": tuning_meta if tuning_enabled else None,
                            "seed_member_metrics": [
                                {
                                    "seed": int(m["seed"]),
                                    "val_rmse": float(m["val_rmse"]),
                                    "val_mae": float(m["val_mae"]),
                                }
                                for m in gbm_members
                            ],
                        },
                        f,
                    )

                target_res["gbm"] = {
                    **gbm_metrics,
                    "residual_quantiles": gbm_q,
                    "tuned_params": gbm_params if tuning_enabled else None,
                    "tuning_meta": tuning_meta if tuning_enabled else None,
                    "quantile_models": list(quantile_models.keys()) if quantile_models else None,
                    "seed_member_metrics": [
                        {
                            "seed": int(m["seed"]),
                            "val_rmse": float(m["val_rmse"]),
                            "val_mae": float(m["val_mae"]),
                        }
                        for m in gbm_members
                    ],
                }
                if backtest_enabled:
                    backtest_payload["targets"].setdefault(target, {})
                    backtest_payload["targets"][target]["gbm"] = walk_forward_horizon_metrics(
                        y_test, gbm_pred_test, horizon, target
                    )

        # ---- LSTM (sequence model) ----
        lstm_cfg = cfg["models"].get("dl_lstm", {})
        if lstm_cfg.get("enabled", True) and (model_filter is None or "lstm" in model_filter):
            if args.skip_existing and (art_dir / f"lstm_{target}.pt").exists():
                print(f"Skipping LSTM for {target} (artifact exists)")
            else:
                params = lstm_cfg.get("params", {})
                # Scale features/targets for DL stability; store scalers with the model.
                x_scaler = StandardScaler.fit(X_train)
                y_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
                X_train_s = x_scaler.transform(X_train)
                X_val_s = x_scaler.transform(X_val)
                X_test_s = x_scaler.transform(X_test)
                y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
                y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
                y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

                model = train_lstm_model(
                    X_train_s,
                    y_train_s,
                    X_val_s,
                    y_val_s,
                    {
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "quantile_levels": quantiles,
                        **params,
                    },
                    device=device,
                )
                batch_size = int(params.get("batch_size", 256))
                val_outputs_s = collect_seq_output_bundle(
                    model,
                    X_val_s,
                    y_val_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                cal_outputs_s = None
                if X_cal is not None and y_cal is not None and len(X_cal) > 0:
                    X_cal_s = x_scaler.transform(X_cal)
                    y_cal_s = y_scaler.transform(y_cal.reshape(-1, 1)).reshape(-1)
                    cal_outputs_s = collect_seq_output_bundle(
                        model,
                        X_cal_s,
                        y_cal_s,
                        lookback_default,
                        horizon,
                        device,
                        batch_size,
                        quantile_levels=quantiles,
                    )
                if val_outputs_s is not None:
                    y_true_val = _inverse_scaled_array(val_outputs_s["y_true"], y_scaler)
                    y_pred_val = _inverse_scaled_array(val_outputs_s["point_pred"], y_scaler)
                    lstm_q = residual_quantiles(y_true_val, y_pred_val, quantiles)
                else:
                    lstm_q = {}

                torch.save(
                    {
                        "model_type": "lstm",
                        "state_dict": model.state_dict(),
                        "feature_cols": feat_cols,
                        "target": target,
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "model_params": {
                            "hidden_size": int(params.get("hidden_size", 128)),
                            "num_layers": int(params.get("num_layers", 2)),
                            "dropout": float(params.get("dropout", 0.1)),
                            "bidirectional": bool(params.get("bidirectional", False)),
                            "attention": bool(params.get("attention", False)),
                            "residual": bool(params.get("residual", False)),
                            "horizon": int(horizon),
                        },
                        "quantiles": quantiles,
                        "quantile_levels": quantiles,
                        "n_quantiles": len(quantiles),
                        "residual_quantiles": lstm_q,
                        "feature_fill_values": feature_fill_values_list,
                        "x_scaler": x_scaler.to_dict(),
                        "y_scaler": y_scaler.to_dict(),
                    },
                    art_dir / f"lstm_{target}.pt",
                )

                test_outputs_s = collect_seq_output_bundle(
                    model,
                    X_test_s,
                    y_test_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                if test_outputs_s is not None:
                    y_true_test = _inverse_scaled_array(test_outputs_s["y_true"], y_scaler)
                    y_pred_test = _inverse_scaled_array(test_outputs_s["point_pred"], y_scaler)
                    lstm_quality = _sequence_model_quality_fields(
                        model_key="lstm",
                        model=model,
                        params=params,
                        target=target,
                        X_train_s=X_train_s,
                        y_train_s=y_train_s,
                        X_test_s=X_test_s,
                        y_test_s=y_test_s,
                        val_outputs_s=val_outputs_s,
                        test_outputs_s=test_outputs_s,
                        y_scaler=y_scaler,
                        lookback=lookback_default,
                        horizon=horizon,
                        n_features=len(feat_cols),
                        device=device,
                        batch_size=batch_size,
                        quantile_levels=quantiles,
                    )
                    lstm_metrics = {**compute_metrics(y_true_test, y_pred_test, target), **lstm_quality}
                    test_outputs = {
                        "y_true": y_true_test,
                        "point_pred": y_pred_test,
                        "quantiles": _inverse_scaled_quantiles(test_outputs_s["quantiles"], y_scaler),
                    }
                    val_outputs = {
                        "y_true": y_true_val,
                        "point_pred": y_pred_val,
                        "quantiles": _inverse_scaled_quantiles(val_outputs_s["quantiles"], y_scaler)
                        if val_outputs_s is not None
                        else {},
                    }
                    cal_outputs = None
                    if cal_outputs_s is not None:
                        cal_outputs = {
                            "y_true": _inverse_scaled_array(cal_outputs_s["y_true"], y_scaler),
                            "point_pred": _inverse_scaled_array(cal_outputs_s["point_pred"], y_scaler),
                            "quantiles": _inverse_scaled_quantiles(cal_outputs_s["quantiles"], y_scaler),
                        }
                    if val_outputs_s is not None and uncertainty_enabled and target in uncertainty_targets:
                        _append_conformal_payload_from_outputs(
                            conformal_payloads=conformal_payloads,
                            target=target,
                            model_key="lstm",
                            horizon=horizon,
                            split_boundaries=split_boundaries,
                            uncertainty_cfg=uncertainty_cfg,
                            quantile_levels=quantiles,
                            val_outputs=val_outputs,
                            test_outputs=test_outputs,
                            cal_outputs=cal_outputs,
                        )
                else:
                    lstm_metrics = {
                        "rmse": None,
                        "mae": None,
                        "mape": None,
                        "smape": None,
                        "model_architecture": _sequence_architecture_summary(
                            "lstm",
                            params,
                            lookback=lookback_default,
                            horizon=horizon,
                            n_features=len(feat_cols),
                        ),
                        "training_summary": getattr(model, "_orius_training_summary", {}),
                    }
                target_res["lstm"] = {**lstm_metrics, "residual_quantiles": lstm_q}
                if backtest_enabled and lstm_metrics.get("rmse") is not None:
                    backtest_payload["targets"].setdefault(target, {})
                    backtest_payload["targets"][target]["lstm"] = walk_forward_horizon_metrics(
                        y_true_test, y_pred_test, horizon, target
                    )

        # ---- TCN (convolutional sequence model) ----
        tcn_cfg = cfg["models"].get("dl_tcn", {})
        if tcn_cfg.get("enabled", False) and (model_filter is None or "tcn" in model_filter):
            if args.skip_existing and (art_dir / f"tcn_{target}.pt").exists():
                print(f"Skipping TCN for {target} (artifact exists)")
            else:
                params = tcn_cfg.get("params", {})
                # Same scaling strategy as LSTM for comparability.
                x_scaler = StandardScaler.fit(X_train)
                y_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
                X_train_s = x_scaler.transform(X_train)
                X_val_s = x_scaler.transform(X_val)
                X_test_s = x_scaler.transform(X_test)
                y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
                y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
                y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

                model = train_tcn_model(
                    X_train_s,
                    y_train_s,
                    X_val_s,
                    y_val_s,
                    {
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "quantile_levels": quantiles,
                        **params,
                    },
                    device=device,
                )
                batch_size = int(params.get("batch_size", 256))
                val_outputs_s = collect_seq_output_bundle(
                    model,
                    X_val_s,
                    y_val_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                cal_outputs_s = None
                if X_cal is not None and y_cal is not None and len(X_cal) > 0:
                    X_cal_s = x_scaler.transform(X_cal)
                    y_cal_s = y_scaler.transform(y_cal.reshape(-1, 1)).reshape(-1)
                    cal_outputs_s = collect_seq_output_bundle(
                        model,
                        X_cal_s,
                        y_cal_s,
                        lookback_default,
                        horizon,
                        device,
                        batch_size,
                        quantile_levels=quantiles,
                    )
                if val_outputs_s is not None:
                    y_true_val = _inverse_scaled_array(val_outputs_s["y_true"], y_scaler)
                    y_pred_val = _inverse_scaled_array(val_outputs_s["point_pred"], y_scaler)
                    tcn_q = residual_quantiles(y_true_val, y_pred_val, quantiles)
                else:
                    tcn_q = {}

                torch.save(
                    {
                        "model_type": "tcn",
                        "state_dict": model.state_dict(),
                        "feature_cols": feat_cols,
                        "target": target,
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "model_params": {
                            "num_channels": list(params.get("num_channels", [32, 32, 32])),
                            "kernel_size": int(params.get("kernel_size", 3)),
                            "dropout": float(params.get("dropout", 0.1)),
                            "dilation_base": int(params.get("dilation_base", 2)),
                            "use_skip_connections": bool(params.get("use_skip_connections", False)),
                        },
                        "quantiles": quantiles,
                        "quantile_levels": quantiles,
                        "n_quantiles": len(quantiles),
                        "residual_quantiles": tcn_q,
                        "feature_fill_values": feature_fill_values_list,
                        "x_scaler": x_scaler.to_dict(),
                        "y_scaler": y_scaler.to_dict(),
                    },
                    art_dir / f"tcn_{target}.pt",
                )

                test_outputs_s = collect_seq_output_bundle(
                    model,
                    X_test_s,
                    y_test_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                if test_outputs_s is not None:
                    y_true_test = _inverse_scaled_array(test_outputs_s["y_true"], y_scaler)
                    y_pred_test = _inverse_scaled_array(test_outputs_s["point_pred"], y_scaler)
                    tcn_quality = _sequence_model_quality_fields(
                        model_key="tcn",
                        model=model,
                        params=params,
                        target=target,
                        X_train_s=X_train_s,
                        y_train_s=y_train_s,
                        X_test_s=X_test_s,
                        y_test_s=y_test_s,
                        val_outputs_s=val_outputs_s,
                        test_outputs_s=test_outputs_s,
                        y_scaler=y_scaler,
                        lookback=lookback_default,
                        horizon=horizon,
                        n_features=len(feat_cols),
                        device=device,
                        batch_size=batch_size,
                        quantile_levels=quantiles,
                    )
                    tcn_metrics = {**compute_metrics(y_true_test, y_pred_test, target), **tcn_quality}
                    test_outputs = {
                        "y_true": y_true_test,
                        "point_pred": y_pred_test,
                        "quantiles": _inverse_scaled_quantiles(test_outputs_s["quantiles"], y_scaler),
                    }
                    val_outputs = {
                        "y_true": y_true_val,
                        "point_pred": y_pred_val,
                        "quantiles": _inverse_scaled_quantiles(val_outputs_s["quantiles"], y_scaler)
                        if val_outputs_s is not None
                        else {},
                    }
                    cal_outputs = None
                    if cal_outputs_s is not None:
                        cal_outputs = {
                            "y_true": _inverse_scaled_array(cal_outputs_s["y_true"], y_scaler),
                            "point_pred": _inverse_scaled_array(cal_outputs_s["point_pred"], y_scaler),
                            "quantiles": _inverse_scaled_quantiles(cal_outputs_s["quantiles"], y_scaler),
                        }
                    if val_outputs_s is not None and uncertainty_enabled and target in uncertainty_targets:
                        _append_conformal_payload_from_outputs(
                            conformal_payloads=conformal_payloads,
                            target=target,
                            model_key="tcn",
                            horizon=horizon,
                            split_boundaries=split_boundaries,
                            uncertainty_cfg=uncertainty_cfg,
                            quantile_levels=quantiles,
                            val_outputs=val_outputs,
                            test_outputs=test_outputs,
                            cal_outputs=cal_outputs,
                        )
                else:
                    tcn_metrics = {
                        "rmse": None,
                        "mae": None,
                        "mape": None,
                        "smape": None,
                        "model_architecture": _sequence_architecture_summary(
                            "tcn",
                            params,
                            lookback=lookback_default,
                            horizon=horizon,
                            n_features=len(feat_cols),
                        ),
                        "training_summary": getattr(model, "_orius_training_summary", {}),
                    }
                target_res["tcn"] = {**tcn_metrics, "residual_quantiles": tcn_q}
                if backtest_enabled and tcn_metrics.get("rmse") is not None:
                    backtest_payload["targets"].setdefault(target, {})
                    backtest_payload["targets"][target]["tcn"] = walk_forward_horizon_metrics(
                        y_true_test, y_pred_test, horizon, target
                    )

        # ---- N-BEATS (MLP sequence baseline) ----
        nbeats_cfg = cfg["models"].get("dl_nbeats", {})
        if nbeats_cfg.get("enabled", False) and (model_filter is None or "nbeats" in model_filter):
            if args.skip_existing and (art_dir / f"nbeats_{target}.pt").exists():
                print(f"Skipping N-BEATS for {target} (artifact exists)")
            else:
                params = nbeats_cfg.get("params", {})
                x_scaler = StandardScaler.fit(X_train)
                y_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
                X_train_s = x_scaler.transform(X_train)
                X_val_s = x_scaler.transform(X_val)
                X_test_s = x_scaler.transform(X_test)
                y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
                y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
                y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

                model = train_nbeats_model(
                    X_train_s,
                    y_train_s,
                    X_val_s,
                    y_val_s,
                    {
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "quantile_levels": quantiles,
                        **params,
                    },
                    device=device,
                )
                batch_size = int(params.get("batch_size", 256))
                val_outputs_s = collect_seq_output_bundle(
                    model,
                    X_val_s,
                    y_val_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                cal_outputs_s = None
                if X_cal is not None and y_cal is not None and len(X_cal) > 0:
                    X_cal_s = x_scaler.transform(X_cal)
                    y_cal_s = y_scaler.transform(y_cal.reshape(-1, 1)).reshape(-1)
                    cal_outputs_s = collect_seq_output_bundle(
                        model,
                        X_cal_s,
                        y_cal_s,
                        lookback_default,
                        horizon,
                        device,
                        batch_size,
                        quantile_levels=quantiles,
                    )
                if val_outputs_s is not None:
                    y_true_val = _inverse_scaled_array(val_outputs_s["y_true"], y_scaler)
                    y_pred_val = _inverse_scaled_array(val_outputs_s["point_pred"], y_scaler)
                    model_q = residual_quantiles(y_true_val, y_pred_val, quantiles)
                else:
                    model_q = {}

                torch.save(
                    {
                        "model_type": "nbeats",
                        "state_dict": model.state_dict(),
                        "feature_cols": feat_cols,
                        "target": target,
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "model_params": {
                            "hidden_dim": int(params.get("hidden_dim", 512)),
                            "num_blocks": int(params.get("num_blocks", 4)),
                            "dropout": float(params.get("dropout", 0.1)),
                            "lookback": int(lookback_default),
                            "horizon": int(horizon),
                        },
                        "quantiles": quantiles,
                        "quantile_levels": quantiles,
                        "n_quantiles": len(quantiles),
                        "residual_quantiles": model_q,
                        "feature_fill_values": feature_fill_values_list,
                        "x_scaler": x_scaler.to_dict(),
                        "y_scaler": y_scaler.to_dict(),
                    },
                    art_dir / f"nbeats_{target}.pt",
                )

                test_outputs_s = collect_seq_output_bundle(
                    model,
                    X_test_s,
                    y_test_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                if test_outputs_s is not None:
                    y_true_test = _inverse_scaled_array(test_outputs_s["y_true"], y_scaler)
                    y_pred_test = _inverse_scaled_array(test_outputs_s["point_pred"], y_scaler)
                    model_quality = _sequence_model_quality_fields(
                        model_key="nbeats",
                        model=model,
                        params=params,
                        target=target,
                        X_train_s=X_train_s,
                        y_train_s=y_train_s,
                        X_test_s=X_test_s,
                        y_test_s=y_test_s,
                        val_outputs_s=val_outputs_s,
                        test_outputs_s=test_outputs_s,
                        y_scaler=y_scaler,
                        lookback=lookback_default,
                        horizon=horizon,
                        n_features=len(feat_cols),
                        device=device,
                        batch_size=batch_size,
                        quantile_levels=quantiles,
                    )
                    model_metrics = {**compute_metrics(y_true_test, y_pred_test, target), **model_quality}
                    test_outputs = {
                        "y_true": y_true_test,
                        "point_pred": y_pred_test,
                        "quantiles": _inverse_scaled_quantiles(test_outputs_s["quantiles"], y_scaler),
                    }
                    val_outputs = {
                        "y_true": y_true_val,
                        "point_pred": y_pred_val,
                        "quantiles": _inverse_scaled_quantiles(val_outputs_s["quantiles"], y_scaler)
                        if val_outputs_s is not None
                        else {},
                    }
                    cal_outputs = None
                    if cal_outputs_s is not None:
                        cal_outputs = {
                            "y_true": _inverse_scaled_array(cal_outputs_s["y_true"], y_scaler),
                            "point_pred": _inverse_scaled_array(cal_outputs_s["point_pred"], y_scaler),
                            "quantiles": _inverse_scaled_quantiles(cal_outputs_s["quantiles"], y_scaler),
                        }
                    if val_outputs_s is not None and uncertainty_enabled and target in uncertainty_targets:
                        _append_conformal_payload_from_outputs(
                            conformal_payloads=conformal_payloads,
                            target=target,
                            model_key="nbeats",
                            horizon=horizon,
                            split_boundaries=split_boundaries,
                            uncertainty_cfg=uncertainty_cfg,
                            quantile_levels=quantiles,
                            val_outputs=val_outputs,
                            test_outputs=test_outputs,
                            cal_outputs=cal_outputs,
                        )
                else:
                    model_metrics = {"rmse": None, "mae": None, "mape": None, "smape": None}
                target_res["nbeats"] = {**model_metrics, "residual_quantiles": model_q}
                if backtest_enabled and model_metrics.get("rmse") is not None:
                    backtest_payload["targets"].setdefault(target, {})
                    backtest_payload["targets"][target]["nbeats"] = walk_forward_horizon_metrics(
                        y_true_test, y_pred_test, horizon, target
                    )

        # ---- TFT (transformer baseline) ----
        tft_cfg = cfg["models"].get("dl_tft", {})
        if tft_cfg.get("enabled", False) and (model_filter is None or "tft" in model_filter):
            if args.skip_existing and (art_dir / f"tft_{target}.pt").exists():
                print(f"Skipping TFT for {target} (artifact exists)")
            else:
                params = tft_cfg.get("params", {})
                x_scaler = StandardScaler.fit(X_train)
                y_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
                X_train_s = x_scaler.transform(X_train)
                X_val_s = x_scaler.transform(X_val)
                X_test_s = x_scaler.transform(X_test)
                y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
                y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
                y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

                model = train_tft_model(
                    X_train_s,
                    y_train_s,
                    X_val_s,
                    y_val_s,
                    {
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "quantile_levels": quantiles,
                        **params,
                    },
                    device=device,
                )
                batch_size = int(params.get("batch_size", 256))
                val_outputs_s = collect_seq_output_bundle(
                    model,
                    X_val_s,
                    y_val_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                cal_outputs_s = None
                if X_cal is not None and y_cal is not None and len(X_cal) > 0:
                    X_cal_s = x_scaler.transform(X_cal)
                    y_cal_s = y_scaler.transform(y_cal.reshape(-1, 1)).reshape(-1)
                    cal_outputs_s = collect_seq_output_bundle(
                        model,
                        X_cal_s,
                        y_cal_s,
                        lookback_default,
                        horizon,
                        device,
                        batch_size,
                        quantile_levels=quantiles,
                    )
                if val_outputs_s is not None:
                    y_true_val = _inverse_scaled_array(val_outputs_s["y_true"], y_scaler)
                    y_pred_val = _inverse_scaled_array(val_outputs_s["point_pred"], y_scaler)
                    model_q = residual_quantiles(y_true_val, y_pred_val, quantiles)
                else:
                    model_q = {}

                torch.save(
                    {
                        "model_type": "tft",
                        "state_dict": model.state_dict(),
                        "feature_cols": feat_cols,
                        "target": target,
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "model_params": {
                            "d_model": int(params.get("d_model", 128)),
                            "n_heads": int(params.get("n_heads", 4)),
                            "num_layers": int(params.get("num_layers", 2)),
                            "dropout": float(params.get("dropout", 0.1)),
                            "dim_feedforward": int(params.get("dim_feedforward", 256)),
                            "horizon": int(horizon),
                        },
                        "quantiles": quantiles,
                        "quantile_levels": quantiles,
                        "n_quantiles": len(quantiles),
                        "residual_quantiles": model_q,
                        "feature_fill_values": feature_fill_values_list,
                        "x_scaler": x_scaler.to_dict(),
                        "y_scaler": y_scaler.to_dict(),
                    },
                    art_dir / f"tft_{target}.pt",
                )

                test_outputs_s = collect_seq_output_bundle(
                    model,
                    X_test_s,
                    y_test_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                if test_outputs_s is not None:
                    y_true_test = _inverse_scaled_array(test_outputs_s["y_true"], y_scaler)
                    y_pred_test = _inverse_scaled_array(test_outputs_s["point_pred"], y_scaler)
                    model_quality = _sequence_model_quality_fields(
                        model_key="tft",
                        model=model,
                        params=params,
                        target=target,
                        X_train_s=X_train_s,
                        y_train_s=y_train_s,
                        X_test_s=X_test_s,
                        y_test_s=y_test_s,
                        val_outputs_s=val_outputs_s,
                        test_outputs_s=test_outputs_s,
                        y_scaler=y_scaler,
                        lookback=lookback_default,
                        horizon=horizon,
                        n_features=len(feat_cols),
                        device=device,
                        batch_size=batch_size,
                        quantile_levels=quantiles,
                    )
                    model_metrics = {**compute_metrics(y_true_test, y_pred_test, target), **model_quality}
                    test_outputs = {
                        "y_true": y_true_test,
                        "point_pred": y_pred_test,
                        "quantiles": _inverse_scaled_quantiles(test_outputs_s["quantiles"], y_scaler),
                    }
                    val_outputs = {
                        "y_true": y_true_val,
                        "point_pred": y_pred_val,
                        "quantiles": _inverse_scaled_quantiles(val_outputs_s["quantiles"], y_scaler)
                        if val_outputs_s is not None
                        else {},
                    }
                    cal_outputs = None
                    if cal_outputs_s is not None:
                        cal_outputs = {
                            "y_true": _inverse_scaled_array(cal_outputs_s["y_true"], y_scaler),
                            "point_pred": _inverse_scaled_array(cal_outputs_s["point_pred"], y_scaler),
                            "quantiles": _inverse_scaled_quantiles(cal_outputs_s["quantiles"], y_scaler),
                        }
                    if val_outputs_s is not None and uncertainty_enabled and target in uncertainty_targets:
                        _append_conformal_payload_from_outputs(
                            conformal_payloads=conformal_payloads,
                            target=target,
                            model_key="tft",
                            horizon=horizon,
                            split_boundaries=split_boundaries,
                            uncertainty_cfg=uncertainty_cfg,
                            quantile_levels=quantiles,
                            val_outputs=val_outputs,
                            test_outputs=test_outputs,
                            cal_outputs=cal_outputs,
                        )
                else:
                    model_metrics = {"rmse": None, "mae": None, "mape": None, "smape": None}
                target_res["tft"] = {**model_metrics, "residual_quantiles": model_q}
                if backtest_enabled and model_metrics.get("rmse") is not None:
                    backtest_payload["targets"].setdefault(target, {})
                    backtest_payload["targets"][target]["tft"] = walk_forward_horizon_metrics(
                        y_true_test, y_pred_test, horizon, target
                    )

        # ---- PatchTST (patch transformer baseline) ----
        patchtst_cfg = cfg["models"].get("dl_patchtst", {})
        if patchtst_cfg.get("enabled", False) and (model_filter is None or "patchtst" in model_filter):
            if args.skip_existing and (art_dir / f"patchtst_{target}.pt").exists():
                print(f"Skipping PatchTST for {target} (artifact exists)")
            else:
                params = patchtst_cfg.get("params", {})
                x_scaler = StandardScaler.fit(X_train)
                y_scaler = StandardScaler.fit(y_train.reshape(-1, 1))
                X_train_s = x_scaler.transform(X_train)
                X_val_s = x_scaler.transform(X_val)
                X_test_s = x_scaler.transform(X_test)
                y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
                y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
                y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

                model = train_patchtst_model(
                    X_train_s,
                    y_train_s,
                    X_val_s,
                    y_val_s,
                    {
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "quantile_levels": quantiles,
                        **params,
                    },
                    device=device,
                )
                batch_size = int(params.get("batch_size", 256))
                val_outputs_s = collect_seq_output_bundle(
                    model,
                    X_val_s,
                    y_val_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                cal_outputs_s = None
                if X_cal is not None and y_cal is not None and len(X_cal) > 0:
                    X_cal_s = x_scaler.transform(X_cal)
                    y_cal_s = y_scaler.transform(y_cal.reshape(-1, 1)).reshape(-1)
                    cal_outputs_s = collect_seq_output_bundle(
                        model,
                        X_cal_s,
                        y_cal_s,
                        lookback_default,
                        horizon,
                        device,
                        batch_size,
                        quantile_levels=quantiles,
                    )
                if val_outputs_s is not None:
                    y_true_val = _inverse_scaled_array(val_outputs_s["y_true"], y_scaler)
                    y_pred_val = _inverse_scaled_array(val_outputs_s["point_pred"], y_scaler)
                    model_q = residual_quantiles(y_true_val, y_pred_val, quantiles)
                else:
                    model_q = {}

                torch.save(
                    {
                        "model_type": "patchtst",
                        "state_dict": model.state_dict(),
                        "feature_cols": feat_cols,
                        "target": target,
                        "lookback": lookback_default,
                        "horizon": horizon,
                        "model_params": {
                            "patch_len": int(params.get("patch_len", 24)),
                            "stride": int(params.get("stride", 12)),
                            "d_model": int(params.get("d_model", 128)),
                            "n_heads": int(params.get("n_heads", 4)),
                            "num_layers": int(params.get("num_layers", 2)),
                            "dropout": float(params.get("dropout", 0.1)),
                            "dim_feedforward": int(params.get("dim_feedforward", 256)),
                            "lookback": int(lookback_default),
                            "horizon": int(horizon),
                        },
                        "quantiles": quantiles,
                        "quantile_levels": quantiles,
                        "n_quantiles": len(quantiles),
                        "residual_quantiles": model_q,
                        "feature_fill_values": feature_fill_values_list,
                        "x_scaler": x_scaler.to_dict(),
                        "y_scaler": y_scaler.to_dict(),
                    },
                    art_dir / f"patchtst_{target}.pt",
                )

                test_outputs_s = collect_seq_output_bundle(
                    model,
                    X_test_s,
                    y_test_s,
                    lookback_default,
                    horizon,
                    device,
                    batch_size,
                    quantile_levels=quantiles,
                )
                if test_outputs_s is not None:
                    y_true_test = _inverse_scaled_array(test_outputs_s["y_true"], y_scaler)
                    y_pred_test = _inverse_scaled_array(test_outputs_s["point_pred"], y_scaler)
                    model_quality = _sequence_model_quality_fields(
                        model_key="patchtst",
                        model=model,
                        params=params,
                        target=target,
                        X_train_s=X_train_s,
                        y_train_s=y_train_s,
                        X_test_s=X_test_s,
                        y_test_s=y_test_s,
                        val_outputs_s=val_outputs_s,
                        test_outputs_s=test_outputs_s,
                        y_scaler=y_scaler,
                        lookback=lookback_default,
                        horizon=horizon,
                        n_features=len(feat_cols),
                        device=device,
                        batch_size=batch_size,
                        quantile_levels=quantiles,
                    )
                    model_metrics = {**compute_metrics(y_true_test, y_pred_test, target), **model_quality}
                    test_outputs = {
                        "y_true": y_true_test,
                        "point_pred": y_pred_test,
                        "quantiles": _inverse_scaled_quantiles(test_outputs_s["quantiles"], y_scaler),
                    }
                    val_outputs = {
                        "y_true": y_true_val,
                        "point_pred": y_pred_val,
                        "quantiles": _inverse_scaled_quantiles(val_outputs_s["quantiles"], y_scaler)
                        if val_outputs_s is not None
                        else {},
                    }
                    cal_outputs = None
                    if cal_outputs_s is not None:
                        cal_outputs = {
                            "y_true": _inverse_scaled_array(cal_outputs_s["y_true"], y_scaler),
                            "point_pred": _inverse_scaled_array(cal_outputs_s["point_pred"], y_scaler),
                            "quantiles": _inverse_scaled_quantiles(cal_outputs_s["quantiles"], y_scaler),
                        }
                    if val_outputs_s is not None and uncertainty_enabled and target in uncertainty_targets:
                        _append_conformal_payload_from_outputs(
                            conformal_payloads=conformal_payloads,
                            target=target,
                            model_key="patchtst",
                            horizon=horizon,
                            split_boundaries=split_boundaries,
                            uncertainty_cfg=uncertainty_cfg,
                            quantile_levels=quantiles,
                            val_outputs=val_outputs,
                            test_outputs=test_outputs,
                            cal_outputs=cal_outputs,
                        )
                else:
                    model_metrics = {"rmse": None, "mae": None, "mape": None, "smape": None}
                target_res["patchtst"] = {**model_metrics, "residual_quantiles": model_q}
                if backtest_enabled and model_metrics.get("rmse") is not None:
                    backtest_payload["targets"].setdefault(target, {})
                    backtest_payload["targets"][target]["patchtst"] = walk_forward_horizon_metrics(
                        y_true_test, y_pred_test, horizon, target
                    )

        report["targets"][target] = target_res

    if uncertainty_enabled and conformal_payloads:
        backtests_dir = Path(
            args.backtests_dir or uncertainty_cfg.get("backtests_dir", "artifacts/backtests")
        )
        backtests_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = Path(
            args.uncertainty_artifacts_dir or uncertainty_cfg.get("artifacts_dir", "artifacts/uncertainty")
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        print("\nConformal Prediction Intervals:")
        for payload in conformal_payloads:
            target = str(payload["target"])
            model_key = str(
                payload.get("artifact_model_key") or _model_artifact_key(str(payload.get("model", "")))
            )
            report_model_key = _model_artifact_key(str(payload.get("model", "")))
            conf_cfg = _conformal_config_for_target(uncertainty_cfg, target)
            cal_path = backtests_dir / f"{model_key}_{target}_calibration.npz"
            test_path = backtests_dir / f"{model_key}_{target}_test.npz"
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.parent.mkdir(parents=True, exist_ok=True)
            method = str(payload.get("method", conf_cfg.method)).lower()
            if method == "cqr":
                np.savez(
                    cal_path,
                    y_true=payload["y_true_cal"],
                    q_lo=payload["q_lo_cal"],
                    q_hi=payload["q_hi_cal"],
                )
                np.savez(
                    test_path,
                    y_true=payload["y_true_test"],
                    q_lo=payload["q_lo_test"],
                    q_hi=payload["q_hi_test"],
                )
            else:
                np.savez(cal_path, y_true=payload["y_true_cal"], y_pred=payload["y_pred_cal"])
                np.savez(test_path, y_true=payload["y_true_test"], y_pred=payload["y_pred_test"])

            ci = ConformalInterval(conf_cfg)
            if method == "cqr":
                ci.fit_calibration_cqr(payload["y_true_cal"], payload["q_lo_cal"], payload["q_hi_cal"])
                interval_metrics = ci.evaluate_intervals_cqr(
                    payload["y_true_test"],
                    payload["q_lo_test"],
                    payload["q_hi_test"],
                    per_horizon=True,
                )
            else:
                ci.fit_calibration(payload["y_true_cal"], payload["y_pred_cal"])
                # Enhanced: per-horizon PICP + MPIW metrics
                interval_metrics = ci.evaluate_intervals(
                    payload["y_true_test"],
                    payload["y_pred_test"],
                    per_horizon=True,
                )

            conformal_path = artifacts_dir / f"{model_key}_{target}_conformal.json"
            meta = {
                "target": payload["target"],
                "model": payload["model"],
                "artifact_model_key": model_key,
                "method": method,
                "alpha": conf_cfg.alpha,
                "q_multiplier": conf_cfg.q_multiplier,
                "horizon": horizon,
                "calibration_source": payload.get("calibration_source", "val"),
                "calibration_rows": int(payload["y_true_cal"].shape[0]),
                "test_rows": int(payload["y_true_test"].shape[0]),
                "split_boundaries": payload.get("split_boundaries", {}),
                "global_coverage": interval_metrics["global_coverage"],
                "global_mean_width": interval_metrics["global_mean_width"],
                "picp_90": interval_metrics.get("picp_90"),
                "mean_interval_width": interval_metrics.get("mean_interval_width"),
                "pinball_loss_q05": interval_metrics.get("pinball_loss_q05"),
                "pinball_loss_q50": interval_metrics.get("pinball_loss_q50"),
                "pinball_loss_q95": interval_metrics.get("pinball_loss_q95"),
                "pinball_loss_mean": interval_metrics.get("pinball_loss_mean"),
                "winkler_score_90": interval_metrics.get("winkler_score_90"),
                "per_horizon_picp": interval_metrics.get("per_horizon_picp", {}),
                "per_horizon_mpiw": interval_metrics.get("per_horizon_mpiw", {}),
            }

            print(
                f"  {model_key}:{target}: Coverage={interval_metrics['global_coverage']:.3f}, "
                f"Width={interval_metrics['global_mean_width']:.2f}"
            )
            save_conformal(conformal_path, ci, meta=meta)
            if model_key == "gbm":
                legacy_conformal_path = artifacts_dir / f"{target}_conformal.json"
                save_conformal(legacy_conformal_path, ci, meta=meta)
                if method == "cqr":
                    np.savez(
                        backtests_dir / f"{target}_calibration.npz",
                        y_true=payload["y_true_cal"],
                        q_lo=payload["q_lo_cal"],
                        q_hi=payload["q_hi_cal"],
                    )
                    np.savez(
                        backtests_dir / f"{target}_test.npz",
                        y_true=payload["y_true_test"],
                        q_lo=payload["q_lo_test"],
                        q_hi=payload["q_hi_test"],
                    )
                else:
                    np.savez(
                        backtests_dir / f"{target}_calibration.npz",
                        y_true=payload["y_true_cal"],
                        y_pred=payload["y_pred_cal"],
                    )
                    np.savez(
                        backtests_dir / f"{target}_test.npz",
                        y_true=payload["y_true_test"],
                        y_pred=payload["y_pred_test"],
                    )

            target_report = report["targets"].setdefault(target, {})
            model_report = target_report.get(report_model_key)
            if isinstance(model_report, dict):
                model_report["uncertainty"] = {
                    "picp_90": interval_metrics.get("picp_90"),
                    "picp_95": interval_metrics.get("picp_95"),
                    "mean_interval_width": interval_metrics.get("mean_interval_width"),
                    "pinball_loss_q05": interval_metrics.get("pinball_loss_q05"),
                    "pinball_loss_q50": interval_metrics.get("pinball_loss_q50"),
                    "pinball_loss_q95": interval_metrics.get("pinball_loss_q95"),
                    "pinball_loss_mean": interval_metrics.get("pinball_loss_mean"),
                    "winkler_score_90": interval_metrics.get("winkler_score_90"),
                }
            print(f"Saved conformal artifacts to {cal_path}, {test_path}, {conformal_path}")

    # Write human-readable comparison report and machine-readable metrics.
    lines = ["# ML vs DL Comparison\n\n"]
    lines.append("## Setup\n")
    lines.append(f"- Targets: **{', '.join(targets)}**\n")
    lines.append(f"- Device: **{device}**\n")
    lines.append(f"- Quantiles: **{quantiles}**\n\n")

    for target, res in report["targets"].items():
        lines.append(f"## Target: {target}\n\n")
        lines.append("| Model | RMSE | MAPE |\n")
        lines.append("|---|---:|---:|\n")
        if "gbm" in res:
            m = res["gbm"]
            lines.append(f"| {m.get('model', 'gbm')} | {m['rmse']:.3f} | {m['mape']:.3f} |\n")
        if "lstm" in res:
            m = res["lstm"]
            if m["rmse"] is None:
                lines.append("| lstm | pending | pending |\n")
            else:
                lines.append(f"| lstm | {m['rmse']:.3f} | {m['mape']:.3f} |\n")
        if "tcn" in res:
            m = res["tcn"]
            if m["rmse"] is None:
                lines.append("| tcn | pending | pending |\n")
            else:
                lines.append(f"| tcn | {m['rmse']:.3f} | {m['mape']:.3f} |\n")
        lines.append("\n")

    (rep_dir / "ml_vs_dl_comparison.md").write_text("".join(lines), encoding="utf-8")
    (rep_dir / "week2_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if backtest_enabled and backtest_payload is not None:
        out_path = Path(
            args.walk_forward_report or backtest_cfg.get("out_path", rep_dir / "walk_forward_report.json")
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(backtest_payload, indent=2), encoding="utf-8")
    print(f"Saved reports to {rep_dir} and models to {art_dir}")


if __name__ == "__main__":
    main()
