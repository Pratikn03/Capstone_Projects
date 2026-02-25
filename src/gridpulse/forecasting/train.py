"""
Forecasting: Model Training Pipeline.

This is the main training orchestration module for GridPulse forecasting models.
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
    python -m gridpulse.forecasting.train --config configs/train_forecast.yaml
    
    # Train US EIA-930 data
    python -m gridpulse.forecasting.train --config configs/train_forecast_eia930.yaml
    
    # Train with Optuna tuning
    python -m gridpulse.forecasting.train --config configs/train_forecast.yaml --tune

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
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

from gridpulse.utils.metrics import rmse, mape, mae, smape, daylight_mape, r2_score
from gridpulse.utils.seed import set_seed
from gridpulse.utils.scaler import StandardScaler
from gridpulse.utils.manifest import create_run_manifest, print_manifest_summary
from gridpulse.forecasting.ml_gbm import train_gbm, predict_gbm
from gridpulse.forecasting.datasets import SeqConfig, TimeSeriesWindowDataset
from gridpulse.forecasting.dl_lstm import LSTMForecaster
from gridpulse.forecasting.dl_tcn import TCNForecaster
from gridpulse.forecasting.backtest import walk_forward_horizon_metrics
from gridpulse.forecasting.evaluate import time_series_cv_score
from gridpulse.forecasting.uncertainty.conformal import ConformalConfig, ConformalInterval, save_conformal

import torch
from torch.utils.data import DataLoader


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


def _enforce_data_contract(features_path: Path) -> dict[str, object]:
    """Validate feature schema and refresh dataset identity manifest before training."""
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run build_features first.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gridpulse.data_pipeline.validate_schema",
            "--in",
            str(features_path),
            "--report",
            "reports/data_quality_report_features.md",
        ],
        check=True,
    )

    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = repo_root / "data" / "dashboard" / "data_manifest.json"
    script_path = repo_root / "scripts" / "build_data_manifest.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--dataset",
            "ALL",
            "--output",
            str(manifest_path),
        ],
        cwd=str(repo_root),
        check=True,
    )

    payload: dict[str, object] = {}
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    return {
        "path": manifest_path,
        "sha256": _sha256_file(manifest_path) if manifest_path.exists() else None,
        "schema_hash": payload.get("schema_hash"),
        "split_boundaries": payload.get("split_boundaries", {}),
    }


def _reshape_horizon(y_true: np.ndarray, y_pred: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray] | None:
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
        return trial.suggest_float(spec["name"], float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
    if kind == "categorical":
        return trial.suggest_categorical(spec["name"], spec.get("choices", []))
    return None


def _resolve_split_cfg(cfg: dict) -> tuple[float, float, float, float]:
    """
    Resolve train/val/test split ratios from config.

    Priority:
    1. cfg["splits"]
    2. cfg["split"]
    3. defaults (0.70 / 0.15 / 0.15)
    """
    split_cfg = cfg.get("splits", cfg.get("split", {})) or {}
    train_ratio = float(split_cfg.get("train_ratio", 0.70))
    val_ratio = float(split_cfg.get("val_ratio", 0.15))
    calibration_ratio = float(split_cfg.get("calibration_ratio", 0.0) or 0.0)

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not (0.0 <= calibration_ratio < 1.0):
        raise ValueError(f"calibration_ratio must be in [0, 1), got {calibration_ratio}")

    test_ratio = 1.0 - train_ratio - val_ratio - calibration_ratio
    if test_ratio < 0.0:
        raise ValueError(
            f"Invalid split ratios: train_ratio({train_ratio}) + "
            f"val_ratio({val_ratio}) + calibration_ratio({calibration_ratio}) exceeds 1.0"
        )
    return train_ratio, val_ratio, test_ratio, calibration_ratio


def _resolve_gap_hours(cfg: dict) -> int:
    split_cfg = cfg.get("splits", cfg.get("split", {})) or {}
    gap_hours = int(split_cfg.get("gap_hours", 0) or 0)
    if gap_hours < 0:
        raise ValueError(f"gap_hours must be >= 0, got {gap_hours}")
    return gap_hours


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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    complete = [t for t in study.trials if t.value is not None]
    complete = sorted(complete, key=lambda t: float(t.value))

    selected = study.best_params
    if selection_mode == "top_pct_median" and complete:
        selected = _aggregate_top_trial_params(complete, param_specs, top_pct=top_pct, min_top_trials=min_top_trials)

    tuning_meta = {
        "enabled": True,
        "n_trials": int(n_trials),
        "metric": metric,
        "selection_mode": selection_mode,
        "select_top_pct": top_pct,
        "min_top_trials": min_top_trials,
        "n_complete_trials": len(complete),
        "best_objective": float(study.best_value) if study.best_value is not None else None,
        "selected_params": selected,
    }
    return {**base_params, **selected}, tuning_meta

def make_xy(df: pd.DataFrame, target: str, targets: list[str]):
    """Split a dataframe into features (X) and a single target (y)."""
    # Drop all targets from features to prevent leakage.
    drop = {"timestamp", *targets}
    feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].to_numpy()
    y = df[target].to_numpy()
    return X, y, feat_cols


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
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    no_improve_count = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            # Model outputs a full sequence; evaluate only forecast horizon.
            pred_seq = model(xb)
            pred_hz = pred_seq[:, -horizon:]
            loss = loss_fn(pred_hz, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
                pred_hz = pred_seq[:, -horizon:]
                val_losses.append(loss_fn(pred_hz, yb).item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        
        # Early stopping check
        if val_loss < best_val - 0.0001:  # min_delta
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        cur_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {ep}/{epochs} train_mse={np.mean(tr_losses):.4f} val_mse={val_loss:.4f} lr={cur_lr:.6f} best={best_val:.4f} patience={no_improve_count}/{patience}")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {ep} — no improvement for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Training complete — best val_mse={best_val:.4f}")
    return model


def train_lstm_model(X_train, y_train, X_val, y_val, cfg: dict, device: str = "cpu"):
    """Train an LSTM sequence model with advanced settings."""
    lookback = int(cfg.get("lookback", 168))
    horizon = int(cfg.get("horizon", 24))
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 100))
    hidden_size = int(cfg.get("hidden_size", 256))
    num_layers = int(cfg.get("num_layers", 3))
    dropout = float(cfg.get("dropout", 0.3))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 1e-3)))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    grad_clip = float(cfg.get("gradient_clip", 1.0))
    
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
    ).to(device)
    model = fit_sequence_model(
        model, train_dl, val_dl, 
        epochs=epochs, lr=lr, horizon=horizon, device=device,
        weight_decay=weight_decay, patience=patience, 
        warmup_epochs=warmup_epochs, grad_clip=grad_clip,
    )
    return model


def train_tcn_model(X_train, y_train, X_val, y_val, cfg: dict, device: str = "cpu"):
    """Train a TCN sequence model with advanced settings."""
    lookback = int(cfg.get("lookback", 168))
    horizon = int(cfg.get("horizon", 24))
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 100))
    num_channels = cfg.get("num_channels", [128, 128, 128, 64])
    kernel_size = int(cfg.get("kernel_size", 5))
    dropout = float(cfg.get("dropout", 0.3))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 1e-3)))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    grad_clip = float(cfg.get("gradient_clip", 1.0))
    
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
    ).to(device)
    model = fit_sequence_model(
        model, train_dl, val_dl, 
        epochs=epochs, lr=lr, horizon=horizon, device=device,
        weight_decay=weight_decay, patience=patience, 
        warmup_epochs=warmup_epochs, grad_clip=grad_clip,
    )
    return model


def collect_seq_preds(model, X: np.ndarray, y: np.ndarray, lookback: int, horizon: int, device: str, batch_size: int):
    """Collect sequential predictions for evaluation."""
    ds = TimeSeriesWindowDataset(X, y, SeqConfig(lookback=lookback, horizon=horizon))
    if len(ds) == 0:
        return None, None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            pred_seq = model(xb).cpu().numpy()
            pred_hz = pred_seq[:, -horizon:]
            preds.append(pred_hz.reshape(-1))
            trues.append(yb.numpy().reshape(-1))
    return np.concatenate(trues), np.concatenate(preds)


def evaluate_seq_model(model, X, y, lookback, horizon, device, batch_size, target: str, y_scaler: StandardScaler | None = None):
    """Evaluate a sequence model and return metrics on original scale."""
    y_true, y_pred = collect_seq_preds(model, X, y, lookback, horizon, device, batch_size)
    if y_true is None:
        return {"rmse": None, "mae": None, "mape": None, "smape": None, "note": "insufficient window"}
    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    return compute_metrics(y_true, y_pred, target)


def main():
    """CLI entrypoint to train GBM/LSTM/TCN models and write reports."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_forecast.yaml")
    p.add_argument("--targets", default=None, help="Comma-separated targets to train (override config)")
    p.add_argument("--models", default=None, help="Comma-separated models to train: gbm,lstm,tcn")
    p.add_argument("--skip-existing", action="store_true", help="Skip training if model artifact already exists")
    p.add_argument("--enable-cv", action="store_true", help="Enable time-series cross-validation (fold count from config)")
    p.add_argument("--use-pipeline", action="store_true", help="Use sklearn Pipeline for GBM (leakage-safe preprocessing)")
    p.add_argument("--tune", action="store_true", help="Force-enable hyperparameter tuning")
    p.add_argument("--no-tune", action="store_true", help="Disable hyperparameter tuning even if enabled in config")
    p.add_argument("--ensemble", action="store_true", help="Train GBM seed ensemble (uses config.seeds)")
    p.add_argument("--max-seeds", type=int, default=None, help="Optional cap on ensemble seed count")
    p.add_argument("--n-trials", type=int, default=None, help="Override Optuna trial count for this run")
    p.add_argument("--top-pct", type=float, default=None, help="Override top-percent trial selection, e.g. 0.30")
    args = p.parse_args()

    # Config-driven training keeps runs reproducible across datasets.
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
    )
    
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    
    # Create run manifest for reproducibility.
    art_dir = Path(cfg["artifacts"]["out_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    cv_cfg = cfg.get("cross_validation", {}) if isinstance(cfg.get("cross_validation"), dict) else {}
    cv_enabled = bool(args.enable_cv or cv_cfg.get("enabled", False))

    # Load prepared features (Parquet) for training.
    features_path = Path(cfg["data"]["processed_path"])
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run build_features first.")
    data_contract = _enforce_data_contract(features_path)

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
        },
        data_manifest_path=data_contract.get("path"),
        data_manifest_sha256=data_contract.get("sha256"),
        split_boundaries=data_contract.get("split_boundaries"),
        schema_hash=str(data_contract.get("schema_hash")) if data_contract.get("schema_hash") is not None else None,
    )
    print_manifest_summary(manifest)

    # Time-ordered split to prevent leakage.
    df = pd.read_parquet(features_path).sort_values("timestamp")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    horizon = int(cfg["task"]["horizon_hours"])
    lookback_default = int(cfg["task"].get("lookback_hours", 168))

    rep_dir = Path(cfg["reports"]["out_dir"])
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
    tuning_enabled = bool((tuning_cfg.get("enabled", False) or args.tune) and not args.no_tune)

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
    uncertainty_cfg = _load_uncertainty_cfg()
    uncertainty_enabled = bool(uncertainty_cfg.get("enabled", False))
    uncertainty_targets = _parse_uncertainty_targets(uncertainty_cfg, targets)
    conformal_payloads: dict[str, dict] = {}

    # Optional model/target filters to avoid retraining everything.
    model_filter = None
    if args.models:
        model_filter = {m.strip().lower() for m in args.models.split(",") if m.strip()}

    def has_gbm_artifact(target_name: str) -> bool:
        return any(art_dir.glob(f"gbm_*_{target_name}.pkl"))

    for target in targets:
        # 1) Build feature/target matrices for the current target.
        X_train, y_train, feat_cols = make_xy(train_df, target, targets)
        X_val, y_val, _ = make_xy(val_df, target, targets)
        X_test, y_test, _ = make_xy(test_df, target, targets)
        X_cal = None
        y_cal = None
        if not calibration_df.empty:
            X_cal, y_cal, _ = make_xy(calibration_df, target, targets)

        target_res = {"n_features": len(feat_cols)}

        # ---- GBM (strong tabular baseline) ----
        gbm_cfg = cfg["models"].get("baseline_gbm", {})
        if gbm_cfg.get("enabled", True) and (model_filter is None or "gbm" in model_filter):
            if args.skip_existing and has_gbm_artifact(target):
                print(f"Skipping GBM for {target} (artifact exists)")
            else:
                gbm_params = dict(gbm_cfg.get("params", {}))
                gbm_params.setdefault("random_state", int(cfg.get("seed", 42)))
                tuning_meta = None
                if tuning_enabled:
                    gbm_params, tuning_meta = tune_gbm_params(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        gbm_params,
                        tuning_cfg,
                        int(cfg.get("seed", 42)),
                    )
                seed_list = ensemble_seeds if args.ensemble else [int(cfg.get("seed", 42))]
                model_kind = "gbm"
                gbm_members: list[dict] = []
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
                    val_preds.append(pred_val_i)
                    test_preds.append(pred_test_i)

                gbm_pred_val = np.mean(np.vstack(val_preds), axis=0)
                gbm_pred_test = np.mean(np.vstack(test_preds), axis=0)
                gbm_pred_cal = None
                if X_cal is not None and len(X_cal) > 0:
                    cal_preds = [predict_gbm(member["model"], X_cal) for member in gbm_members]
                    gbm_pred_cal = np.mean(np.vstack(cal_preds), axis=0)
                gbm_metrics = {
                    **compute_metrics(y_test, gbm_pred_test, target),
                    "model": f"gbm_{model_kind}",
                    "ensemble_members": len(gbm_members),
                }
                
                # Optional: time-series cross-validation
                if cv_enabled:
                    cv_folds = int(cv_cfg.get("n_folds", 5))
                    print(f"Running {cv_folds}-fold CV for GBM on {target}...")
                    # CV on combined train+val for robust evaluation
                    X_trainval = np.vstack([X_train, X_val])
                    y_trainval = np.concatenate([y_train, y_val])
                    
                    cv_results = time_series_cv_score(
                        X=X_trainval,
                        y=y_trainval,
                        train_fn=lambda X, y: train_gbm(
                            X, y, gbm_params, use_pipeline=args.use_pipeline, preprocessing="standard" if args.use_pipeline else None
                        )[1],
                        predict_fn=predict_gbm,
                        n_splits=cv_folds,
                        target=target,
                    )
                    
                    # Save CV results
                    cv_path = rep_dir / f"{target}_gbm_cv_results.json"
                    cv_path.write_text(json.dumps(cv_results, indent=2), encoding="utf-8")
                    print(f"  CV RMSE: {cv_results['aggregated']['rmse_mean']:.2f} ± {cv_results['aggregated']['rmse_std']:.2f}")
                    gbm_metrics["cv_results"] = cv_results["aggregated"]
                
                gbm_q = residual_quantiles(y_val, gbm_pred_val, quantiles)

                # Optional: train quantile GBM models for calibrated intervals.
                quantile_models = {}
                quant_cfg = cfg["models"].get("quantile_gbm", {})
                quantile_predictions: dict[str, dict[str, np.ndarray]] = {}
                if quant_cfg.get("enabled", False):
                    lgb = _try_lightgbm(required=True)
                    q_params_base = dict(quant_cfg.get("params", {}))
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
                            quantile_predictions[str(q)]["calibration"] = np.asarray(q_model.predict(X_cal), dtype=float)

                if uncertainty_enabled and target in uncertainty_targets and target not in conformal_payloads:
                    conf_meta_cfg = uncertainty_cfg.get("conformal", {}) if isinstance(uncertainty_cfg.get("conformal"), dict) else {}
                    conf_method = str(conf_meta_cfg.get("method", "residual")).lower()
                    cal_split = str(uncertainty_cfg.get("calibration_split", "val")).lower()
                    calibration_source = "val"
                    if cal_split == "test":
                        cal_true = y_test
                        cal_pred = gbm_pred_test
                        test_true = y_val
                        test_pred = gbm_pred_val
                        calibration_source = "test"
                    elif cal_split == "calibration":
                        if gbm_pred_cal is not None and y_cal is not None and len(y_cal) > 0:
                            cal_true = y_cal
                            cal_pred = gbm_pred_cal
                            calibration_source = "calibration"
                        else:
                            print(
                                f"Warning: target={target} requested calibration split but calibration set is empty; "
                                "falling back to validation split."
                            )
                            cal_true = y_val
                            cal_pred = gbm_pred_val
                            calibration_source = "val_fallback_missing_calibration"
                        test_true = y_test
                        test_pred = gbm_pred_test
                    else:
                        cal_true = y_val
                        cal_pred = gbm_pred_val
                        test_true = y_test
                        test_pred = gbm_pred_test
                        calibration_source = "val"

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
                        cal_q_lo_src = quantile_predictions[lo_key].get("calibration" if calibration_source.startswith("calibration") else calibration_source.replace("_fallback_missing_calibration", "val"))
                        cal_q_hi_src = quantile_predictions[hi_key].get("calibration" if calibration_source.startswith("calibration") else calibration_source.replace("_fallback_missing_calibration", "val"))
                        if calibration_source == "test":
                            cal_q_lo_src = quantile_predictions[lo_key].get("test")
                            cal_q_hi_src = quantile_predictions[hi_key].get("test")
                            test_q_lo_src = quantile_predictions[lo_key].get("val")
                            test_q_hi_src = quantile_predictions[hi_key].get("val")
                        elif calibration_source.startswith("calibration"):
                            cal_q_lo_src = quantile_predictions[lo_key].get("calibration")
                            cal_q_hi_src = quantile_predictions[hi_key].get("calibration")
                            test_q_lo_src = quantile_predictions[lo_key].get("test")
                            test_q_hi_src = quantile_predictions[hi_key].get("test")
                        else:
                            cal_q_lo_src = quantile_predictions[lo_key].get("val")
                            cal_q_hi_src = quantile_predictions[hi_key].get("val")
                            test_q_lo_src = quantile_predictions[lo_key].get("test")
                            test_q_hi_src = quantile_predictions[hi_key].get("test")

                        if cal_q_lo_src is None or cal_q_hi_src is None or test_q_lo_src is None or test_q_hi_src is None:
                            raise RuntimeError(
                                f"CQR configured for target={target} but missing quantile predictions for "
                                f"calibration_source={calibration_source}."
                            )
                        cal_q_lo = _reshape_horizon(cal_true, np.asarray(cal_q_lo_src, dtype=float), horizon)
                        cal_q_hi = _reshape_horizon(cal_true, np.asarray(cal_q_hi_src, dtype=float), horizon)
                        test_q_lo = _reshape_horizon(test_true, np.asarray(test_q_lo_src, dtype=float), horizon)
                        test_q_hi = _reshape_horizon(test_true, np.asarray(test_q_hi_src, dtype=float), horizon)
                        if cal_q_lo is None or cal_q_hi is None or test_q_lo is None or test_q_hi is None:
                            raise RuntimeError(f"CQR reshape failed for target={target} due horizon alignment.")
                        conformal_payloads[target] = {
                            "target": target,
                            "model": f"gbm_{model_kind}",
                            "calibration_source": calibration_source,
                            "split_boundaries": split_boundaries,
                            "method": "cqr",
                            "quantile_lo_key": lo_key,
                            "quantile_hi_key": hi_key,
                            "y_true_cal": cal[0],
                            "q_lo_cal": cal_q_lo[1],
                            "q_hi_cal": cal_q_hi[1],
                            "y_true_test": test[0],
                            "q_lo_test": test_q_lo[1],
                            "q_hi_test": test_q_hi[1],
                        }
                    else:
                        conformal_payloads[target] = {
                            "target": target,
                            "model": f"gbm_{model_kind}",
                            "calibration_source": calibration_source,
                            "split_boundaries": split_boundaries,
                            "method": "residual",
                            "y_true_cal": cal[0],
                            "y_pred_cal": cal[1],
                            "y_true_test": test[0],
                            "y_pred_test": test[1],
                        }

                import pickle
                with open(art_dir / f"{gbm_metrics['model']}_{target}.pkl", "wb") as f:
                    pickle.dump({
                        "model_type": "gbm",
                        # Keep single-model key for compatibility; use ensemble list when present.
                        "model": gbm_members[0]["model"],
                        "ensemble_models": [m["model"] for m in gbm_members] if len(gbm_members) > 1 else None,
                        "ensemble_seeds": [m["seed"] for m in gbm_members],
                        "feature_cols": feat_cols,
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
                    }, f)

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

                model = train_lstm_model(X_train_s, y_train_s, X_val_s, y_val_s, {
                    "lookback": lookback_default,
                    "horizon": horizon,
                    **params,
                }, device=device)
                batch_size = int(params.get("batch_size", 256))
                y_true_val_s, y_pred_val_s = collect_seq_preds(model, X_val_s, y_val_s, lookback_default, horizon, device, batch_size)
                if y_true_val_s is not None:
                    y_true_val = y_scaler.inverse_transform(y_true_val_s.reshape(-1, 1)).reshape(-1)
                    y_pred_val = y_scaler.inverse_transform(y_pred_val_s.reshape(-1, 1)).reshape(-1)
                    lstm_q = residual_quantiles(y_true_val, y_pred_val, quantiles)
                else:
                    lstm_q = {}

                torch.save({
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
                        "horizon": int(horizon),
                    },
                    "quantiles": quantiles,
                    "residual_quantiles": lstm_q,
                    "x_scaler": x_scaler.to_dict(),
                    "y_scaler": y_scaler.to_dict(),
                }, art_dir / f"lstm_{target}.pt")

                y_true_test_s, y_pred_test_s = collect_seq_preds(model, X_test_s, y_test_s, lookback_default, horizon, device, batch_size)
                if y_true_test_s is not None:
                    y_true_test = y_scaler.inverse_transform(y_true_test_s.reshape(-1, 1)).reshape(-1)
                    y_pred_test = y_scaler.inverse_transform(y_pred_test_s.reshape(-1, 1)).reshape(-1)
                    lstm_metrics = compute_metrics(y_true_test, y_pred_test, target)
                else:
                    lstm_metrics = {"rmse": None, "mae": None, "mape": None, "smape": None}
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

                model = train_tcn_model(X_train_s, y_train_s, X_val_s, y_val_s, {
                    "lookback": lookback_default,
                    "horizon": horizon,
                    **params,
                }, device=device)
                batch_size = int(params.get("batch_size", 256))
                y_true_val_s, y_pred_val_s = collect_seq_preds(model, X_val_s, y_val_s, lookback_default, horizon, device, batch_size)
                if y_true_val_s is not None:
                    y_true_val = y_scaler.inverse_transform(y_true_val_s.reshape(-1, 1)).reshape(-1)
                    y_pred_val = y_scaler.inverse_transform(y_pred_val_s.reshape(-1, 1)).reshape(-1)
                    tcn_q = residual_quantiles(y_true_val, y_pred_val, quantiles)
                else:
                    tcn_q = {}

                torch.save({
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
                    },
                    "quantiles": quantiles,
                    "residual_quantiles": tcn_q,
                    "x_scaler": x_scaler.to_dict(),
                    "y_scaler": y_scaler.to_dict(),
                }, art_dir / f"tcn_{target}.pt")

                y_true_test_s, y_pred_test_s = collect_seq_preds(model, X_test_s, y_test_s, lookback_default, horizon, device, batch_size)
                if y_true_test_s is not None:
                    y_true_test = y_scaler.inverse_transform(y_true_test_s.reshape(-1, 1)).reshape(-1)
                    y_pred_test = y_scaler.inverse_transform(y_pred_test_s.reshape(-1, 1)).reshape(-1)
                    tcn_metrics = compute_metrics(y_true_test, y_pred_test, target)
                else:
                    tcn_metrics = {"rmse": None, "mae": None, "mape": None, "smape": None}
                target_res["tcn"] = {**tcn_metrics, "residual_quantiles": tcn_q}
                if backtest_enabled and tcn_metrics.get("rmse") is not None:
                    backtest_payload["targets"].setdefault(target, {})
                    backtest_payload["targets"][target]["tcn"] = walk_forward_horizon_metrics(
                        y_true_test, y_pred_test, horizon, target
                    )

        report["targets"][target] = target_res

    if uncertainty_enabled and conformal_payloads:
        multi = len(conformal_payloads) > 1
        cal_template = str(uncertainty_cfg.get("calibration_npz", "artifacts/backtests/calibration.npz"))
        test_template = str(uncertainty_cfg.get("test_npz", "artifacts/backtests/test.npz"))
        conf_cfg = ConformalConfig(**(uncertainty_cfg.get("conformal", {}) or {}))
        artifacts_dir = Path(uncertainty_cfg.get("artifacts_dir", "artifacts/uncertainty"))
        
        print("\nConformal Prediction Intervals:")
        for target, payload in conformal_payloads.items():
            cal_path = _format_uncertainty_path(cal_template, target, "calibration", multi)
            test_path = _format_uncertainty_path(test_template, target, "test", multi)
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
            
            conformal_path = artifacts_dir / f"{payload['target']}_conformal.json"
            meta = {
                "target": payload["target"],
                "model": payload["model"],
                "method": method,
                "horizon": horizon,
                "calibration_source": payload.get("calibration_source", "val"),
                "calibration_rows": int(payload["y_true_cal"].shape[0]),
                "test_rows": int(payload["y_true_test"].shape[0]),
                "split_boundaries": payload.get("split_boundaries", {}),
                "global_coverage": interval_metrics["global_coverage"],
                "global_mean_width": interval_metrics["global_mean_width"],
                "per_horizon_picp": interval_metrics.get("per_horizon_picp", {}),
                "per_horizon_mpiw": interval_metrics.get("per_horizon_mpiw", {}),
            }
            
            print(f"  {target}: Coverage={interval_metrics['global_coverage']:.3f}, Width={interval_metrics['global_mean_width']:.2f}")
            save_conformal(conformal_path, ci, meta=meta)
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
        out_path = Path(backtest_cfg.get("out_path", rep_dir / "walk_forward_report.json"))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(backtest_payload, indent=2), encoding="utf-8")
    print("Saved report to reports/ and models to artifacts/models")


if __name__ == "__main__":
    main()
