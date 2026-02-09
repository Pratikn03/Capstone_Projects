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
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml

from gridpulse.utils.metrics import rmse, mape, mae, smape, daylight_mape, r2_score
from gridpulse.utils.seed import set_seed
from gridpulse.utils.scaler import StandardScaler
from gridpulse.utils.manifest import create_run_manifest, print_manifest_summary
from gridpulse.forecasting.ml_gbm import train_gbm, predict_gbm, extract_base_model
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

def _try_optuna():
    """
    Import Optuna for hyperparameter optimization.
    
    Optuna is optional - if not installed, tuning will be skipped
    and default hyperparameters will be used.
    """
    try:
        import optuna  # type: ignore
        return optuna
    except Exception:
        return None


def _try_lightgbm():
    """Import LightGBM if available (needed for quantile GBM and tuning)."""
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception:
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
    optuna = _try_optuna()
    lgb = _try_lightgbm()
    if optuna is None or lgb is None:
        print("Optuna or LightGBM not available; skipping tuning.")
        return base_params

    params_cfg = tuning_cfg.get("params", {}).get("baseline_gbm", {})
    n_trials = int(tuning_cfg.get("n_trials", 20))
    metric = tuning_cfg.get("metric", "val_loss")

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
    best = study.best_params
    return {**base_params, **best}

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
    p.add_argument("--enable-cv", action="store_true", help="Enable time-series cross-validation (5-fold)")
    p.add_argument("--use-pipeline", action="store_true", help="Use sklearn Pipeline for GBM (leakage-safe preprocessing)")
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
    
    # Create run manifest for reproducibility
    art_dir = Path(cfg["artifacts"]["out_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating run manifest for reproducibility...")
    manifest = create_run_manifest(
        config_path=args.config,
        output_dir=art_dir,
        extra_metadata={
            "enable_cv": args.enable_cv,
            "use_pipeline": args.use_pipeline,
            "seed": seed,
        },
    )
    print_manifest_summary(manifest)
    
    # Load prepared features (Parquet) for training.
    features_path = Path(cfg["data"]["processed_path"])
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run build_features first.")

    # Time-ordered split to prevent leakage.
    df = pd.read_parquet(features_path).sort_values("timestamp")
    n = len(df)
    train_df = df.iloc[: int(n * 0.7)]
    val_df = df.iloc[int(n * 0.7): int(n * 0.85)]
    test_df = df.iloc[int(n * 0.85):]

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
    tuning_cfg = cfg.get("tuning", {}) or {}
    tuning_enabled = bool(tuning_cfg.get("enabled", False))
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

        target_res = {"n_features": len(feat_cols)}

        # ---- GBM (strong tabular baseline) ----
        gbm_cfg = cfg["models"].get("baseline_gbm", {})
        if gbm_cfg.get("enabled", True) and (model_filter is None or "gbm" in model_filter):
            if args.skip_existing and has_gbm_artifact(target):
                print(f"Skipping GBM for {target} (artifact exists)")
            else:
                gbm_params = dict(gbm_cfg.get("params", {}))
                gbm_params.setdefault("random_state", int(cfg.get("seed", 42)))
                if tuning_enabled:
                    gbm_params = tune_gbm_params(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        gbm_params,
                        tuning_cfg,
                        int(cfg.get("seed", 42)),
                    )
                
                # Train with optional sklearn Pipeline for leakage-safe preprocessing
                model_kind, gbm = train_gbm(
                    X_train, 
                    y_train, 
                    gbm_params, 
                    use_pipeline=args.use_pipeline,
                    preprocessing="standard" if args.use_pipeline else None,
                )
                
                gbm_pred_test = predict_gbm(gbm, X_test)
                gbm_pred_val = predict_gbm(gbm, X_val)
                gbm_metrics = {
                    **compute_metrics(y_test, gbm_pred_test, target),
                    "model": f"gbm_{model_kind}",
                }
                
                # Optional: time-series cross-validation
                if args.enable_cv:
                    print(f"Running 5-fold CV for GBM on {target}...")
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
                        n_splits=5,
                        target=target,
                    )
                    
                    # Save CV results
                    cv_path = rep_dir / f"{target}_gbm_cv_results.json"
                    cv_path.write_text(json.dumps(cv_results, indent=2), encoding="utf-8")
                    print(f"  CV RMSE: {cv_results['aggregated']['rmse_mean']:.2f} ± {cv_results['aggregated']['rmse_std']:.2f}")
                    gbm_metrics["cv_results"] = cv_results["aggregated"]
                
                if uncertainty_enabled and target in uncertainty_targets and target not in conformal_payloads:
                    cal_split = str(uncertainty_cfg.get("calibration_split", "val")).lower()
                    if cal_split == "test":
                        cal = _reshape_horizon(y_test, gbm_pred_test, horizon)
                        test = _reshape_horizon(y_val, gbm_pred_val, horizon)
                    else:
                        cal = _reshape_horizon(y_val, gbm_pred_val, horizon)
                        test = _reshape_horizon(y_test, gbm_pred_test, horizon)
                    if cal is not None and test is not None:
                        conformal_payloads[target] = {
                            "target": target,
                            "model": f"gbm_{model_kind}",
                            "y_true_cal": cal[0],
                            "y_pred_cal": cal[1],
                            "y_true_test": test[0],
                            "y_pred_test": test[1],
                        }
                gbm_q = residual_quantiles(y_val, gbm_pred_val, quantiles)

                # Optional: train quantile GBM models for calibrated intervals.
                quantile_models = {}
                quant_cfg = cfg["models"].get("quantile_gbm", {})
                if quant_cfg.get("enabled", False):
                    lgb = _try_lightgbm()
                    if lgb is None:
                        print("Quantile GBM enabled but LightGBM not available; skipping quantile models.")
                    else:
                        q_params_base = dict(quant_cfg.get("params", {}))
                        for q in quantiles:
                            params = {**q_params_base, "objective": "quantile", "alpha": float(q)}
                            params.setdefault("random_state", int(cfg.get("seed", 42)))
                            q_model = lgb.LGBMRegressor(**params)
                            q_model.fit(X_train, y_train)
                            quantile_models[str(q)] = q_model

                import pickle
                with open(art_dir / f"{gbm_metrics['model']}_{target}.pkl", "wb") as f:
                    pickle.dump({
                        "model_type": "gbm",
                        "model": gbm,
                        "feature_cols": feat_cols,
                        "target": target,
                        "quantiles": quantiles,
                        "residual_quantiles": gbm_q,
                        "quantile_models": quantile_models if quantile_models else None,
                        "tuned_params": gbm_params if tuning_enabled else None,
                    }, f)

                target_res["gbm"] = {
                    **gbm_metrics,
                    "residual_quantiles": gbm_q,
                    "tuned_params": gbm_params if tuning_enabled else None,
                    "quantile_models": list(quantile_models.keys()) if quantile_models else None,
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
            np.savez(cal_path, y_true=payload["y_true_cal"], y_pred=payload["y_pred_cal"])
            np.savez(test_path, y_true=payload["y_true_test"], y_pred=payload["y_pred_test"])

            ci = ConformalInterval(conf_cfg)
            ci.fit_calibration(payload["y_true_cal"], payload["y_pred_cal"])
            
            # Enhanced: per-horizon PICP + MPIW metrics
            interval_metrics = ci.evaluate_intervals(
                payload["y_true_test"], 
                payload["y_pred_test"],
                per_horizon=True
            )
            
            conformal_path = artifacts_dir / f"{payload['target']}_conformal.json"
            meta = {
                "target": payload["target"],
                "model": payload["model"],
                "horizon": horizon,
                "calibration_rows": int(payload["y_true_cal"].shape[0]),
                "test_rows": int(payload["y_true_test"].shape[0]),
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
