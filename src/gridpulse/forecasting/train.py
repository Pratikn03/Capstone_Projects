"""Forecasting: train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml

from gridpulse.utils.metrics import rmse, mape, mae, smape, daylight_mape
from gridpulse.utils.seed import set_seed
from gridpulse.utils.scaler import StandardScaler
from gridpulse.forecasting.ml_gbm import train_gbm, predict_gbm
from gridpulse.forecasting.datasets import SeqConfig, TimeSeriesWindowDataset
from gridpulse.forecasting.dl_lstm import LSTMForecaster
from gridpulse.forecasting.dl_tcn import TCNForecaster
from gridpulse.forecasting.backtest import walk_forward_horizon_metrics

import torch
from torch.utils.data import DataLoader

def make_xy(df: pd.DataFrame, target: str, targets: list[str]):
    # Key: prepare features/targets and train or evaluate models
    # Drop all targets from features to prevent leakage.
    drop = {"timestamp", *targets}
    feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].to_numpy()
    y = df[target].to_numpy()
    return X, y, feat_cols


def residual_quantiles(y_true: np.ndarray, y_pred: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    return {str(q): float(np.quantile(resid, q)) for q in quantiles}

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> dict:
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if target == "solar_mw":
        # Solar MAPE explodes at night; use daylight-only MAPE.
        out["daylight_mape"] = daylight_mape(y_true, y_pred)
    return out


def fit_sequence_model(model, train_dl, val_dl, epochs: int, lr: float, horizon: int, device: str):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience = 3
    wait = 0

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
            opt.step()
            tr_losses.append(loss.item())

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
        # Early stopping on validation loss for stability.
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        print(f"Epoch {ep}/{epochs} train_mse={np.mean(tr_losses):.4f} val_mse={val_loss:.4f}")
        if wait >= patience:
            print(f"Early stopping at epoch {ep} (best_val={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_lstm_model(X_train, y_train, X_val, y_val, cfg: dict, device: str = "cpu"):
    lookback = int(cfg.get("lookback", 168))
    horizon = int(cfg.get("horizon", 24))
    batch_size = int(cfg.get("batch_size", 256))
    epochs = int(cfg.get("epochs", 8))
    hidden_size = int(cfg.get("hidden_size", 128))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.1))
    lr = float(cfg.get("lr", 1e-3))

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
    model = fit_sequence_model(model, train_dl, val_dl, epochs=epochs, lr=lr, horizon=horizon, device=device)
    return model


def train_tcn_model(X_train, y_train, X_val, y_val, cfg: dict, device: str = "cpu"):
    lookback = int(cfg.get("lookback", 168))
    horizon = int(cfg.get("horizon", 24))
    batch_size = int(cfg.get("batch_size", 256))
    epochs = int(cfg.get("epochs", 8))
    num_channels = cfg.get("num_channels", [32, 32, 32])
    kernel_size = int(cfg.get("kernel_size", 3))
    dropout = float(cfg.get("dropout", 0.1))
    lr = float(cfg.get("lr", 1e-3))

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
    model = fit_sequence_model(model, train_dl, val_dl, epochs=epochs, lr=lr, horizon=horizon, device=device)
    return model


def collect_seq_preds(model, X: np.ndarray, y: np.ndarray, lookback: int, horizon: int, device: str, batch_size: int):
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
    y_true, y_pred = collect_seq_preds(model, X, y, lookback, horizon, device, batch_size)
    if y_true is None:
        return {"rmse": None, "mae": None, "mape": None, "smape": None, "note": "insufficient window"}
    if y_scaler is not None:
        y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1)
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    return compute_metrics(y_true, y_pred, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_forecast.yaml")
    p.add_argument("--targets", default=None, help="Comma-separated targets to train (override config)")
    p.add_argument("--models", default=None, help="Comma-separated models to train: gbm,lstm,tcn")
    p.add_argument("--skip-existing", action="store_true", help="Skip training if model artifact already exists")
    args = p.parse_args()

    # Config-driven training keeps runs reproducible across datasets.
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
    )
    set_seed(int(cfg.get("seed", 42)))
    features_path = Path(cfg["data"]["processed_path"])
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run build_features first.")

    # Time-ordered split to prevent leakage.
    df = pd.read_parquet(features_path).sort_values("timestamp")
    n = len(df)
    train_df = df.iloc[: int(n * 0.7)]
    val_df = df.iloc[int(n * 0.7): int(n * 0.85)]
    test_df = df.iloc[int(n * 0.85):]

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

    art_dir = Path(cfg["artifacts"]["out_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(cfg["reports"]["out_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "device": device,
        "quantiles": quantiles,
        "targets": {},
    }
    backtest_cfg = cfg.get("backtest", {})
    backtest_enabled = bool(backtest_cfg.get("enabled", False))
    backtest_payload = {"targets": {}} if backtest_enabled else None

    # Optional model/target filters to avoid retraining everything.
    model_filter = None
    if args.models:
        model_filter = {m.strip().lower() for m in args.models.split(",") if m.strip()}

    def has_gbm_artifact(target_name: str) -> bool:
        return any(art_dir.glob(f"gbm_*_{target_name}.pkl"))

    for target in targets:
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
                model_kind, gbm = train_gbm(X_train, y_train, gbm_params)
                gbm_pred_test = predict_gbm(gbm, X_test)
                gbm_pred_val = predict_gbm(gbm, X_val)
                gbm_metrics = {
                    **compute_metrics(y_test, gbm_pred_test, target),
                    "model": f"gbm_{model_kind}",
                }
                gbm_q = residual_quantiles(y_val, gbm_pred_val, quantiles)

                import pickle
                with open(art_dir / f"{gbm_metrics['model']}_{target}.pkl", "wb") as f:
                    pickle.dump({
                        "model_type": "gbm",
                        "model": gbm,
                        "feature_cols": feat_cols,
                        "target": target,
                        "quantiles": quantiles,
                        "residual_quantiles": gbm_q,
                    }, f)

                target_res["gbm"] = {**gbm_metrics, "residual_quantiles": gbm_q}
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
