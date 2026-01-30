from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gridpulse.anomaly.isolation_forest import fit_iforest, predict_iforest


def _load_config(cfg_path: str | Path | None) -> dict:
    if cfg_path is None:
        cfg_path = Path("configs/anomaly.yaml")
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        return {
            "residual_rules": {"z_threshold": 3.0, "window": 168},
            "isolation_forest": {"enabled": True, "contamination": 0.01, "random_state": 42},
        }
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _residual_z_flags(residuals: np.ndarray, window: int, z_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1:
        window = max(3, len(residuals) // 10)
    s = pd.Series(residuals)
    mean = s.rolling(window=window, min_periods=max(10, window // 2)).mean()
    std = s.rolling(window=window, min_periods=max(10, window // 2)).std().replace(0, np.nan)
    z = (s - mean) / std
    z = z.fillna(0.0).to_numpy()
    flags = np.abs(z) >= z_threshold
    return flags.astype(bool), z


def detect_anomalies(
    actual: np.ndarray | list[float],
    forecast: np.ndarray | list[float],
    features: Optional[pd.DataFrame | np.ndarray | list[list[float]]] = None,
    config_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Detect anomalies using residual z-score + optional Isolation Forest."""
    cfg = _load_config(config_path)
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if actual.shape != forecast.shape:
        raise ValueError("actual and forecast must have the same shape")

    residuals = actual - forecast
    z_cfg = cfg.get("residual_rules", {})
    z_threshold = float(z_cfg.get("z_threshold", 3.0))
    window = int(z_cfg.get("window", 168))
    z_flags, z_scores = _residual_z_flags(residuals, window, z_threshold)

    iforest_cfg = cfg.get("isolation_forest", {})
    iforest_enabled = bool(iforest_cfg.get("enabled", True))
    iforest_flags = np.zeros_like(z_flags, dtype=bool)

    if iforest_enabled and features is not None:
        if isinstance(features, pd.DataFrame):
            X = features.to_numpy()
        else:
            X = np.asarray(features, dtype=float)

        # augment with actual/forecast/residual for a stronger multivariate signal
        X_aug = np.column_stack([X, actual, forecast, residuals])
        model = fit_iforest(
            X_aug,
            contamination=float(iforest_cfg.get("contamination", 0.01)),
            random_state=int(iforest_cfg.get("random_state", 42)),
        )
        preds = predict_iforest(model, X_aug)
        iforest_flags = preds == -1

    combined = z_flags | iforest_flags

    return {
        "residuals": residuals.tolist(),
        "z_scores": z_scores.tolist(),
        "residual_z": z_flags.tolist(),
        "iforest": iforest_flags.tolist(),
        "combined": combined.tolist(),
    }
