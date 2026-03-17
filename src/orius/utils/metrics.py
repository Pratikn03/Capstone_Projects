"""
Utilities: Common regression metrics for forecast evaluation.

This module provides standard metrics for evaluating time series forecasts:
- RMSE: Penalizes large errors (quadratic loss)
- MAE: Robust to outliers (linear loss)
- MAPE/sMAPE: Percentage-based, useful for comparing across scales
- R²: Coefficient of determination (explained variance)

All functions accept numpy arrays or lists and return Python floats.
"""
import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error.
    
    RMSE = sqrt(mean((y_true - y_pred)²))
    
    Penalizes large deviations more than small ones due to squaring.
    Useful when large errors are particularly undesirable.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error.
    
    MAE = mean(|y_true - y_pred|)
    
    More robust to outliers than RMSE. Represents average absolute deviation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination (R²).
    
    R² = 1 - SS_res / SS_tot
    
    Measures proportion of variance explained by the model.
    R² = 1 is perfect prediction; R² = 0 means model predicts the mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def smape(y_true, y_pred) -> float:
    """Symmetric Mean Absolute Percentage Error.
    
    sMAPE = mean(2 * |y_pred - y_true| / (|y_true| + |y_pred|))
    
    Bounded between 0 and 2 (or 0-200%). Symmetric: treats over/under
    predictions equally. Preferred over MAPE when values can be near zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error.
    
    MAPE = mean(|y_true - y_pred| / |y_true|)
    
    Intuitive percentage error. Caution: explodes when y_true approaches zero.
    Use safe denominator clipping to prevent division by zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def daylight_mape(y_true, y_pred) -> float:
    """MAPE computed only on daylight (non-zero) values.
    
    For solar forecasting, this avoids the divide-by-zero issue at night
    by only computing MAPE when actual solar generation > 0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return mape(y_true[mask], y_pred[mask])
