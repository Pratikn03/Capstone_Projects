"""Gradient boosting forecaster (LightGBM preferred; XGBoost fallback).

This module is designed to be simple and reliable for Week-1 baseline.
"""
from __future__ import annotations
from typing import Tuple, Any, Dict

import numpy as np

def _try_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None

def _try_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None

def train_gbm(X_train, y_train, params: Dict[str, Any] | None = None) -> Tuple[str, Any]:
    params = params or {}
    lgb = _try_lightgbm()
    if lgb is not None:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return "lightgbm", model

    xgb = _try_xgboost()
    if xgb is not None:
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return "xgboost", model

    # ultimate fallback: sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    return "sklearn_hgbrt", model

def predict_gbm(model, X) -> np.ndarray:
    return model.predict(X)
