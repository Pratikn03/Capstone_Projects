"""Gradient boosting forecaster (LightGBM preferred; XGBoost fallback).

This module provides leakage-safe GBM training with sklearn Pipeline support.
"""
from __future__ import annotations
from typing import Tuple, Any, Dict

import numpy as np


def _try_lightgbm():
    """Import LightGBM if available (preferred GBM backend)."""
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None


def _try_xgboost():
    """Import XGBoost if available (secondary GBM backend)."""
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None


def _try_sklearn_pipeline():
    """Import sklearn Pipeline for preprocessing."""
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline, StandardScaler
    except Exception:
        return None, None


def train_gbm(
    X_train, 
    y_train, 
    params: Dict[str, Any] | None = None, 
    use_pipeline: bool = False,
    preprocessing: str | None = None,
) -> Tuple[str, Any]:
    """
    Train a gradient boosting model with optional sklearn Pipeline.
    
    Args:
        X_train: Training features
        y_train: Training targets
        params: Model hyperparameters
        use_pipeline: Whether to wrap model in sklearn Pipeline
        preprocessing: Type of preprocessing ('standard', 'minmax', or None)
    
    Returns:
        Tuple of (model_kind: str, model: Any)
        
    Note:
        If use_pipeline=True, returns a Pipeline that handles preprocessing.
        This ensures leakage-safe fit/transform for all downstream operations.
    """
    params = params or {}
    lgb = _try_lightgbm()
    
    # Build base model
    if lgb is not None:
        base_model = lgb.LGBMRegressor(**params)
        model_kind = "lightgbm"
    else:
        xgb = _try_xgboost()
        if xgb is not None:
            base_model = xgb.XGBRegressor(**params)
            model_kind = "xgboost"
        else:
            # Ultimate fallback
            from sklearn.ensemble import HistGradientBoostingRegressor
            base_model = HistGradientBoostingRegressor()
            model_kind = "sklearn_hgbrt"
    
    # Optionally wrap in Pipeline for preprocessing
    if use_pipeline:
        Pipeline, StandardScaler = _try_sklearn_pipeline()
        if Pipeline is not None and preprocessing:
            steps = []
            
            if preprocessing == "standard":
                steps.append(("scaler", StandardScaler()))
            elif preprocessing == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                steps.append(("scaler", MinMaxScaler()))
            
            steps.append(("model", base_model))
            pipeline = Pipeline(steps)
            pipeline.fit(X_train, y_train)
            return model_kind, pipeline
    
    # Standard training without Pipeline
    base_model.fit(X_train, y_train)
    return model_kind, base_model


def predict_gbm(model, X) -> np.ndarray:
    """Predict using a trained GBM model (Pipeline-aware)."""
    return model.predict(X)


def extract_base_model(model):
    """
    Extract the base GBM model from a Pipeline if wrapped.
    
    Useful for SHAP analysis and feature importance which need the raw model.
    """
    # Check if this is a sklearn Pipeline
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model

