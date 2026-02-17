"""
Advanced Forecasting Baselines: Prophet, N-BEATS, AutoML

This module provides state-of-the-art baseline models for energy forecasting
comparison against the primary GBM models.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# BASE CLASS
# =============================================================================

@dataclass
class BaselineConfig:
    """Configuration for baseline models."""
    target: str = "load_mw"
    horizon: int = 24
    freq: str = "h"
    n_jobs: int = -1
    verbose: bool = False


class BaselineModel(ABC):
    """Abstract base class for forecasting baselines."""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self._fitted = False
        
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaselineModel":
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass


# =============================================================================
# PROPHET BASELINE
# =============================================================================

@dataclass
class ProphetConfig(BaselineConfig):
    """Prophet-specific configuration."""
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = True
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    holidays_country: Optional[str] = "DE"
    add_regressors: List[str] = field(default_factory=list)
    uncertainty_samples: int = 1000
    interval_width: float = 0.90


class ProphetBaseline(BaselineModel):
    """
    Facebook Prophet baseline for time-series forecasting.
    
    Prophet is robust to missing data and shifts in the trend,
    and typically handles outliers well.
    
    Reference: Taylor & Letham (2018) - Forecasting at Scale
    """
    
    def __init__(self, config: ProphetConfig):
        super().__init__(config)
        self.config: ProphetConfig = config
        self.model = None
        self._regressors_fitted: List[str] = []
        
    def name(self) -> str:
        return "prophet"
    
    def fit(self, df: pd.DataFrame) -> "ProphetBaseline":
        """
        Fit Prophet model.
        
        Args:
            df: DataFrame with 'timestamp' and target column
            
        Returns:
            Self for chaining
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Run: pip install prophet")
        
        # Prepare data in Prophet format
        prophet_df = self._prepare_prophet_df(df)
        
        # Initialize model
        self.model = Prophet(
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            uncertainty_samples=self.config.uncertainty_samples,
            interval_width=self.config.interval_width,
        )
        
        # Add country holidays if specified
        if self.config.holidays_country:
            self.model.add_country_holidays(country_name=self.config.holidays_country)
        
        # Add extra regressors if present in data
        for reg in self.config.add_regressors:
            if reg in df.columns:
                self.model.add_regressor(reg)
                self._regressors_fitted.append(reg)
        
        # Fit
        self.model.fit(prophet_df, iter=300)
        self._fitted = True
        
        logger.info(f"Prophet fitted on {len(prophet_df)} samples")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        future = self._prepare_prophet_df(df, include_y=False)
        forecast = self.model.predict(future)
        
        return forecast["yhat"].values
    
    def predict_intervals(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty intervals.
        
        Returns:
            Tuple of (point_forecast, lower_bound, upper_bound)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        future = self._prepare_prophet_df(df, include_y=False)
        forecast = self.model.predict(future)
        
        return (
            forecast["yhat"].values,
            forecast["yhat_lower"].values,
            forecast["yhat_upper"].values,
        )
    
    def _prepare_prophet_df(self, df: pd.DataFrame, include_y: bool = True) -> pd.DataFrame:
        """Convert DataFrame to Prophet format (ds, y)."""
        # Find timestamp column
        ts_col = None
        for col in ["timestamp", "utc_timestamp", "datetime", "date"]:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col is None and df.index.name in ["timestamp", "utc_timestamp"]:
            df = df.reset_index()
            ts_col = df.columns[0]
        
        if ts_col is None:
            raise ValueError("No timestamp column found")
        
        result = pd.DataFrame({"ds": pd.to_datetime(df[ts_col])})
        
        if include_y and self.config.target in df.columns:
            result["y"] = df[self.config.target].values
        
        # Add regressors
        for reg in self._regressors_fitted:
            if reg in df.columns:
                result[reg] = df[reg].values
        
        return result


# =============================================================================
# N-BEATS BASELINE (via Darts)
# =============================================================================

@dataclass
class NBEATSConfig(BaselineConfig):
    """N-BEATS specific configuration."""
    input_chunk_length: int = 168  # 1 week of hourly data
    output_chunk_length: int = 24  # 24-hour forecast
    num_stacks: int = 30
    num_blocks: int = 1
    num_layers: int = 4
    layer_widths: int = 256
    expansion_coefficient_dim: int = 5
    trend_polynomial_degree: int = 2
    batch_size: int = 32
    n_epochs: int = 100
    learning_rate: float = 1e-3
    dropout: float = 0.1
    random_state: int = 42
    generic_architecture: bool = True  # Use generic (vs interpretable)


class NBEATSBaseline(BaselineModel):
    """
    N-BEATS (Neural Basis Expansion Analysis) baseline.
    
    N-BEATS is a pure deep learning architecture for time series forecasting
    that achieved state-of-the-art on M4 competition.
    
    Reference: Oreshkin et al. (2020) - N-BEATS: Neural basis expansion 
               analysis for interpretable time series forecasting
    """
    
    def __init__(self, config: NBEATSConfig):
        super().__init__(config)
        self.config: NBEATSConfig = config
        self.model = None
        self.scaler = None
        
    def name(self) -> str:
        return "nbeats"
    
    def fit(self, df: pd.DataFrame) -> "NBEATSBaseline":
        """
        Fit N-BEATS model using Darts library.
        
        Args:
            df: DataFrame with timestamp and target column
            
        Returns:
            Self for chaining
        """
        try:
            from darts import TimeSeries
            from darts.models import NBEATSModel
            from darts.dataprocessing.transformers import Scaler
        except ImportError:
            raise ImportError("Darts not installed. Run: pip install darts")
        
        # Convert to Darts TimeSeries
        series = self._to_darts_series(df)
        
        # Scale the data
        self.scaler = Scaler()
        series_scaled = self.scaler.fit_transform(series)
        
        # Initialize N-BEATS model
        self.model = NBEATSModel(
            input_chunk_length=self.config.input_chunk_length,
            output_chunk_length=self.config.output_chunk_length,
            num_stacks=self.config.num_stacks,
            num_blocks=self.config.num_blocks,
            num_layers=self.config.num_layers,
            layer_widths=self.config.layer_widths,
            expansion_coefficient_dim=self.config.expansion_coefficient_dim,
            trend_polynomial_degree=self.config.trend_polynomial_degree,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            optimizer_kwargs={"lr": self.config.learning_rate},
            dropout=self.config.dropout,
            random_state=self.config.random_state,
            generic_architecture=self.config.generic_architecture,
            force_reset=True,
            pl_trainer_kwargs={
                "accelerator": "auto",
                "enable_progress_bar": self.config.verbose,
            },
        )
        
        # Fit
        self.model.fit(series_scaled, verbose=self.config.verbose)
        self._fitted = True
        
        logger.info(f"N-BEATS fitted on {len(series)} samples")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the horizon."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        try:
            from darts import TimeSeries
        except ImportError:
            raise ImportError("Darts not installed")
        
        # For prediction, we need the last input_chunk_length points as context
        series = self._to_darts_series(df)
        series_scaled = self.scaler.transform(series)
        
        # Predict
        pred_scaled = self.model.predict(
            n=self.config.output_chunk_length,
            series=series_scaled,
        )
        
        # Inverse transform
        pred = self.scaler.inverse_transform(pred_scaled)
        
        return pred.values().flatten()
    
    def _to_darts_series(self, df: pd.DataFrame):
        """Convert pandas DataFrame to Darts TimeSeries."""
        from darts import TimeSeries
        
        # Find timestamp column
        ts_col = None
        for col in ["timestamp", "utc_timestamp", "datetime", "date"]:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col is None and df.index.name:
            df = df.reset_index()
            ts_col = df.columns[0]
        
        df_copy = df.copy()
        df_copy[ts_col] = pd.to_datetime(df_copy[ts_col])
        df_copy = df_copy.set_index(ts_col)
        
        return TimeSeries.from_dataframe(
            df_copy[[self.config.target]],
            freq=self.config.freq,
        )


# =============================================================================
# AUTOML BASELINE (via FLAML)
# =============================================================================

@dataclass
class AutoMLConfig(BaselineConfig):
    """AutoML (FLAML) specific configuration."""
    time_budget: int = 300  # 5 minutes
    metric: str = "rmse"
    task: str = "ts_forecast"
    estimator_list: List[str] = field(default_factory=lambda: [
        "lgbm", "xgboost", "rf", "extra_tree", "xgb_limitdepth"
    ])
    log_file_name: str = "flaml_automl.log"
    seed: int = 42
    period: int = 24  # Seasonal period (hourly daily cycle)


class AutoMLBaseline(BaselineModel):
    """
    AutoML baseline using FLAML (Fast Lightweight AutoML).
    
    FLAML automatically searches for the best model and hyperparameters
    within a given time budget.
    
    Reference: Wang et al. (2021) - FLAML: A Fast and Lightweight AutoML Library
    """
    
    def __init__(self, config: AutoMLConfig):
        super().__init__(config)
        self.config: AutoMLConfig = config
        self.model = None
        self._feature_cols: List[str] = []
        
    def name(self) -> str:
        return "automl_flaml"
    
    def fit(self, df: pd.DataFrame) -> "AutoMLBaseline":
        """
        Fit AutoML model.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Self for chaining
        """
        try:
            from flaml import AutoML
        except ImportError:
            raise ImportError("FLAML not installed. Run: pip install flaml[automl]")
        
        # Prepare features
        X, y = self._prepare_features(df)
        
        # Initialize AutoML
        self.model = AutoML()
        
        # Define settings
        settings = {
            "time_budget": self.config.time_budget,
            "metric": self.config.metric,
            "task": "regression",  # Time series as regression with lag features
            "estimator_list": self.config.estimator_list,
            "log_file_name": self.config.log_file_name,
            "seed": self.config.seed,
            "verbose": 1 if self.config.verbose else 0,
        }
        
        # Fit
        self.model.fit(X, y, **settings)
        self._fitted = True
        
        logger.info(f"AutoML fitted. Best model: {self.model.best_estimator}")
        logger.info(f"Best config: {self.model.best_config}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X, _ = self._prepare_features(df, include_target=False)
        return self.model.predict(X)
    
    def _prepare_features(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare feature matrix for AutoML.
        
        Automatically selects numeric columns as features.
        """
        # Select numeric columns (excluding target for X)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.config.target in numeric_cols:
            numeric_cols.remove(self.config.target)
        
        # Remove timestamp-like columns
        exclude_patterns = ["timestamp", "date", "time", "index"]
        feature_cols = [
            c for c in numeric_cols
            if not any(p in c.lower() for p in exclude_patterns)
        ]
        
        if not self._feature_cols:
            self._feature_cols = feature_cols
        
        X = df[self._feature_cols].copy()
        
        # Fill NaN with forward fill then 0
        X = X.ffill().fillna(0)
        
        y = None
        if include_target and self.config.target in df.columns:
            y = df[self.config.target].copy()
        
        return X, y
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the best model."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        
        try:
            importance = self.model.feature_importances_
            return dict(zip(self._feature_cols, importance))
        except AttributeError:
            return {}


# =============================================================================
# ENSEMBLE BASELINE
# =============================================================================

class EnsembleBaseline(BaselineModel):
    """
    Ensemble of multiple baseline models with weighted averaging.
    """
    
    def __init__(
        self,
        models: List[BaselineModel],
        weights: Optional[List[float]] = None,
        config: Optional[BaselineConfig] = None,
    ):
        super().__init__(config or BaselineConfig())
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
    def name(self) -> str:
        model_names = [m.name() for m in self.models]
        return f"ensemble({','.join(model_names)})"
    
    def fit(self, df: pd.DataFrame) -> "EnsembleBaseline":
        """Fit all component models."""
        for model in self.models:
            logger.info(f"Fitting {model.name()}...")
            model.fit(df)
        
        self._fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(df)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def train_all_baselines(
    train_df: pd.DataFrame,
    target: str = "load_mw",
    time_budget: int = 300,
    verbose: bool = True,
) -> Dict[str, BaselineModel]:
    """
    Train all available baseline models.
    
    Args:
        train_df: Training DataFrame
        target: Target column name
        time_budget: Time budget for AutoML in seconds
        verbose: Print progress
        
    Returns:
        Dictionary of fitted models
    """
    models = {}
    
    # Prophet
    try:
        if verbose:
            logger.info("Training Prophet...")
        prophet_config = ProphetConfig(target=target)
        prophet = ProphetBaseline(prophet_config)
        prophet.fit(train_df)
        models["prophet"] = prophet
    except Exception as e:
        logger.warning(f"Prophet training failed: {e}")
    
    # N-BEATS
    try:
        if verbose:
            logger.info("Training N-BEATS...")
        nbeats_config = NBEATSConfig(target=target, n_epochs=50)
        nbeats = NBEATSBaseline(nbeats_config)
        nbeats.fit(train_df)
        models["nbeats"] = nbeats
    except Exception as e:
        logger.warning(f"N-BEATS training failed: {e}")
    
    # AutoML
    try:
        if verbose:
            logger.info("Training AutoML...")
        automl_config = AutoMLConfig(target=target, time_budget=time_budget)
        automl = AutoMLBaseline(automl_config)
        automl.fit(train_df)
        models["automl"] = automl
    except Exception as e:
        logger.warning(f"AutoML training failed: {e}")
    
    return models


def evaluate_baselines(
    models: Dict[str, BaselineModel],
    test_df: pd.DataFrame,
    target: str = "load_mw",
) -> pd.DataFrame:
    """
    Evaluate all baseline models on test data.
    
    Returns:
        DataFrame with metrics for each model
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = []
    y_true = test_df[target].values
    
    for name, model in models.items():
        try:
            y_pred = model.predict(test_df)
            
            # Align lengths (some models may predict different horizons)
            min_len = min(len(y_true), len(y_pred))
            y_true_aligned = y_true[:min_len]
            y_pred_aligned = y_pred[:min_len]
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
            mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
            r2 = r2_score(y_true_aligned, y_pred_aligned)
            
            # MAPE (avoid division by zero)
            mask = y_true_aligned != 0
            mape = np.mean(np.abs((y_true_aligned[mask] - y_pred_aligned[mask]) / y_true_aligned[mask])) * 100
            
            results.append({
                "model": name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape,
                "n_samples": min_len,
            })
        except Exception as e:
            logger.warning(f"Evaluation failed for {name}: {e}")
            results.append({
                "model": name,
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "mape": np.nan,
                "n_samples": 0,
                "error": str(e),
            })
    
    return pd.DataFrame(results)
