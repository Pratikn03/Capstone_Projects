"""
Prometheus Metrics Integration for GridPulse API.

This module provides comprehensive metrics collection for monitoring
the GridPulse energy forecasting and optimization platform.
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    multiprocess,
    REGISTRY,
)

# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

# Request Metrics
REQUESTS_TOTAL = Counter(
    "gridpulse_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"],
)

REQUEST_DURATION = Histogram(
    "gridpulse_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUEST_ERRORS = Counter(
    "gridpulse_request_errors_total",
    "Total number of request errors",
    ["endpoint", "method", "error_type"],
)

REQUESTS_IN_PROGRESS = Gauge(
    "gridpulse_requests_in_progress",
    "Number of requests currently being processed",
    ["endpoint"],
)

# Forecasting Metrics
FORECAST_DURATION = Histogram(
    "gridpulse_forecast_duration_seconds",
    "Time to generate forecast",
    ["model", "target", "region"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

FORECAST_MAPE = Gauge(
    "gridpulse_forecast_mape",
    "Mean Absolute Percentage Error of forecasts",
    ["model", "target", "region"],
)

FORECAST_RMSE = Gauge(
    "gridpulse_forecast_rmse",
    "Root Mean Square Error of forecasts",
    ["model", "target", "region"],
)

FORECAST_VALUE = Gauge(
    "gridpulse_forecast_value",
    "Latest forecast value",
    ["target", "region", "horizon"],
)

ACTUAL_VALUE = Gauge(
    "gridpulse_actual_value",
    "Latest actual value",
    ["target", "region"],
)

FORECAST_DRIFT_SCORE = Gauge(
    "gridpulse_forecast_drift_score",
    "Model drift score (0=no drift, 1=severe drift)",
    ["model", "target"],
)

# Uncertainty Metrics
PREDICTION_INTERVAL_COVERAGE = Gauge(
    "gridpulse_prediction_interval_coverage",
    "Prediction Interval Coverage Probability",
    ["target", "region", "confidence_level"],
)

PREDICTION_INTERVAL_WIDTH = Gauge(
    "gridpulse_prediction_interval_width",
    "Mean Prediction Interval Width",
    ["target", "region"],
)

# Optimization Metrics
OPTIMIZATION_DURATION = Histogram(
    "gridpulse_optimization_duration_seconds",
    "Time to solve optimization problem",
    ["mode", "solver"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

OPTIMIZATION_INFEASIBLE = Counter(
    "gridpulse_optimization_infeasible_total",
    "Count of infeasible optimization problems",
    ["mode", "reason"],
)

OPTIMIZATION_COST_BASELINE = Gauge(
    "gridpulse_optimization_cost_baseline",
    "Baseline cost without optimization",
    ["region"],
)

OPTIMIZATION_COST_OPTIMIZED = Gauge(
    "gridpulse_optimization_cost_optimized",
    "Optimized cost",
    ["region"],
)

OPTIMIZATION_COST_SAVINGS = Gauge(
    "gridpulse_optimization_cost_savings",
    "Cost savings from optimization",
    ["region"],
)

OPTIMIZATION_CARBON_REDUCTION = Gauge(
    "gridpulse_optimization_carbon_reduction_kg",
    "Carbon reduction in kg",
    ["region"],
)

# Battery Metrics
BATTERY_SOC = Gauge(
    "gridpulse_battery_soc_mwh",
    "Battery State of Charge in MWh",
    ["battery_id"],
)

BATTERY_CHARGE_POWER = Gauge(
    "gridpulse_battery_charge_power_mw",
    "Current battery charge power in MW",
    ["battery_id"],
)

BATTERY_DISCHARGE_POWER = Gauge(
    "gridpulse_battery_discharge_power_mw",
    "Current battery discharge power in MW",
    ["battery_id"],
)

BATTERY_CYCLES = Counter(
    "gridpulse_battery_cycles_total",
    "Total battery charge/discharge cycles",
    ["battery_id"],
)

# Anomaly Detection Metrics
ANOMALIES_DETECTED = Counter(
    "gridpulse_anomalies_detected_total",
    "Total anomalies detected",
    ["target", "region", "detector"],
)

ANOMALY_SCORE = Gauge(
    "gridpulse_anomaly_score",
    "Latest anomaly score",
    ["target", "region"],
)

# Streaming Metrics
KAFKA_CONSUMER_LAG = Gauge(
    "gridpulse_kafka_consumer_lag",
    "Kafka consumer lag in messages",
    ["topic", "partition"],
)

KAFKA_MESSAGES_CONSUMED = Counter(
    "gridpulse_kafka_messages_consumed_total",
    "Total messages consumed from Kafka",
    ["topic"],
)

STREAMING_PROCESSING_DELAY = Gauge(
    "gridpulse_streaming_processing_delay_seconds",
    "Delay between event timestamp and processing time",
    ["topic"],
)

# System Health Metrics
WATCHDOG_LAST_BEAT = Gauge(
    "gridpulse_watchdog_last_beat_seconds",
    "Seconds since last watchdog heartbeat",
)

SAFETY_VIOLATIONS = Counter(
    "gridpulse_safety_violations_total",
    "Total safety violations caught by BMS",
    ["violation_type"],
)

# Model Registry Metrics
MODELS_LOADED = Gauge(
    "gridpulse_models_loaded",
    "Number of models currently loaded",
    ["model_type"],
)

MODEL_LAST_TRAINED = Gauge(
    "gridpulse_model_last_trained_timestamp",
    "Timestamp of last model training",
    ["model", "target"],
)

# Service Info
SERVICE_INFO = Info(
    "gridpulse_service",
    "GridPulse service information",
)


# =============================================================================
# MIDDLEWARE & DECORATORS
# =============================================================================

def track_request_metrics(endpoint: str):
    """Decorator to track request metrics for an endpoint."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            method = "POST"  # Default, could be extracted from request
            
            REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                REQUESTS_TOTAL.labels(
                    endpoint=endpoint, method=method, status="success"
                ).inc()
                return result
            except Exception as e:
                REQUESTS_TOTAL.labels(
                    endpoint=endpoint, method=method, status="error"
                ).inc()
                REQUEST_ERRORS.labels(
                    endpoint=endpoint, method=method, error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.perf_counter() - start_time
                REQUEST_DURATION.labels(endpoint=endpoint, method=method).observe(duration)
                REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()
        
        return wrapper
    return decorator


def track_forecast_metrics(model: str, target: str, region: str = "DE"):
    """Decorator to track forecast generation metrics."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                FORECAST_DURATION.labels(
                    model=model, target=target, region=region
                ).observe(duration)
                return result
            except Exception:
                raise
        
        return wrapper
    return decorator


def track_optimization_metrics(mode: str, solver: str = "highs"):
    """Decorator to track optimization metrics."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                OPTIMIZATION_DURATION.labels(mode=mode, solver=solver).observe(duration)
                return result
            except Exception:
                raise
        
        return wrapper
    return decorator


# =============================================================================
# METRIC UPDATE FUNCTIONS
# =============================================================================

def update_forecast_metrics(
    model: str,
    target: str,
    region: str,
    mape: float,
    rmse: float,
    forecast_value: Optional[float] = None,
    actual_value: Optional[float] = None,
    drift_score: Optional[float] = None,
):
    """Update forecast-related metrics."""
    FORECAST_MAPE.labels(model=model, target=target, region=region).set(mape)
    FORECAST_RMSE.labels(model=model, target=target, region=region).set(rmse)
    
    if forecast_value is not None:
        FORECAST_VALUE.labels(target=target, region=region, horizon="1h").set(forecast_value)
    
    if actual_value is not None:
        ACTUAL_VALUE.labels(target=target, region=region).set(actual_value)
    
    if drift_score is not None:
        FORECAST_DRIFT_SCORE.labels(model=model, target=target).set(drift_score)


def update_optimization_metrics(
    region: str,
    baseline_cost: float,
    optimized_cost: float,
    carbon_reduction: float,
):
    """Update optimization-related metrics."""
    OPTIMIZATION_COST_BASELINE.labels(region=region).set(baseline_cost)
    OPTIMIZATION_COST_OPTIMIZED.labels(region=region).set(optimized_cost)
    OPTIMIZATION_COST_SAVINGS.labels(region=region).set(baseline_cost - optimized_cost)
    OPTIMIZATION_CARBON_REDUCTION.labels(region=region).set(carbon_reduction)


def update_battery_metrics(
    battery_id: str,
    soc_mwh: float,
    charge_mw: float,
    discharge_mw: float,
):
    """Update battery-related metrics."""
    BATTERY_SOC.labels(battery_id=battery_id).set(soc_mwh)
    BATTERY_CHARGE_POWER.labels(battery_id=battery_id).set(charge_mw)
    BATTERY_DISCHARGE_POWER.labels(battery_id=battery_id).set(discharge_mw)


def update_streaming_metrics(
    topic: str,
    consumer_lag: int,
    processing_delay: float,
):
    """Update streaming-related metrics."""
    KAFKA_CONSUMER_LAG.labels(topic=topic, partition="0").set(consumer_lag)
    STREAMING_PROCESSING_DELAY.labels(topic=topic).set(processing_delay)


def record_anomaly(target: str, region: str, detector: str, score: float):
    """Record an anomaly detection."""
    ANOMALIES_DETECTED.labels(target=target, region=region, detector=detector).inc()
    ANOMALY_SCORE.labels(target=target, region=region).set(score)


def record_safety_violation(violation_type: str):
    """Record a safety violation."""
    SAFETY_VIOLATIONS.labels(violation_type=violation_type).inc()


def update_watchdog_heartbeat():
    """Update watchdog heartbeat timestamp."""
    WATCHDOG_LAST_BEAT.set(0)  # Reset to 0 on heartbeat


def set_service_info(version: str, region: str, environment: str):
    """Set service information."""
    SERVICE_INFO.info({
        "version": version,
        "region": region,
        "environment": environment,
    })


# =============================================================================
# METRICS ENDPOINT
# =============================================================================

def get_metrics() -> bytes:
    """
    Generate latest metrics in Prometheus format.
    
    Returns:
        Bytes of metrics in Prometheus exposition format
    """
    return generate_latest(REGISTRY)


def get_metrics_content_type() -> str:
    """Get the content type for metrics response."""
    return CONTENT_TYPE_LATEST


# =============================================================================
# MULTIPROCESS MODE (for Gunicorn)
# =============================================================================

def setup_multiprocess_metrics(prometheus_multiproc_dir: str):
    """
    Setup metrics for multiprocess mode (Gunicorn workers).
    
    Args:
        prometheus_multiproc_dir: Directory for multiprocess metric files
    """
    import os
    os.environ["prometheus_multiproc_dir"] = prometheus_multiproc_dir
    
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry
