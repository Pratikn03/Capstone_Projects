"""
Locust Load Testing for GridPulse API

This module provides comprehensive load testing scenarios using Locust,
allowing simulation of realistic user behavior and traffic patterns.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8000

Web UI available at: http://localhost:8089
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

from locust import HttpUser, TaskSet, task, between, events, tag
from locust.runners import MasterRunner

# Configuration
API_KEY = os.environ.get("GRIDPULSE_API_KEY", "test-api-key")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_load_forecast(horizon: int = 24) -> List[float]:
    """Generate realistic load demand forecast."""
    base_load = 45000
    return [
        base_load + random.gauss(0, 2000) + 5000 * abs(random.gauss(0, 1))
        for _ in range(horizon)
    ]


def generate_renewable_forecast(horizon: int = 24, scale: float = 5000) -> List[float]:
    """Generate renewable generation forecast."""
    return [max(0, random.gauss(scale / 2, scale / 3)) for _ in range(horizon)]


def generate_price_forecast(horizon: int = 24) -> List[float]:
    """Generate price forecast with daily pattern."""
    return [
        50 + 20 * abs(random.gauss(0, 1)) + 10 * (1 if 8 <= i % 24 <= 20 else 0)
        for i in range(horizon)
    ]


def generate_time_series(length: int = 168, base: float = 45000) -> List[float]:
    """Generate time series data for anomaly detection."""
    return [base + random.gauss(0, 1000) for _ in range(length)]


# =============================================================================
# TASK SETS
# =============================================================================

class ForecastTasks(TaskSet):
    """Tasks related to forecasting endpoints."""
    
    @task(10)
    @tag("forecast", "critical")
    def forecast_load(self):
        """Test load forecasting endpoint."""
        payload = {
            "region": random.choice(["DE", "US"]),
            "target": "load_mw",
            "horizon": 24,
            "include_intervals": True,
        }
        with self.client.post(
            "/forecast/predict",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/forecast/predict [load]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" not in data:
                    response.failure("Missing predictions in response")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(5)
    @tag("forecast")
    def forecast_wind(self):
        """Test wind forecasting endpoint."""
        payload = {
            "region": random.choice(["DE", "US"]),
            "target": "wind_mw",
            "horizon": 24,
        }
        self.client.post(
            "/forecast/predict",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/forecast/predict [wind]",
        )
    
    @task(5)
    @tag("forecast")
    def forecast_solar(self):
        """Test solar forecasting endpoint."""
        payload = {
            "region": random.choice(["DE", "US"]),
            "target": "solar_mw",
            "horizon": 24,
        }
        self.client.post(
            "/forecast/predict",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/forecast/predict [solar]",
        )
    
    @task(2)
    @tag("forecast", "intervals")
    def forecast_with_intervals(self):
        """Test forecasting with prediction intervals."""
        payload = {
            "region": "DE",
            "target": "load_mw",
            "horizon": 48,
            "include_intervals": True,
            "confidence_level": 0.90,
        }
        self.client.post(
            "/forecast/intervals",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/forecast/intervals",
        )


class OptimizationTasks(TaskSet):
    """Tasks related to optimization endpoints."""
    
    @task(8)
    @tag("optimize", "critical")
    def optimize_dispatch_robust(self):
        """Test robust dispatch optimization."""
        horizon = 24
        payload = {
            "load_forecast": generate_load_forecast(horizon),
            "wind_forecast": generate_renewable_forecast(horizon, 5000),
            "solar_forecast": generate_renewable_forecast(horizon, 3000),
            "price_forecast": generate_price_forecast(horizon),
            "battery_capacity_mwh": 100,
            "battery_max_power_mw": 50,
            "initial_soc_mwh": 50,
            "optimization_mode": "robust",
        }
        with self.client.post(
            "/optimize/dispatch",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/optimize/dispatch [robust]",
            timeout=30,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if not data.get("feasible", True):
                    response.failure("Optimization infeasible")
            elif response.status_code != 200:
                response.failure(f"Status {response.status_code}")
    
    @task(4)
    @tag("optimize")
    def optimize_dispatch_deterministic(self):
        """Test deterministic dispatch optimization."""
        horizon = 24
        payload = {
            "load_forecast": generate_load_forecast(horizon),
            "wind_forecast": generate_renewable_forecast(horizon, 5000),
            "solar_forecast": generate_renewable_forecast(horizon, 3000),
            "price_forecast": generate_price_forecast(horizon),
            "battery_capacity_mwh": 100,
            "battery_max_power_mw": 50,
            "initial_soc_mwh": 50,
            "optimization_mode": "deterministic",
        }
        self.client.post(
            "/optimize/dispatch",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/optimize/dispatch [deterministic]",
            timeout=30,
        )
    
    @task(2)
    @tag("optimize", "impact")
    def get_impact_metrics(self):
        """Test impact metrics endpoint."""
        self.client.get(
            "/optimize/impact",
            headers={"X-API-Key": API_KEY},
            name="/optimize/impact",
        )


class AnomalyTasks(TaskSet):
    """Tasks related to anomaly detection endpoints."""
    
    @task(5)
    @tag("anomaly")
    def detect_anomalies(self):
        """Test anomaly detection endpoint."""
        payload = {
            "values": generate_time_series(168),
            "target": "load_mw",
            "detector": random.choice(["zscore", "isolation_forest"]),
        }
        self.client.post(
            "/anomaly/detect",
            json=payload,
            headers={"X-API-Key": API_KEY},
            name="/anomaly/detect",
        )
    
    @task(2)
    @tag("anomaly")
    def get_anomaly_summary(self):
        """Test anomaly summary endpoint."""
        self.client.get(
            "/anomaly/summary",
            headers={"X-API-Key": API_KEY},
            name="/anomaly/summary",
        )


class MonitoringTasks(TaskSet):
    """Tasks related to monitoring endpoints."""
    
    @task(10)
    @tag("monitoring", "health")
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health", name="/health")
    
    @task(5)
    @tag("monitoring", "health")
    def ready_check(self):
        """Test readiness endpoint."""
        self.client.get("/ready", name="/ready")
    
    @task(3)
    @tag("monitoring")
    def get_drift_metrics(self):
        """Test drift monitoring endpoint."""
        self.client.get(
            "/monitor/drift",
            headers={"X-API-Key": API_KEY},
            name="/monitor/drift",
        )
    
    @task(2)
    @tag("monitoring", "metrics")
    def get_prometheus_metrics(self):
        """Test Prometheus metrics endpoint."""
        self.client.get("/metrics", name="/metrics")
    
    @task(1)
    @tag("monitoring")
    def get_research_metrics(self):
        """Test research metrics endpoint."""
        self.client.get(
            "/monitor/research-metrics",
            headers={"X-API-Key": API_KEY},
            name="/monitor/research-metrics",
        )


# =============================================================================
# USER CLASSES
# =============================================================================

class GridPulseUser(HttpUser):
    """
    Standard GridPulse API user simulating typical operator behavior.
    
    This user performs a mix of forecasting, optimization, and monitoring
    tasks with realistic wait times between requests.
    """
    wait_time = between(1, 5)
    
    tasks = {
        ForecastTasks: 4,
        OptimizationTasks: 3,
        AnomalyTasks: 2,
        MonitoringTasks: 1,
    }
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Verify API is accessible
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("API not healthy")


class HeavyForecastUser(HttpUser):
    """
    User that heavily uses forecasting endpoints.
    Simulates automated forecast consumers.
    """
    wait_time = between(0.5, 2)
    tasks = [ForecastTasks]
    weight = 2


class OptimizationUser(HttpUser):
    """
    User focused on optimization endpoints.
    Simulates dispatch planning systems.
    """
    wait_time = between(2, 10)
    tasks = [OptimizationTasks]
    weight = 1


class MonitoringBot(HttpUser):
    """
    Monitoring bot that frequently checks health and metrics.
    Simulates monitoring systems like Prometheus.
    """
    wait_time = between(5, 15)
    tasks = [MonitoringTasks]
    weight = 1


# =============================================================================
# EVENT HOOKS
# =============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 60)
    print("GridPulse Load Test Starting")
    print(f"Target Host: {environment.host}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("=" * 60)
    print("GridPulse Load Test Complete")
    if environment.stats.total.num_requests > 0:
        print(f"Total Requests: {environment.stats.total.num_requests}")
        print(f"Failure Rate: {environment.stats.total.fail_ratio * 100:.2f}%")
        print(f"Avg Response Time: {environment.stats.total.avg_response_time:.2f}ms")
        print(f"95th Percentile: {environment.stats.total.get_response_time_percentile(0.95):.2f}ms")
    print("=" * 60)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """Log slow requests for debugging."""
    if response_time > 5000:  # > 5 seconds
        print(f"SLOW REQUEST: {name} took {response_time}ms")


# =============================================================================
# CUSTOM LOAD SHAPES
# =============================================================================

class StepLoadShape:
    """
    Step load shape that increases users in discrete steps.
    """
    step_time = 30
    step_load = 10
    spawn_rate = 5
    time_limit = 600  # 10 minutes
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        return (current_step * self.step_load, self.spawn_rate)


class DoubleWaveLoadShape:
    """
    Double wave pattern simulating morning and evening peak loads.
    """
    min_users = 10
    max_users = 100
    
    def tick(self):
        import math
        run_time = self.get_run_time()
        
        # Two peaks per hour (simulating morning/evening)
        user_count = int(
            self.min_users + 
            (self.max_users - self.min_users) * 
            (1 + math.sin(run_time * math.pi / 900)) / 2  # 15 minute cycle
        )
        
        return (user_count, user_count)
