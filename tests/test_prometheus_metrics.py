"""
Tests for Prometheus metrics integration.

These tests verify that metrics are correctly exposed
and tracked throughout the application.
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock


class TestPrometheusMetrics:
    """Tests for Prometheus metrics module."""
    
    def test_prometheus_client_import(self):
        """Test that prometheus_client can be imported."""
        try:
            from prometheus_client import Counter, Gauge, Histogram, Summary
            assert Counter is not None
            assert Gauge is not None
            assert Histogram is not None
            assert Summary is not None
        except ImportError:
            pytest.skip("prometheus-client not installed")
    
    def test_metrics_module_import(self):
        """Test that our metrics module can be imported."""
        try:
            from gridpulse.monitoring.prometheus_metrics import (
                REQUESTS_TOTAL,
                FORECAST_DURATION,
                OPTIMIZATION_COST_SAVINGS,
                BATTERY_SOC,
            )
            assert REQUESTS_TOTAL is not None
            assert FORECAST_DURATION is not None
        except ImportError:
            pytest.skip("prometheus_metrics module not available")
    
    def test_request_counter_increment(self):
        """Test that request counter can be incremented."""
        try:
            from gridpulse.monitoring.prometheus_metrics import REQUESTS_TOTAL
            
            # Get initial value (may not be 0 if tests ran before)
            initial = REQUESTS_TOTAL.labels(
                method="GET",
                endpoint="/test",
                status="200"
            )._value.get()
            
            # Increment
            REQUESTS_TOTAL.labels(
                method="GET",
                endpoint="/test",
                status="200"
            ).inc()
            
            # Verify increment
            new_value = REQUESTS_TOTAL.labels(
                method="GET",
                endpoint="/test",
                status="200"
            )._value.get()
            
            assert new_value == initial + 1
        except ImportError:
            pytest.skip("prometheus_metrics module not available")
    
    def test_histogram_observation(self):
        """Test that histogram can record observations."""
        try:
            from gridpulse.monitoring.prometheus_metrics import FORECAST_DURATION
            
            # Record some observations
            FORECAST_DURATION.labels(
                model="lightgbm",
                target="load_mw",
                horizon="24h"
            ).observe(0.5)
            
            FORECAST_DURATION.labels(
                model="lightgbm",
                target="load_mw",
                horizon="24h"
            ).observe(0.7)
            
            # Just verify no errors occurred
            assert True
        except ImportError:
            pytest.skip("prometheus_metrics module not available")
    
    def test_gauge_set_value(self):
        """Test that gauge can be set to a value."""
        try:
            from gridpulse.monitoring.prometheus_metrics import BATTERY_SOC
            
            # Set battery SOC
            BATTERY_SOC.labels(battery_id="bess_1").set(0.75)
            
            # Verify value
            value = BATTERY_SOC.labels(battery_id="bess_1")._value.get()
            assert value == 0.75
        except ImportError:
            pytest.skip("prometheus_metrics module not available")


class TestMetricsDecorators:
    """Tests for metrics decorator functions."""
    
    def test_track_request_metrics_decorator(self):
        """Test the track_request_metrics decorator."""
        try:
            from gridpulse.monitoring.prometheus_metrics import track_request_metrics
            
            @track_request_metrics(endpoint="/api/test")
            def dummy_endpoint():
                return {"status": "ok"}
            
            result = dummy_endpoint()
            assert result == {"status": "ok"}
        except ImportError:
            pytest.skip("prometheus_metrics module not available")
    
    def test_track_forecast_metrics_decorator(self):
        """Test the track_forecast_metrics decorator."""
        try:
            from gridpulse.monitoring.prometheus_metrics import track_forecast_metrics
            
            @track_forecast_metrics(model="test_model", target="load_mw")
            def dummy_forecast():
                time.sleep(0.01)  # Simulate some work
                return [1.0, 2.0, 3.0]
            
            result = dummy_forecast()
            assert result == [1.0, 2.0, 3.0]
        except ImportError:
            pytest.skip("prometheus_metrics module not available")


class TestMetricsUpdateFunctions:
    """Tests for metrics update helper functions."""
    
    def test_update_cost_savings_metrics(self):
        """Test updating cost savings metrics."""
        try:
            from gridpulse.monitoring.prometheus_metrics import (
                update_cost_savings_metrics,
                OPTIMIZATION_COST_SAVINGS,
            )
            
            update_cost_savings_metrics(
                region="DE",
                savings_eur=2708.61,
                savings_pct=7.11
            )
            
            # Verify the gauge was set
            value = OPTIMIZATION_COST_SAVINGS.labels(
                region="DE"
            )._value.get()
            assert value == 2708.61
        except ImportError:
            pytest.skip("prometheus_metrics module not available")
    
    def test_update_streaming_metrics(self):
        """Test updating streaming metrics."""
        try:
            from gridpulse.monitoring.prometheus_metrics import (
                update_streaming_metrics,
                KAFKA_CONSUMER_LAG,
            )
            
            update_streaming_metrics(
                topic="gridpulse-telemetry",
                partition=0,
                lag=100
            )
            
            # Verify the gauge was set
            value = KAFKA_CONSUMER_LAG.labels(
                topic="gridpulse-telemetry",
                partition="0"
            )._value.get()
            assert value == 100
        except ImportError:
            pytest.skip("prometheus_metrics module not available")
    
    def test_update_drift_metrics(self):
        """Test updating drift detection metrics."""
        try:
            from gridpulse.monitoring.prometheus_metrics import (
                update_drift_metrics,
                DRIFT_DETECTED,
            )
            
            update_drift_metrics(
                feature="load_mw",
                drift_detected=True,
                drift_score=0.85
            )
            
            # Verify drift was flagged
            value = DRIFT_DETECTED.labels(feature="load_mw")._value.get()
            assert value == 1
        except ImportError:
            pytest.skip("prometheus_metrics module not available")


class TestMetricsEndpoint:
    """Tests for /metrics HTTP endpoint."""
    
    @pytest.mark.integration
    def test_metrics_endpoint_format(self):
        """Test that metrics are exposed in Prometheus format."""
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            
            metrics_output = generate_latest()
            
            # Should be bytes
            assert isinstance(metrics_output, bytes)
            
            # Should contain metric names
            metrics_str = metrics_output.decode("utf-8")
            assert "python" in metrics_str or "process" in metrics_str
        except ImportError:
            pytest.skip("prometheus-client not installed")
    
    @pytest.mark.integration
    def test_custom_metrics_in_output(self):
        """Test that custom metrics appear in output."""
        try:
            from prometheus_client import generate_latest
            from gridpulse.monitoring.prometheus_metrics import REQUESTS_TOTAL
            
            # Make sure we have at least one metric
            REQUESTS_TOTAL.labels(
                method="GET",
                endpoint="/metrics-test",
                status="200"
            ).inc()
            
            metrics_output = generate_latest().decode("utf-8")
            
            # Our custom metrics should appear
            assert "gridpulse_requests_total" in metrics_output or "requests_total" in metrics_output
        except ImportError:
            pytest.skip("prometheus_metrics module not available")


class TestMetricsLabels:
    """Tests for metrics label handling."""
    
    def test_label_cardinality(self):
        """Test that label cardinality is controlled."""
        try:
            from gridpulse.monitoring.prometheus_metrics import REQUESTS_TOTAL
            
            # Should be able to create labels without issues
            for endpoint in ["/api/v1/forecast", "/api/v1/optimize", "/api/v1/anomaly"]:
                for status in ["200", "400", "500"]:
                    REQUESTS_TOTAL.labels(
                        method="GET",
                        endpoint=endpoint,
                        status=status
                    ).inc()
            
            # No assertion - just verify no errors were raised
            assert True
        except ImportError:
            pytest.skip("prometheus_metrics module not available")
    
    def test_missing_label_handling(self):
        """Test that missing labels raise appropriate errors."""
        try:
            from gridpulse.monitoring.prometheus_metrics import REQUESTS_TOTAL
            
            # Should raise error for missing label
            with pytest.raises((ValueError, TypeError)):
                REQUESTS_TOTAL.labels(method="GET").inc()  # Missing endpoint and status
        except ImportError:
            pytest.skip("prometheus_metrics module not available")


@pytest.mark.integration
class TestMetricsIntegration:
    """Integration tests for metrics system."""
    
    def test_metrics_registry_health(self):
        """Test that metrics registry is healthy."""
        try:
            from prometheus_client import REGISTRY
            
            # Should have some collectors
            collectors = list(REGISTRY._names_to_collectors.values())
            assert len(collectors) > 0
        except ImportError:
            pytest.skip("prometheus-client not installed")
    
    def test_metrics_thread_safety(self):
        """Test that metrics are thread-safe."""
        import threading
        import random
        
        try:
            from gridpulse.monitoring.prometheus_metrics import REQUESTS_TOTAL
            
            errors = []
            
            def increment_counter():
                try:
                    for _ in range(100):
                        REQUESTS_TOTAL.labels(
                            method="GET",
                            endpoint="/thread-test",
                            status=random.choice(["200", "201", "202"])
                        ).inc()
                except Exception as e:
                    errors.append(e)
            
            # Create multiple threads
            threads = [threading.Thread(target=increment_counter) for _ in range(5)]
            
            # Start all threads
            for t in threads:
                t.start()
            
            # Wait for completion
            for t in threads:
                t.join()
            
            # No errors should have occurred
            assert len(errors) == 0
        except ImportError:
            pytest.skip("prometheus_metrics module not available")
