"""Comprehensive tests for forecasting evaluation metrics and CV."""
from __future__ import annotations

import numpy as np
import pytest

from orius.utils.metrics import daylight_mape, mae, mape, r2_score, rmse, smape
from orius.forecasting.evaluate import evaluate_model_cv, multi_horizon_cv_score, time_series_cv_score
from orius.forecasting.backtest import multi_horizon_metrics, walk_forward_horizon_metrics


class TestRMSE:
    def test_perfect(self):
        assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_known_value(self):
        assert rmse([10.0], [20.0]) == pytest.approx(10.0)

    def test_symmetric(self):
        assert rmse([1.0, 2.0], [3.0, 4.0]) == rmse([3.0, 4.0], [1.0, 2.0])


class TestMAE:
    def test_perfect(self):
        assert mae([5.0, 10.0], [5.0, 10.0]) == pytest.approx(0.0)

    def test_known(self):
        assert mae([10.0, 20.0], [12.0, 22.0]) == pytest.approx(2.0)


class TestMAPE:
    def test_perfect(self):
        assert mape([10.0, 20.0], [10.0, 20.0]) == pytest.approx(0.0)

    def test_known(self):
        val = mape([100.0], [110.0])
        assert val == pytest.approx(0.10)


class TestSMAPE:
    def test_perfect(self):
        assert smape([10.0, 20.0], [10.0, 20.0]) == pytest.approx(0.0)

    def test_symmetric_property(self):
        s1 = smape([10.0], [12.0])
        s2 = smape([12.0], [10.0])
        assert s1 == pytest.approx(s2)


class TestR2:
    def test_perfect(self):
        assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_mean_predictor(self):
        y = [1.0, 2.0, 3.0]
        assert r2_score(y, [2.0, 2.0, 2.0]) == pytest.approx(0.0)

    def test_constant_target(self):
        assert r2_score([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == pytest.approx(0.0)


class TestDaylightMAPE:
    def test_positive_values_only(self):
        val = daylight_mape(np.array([0.0, 100.0, 200.0, 0.0]), np.array([0.0, 110.0, 190.0, 0.0]))
        assert val >= 0.0


class TestTimeSeriesCV:
    def _data(self, n=200):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (n, 5))
        y = X[:, 0] * 2 + rng.normal(0, 0.1, n)
        return X, y

    def test_basic_cv(self):
        X, y = self._data()
        from sklearn.linear_model import LinearRegression
        result = time_series_cv_score(
            X, y,
            train_fn=lambda X, y: LinearRegression().fit(X, y),
            predict_fn=lambda m, X: m.predict(X),
            n_splits=3,
        )
        assert result["n_splits"] == 3
        assert len(result["fold_metrics"]) == 3
        assert "aggregated" in result

    def test_fold_metrics_have_required_keys(self):
        X, y = self._data()
        from sklearn.linear_model import LinearRegression
        result = time_series_cv_score(
            X, y,
            train_fn=lambda X, y: LinearRegression().fit(X, y),
            predict_fn=lambda m, X: m.predict(X),
            n_splits=2,
        )
        for fm in result["fold_metrics"]:
            assert "rmse" in fm
            assert "mae" in fm
            assert "r2" in fm

    def test_aggregated_stats(self):
        X, y = self._data()
        from sklearn.linear_model import LinearRegression
        result = time_series_cv_score(
            X, y,
            train_fn=lambda X, y: LinearRegression().fit(X, y),
            predict_fn=lambda m, X: m.predict(X),
            n_splits=3,
        )
        agg = result["aggregated"]
        assert "rmse_mean" in agg
        assert "rmse_std" in agg


class TestMultiHorizonCV:
    def test_basic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.1, 200)
        from sklearn.linear_model import LinearRegression
        result = multi_horizon_cv_score(
            X, y,
            train_fn=lambda X, y: LinearRegression().fit(X, y),
            predict_fn=lambda m, X: m.predict(X),
            horizon=4, n_splits=2,
        )
        assert result["horizon"] == 4
        assert len(result["fold_results"]) == 2


class TestEvaluateModelCV:
    def test_wrapper(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.1, 200)
        from sklearn.linear_model import LinearRegression
        result = evaluate_model_cv(
            "linear",
            X, y,
            train_fn=lambda X, y: LinearRegression().fit(X, y),
            predict_fn=lambda m, X: m.predict(X),
            n_splits=2, horizon=4,
        )
        assert result["model_type"] == "linear"
        assert "cv_standard" in result
        assert "cv_multi_horizon" in result


class TestWalkForwardHorizonMetrics:
    def test_basic(self):
        y_true = np.arange(48, dtype=float)
        y_pred = y_true + np.random.default_rng(0).normal(0, 0.5, 48)
        result = walk_forward_horizon_metrics(y_true, y_pred, horizon=24, target="load_mw")
        assert "per_horizon" in result
        assert "1" in result["per_horizon"]
        assert "24" in result["per_horizon"]

    def test_per_step_metrics(self):
        y = np.ones(48)
        result = walk_forward_horizon_metrics(y, y + 1, horizon=12, target="load_mw")
        for step_key, metrics in result["per_horizon"].items():
            assert "rmse" in metrics
            assert "mae" in metrics

    def test_solar_daylight_mape(self):
        rng = np.random.default_rng(42)
        y = np.abs(rng.normal(100, 10, 48))
        result = walk_forward_horizon_metrics(y, y + 5, horizon=24, target="solar_mw")
        assert "daylight_mape" in result["per_horizon"]["1"]


class TestMultiHorizonMetrics:
    def test_multiple_horizons(self):
        rng = np.random.default_rng(42)
        y = rng.normal(100, 5, 96)
        result = multi_horizon_metrics(y, y + 1, horizons=[12, 24, 48], target="load_mw")
        assert "12" in result["results"]
        assert "24" in result["results"]

    def test_summary_includes_key_metrics(self):
        y = np.ones(48)
        result = multi_horizon_metrics(y, y + 2, horizons=[24], target="load_mw")
        summary = result["results"]["24"]["summary"]
        assert summary["rmse"] is not None
        assert summary["mae"] is not None
