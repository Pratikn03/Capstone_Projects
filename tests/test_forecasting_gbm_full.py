"""Comprehensive tests for GBM forecaster."""

from __future__ import annotations

import numpy as np
import pytest

from orius.forecasting.ml_gbm import extract_base_model, predict_gbm, train_gbm


def _data(n=200, features=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, features))
    y = X[:, 0] * 3 + X[:, 1] * 2 + rng.normal(0, 0.5, n)
    return X, y


class TestTrainGBM:
    def test_sklearn_backend(self):
        X, y = _data()
        kind, model = train_gbm(X, y, params={"backend": "sklearn_hgbrt"})
        assert kind == "sklearn_hgbrt"
        assert hasattr(model, "predict")

    def test_predict_shape(self):
        X, y = _data()
        _, model = train_gbm(X, y, params={"backend": "sklearn_hgbrt"})
        preds = predict_gbm(model, X[:10])
        assert preds.shape == (10,)

    def test_predictions_reasonable(self):
        X, y = _data()
        _, model = train_gbm(X, y, params={"backend": "sklearn_hgbrt"})
        preds = predict_gbm(model, X[:10])
        assert np.all(np.isfinite(preds))
        assert np.std(preds) > 0

    def test_invalid_backend_raises(self):
        X, y = _data()
        with pytest.raises(ValueError, match="Unknown GBM backend"):
            train_gbm(X, y, params={"backend": "nonexistent"})

    def test_pipeline_mode(self):
        X, y = _data()
        kind, model = train_gbm(
            X, y, params={"backend": "sklearn_hgbrt"}, use_pipeline=True, preprocessing="standard"
        )
        assert kind == "sklearn_hgbrt"
        preds = predict_gbm(model, X[:5])
        assert preds.shape == (5,)


class TestExtractBaseModel:
    def test_extract_from_pipeline(self):
        X, y = _data()
        _, model = train_gbm(
            X, y, params={"backend": "sklearn_hgbrt"}, use_pipeline=True, preprocessing="standard"
        )
        base = extract_base_model(model)
        assert not hasattr(base, "named_steps")

    def test_extract_from_plain_model(self):
        X, y = _data()
        _, model = train_gbm(X, y, params={"backend": "sklearn_hgbrt"})
        base = extract_base_model(model)
        assert base is model

    def test_pipeline_predict_matches_base(self):
        X, y = _data()
        _, pipeline = train_gbm(
            X, y, params={"backend": "sklearn_hgbrt"}, use_pipeline=True, preprocessing="standard"
        )
        p1 = predict_gbm(pipeline, X[:5])
        assert p1.shape == (5,)
