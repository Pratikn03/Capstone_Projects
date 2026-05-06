from __future__ import annotations

import numpy as np
import pandas as pd

from orius.forecasting import train


def test_generalization_summary_uses_train_validation_and_test_rmse() -> None:
    summary = train._generalization_summary(
        {"rmse": 10.0},
        {"rmse": 12.0},
        {"rmse": 13.2},
    )

    assert summary["train_validation_rmse_ratio"] == 1.2
    assert summary["validation_test_rmse_ratio"] == 1.1
    assert summary["train_test_rmse_ratio"] == 1.32


def test_prediction_latency_summary_reports_p95_per_sample_ms() -> None:
    X = np.ones((8, 3), dtype=np.float32)

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return batch.sum(axis=1)

    summary = train._prediction_latency_summary(predict_fn, X, repeats=3, max_rows=8)

    assert summary["sample_rows"] == 8
    assert summary["repeats"] == 3
    assert summary["p95_per_sample_ms"] >= 0.0
    assert summary["p95_batch_ms"] >= summary["mean_batch_ms"] * 0.0


def test_training_history_summary_flags_gradient_instability() -> None:
    history = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "val_loss": 1.1,
            "mean_grad_norm": 0.8,
            "max_grad_norm": 1.2,
            "clipped_batches": 1,
            "num_batches": 4,
        },
        {
            "epoch": 2,
            "train_loss": float("nan"),
            "val_loss": 1.4,
            "mean_grad_norm": 4.0,
            "max_grad_norm": 9.5,
            "clipped_batches": 4,
            "num_batches": 4,
        },
    ]

    summary = train._training_history_summary(history)

    assert summary["epochs_ran"] == 2
    assert summary["non_finite_loss"] is True
    assert summary["gradient_clipped_fraction"] == 0.625
    assert summary["max_grad_norm"] == 9.5


def test_sequence_architecture_summary_records_regularization_and_capacity() -> None:
    summary = train._sequence_architecture_summary(
        "lstm",
        {
            "epochs": 8,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "dropout": 0.2,
            "gradient_clip": 0.5,
            "early_stopping": {"patience": 3},
            "warmup_epochs": 2,
        },
        lookback=24,
        horizon=6,
        n_features=12,
    )

    assert summary["model_type"] == "lstm"
    assert summary["n_features"] == 12
    assert summary["dropout"] == 0.2
    assert summary["gradient_clip"] == 0.5
    assert summary["early_stopping_patience"] == 3


def test_select_torch_device_prefers_mps_when_cuda_is_unavailable(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeBackends:
        mps = FakeMps()

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()

    monkeypatch.setattr(train, "_ensure_torch", lambda: None)
    monkeypatch.setattr(train, "torch", FakeTorch())

    assert train._select_torch_device() == "mps"


def test_select_torch_device_respects_cpu_override(monkeypatch) -> None:
    called = False

    def fake_ensure_torch() -> None:
        nonlocal called
        called = True

    monkeypatch.setenv("ORIUS_TORCH_DEVICE", "cpu")
    monkeypatch.setattr(train, "_ensure_torch", fake_ensure_torch)

    assert train._select_torch_device() == "cpu"
    assert called is True


def test_apply_gbm_thread_limits_sets_lightgbm_n_jobs() -> None:
    params = train._apply_gbm_thread_limits({"learning_rate": 0.03}, 2)

    assert params["learning_rate"] == 0.03
    assert params["n_jobs"] == 2


def test_sort_training_frame_uses_configured_order_columns() -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["b", "a", "a"],
            "step_index": [2, 2, 1],
            "target": [3.0, 2.0, 1.0],
        }
    )

    ordered = train._sort_training_frame(df, {"order_cols": ["scenario_id", "step_index"]})

    assert ordered["target"].tolist() == [1.0, 2.0, 3.0]


def test_make_xy_drops_configured_order_columns() -> None:
    df = pd.DataFrame(
        {
            "scenario_id": ["a", "b"],
            "step_index": [1, 2],
            "feature": [10.0, 20.0],
            "target": [1.0, 2.0],
        }
    )

    X, y, feat_cols = train.make_xy(
        df,
        "target",
        ["target"],
        non_feature_cols={"scenario_id", "step_index"},
    )

    assert feat_cols == ["feature"]
    assert X.shape == (2, 1)
    assert y.tolist() == [1.0, 2.0]
