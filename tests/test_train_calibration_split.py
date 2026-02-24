"""Tests for optional calibration split ratio parsing in training config."""
from __future__ import annotations

import pytest

from gridpulse.forecasting.train import _resolve_split_cfg


def test_resolve_split_cfg_defaults_calibration_ratio_zero() -> None:
    train_ratio, val_ratio, test_ratio, calibration_ratio = _resolve_split_cfg({"splits": {"train_ratio": 0.7, "val_ratio": 0.15}})
    assert train_ratio == 0.7
    assert val_ratio == 0.15
    assert calibration_ratio == 0.0
    assert abs(test_ratio - 0.15) < 1e-12


def test_resolve_split_cfg_supports_calibration_ratio() -> None:
    train_ratio, val_ratio, test_ratio, calibration_ratio = _resolve_split_cfg(
        {"splits": {"train_ratio": 0.7, "calibration_ratio": 0.1, "val_ratio": 0.1}}
    )
    assert train_ratio == 0.7
    assert calibration_ratio == 0.1
    assert val_ratio == 0.1
    assert abs(test_ratio - 0.1) < 1e-12


def test_resolve_split_cfg_rejects_invalid_total_ratio() -> None:
    with pytest.raises(ValueError):
        _resolve_split_cfg({"splits": {"train_ratio": 0.8, "calibration_ratio": 0.2, "val_ratio": 0.1}})
