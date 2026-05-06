"""Configuration helpers for the forecasting training CLI."""

from __future__ import annotations


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int | float | str | bytes | bytearray):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value).__name__}")


def _as_float(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int | float | str | bytes | bytearray):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value).__name__}")


def resolve_split_cfg(cfg: dict) -> tuple[float, float, float, float]:
    """Resolve train/validation/test/calibration split ratios from config."""

    split_cfg = cfg.get("splits", cfg.get("split", {})) or {}
    train_ratio = float(split_cfg.get("train_ratio", 0.70))
    val_ratio = float(split_cfg.get("val_ratio", 0.15))
    calibration_ratio = float(split_cfg.get("calibration_ratio", 0.0) or 0.0)

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not (0.0 <= calibration_ratio < 1.0):
        raise ValueError(f"calibration_ratio must be in [0, 1), got {calibration_ratio}")

    test_ratio = 1.0 - train_ratio - val_ratio - calibration_ratio
    if test_ratio < 0.0:
        raise ValueError(
            f"Invalid split ratios: train_ratio({train_ratio}) + "
            f"val_ratio({val_ratio}) + calibration_ratio({calibration_ratio}) exceeds 1.0"
        )
    return train_ratio, val_ratio, test_ratio, calibration_ratio


def resolve_gap_hours(cfg: dict) -> int:
    """Resolve non-negative split gap hours from config."""

    split_cfg = cfg.get("splits", cfg.get("split", {})) or {}
    gap_hours = int(split_cfg.get("gap_hours", 0) or 0)
    if gap_hours < 0:
        raise ValueError(f"gap_hours must be >= 0, got {gap_hours}")
    return gap_hours


def resolve_task_horizon(task_cfg: dict[str, object]) -> int:
    """Resolve forecast horizon for training/evaluation."""

    if "horizon_steps" in task_cfg:
        return max(1, _as_int(task_cfg.get("horizon_steps"), 1))
    horizon_raw = _as_float(task_cfg.get("horizon_hours"), 1.0)
    return max(1, int(round(horizon_raw)))


def resolve_task_lookback(task_cfg: dict[str, object]) -> int:
    """Resolve sequence-model lookback window."""

    if "lookback_steps" in task_cfg:
        return max(1, _as_int(task_cfg.get("lookback_steps"), 1))
    lookback_raw = _as_float(task_cfg.get("lookback_hours"), 168.0)
    return max(1, int(round(lookback_raw)))
