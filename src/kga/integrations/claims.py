"""Deterministic promotion rules for integrated KGA research evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _as_float(value: object, default: float) -> float:
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        try:
            numeric = float(value)
        except OverflowError:
            return default
        if math.isfinite(numeric):
            return numeric
    return default


def _as_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def assess_promotion(summary: Mapping[str, object]) -> dict[str, object]:
    """Return whether an integrated run clears the predeclared evidence bar."""

    mode = str(summary.get("mode", ""))
    alpha = _as_float(summary.get("alpha"), 0.0)
    regret_kga = _as_float(summary.get("regret_kga"), float("inf"))
    regret_adapt = _as_float(summary.get("regret_always_adapt"), float("-inf"))
    regret_freeze = _as_float(summary.get("regret_always_freeze"), float("-inf"))
    raw_integrity_failures = summary.get("integrity_failures")
    integrity_failures = _as_str_list(raw_integrity_failures)
    requirements = (
        (0.0 < alpha < 1.0, "invalid_alpha"),
        (
            isinstance(raw_integrity_failures, list) and all(isinstance(item, str) for item in raw_integrity_failures),
            "invalid_integrity_failures",
        ),
        (mode in {"label_free", "target_label_light"}, "invalid_evaluation_mode"),
        (
            mode != "label_free" or summary.get("frozen_estimator_verified") is True,
            "unverified_frozen_estimator",
        ),
        (
            _as_int(summary.get("held_out_natural_datasets"), 0) >= 2,
            "fewer_than_two_heldout_natural_datasets",
        ),
        (summary.get("frozen_before_scoring") is True, "configuration_not_frozen_before_scoring"),
        (_as_int(summary.get("independent_splits"), 0) >= 3, "fewer_than_three_independent_splits"),
        (regret_kga < regret_adapt, "does_not_beat_always_adapt_regret"),
        (regret_kga < regret_freeze, "does_not_beat_always_freeze_regret"),
        (0.0 <= _as_float(summary.get("false_adapt_rate"), float("inf")) <= alpha, "false_adapt_exceeds_alpha"),
        (0.20 <= _as_float(summary.get("coverage"), 0.0) <= 1.0, "coverage_below_20_percent"),
        (summary.get("confidence_intervals_complete") is True, "confidence_intervals_incomplete"),
        (summary.get("strong_baselines_complete") is True, "strong_baselines_incomplete"),
        (summary.get("required_tracks_complete") is True, "required_tracks_incomplete"),
        (not integrity_failures, "integrity_failure"),
    )
    reasons = [reason for passed, reason in requirements if not passed]
    if mode == "label_free":
        claim_kind = "label_free"
    elif mode == "target_label_light":
        claim_kind = "target_label_light"
    else:
        claim_kind = "retrospective_only"
    return {
        "eligible": not reasons,
        "claim_kind": claim_kind,
        "reasons": reasons,
    }
