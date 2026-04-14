from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_STATUS_ORDER = {"nominal": 0, "watch": 1, "degraded": 2, "invalid": 3}


@dataclass
class CoverageWindowStats:
    count: int = 0
    covered_count: int = 0
    miss_count: int = 0
    avg_interval_width: float = 0.0
    avg_abs_residual: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        coverage = float(self.covered_count / self.count) if self.count > 0 else 0.0
        return {
            "count": int(self.count),
            "covered_count": int(self.covered_count),
            "miss_count": int(self.miss_count),
            "empirical_coverage": coverage,
            "avg_interval_width": float(self.avg_interval_width),
            "avg_abs_residual": float(self.avg_abs_residual),
        }


@dataclass
class GroupCoverageStats(CoverageWindowStats):
    group_key: str = "unknown"
    target_coverage: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "group_key": self.group_key,
                "target_coverage": float(self.target_coverage),
                "under_coverage_gap": max(0.0, float(self.target_coverage) - float(payload["empirical_coverage"])),
            }
        )
        return payload


@dataclass
class AdaptiveQuantileState:
    mode: str = "fixed"
    target_alpha: float = 0.1
    effective_alpha: float = 0.1
    effective_quantile: float = 1.0
    step_size: float = 0.02
    updates: int = 0
    instability: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_alpha < 1.0:
            raise ValueError("target_alpha must be in (0,1)")
        if self.step_size < 0.0:
            raise ValueError("step_size must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_alpha": float(self.target_alpha),
            "effective_alpha": float(self.effective_alpha),
            "effective_quantile": float(self.effective_quantile),
            "step_size": float(self.step_size),
            "updates": int(self.updates),
            "instability": float(self.instability),
        }


@dataclass
class ShiftValidityState:
    validity_score: float = 1.0
    validity_status: str = "nominal"
    normalized_residual: float = 0.0
    under_coverage_gap: float = 0.0
    drift_magnitude: float = 0.0
    adaptation_instability: float = 0.0
    reliability_score: float = 1.0
    component_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validity_score = float(min(1.0, max(0.0, self.validity_score)))
        if self.validity_status not in _STATUS_ORDER:
            raise ValueError(f"unknown validity_status {self.validity_status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "validity_score": float(self.validity_score),
            "validity_status": self.validity_status,
            "normalized_residual": float(self.normalized_residual),
            "under_coverage_gap": float(self.under_coverage_gap),
            "drift_magnitude": float(self.drift_magnitude),
            "adaptation_instability": float(self.adaptation_instability),
            "reliability_score": float(self.reliability_score),
            "component_scores": {k: float(v) for k, v in sorted(self.component_scores.items())},
        }


@dataclass
class ShiftAwareIntervalDecision:
    lower: float
    upper: float
    base_half_width: float
    adjusted_half_width: float
    inflation_multiplier: float
    adaptive_quantile: float
    validity_score: float
    validity_status: str
    under_coverage_gap: float
    applied_policy: str
    coverage_group_key: str = "unknown"
    shift_alert_flag: bool = False

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("lower must be <= upper")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "base_half_width": float(self.base_half_width),
            "adjusted_half_width": float(self.adjusted_half_width),
            "inflation_multiplier": float(self.inflation_multiplier),
            "adaptive_quantile": float(self.adaptive_quantile),
            "validity_score": float(self.validity_score),
            "validity_status": self.validity_status,
            "under_coverage_gap": float(self.under_coverage_gap),
            "applied_policy": self.applied_policy,
            "coverage_group_key": self.coverage_group_key,
            "shift_alert_flag": bool(self.shift_alert_flag),
        }


@dataclass
class ShiftAwareConfig:
    enabled: bool = False
    policy_mode: str = "legacy_rac_cert"
    aci_mode: str = "fixed"
    adaptation_step: float = 0.02
    target_alpha: float = 0.1
    min_quantile: float = 0.5
    max_quantile: float = 4.0
    reliability_bins: int = 4
    volatility_bins: int = 4
    watch_threshold: float = 0.75
    degraded_threshold: float = 0.5
    invalid_threshold: float = 0.3
    validity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "reliability": 0.25,
            "residual": 0.25,
            "drift": 0.2,
            "subgroup": 0.2,
            "adaptation": 0.1,
        }
    )
    linear_reliability_gain: float = 0.8
    linear_validity_gain: float = 1.0
    linear_drift_gain: float = 0.5
    linear_under_coverage_gain: float = 1.0
    widening_cap: float = 4.0

    def __post_init__(self) -> None:
        if self.reliability_bins < 1 or self.volatility_bins < 1:
            raise ValueError("bin counts must be >= 1")
        if not (0.0 <= self.invalid_threshold <= self.degraded_threshold <= self.watch_threshold <= 1.0):
            raise ValueError("thresholds must satisfy invalid <= degraded <= watch <= 1")

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> "ShiftAwareConfig":
        data = dict(cfg or {})
        return cls(
            enabled=bool(data.get("enable", data.get("enabled", False))),
            policy_mode=str(data.get("policy_mode", "legacy_rac_cert")),
            aci_mode=str(data.get("aci_mode", "fixed")),
            adaptation_step=float(data.get("adaptation_step_size", data.get("adaptation_step", 0.02))),
            target_alpha=float(data.get("target_alpha", 0.1)),
            min_quantile=float(data.get("min_quantile", 0.5)),
            max_quantile=float(data.get("max_quantile", 4.0)),
            reliability_bins=int(data.get("reliability_bin_count", data.get("reliability_bins", 4))),
            volatility_bins=int(data.get("volatility_bin_count", data.get("volatility_bins", 4))),
            watch_threshold=float(data.get("watch_threshold", 0.75)),
            degraded_threshold=float(data.get("degraded_threshold", 0.5)),
            invalid_threshold=float(data.get("invalid_threshold", 0.3)),
            validity_weights=dict(data.get("validity_score_weights", data.get("validity_weights", {})) or cls().validity_weights),
            widening_cap=float(data.get("widening_cap", data.get("widening_caps", 4.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "policy_mode": self.policy_mode,
            "aci_mode": self.aci_mode,
            "adaptation_step": float(self.adaptation_step),
            "target_alpha": float(self.target_alpha),
            "min_quantile": float(self.min_quantile),
            "max_quantile": float(self.max_quantile),
            "reliability_bins": int(self.reliability_bins),
            "volatility_bins": int(self.volatility_bins),
            "watch_threshold": float(self.watch_threshold),
            "degraded_threshold": float(self.degraded_threshold),
            "invalid_threshold": float(self.invalid_threshold),
            "validity_weights": {k: float(v) for k, v in sorted(self.validity_weights.items())},
            "widening_cap": float(self.widening_cap),
        }
