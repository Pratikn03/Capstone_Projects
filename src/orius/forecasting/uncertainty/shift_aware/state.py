from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ValidityStatus = Literal["nominal", "watch", "degraded", "invalid"]


@dataclass
class CoverageWindowStats:
    n: int = 0
    covered: int = 0
    miss: int = 0
    target_coverage: float = 0.9
    mean_width: float = 0.0
    mean_abs_residual: float = 0.0

    def __post_init__(self) -> None:
        self.target_coverage = float(min(max(self.target_coverage, 1.0e-6), 1.0 - 1.0e-6))

    @property
    def empirical_coverage(self) -> float:
        return float(self.covered / self.n) if self.n else 1.0

    @property
    def under_coverage_gap(self) -> float:
        return max(0.0, self.target_coverage - self.empirical_coverage)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["empirical_coverage"] = self.empirical_coverage
        payload["under_coverage_gap"] = self.under_coverage_gap
        return payload


@dataclass
class GroupCoverageStats:
    group_key: str
    count: int = 0
    covered: int = 0
    miss_count: int = 0
    avg_interval_width: float = 0.0
    avg_abs_residual: float = 0.0
    target_coverage: float = 0.9

    @property
    def empirical_coverage(self) -> float:
        return float(self.covered / self.count) if self.count else 1.0

    @property
    def under_coverage_gap(self) -> float:
        return max(0.0, float(self.target_coverage) - self.empirical_coverage)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["empirical_coverage"] = self.empirical_coverage
        payload["under_coverage_gap"] = self.under_coverage_gap
        return payload


@dataclass
class AdaptiveQuantileState:
    mode: Literal["fixed", "aci_basic", "aci_clipped"] = "fixed"
    base_alpha: float = 0.1
    effective_alpha: float = 0.1
    learning_rate: float = 0.01
    alpha_min: float = 0.01
    alpha_max: float = 0.5
    updates: int = 0
    miss_streak: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShiftValidityState:
    validity_score: float = 1.0
    validity_status: ValidityStatus = "nominal"
    normalized_residual: float = 0.0
    subgroup_gap: float = 0.0
    drift_score: float = 0.0
    adaptation_instability: float = 0.0
    reliability_interaction: float = 0.0
    shift_alert_flag: bool = False

    def __post_init__(self) -> None:
        self.validity_score = float(min(max(self.validity_score, 0.0), 1.0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShiftAwareIntervalDecision:
    lower: float
    upper: float
    base_half_width: float
    adjusted_half_width: float
    inflation_multiplier: float
    adaptive_quantile: float
    validity_score: float
    validity_status: ValidityStatus
    under_coverage_gap: float
    applied_policy: str
    coverage_group_key: str = "global"
    shift_alert_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShiftAwareConfig:
    enabled: bool = False
    policy_mode: Literal[
        "legacy_rac_cert", "shift_aware_linear", "shift_aware_piecewise", "shift_aware_mondrian"
    ] = "legacy_rac_cert"
    aci_mode: Literal["fixed", "aci_basic", "aci_clipped"] = "fixed"
    adaptation_step: float = 0.01
    alpha: float = 0.1
    alpha_min: float = 0.02
    alpha_max: float = 0.3
    reliability_bins: int = 5
    volatility_bins: int = 5
    coverage_window_size: int = 128
    coverage_target: float = 0.9
    watch_threshold: float = 0.75
    degraded_threshold: float = 0.55
    invalid_threshold: float = 0.35
    max_inflation_multiplier: float = 4.0
    publication_dir: str = "reports/publication"
    runtime_state_path: str | None = None
    artifact_toggles: dict[str, bool] = field(default_factory=lambda: {"enabled": True})
    validity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "residual": 0.25,
            "subgroup": 0.25,
            "drift": 0.2,
            "adaptation": 0.15,
            "reliability": 0.15,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> ShiftAwareConfig:
        if not cfg:
            return cls()
        raw = dict(cfg)
        # Backward-compatible aliases from YAML / previous drafts.
        aliases: dict[str, str] = {
            "enable": "enabled",
            "adaptation_step_size": "adaptation_step",
            "reliability_bin_count": "reliability_bins",
            "volatility_bin_count": "volatility_bins",
            "validity_score_weights": "validity_weights",
        }
        for src, dst in aliases.items():
            if src in raw and dst not in raw:
                raw[dst] = raw[src]
        thresholds = dict(raw.get("thresholds", {})) if isinstance(raw.get("thresholds"), dict) else {}
        widening_caps = (
            dict(raw.get("widening_caps", {})) if isinstance(raw.get("widening_caps"), dict) else {}
        )
        if "watch" in thresholds:
            raw["watch_threshold"] = thresholds["watch"]
        if "degraded" in thresholds:
            raw["degraded_threshold"] = thresholds["degraded"]
        if "invalid" in thresholds:
            raw["invalid_threshold"] = thresholds["invalid"]
        if "max_inflation_multiplier" in widening_caps:
            raw["max_inflation_multiplier"] = widening_caps["max_inflation_multiplier"]
        if "artifact_output_toggles" in raw and "artifact_toggles" not in raw:
            raw["artifact_toggles"] = raw["artifact_output_toggles"]
        weights = dict(raw.get("validity_weights", {}))
        defaults = cls().to_dict()
        merged_weights = dict(defaults["validity_weights"])
        merged_weights.update(weights)
        raw["validity_weights"] = merged_weights
        return cls(**{k: v for k, v in raw.items() if k in defaults})
