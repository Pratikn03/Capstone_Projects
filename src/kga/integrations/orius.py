"""kga.integrations.orius -- Native K-Bound (KGA) + ORIUS DC3S integration bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from orius.kbound.mind import (
    KBoundMindEvaluation,
    ProactiveKBoundGovernor,
    UniversalPhysicalAIShield,
    VelocityAwareUniversalShield,
    evaluate_kbound_mind,
)


class KGAORIUSShieldBridge:
    """High-level bridge connecting K-Bound epistemic decisions to ORIUS DC3S shields."""

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        k_tighten: float = 1.2,
        delta_hat: float = 0.045,
        w_threshold: float = 0.65,
    ) -> None:
        self.universal_shield = UniversalPhysicalAIShield(
            alpha=alpha,
            k_tighten=k_tighten,
            delta_hat=delta_hat,
            w_threshold=w_threshold,
        )

    def evaluate_epistemic_gate(
        self,
        *,
        reliability_w: float,
        calib_residuals: Sequence[float] | None = None,
        residual: float | None = None,
    ) -> KBoundMindEvaluation:
        return evaluate_kbound_mind(
            reliability_w=reliability_w,
            delta_hat=self.universal_shield.delta_hat,
            calib_residuals=calib_residuals,
            alpha=self.universal_shield.alpha,
            w_threshold=self.universal_shield.w_threshold,
            residual=residual,
        )

    def filter_actuation(self, *args: Any, **kwargs: Any) -> tuple[float, float, bool, float]:
        return self.universal_shield.filter_actuation(*args, **kwargs)


__all__ = [
    "KBoundMindEvaluation",
    "KGAORIUSShieldBridge",
    "ProactiveKBoundGovernor",
    "UniversalPhysicalAIShield",
    "VelocityAwareUniversalShield",
    "evaluate_kbound_mind",
]
