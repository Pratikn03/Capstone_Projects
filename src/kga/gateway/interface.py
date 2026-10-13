"""kga.gateway.interface -- Core protocols and containers for the Autonomous Safety Gate."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

from kga.policy import Decision


class FallbackReason(str, enum.Enum):
    """Explicit cause for falling back to the frozen base model."""
    NONE = "none"  # Not a fallback; adaptation was certified and served
    ZERO_CONTAINMENT = "zero_containment"  # Delta interval contains 0 (uncertainty)
    ESTIMATED_HARMFUL = "estimated_harmful"  # Upper bound is strictly negative (FREEZE)
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"  # Martingale drift monitor or error latch tripped
    OOD_EVIDENCE = "ood_evidence"  # Feature vector Z lies outside calibration support
    NUMERICAL_INSTABILITY = "numerical_instability"  # NaN/Inf or extreme entropy collapse
    LATENCY_TIMEOUT = "latency_timeout"  # Adaptation step exceeded latency SLA budget
    SAMPLE_SIZE_STARVATION = "sample_size_starvation"  # Batch size m < m_min for Hoeffding concentration
    ADAPTATION_EXCEPTION = "adaptation_exception"  # Adapter crashed during forward/update step
    SHADOW_MODE = "shadow_mode"  # Shadow execution mode intentionally serves frozen base


@runtime_checkable
class Predictor(Protocol):
    """Protocol for model inference."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for extracting label-free evidence vector Z from calibration and test scores."""
    def __call__(self, calib_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class GatewayDecision:
    """Comprehensive decision and audit record emitted by the SafeInferenceGateway."""
    action: Decision
    served_model: str  # "frozen_base" (f0) or "adapted_candidate" (fa)
    fallback_reason: FallbackReason
    delta_hat: float
    epsilon: float
    sampling_radius: float  # b(m, delta)
    lower_bound: float  # delta_hat - (epsilon + sampling_radius)
    upper_bound: float  # delta_hat + (epsilon + sampling_radius)
    sample_size: int  # m
    input_hash: str  # SHA-256 of test input array
    latency_ms: float
    guard_status: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        """True if the gateway routed traffic to the frozen base model."""
        return self.served_model == "frozen_base"

    @property
    def is_adapted(self) -> bool:
        """True if the gateway verified positive benefit and routed to the adapted model."""
        return self.served_model == "adapted_candidate"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision record to a JSON-compatible dictionary."""
        return {
            "action": self.action.value,
            "served_model": self.served_model,
            "fallback_reason": self.fallback_reason.value,
            "delta_hat": float(self.delta_hat),
            "epsilon": float(self.epsilon),
            "sampling_radius": float(self.sampling_radius),
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "sample_size": int(self.sample_size),
            "input_hash": self.input_hash,
            "latency_ms": float(self.latency_ms),
            "guard_status": self.guard_status,
            "extra": self.extra,
        }
