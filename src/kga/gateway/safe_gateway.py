"""kga.gateway.safe_gateway -- Enterprise-grade Safe Inference Gateway for model adaptation."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from kga.gateway.interface import FallbackReason, GatewayDecision, Predictor
from kga.gateway.modes import DeploymentMode
from kga.guards.circuit_breaker import CircuitBreaker, CircuitState
from kga.guards.evidence_support import EvidenceSupportGuard
from kga.guards.numerical_health import NumericalHealthGuard
from kga.guards.streaming_monitor import StreamingDriftMonitor
from kga.observability.audit_logger import AuditLogger
from kga.observability.metrics import METRICS
from kga.policy import Decision


class SafeInferenceGateway:
    """Audit-grounded, fail-closed safety gate and circuit breaker for test-time adaptation.
    
    Routes traffic between a reliable frozen base model (f0) and a candidate adapted
    model (fa). Upholds the invariant that live user traffic defaults to f0 unless
    a strict mathematical certificate guarantees positive adaptation benefit.
    """

    def __init__(
        self,
        base_model: Predictor,
        candidate_adapter: Predictor,
        benefit_estimator: Callable[[np.ndarray], float],
        feature_extractor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        calibration_epsilon: float,
        alpha: float = 0.10,
        delta: float = 0.05,
        m_min: int = 16,
        decision_margin: float = 0.0,
        max_latency_ms: float = 100.0,
        mode: DeploymentMode = DeploymentMode.INLINE_GATE,
        circuit_breaker: Optional[CircuitBreaker] = None,
        evidence_guard: Optional[EvidenceSupportGuard] = None,
        numerical_guard: Optional[NumericalHealthGuard] = None,
        drift_monitor: Optional[StreamingDriftMonitor] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self.base_model = base_model
        self.candidate_adapter = candidate_adapter
        self.benefit_estimator = benefit_estimator
        self.feature_extractor = feature_extractor
        
        self.calibration_epsilon = float(calibration_epsilon)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.m_min = max(1, int(m_min))
        self.decision_margin = max(0.0, float(decision_margin))
        self.max_latency_ms = float(max_latency_ms)
        self.mode = mode

        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.evidence_guard = evidence_guard
        self.numerical_guard = numerical_guard or NumericalHealthGuard()
        self.drift_monitor = drift_monitor or StreamingDriftMonitor(alpha=alpha)
        self.audit_logger = audit_logger

    def predict(
        self,
        x: np.ndarray,
        request_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, GatewayDecision]:
        """Execute safe model inference with audit certification and automated fail-closed fallback."""
        t_start = time.monotonic()
        x_arr = np.asarray(x)
        m = int(x_arr.shape[0]) if x_arr.ndim > 0 else 1
        
        # Zero-trust input hashing
        input_hash = hashlib.sha256(x_arr.tobytes()[:4096]).hexdigest()

        # Step 1: Baseline inference on frozen model f0 (guaranteed SLA)
        y_base = self.base_model(x_arr)

        # Step 2: Check sample size starvation for concentration bound
        if m < self.m_min:
            METRICS.inc_counter("kga_requests_total", action=Decision.ABSTAIN.value, served="frozen_base", reason=FallbackReason.SAMPLE_SIZE_STARVATION.value)
            decision = self._create_fallback_decision(
                action=Decision.ABSTAIN,
                reason=FallbackReason.SAMPLE_SIZE_STARVATION,
                m=m,
                input_hash=input_hash,
                t_start=t_start,
                guard_info={"m": m, "m_min": self.m_min},
            )
            self._log_and_emit(decision, request_id)
            return y_base, decision

        # Step 3: Check circuit breaker state
        if not self.circuit_breaker.allow_adaptation_attempt():
            METRICS.inc_counter("kga_requests_total", action=Decision.ABSTAIN.value, served="frozen_base", reason=FallbackReason.CIRCUIT_BREAKER_OPEN.value)
            decision = self._create_fallback_decision(
                action=Decision.ABSTAIN,
                reason=FallbackReason.CIRCUIT_BREAKER_OPEN,
                m=m,
                input_hash=input_hash,
                t_start=t_start,
                guard_info={"circuit_state": self.circuit_breaker.state.value},
            )
            self._log_and_emit(decision, request_id)
            return y_base, decision

        # Step 4: Attempt candidate adaptation with SLA watchdog
        t_adapt_start = time.monotonic()
        try:
            y_adapted = self.candidate_adapter(x_arr)
            adapt_latency_ms = (time.monotonic() - t_adapt_start) * 1000.0
        except Exception as ex:
            self.circuit_breaker.record_failure(f"Adapter exception: {ex}")
            METRICS.inc_counter("kga_requests_total", action=Decision.ABSTAIN.value, served="frozen_base", reason=FallbackReason.ADAPTATION_EXCEPTION.value)
            decision = self._create_fallback_decision(
                action=Decision.ABSTAIN,
                reason=FallbackReason.ADAPTATION_EXCEPTION,
                m=m,
                input_hash=input_hash,
                t_start=t_start,
                guard_info={"exception": str(ex)},
            )
            self._log_and_emit(decision, request_id)
            return y_base, decision

        if adapt_latency_ms > self.max_latency_ms:
            self.circuit_breaker.record_failure(f"Adaptation latency {adapt_latency_ms:.1f}ms > SLA {self.max_latency_ms:.1f}ms")
            METRICS.inc_counter("kga_requests_total", action=Decision.ABSTAIN.value, served="frozen_base", reason=FallbackReason.LATENCY_TIMEOUT.value)
            decision = self._create_fallback_decision(
                action=Decision.ABSTAIN,
                reason=FallbackReason.LATENCY_TIMEOUT,
                m=m,
                input_hash=input_hash,
                t_start=t_start,
                guard_info={"latency_ms": adapt_latency_ms, "sla_ms": self.max_latency_ms},
            )
            self._log_and_emit(decision, request_id)
            return y_base, decision

        # Step 5: Numerical health guard on candidate predictions
        health = self.numerical_guard.check_probabilities(y_adapted)
        if not health.is_healthy:
            self.circuit_breaker.record_failure(f"Unhealthy candidate outputs: {health.rejection_reason}")
            METRICS.inc_counter("kga_requests_total", action=Decision.ABSTAIN.value, served="frozen_base", reason=FallbackReason.NUMERICAL_INSTABILITY.value)
            decision = self._create_fallback_decision(
                action=Decision.ABSTAIN,
                reason=FallbackReason.NUMERICAL_INSTABILITY,
                m=m,
                input_hash=input_hash,
                t_start=t_start,
                guard_info={"health_reason": health.rejection_reason},
            )
            self._log_and_emit(decision, request_id)
            return y_base, decision

        # Step 6: Extract label-free evidence vector Z
        z_vector = self.feature_extractor(x_arr, y_base, y_adapted)

        # Step 7: Check evidence support against calibration manifold
        guard_info: Dict[str, Any] = {}
        if self.evidence_guard is not None:
            support = self.evidence_guard.check(z_vector)
            guard_info["support"] = {
                "is_supported": support.is_supported,
                "mahalanobis": support.mahalanobis_distance,
                "max_zscore": support.max_zscore,
            }
            if not support.is_supported:
                METRICS.inc_counter("kga_requests_total", action=Decision.ABSTAIN.value, served="frozen_base", reason=FallbackReason.OOD_EVIDENCE.value)
                decision = self._create_fallback_decision(
                    action=Decision.ABSTAIN,
                    reason=FallbackReason.OOD_EVIDENCE,
                    m=m,
                    input_hash=input_hash,
                    t_start=t_start,
                    guard_info=guard_info,
                )
                self._log_and_emit(decision, request_id)
                return y_base, decision

        # Step 8: Compute benefit estimate and compound concentration interval
        delta_hat = float(self.benefit_estimator(z_vector))
        # Hoeffding concentration radius: b(m, delta) = sqrt(ln(2/delta) / (2m))
        b_radius = float(math.sqrt(math.log(2.0 / self.delta) / (2.0 * m)))
        total_radius = float(self.calibration_epsilon + b_radius)

        lower_bound = float(delta_hat - total_radius)
        upper_bound = float(delta_hat + total_radius)

        # Step 9: Evaluate trichotomy decision rule
        if lower_bound > self.decision_margin:
            action = Decision.ADAPT
            fallback_reason = FallbackReason.NONE
        elif upper_bound < -self.decision_margin:
            action = Decision.FREEZE
            fallback_reason = FallbackReason.ESTIMATED_HARMFUL
        else:
            action = Decision.ABSTAIN
            fallback_reason = FallbackReason.ZERO_CONTAINMENT

        # Step 10: Update streaming drift monitor
        monitor_status = self.drift_monitor.update_from_margin(delta_sample=delta_hat)
        guard_info["drift_monitor"] = {
            "wealth": monitor_status.wealth,
            "threshold": monitor_status.threshold,
            "tripped": monitor_status.tripped,
        }
        if monitor_status.tripped:
            self.circuit_breaker.trip(f"Drift monitor alarm: {monitor_status.trip_reason}")

        # Step 11: Route traffic according to operational mode and decision
        total_latency_ms = (time.monotonic() - t_start) * 1000.0

        if self.mode == DeploymentMode.SHADOW_CANARY:
            # Shadow mode evaluates adaptation but serves frozen base model to user
            served_model = "frozen_base"
            effective_fallback = FallbackReason.SHADOW_MODE
            served_predictions = y_base
        elif action == Decision.ADAPT and self.circuit_breaker.allow_adaptation_attempt():
            served_model = "adapted_candidate"
            effective_fallback = FallbackReason.NONE
            served_predictions = y_adapted
            self.circuit_breaker.record_success()
        else:
            served_model = "frozen_base"
            effective_fallback = fallback_reason
            served_predictions = y_base

        decision = GatewayDecision(
            action=action,
            served_model=served_model,
            fallback_reason=effective_fallback,
            delta_hat=delta_hat,
            epsilon=self.calibration_epsilon,
            sampling_radius=b_radius,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            sample_size=m,
            input_hash=input_hash,
            latency_ms=total_latency_ms,
            guard_status=guard_info,
        )

        # Observability updates
        METRICS.inc_counter("kga_requests_total", action=action.value, served=served_model, reason=effective_fallback.value)
        METRICS.set_gauge("kga_certificate_lower_bound", lower_bound)
        METRICS.set_gauge("kga_certificate_radius", total_radius)
        METRICS.set_gauge("kga_drift_wealth", monitor_status.wealth)
        self._log_and_emit(decision, request_id)

        return served_predictions, decision

    def _create_fallback_decision(
        self,
        action: Decision,
        reason: FallbackReason,
        m: int,
        input_hash: str,
        t_start: float,
        guard_info: Dict[str, Any],
    ) -> GatewayDecision:
        latency_ms = (time.monotonic() - t_start) * 1000.0
        return GatewayDecision(
            action=action,
            served_model="frozen_base",
            fallback_reason=reason,
            delta_hat=0.0,
            epsilon=self.calibration_epsilon,
            sampling_radius=float("inf") if m < self.m_min else 0.0,
            lower_bound=-float("inf"),
            upper_bound=float("inf"),
            sample_size=m,
            input_hash=input_hash,
            latency_ms=latency_ms,
            guard_status=guard_info,
        )

    def _log_and_emit(self, decision: GatewayDecision, request_id: Optional[str]) -> None:
        if self.audit_logger is not None:
            self.audit_logger.log_decision(decision, request_id=request_id)
