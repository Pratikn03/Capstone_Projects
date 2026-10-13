"""kga.gateway.async_gateway -- Asynchronous non-blocking Safe Inference Gateway."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from kga.gateway.interface import FallbackReason, GatewayDecision, Predictor
from kga.gateway.modes import DeploymentMode
from kga.guards.circuit_breaker import CircuitBreaker
from kga.guards.evidence_support import EvidenceSupportGuard
from kga.guards.numerical_health import NumericalHealthGuard
from kga.guards.streaming_monitor import StreamingDriftMonitor
from kga.observability.audit_logger import AuditLogger
from kga.observability.metrics import METRICS
from kga.policy import Decision


class AsyncSafeInferenceGateway:
    """Non-blocking asynchronous safety gateway for high-throughput ASGI/FastAPI pipelines."""

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

    async def predict(
        self,
        x: np.ndarray,
        request_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, GatewayDecision]:
        """Asynchronously execute safe inference with asyncio watchdog timeouts."""
        t_start = time.monotonic()
        x_arr = np.asarray(x)
        m = int(x_arr.shape[0]) if x_arr.ndim > 0 else 1
        input_hash = hashlib.sha256(x_arr.tobytes()[:4096]).hexdigest()

        # Step 1: Base model inference (fast path)
        y_base = await asyncio.to_thread(self.base_model, x_arr)

        # Step 2: Sample size check
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

        # Step 3: Circuit breaker check
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

        # Step 4: Run candidate adaptation under asyncio timeout SLA
        timeout_sec = self.max_latency_ms / 1000.0
        try:
            y_adapted = await asyncio.wait_for(
                asyncio.to_thread(self.candidate_adapter, x_arr),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            self.circuit_breaker.record_failure(f"Async adaptation timeout > SLA {self.max_latency_ms:.1f}ms")
            METRICS.inc_counter("kga_requests_total", action=Decision.ABSTAIN.value, served="frozen_base", reason=FallbackReason.LATENCY_TIMEOUT.value)
            decision = self._create_fallback_decision(
                action=Decision.ABSTAIN,
                reason=FallbackReason.LATENCY_TIMEOUT,
                m=m,
                input_hash=input_hash,
                t_start=t_start,
                guard_info={"timeout_sec": timeout_sec},
            )
            self._log_and_emit(decision, request_id)
            return y_base, decision
        except Exception as ex:
            self.circuit_breaker.record_failure(f"Async adapter exception: {ex}")
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

        # Step 5: Numerical health check
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

        # Step 6 & 7: Feature extraction and OOD guard
        z_vector = await asyncio.to_thread(self.feature_extractor, x_arr, y_base, y_adapted)
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

        # Step 8: Benefit estimation and compound radius
        delta_hat = float(await asyncio.to_thread(self.benefit_estimator, z_vector))
        b_radius = float(math.sqrt(math.log(2.0 / self.delta) / (2.0 * m)))
        total_radius = float(self.calibration_epsilon + b_radius)

        lower_bound = float(delta_hat - total_radius)
        upper_bound = float(delta_hat + total_radius)

        # Step 9: Decision rule
        if lower_bound > self.decision_margin:
            action = Decision.ADAPT
            fallback_reason = FallbackReason.NONE
        elif upper_bound < -self.decision_margin:
            action = Decision.FREEZE
            fallback_reason = FallbackReason.ESTIMATED_HARMFUL
        else:
            action = Decision.ABSTAIN
            fallback_reason = FallbackReason.ZERO_CONTAINMENT

        # Step 10: Drift monitor
        monitor_status = self.drift_monitor.update_from_margin(delta_sample=delta_hat)
        if monitor_status.tripped:
            self.circuit_breaker.trip(f"Drift monitor alarm: {monitor_status.trip_reason}")

        # Step 11: Route
        total_latency_ms = (time.monotonic() - t_start) * 1000.0
        if self.mode == DeploymentMode.SHADOW_CANARY:
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

        METRICS.inc_counter("kga_requests_total", action=action.value, served=served_model, reason=effective_fallback.value)
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
