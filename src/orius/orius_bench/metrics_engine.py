"""ORIUS-Bench standardised metrics engine.

Computes the seven Paper-4 benchmark metrics across an episode trajectory:

1. TSVR  – True-State Violation Rate
2. OASG  – Observation-Action Safety Gap
3. CVA   – Certificate Validity Accuracy
4. GDQ   – Graceful Degradation Quality
5. IR    – Intervention Rate
6. AC    – Audit Completeness
7. RL    – Recovery Latency (steps to regain valid certificate)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class StepRecord:
    """One timestep worth of episode data."""

    step: int
    true_state: dict[str, float]
    observed_state: dict[str, float]
    action: dict[str, float]
    true_constraint_violated: bool | None = None
    observed_constraint_satisfied: bool | None = None
    true_margin: float | None = None
    observed_margin: float | None = None
    intervened: bool = False
    fallback_used: bool = False
    certificate_valid: bool = True
    certificate_predicted_valid: bool = True
    useful_work: float = 0.0
    audit_fields_present: int = 0
    audit_fields_required: int = 1
    latency_us: float | None = None
    domain_metrics: dict[str, float] = field(default_factory=dict)
    # Legacy compatibility input fields. New code should use the canonical
    # domain-agnostic fields above.
    soc_after: float | None = None
    soc_min: float = 0.1
    soc_max: float = 0.9
    fallback_active: bool = False

    def resolved_true_constraint_violated(self) -> bool:
        if self.true_constraint_violated is not None:
            return bool(self.true_constraint_violated)
        if self.soc_after is None:
            return False
        return bool(self.soc_after < self.soc_min or self.soc_after > self.soc_max)

    def resolved_observed_constraint_satisfied(self) -> bool | None:
        if self.observed_constraint_satisfied is not None:
            return bool(self.observed_constraint_satisfied)
        return bool(self.certificate_predicted_valid)

    def resolved_fallback_used(self) -> bool:
        return bool(self.fallback_used or self.fallback_active)

    def resolved_intervened(self) -> bool:
        return bool(self.intervened or self.resolved_fallback_used())


@dataclass
class BenchmarkMetrics:
    """Full suite of ORIUS-Bench metrics for one episode."""

    tsvr: float
    oasg: float
    cva: float
    gdq: float
    intervention_rate: float
    audit_completeness: float
    recovery_latency: float
    n_steps: int
    raw: dict[str, Any] | None = None


def compute_tsvr(records: Sequence[StepRecord]) -> float:
    """True-State Violation Rate: fraction of steps with true constraint violation."""
    if not records:
        return 0.0
    violations = sum(1 for r in records if r.resolved_true_constraint_violated())
    return violations / len(records)


def compute_oasg(records: Sequence[StepRecord], window: int = 12) -> float:
    """Observation-Action Safety Gap.

    Fraction of comparable steps where the observed state appears safe while
    the true state is already violating the constraint surface.
    """
    del window
    comparable = [
        r for r in records
        if r.resolved_observed_constraint_satisfied() is not None
    ]
    if not comparable:
        return 0.0
    gaps = sum(
        1
        for r in comparable
        if bool(r.resolved_observed_constraint_satisfied()) and r.resolved_true_constraint_violated()
    )
    return gaps / len(comparable)


def compute_cva(records: Sequence[StepRecord]) -> float:
    """Certificate Validity Accuracy: fraction of steps where the
    certificate prediction matched ground truth."""
    if not records:
        return 1.0
    correct = sum(
        1 for r in records if r.certificate_valid == r.certificate_predicted_valid
    )
    return correct / len(records)


def compute_gdq(records: Sequence[StepRecord]) -> float:
    """Graceful Degradation Quality.

    GDQ = useful_work_fraction × (1 − TSVR) × descent_stability
    where descent_stability measures monotonic decrease in action magnitude
    during fallback.
    """
    if not records:
        return 0.0

    total_work = sum(r.useful_work for r in records)
    max_possible = len(records)  # normalized unit work per step
    uw_frac = min(1.0, total_work / max_possible) if max_possible > 0 else 0.0

    tsvr = compute_tsvr(records)

    # Descent stability: fraction of consecutive fallback steps with
    # non-increasing action magnitude (domain-agnostic: sum of abs of numeric values)
    fb_steps = [r for r in records if r.resolved_fallback_used()]
    if len(fb_steps) < 2:
        descent = 1.0
    else:
        def _action_magnitude(action: dict[str, float]) -> float:
            return sum(
                abs(float(v)) for v in action.values()
                if isinstance(v, (int, float)) or (hasattr(v, "__float__") and not isinstance(v, bool))
            )

        magnitudes = [_action_magnitude(r.action) for r in fb_steps]
        monotonic = sum(
            1 for i in range(1, len(magnitudes)) if magnitudes[i] <= magnitudes[i - 1] + 1e-9
        )
        descent = monotonic / (len(magnitudes) - 1)

    return float(uw_frac * (1.0 - tsvr) * descent)


def compute_intervention_rate(records: Sequence[StepRecord]) -> float:
    """Fraction of steps where the fallback overrode the controller."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.resolved_intervened()) / len(records)


def compute_audit_completeness(records: Sequence[StepRecord]) -> float:
    """Fraction of audit fields present across all steps."""
    if not records:
        return 1.0
    total_required = sum(r.audit_fields_required for r in records)
    if total_required == 0:
        return 1.0
    total_present = sum(r.audit_fields_present for r in records)
    return float(np.clip(total_present / total_required, 0.0, 1.0))


def compute_recovery_latency(records: Sequence[StepRecord]) -> float:
    """Average number of steps to regain a valid certificate after expiry."""
    recovery_times: list[int] = []
    in_expired = False
    expire_start = 0

    for r in records:
        if not r.certificate_valid and not in_expired:
            in_expired = True
            expire_start = r.step
        elif r.certificate_valid and in_expired:
            recovery_times.append(r.step - expire_start)
            in_expired = False

    if not recovery_times:
        return 0.0
    return float(np.mean(recovery_times))


def compute_all_metrics(records: Sequence[StepRecord]) -> BenchmarkMetrics:
    """Compute the full seven-metric suite and return a ``BenchmarkMetrics``."""
    return BenchmarkMetrics(
        tsvr=compute_tsvr(records),
        oasg=compute_oasg(records),
        cva=compute_cva(records),
        gdq=compute_gdq(records),
        intervention_rate=compute_intervention_rate(records),
        audit_completeness=compute_audit_completeness(records),
        recovery_latency=compute_recovery_latency(records),
        n_steps=len(records),
    )
