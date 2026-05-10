"""Graceful-degradation policy comparisons for theorem T8.

The helpers in this module keep the theorem surface narrow: T8 is a paired
trace comparison under the same admissible fault sequence. A policy passes
only when it weakly reduces true-state violations and preserves a declared
fraction of useful work.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyOutcome:
    """Trajectory summary used by the T8 dominance check."""

    name: str
    violations: tuple[bool, ...]
    useful_work: float
    fallback_count: int = 0

    @property
    def tsvr(self) -> float:
        if not self.violations:
            return 0.0
        return sum(1 for value in self.violations if value) / len(self.violations)


class BlindPersistencePolicy:
    """Policy model for replaying the commanded action without repair."""

    name = "Blind"

    def summarize(self, hazard_trace: Sequence[bool], work_trace: Sequence[float]) -> PolicyOutcome:
        return PolicyOutcome(
            name=self.name,
            violations=tuple(bool(x) for x in hazard_trace),
            useful_work=float(sum(work_trace)),
            fallback_count=0,
        )


class ImmediateShutdownPolicy:
    """Policy model that denies all work after the first degraded observation."""

    name = "Shutdown"

    def summarize(self, hazard_trace: Sequence[bool], work_trace: Sequence[float]) -> PolicyOutcome:
        return PolicyOutcome(
            name=self.name,
            violations=tuple(False for _ in hazard_trace),
            useful_work=0.0,
            fallback_count=len(hazard_trace),
        )


class RampDownPolicy:
    """Simple policy that halves useful work and partially reduces violations."""

    name = "Ramp"

    def summarize(self, hazard_trace: Sequence[bool], work_trace: Sequence[float]) -> PolicyOutcome:
        violations = tuple(bool(value) and idx % 2 == 0 for idx, value in enumerate(hazard_trace))
        return PolicyOutcome(
            name=self.name,
            violations=violations,
            useful_work=0.5 * float(sum(work_trace)),
            fallback_count=sum(1 for value in hazard_trace if value),
        )


class ORIUSGracefulPolicy:
    """Policy model for certified repair/fallback with retained useful work."""

    name = "ORIUS"

    def summarize(self, hazard_trace: Sequence[bool], work_trace: Sequence[float]) -> PolicyOutcome:
        retained_work = 0.0
        violations: list[bool] = []
        fallback_count = 0
        for hazard, work in zip(hazard_trace, work_trace, strict=True):
            if hazard:
                fallback_count += 1
                retained_work += 0.35 * float(work)
                violations.append(False)
            else:
                retained_work += float(work)
                violations.append(False)
        return PolicyOutcome(
            name=self.name,
            violations=tuple(violations),
            useful_work=retained_work,
            fallback_count=fallback_count,
        )


def graceful_dominance_with_useful_work(
    graceful: PolicyOutcome,
    uncontrolled: PolicyOutcome,
    *,
    lambda_work: float,
) -> dict[str, float | bool]:
    """Evaluate the T8 two-objective dominance relation."""

    if not (0.0 <= lambda_work <= 1.0):
        raise ValueError("lambda_work must lie in [0, 1].")
    graceful_violations = sum(1 for value in graceful.violations if value)
    uncontrolled_violations = sum(1 for value in uncontrolled.violations if value)
    required_work = float(lambda_work) * float(uncontrolled.useful_work)
    safety_dominates = graceful_violations <= uncontrolled_violations
    work_preserved = float(graceful.useful_work) + 1e-12 >= required_work
    return {
        "safety_dominates": bool(safety_dominates),
        "work_preserved": bool(work_preserved),
        "passes": bool(safety_dominates and work_preserved),
        "graceful_violation_count": float(graceful_violations),
        "uncontrolled_violation_count": float(uncontrolled_violations),
        "useful_work_fraction": float(graceful.useful_work / max(uncontrolled.useful_work, 1e-12)),
        "lambda_work": float(lambda_work),
    }


def evaluate_policy_frontier(
    hazard_trace: Sequence[bool],
    work_trace: Sequence[float],
    *,
    lambda_work: float = 0.25,
    policies: Iterable[object] | None = None,
) -> list[dict[str, float | int | str | bool]]:
    """Evaluate the standard T8 policy frontier on one paired trace."""

    if len(hazard_trace) != len(work_trace):
        raise ValueError("hazard_trace and work_trace must have the same length.")
    policy_list = list(
        policies
        or [
            BlindPersistencePolicy(),
            ImmediateShutdownPolicy(),
            RampDownPolicy(),
            ORIUSGracefulPolicy(),
        ]
    )
    uncontrolled = BlindPersistencePolicy().summarize(hazard_trace, work_trace)
    rows: list[dict[str, float | int | str | bool]] = []
    for policy in policy_list:
        outcome = policy.summarize(hazard_trace, work_trace)
        dominance = graceful_dominance_with_useful_work(
            outcome,
            uncontrolled,
            lambda_work=lambda_work,
        )
        rows.append(
            {
                "policy": outcome.name,
                "tsvr": outcome.tsvr,
                "work": outcome.useful_work,
                "fallback": outcome.fallback_count,
                "useful_work_fraction": dominance["useful_work_fraction"],
                "pass": dominance["passes"],
            }
        )
    return rows
