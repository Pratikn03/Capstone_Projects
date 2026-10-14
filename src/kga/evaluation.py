"""Descriptive, post-outcome evaluation of KGA routing actions.

The summaries here equally weight supplied records.  They do not establish
confidence, guarantees, protocol validity, or an inferential sample size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

_ACTIONS = frozenset({"ADAPT", "FREEZE", "ABSTAIN"})


@dataclass(frozen=True)
class RecordContribution:
    """One record's additive contributions to descriptive mean regrets."""

    action: str
    delta: float
    regret: float
    harmful_adapt_regret: float
    missed_benefit_freeze_regret: float
    abstain_opportunity_regret: float
    always_adapt_regret: float
    always_freeze_regret: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Equally weighted descriptive summary over the supplied records."""

    n: int
    adapt_count: int
    freeze_count: int
    abstain_count: int
    adapt_rate: float
    freeze_rate: float
    abstain_rate: float
    commitment_count: int
    commitment_rate: float
    false_adapt_count: int
    false_freeze_count: int
    false_adapt_rate_all: float
    false_freeze_rate_all: float
    false_adapt_rate_conditional: float | None
    false_freeze_rate_conditional: float | None
    directional_accuracy_commitments: float | None
    mean_regret: float
    mean_regret_commitments: float | None
    mean_harmful_adapt_regret: float
    mean_missed_benefit_freeze_regret: float
    mean_abstain_opportunity_regret: float
    mean_always_adapt_regret: float
    mean_always_freeze_regret: float
    always_adapt_minus_kga_regret: float
    always_freeze_minus_kga_regret: float
    description: str


@dataclass(frozen=True)
class EvaluationReport:
    """Immutable per-record contributions and their descriptive summary."""

    records: tuple[RecordContribution, ...]
    summary: EvaluationSummary


def _materialize_inputs(actions: Sequence[str], deltas: Sequence[float]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise TypeError("actions must be a one-dimensional sequence")
    if isinstance(deltas, (str, bytes)) or not isinstance(deltas, Sequence):
        raise TypeError("deltas must be a one-dimensional sequence")
    if not actions:
        raise ValueError("at least one record is required")
    if len(actions) != len(deltas):
        raise ValueError("actions and deltas must have the same length")

    copied_actions: list[str] = []
    copied_deltas: list[float] = []
    for action, delta in zip(actions, deltas, strict=True):
        if isinstance(action, bool) or not isinstance(action, str):
            raise TypeError("each action must be a string")
        if action not in _ACTIONS:
            raise ValueError("action must be ADAPT, FREEZE, or ABSTAIN")
        if isinstance(delta, bool) or not isinstance(delta, Real):
            raise TypeError("each delta must be a real, non-boolean number")
        value = float(delta)
        if not math.isfinite(value):
            raise ValueError("delta must be finite")
        if value < -1.0 or value > 1.0:
            raise ValueError("delta must be in [-1, 1]")
        copied_actions.append(action)
        copied_deltas.append(value)
    return tuple(copied_actions), tuple(copied_deltas)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def evaluate_actions(actions: Sequence[str], deltas: Sequence[float]) -> EvaluationReport:
    """Compute faithful post-outcome descriptive routing metrics.

    ``delta`` is the cell benefit (adapt accuracy minus freeze accuracy).
    Positive baseline-minus-KGA regret gaps favor KGA.
    """

    copied_actions, copied_deltas = _materialize_inputs(actions, deltas)
    records: list[RecordContribution] = []
    for action, delta in zip(copied_actions, copied_deltas, strict=True):
        optimal_benefit = max(delta, 0.0)
        regret = optimal_benefit - delta * (action == "ADAPT")
        records.append(
            RecordContribution(
                action=action,
                delta=delta,
                regret=regret,
                harmful_adapt_regret=max(-delta, 0.0) * (action == "ADAPT"),
                missed_benefit_freeze_regret=optimal_benefit * (action == "FREEZE"),
                abstain_opportunity_regret=optimal_benefit * (action == "ABSTAIN"),
                always_adapt_regret=max(-delta, 0.0),
                always_freeze_regret=optimal_benefit,
            )
        )

    frozen_records = tuple(records)
    n = len(frozen_records)
    adapt_count = copied_actions.count("ADAPT")
    freeze_count = copied_actions.count("FREEZE")
    abstain_count = copied_actions.count("ABSTAIN")
    commitment_count = adapt_count + freeze_count
    false_adapt_count = sum(
        action == "ADAPT" and delta <= 0.0 for action, delta in zip(copied_actions, copied_deltas, strict=True)
    )
    false_freeze_count = sum(
        action == "FREEZE" and delta >= 0.0 for action, delta in zip(copied_actions, copied_deltas, strict=True)
    )
    correct_commitments = commitment_count - false_adapt_count - false_freeze_count
    mean_regret = _mean([record.regret for record in frozen_records])
    always_adapt = _mean([record.always_adapt_regret for record in frozen_records])
    always_freeze = _mean([record.always_freeze_regret for record in frozen_records])
    committed_regrets = [record.regret for record in frozen_records if record.action != "ABSTAIN"]

    summary = EvaluationSummary(
        n=n,
        adapt_count=adapt_count,
        freeze_count=freeze_count,
        abstain_count=abstain_count,
        adapt_rate=adapt_count / n,
        freeze_rate=freeze_count / n,
        abstain_rate=abstain_count / n,
        commitment_count=commitment_count,
        commitment_rate=commitment_count / n,
        false_adapt_count=false_adapt_count,
        false_freeze_count=false_freeze_count,
        false_adapt_rate_all=false_adapt_count / n,
        false_freeze_rate_all=false_freeze_count / n,
        false_adapt_rate_conditional=(false_adapt_count / adapt_count if adapt_count else None),
        false_freeze_rate_conditional=(false_freeze_count / freeze_count if freeze_count else None),
        directional_accuracy_commitments=(correct_commitments / commitment_count if commitment_count else None),
        mean_regret=mean_regret,
        mean_regret_commitments=(_mean(committed_regrets) if committed_regrets else None),
        mean_harmful_adapt_regret=_mean([record.harmful_adapt_regret for record in frozen_records]),
        mean_missed_benefit_freeze_regret=_mean([record.missed_benefit_freeze_regret for record in frozen_records]),
        mean_abstain_opportunity_regret=_mean([record.abstain_opportunity_regret for record in frozen_records]),
        mean_always_adapt_regret=always_adapt,
        mean_always_freeze_regret=always_freeze,
        always_adapt_minus_kga_regret=always_adapt - mean_regret,
        always_freeze_minus_kga_regret=always_freeze - mean_regret,
        description=(
            "Equally weighted descriptive record summary; n is the number of "
            "records, not an inferential independent sample size."
        ),
    )
    return EvaluationReport(records=frozen_records, summary=summary)
