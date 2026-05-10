"""Two-state indistinguishability lower-bound helpers (T10)."""

from __future__ import annotations

from collections.abc import Sequence


def estimate_total_variation(p0: Sequence[float], p1: Sequence[float]) -> float:
    if len(p0) != len(p1):
        raise ValueError("distributions must have same support length")
    return 0.5 * sum(abs(a - b) for a, b in zip(p0, p1, strict=True))


def two_state_lower_bound(tv: float, disjoint_safe_sets: bool) -> float:
    if not disjoint_safe_sets:
        return 0.0
    clipped = max(0.0, min(1.0, tv))
    return (1.0 - clipped) / 2.0


def build_boundary_pair(x0: object, x1: object, c0: set[object], c1: set[object]) -> dict[str, object]:
    return {"x0": x0, "x1": x1, "disjoint_safe_sets": len(c0 & c1) == 0}


def evaluate_boundary_policy_risk(tv: float, disjoint_safe_sets: bool) -> dict[str, float]:
    return {"lower_bound": two_state_lower_bound(tv, disjoint_safe_sets), "tv": max(0.0, min(1.0, tv))}
