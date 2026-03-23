"""SOTA baseline safety-filter wrappers for the ORIUS framework.

Three alternative safety strategies, each wrapping any DomainAdapter:

TubeMPCWrapper
    Fixed-tube strategy: builds uncertainty sets with a constant reliability
    floor ``w_floor`` instead of the live OQE score.  Represents Tube MPC's
    constant disturbance tube — does not adapt to telemetry quality.
    Failure mode: at high fault rates the fixed tube is the wrong size;
    across domains with different fault characteristics the single floor
    cannot simultaneously be tight in clean periods and protective in
    degraded ones.

CBFWrapper
    Control Barrier Function filter evaluated on the *observed* state.
    Sets ``reliability_w = 1.0`` (full trust in observations) so the
    uncertainty set is not inflated.  Exposes the OASG: h(x_obs) >= 0
    does not guarantee h(x_true) >= 0 under degraded telemetry.

LagrangianWrapper
    Lagrangian / Safe-RL analog.  Uses observed-state uncertainty (w=1.0)
    AND relaxes the tightened action set by ``soft_margin``, representing
    the residual softness of a Lagrangian penalty that was trained to
    reduce but not hard-clip constraint violations.  At deployment, the
    penalty signal comes from x_obs and cannot prevent true-state violations.

All three wrappers are domain-agnostic: they delegate to the wrapped
adapter for all domain-specific logic and override only the
reliability-handling or action-repair stage.

Usage::

    from orius.sota_baselines import wrap_adapter
    wrapped = wrap_adapter(my_adapter, "tube_mpc", w_floor=0.5)
    wrapped = wrap_adapter(my_adapter, "cbf")
    wrapped = wrap_adapter(my_adapter, "lagrangian", soft_margin=0.25)
    original = wrap_adapter(my_adapter, "dc3s")   # no-op, returns adapter
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class _AdapterProxy:
    """Thin proxy that forwards any attribute not overridden to the wrapped adapter."""

    def __init__(self, adapter: Any) -> None:
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_adapter", adapter)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_adapter"), name)


class TubeMPCWrapper(_AdapterProxy):
    """Fixed-reliability-floor wrapper (Tube MPC analogue).

    Delegates all domain logic to the wrapped adapter but overrides
    ``build_uncertainty_set`` to use ``reliability_w = w_floor`` regardless
    of the actual OQE score.  The tube width is constant across all
    telemetry conditions.

    Args:
        adapter: Any domain adapter implementing the DomainAdapter interface.
        w_floor: Fixed reliability score used for all uncertainty computations.
            Default 0.5 — a neutral mid-range value that neither trusts
            observations fully nor inflates aggressively.
    """

    def __init__(self, adapter: Any, w_floor: float = 0.5) -> None:
        super().__init__(adapter)
        object.__setattr__(self, "_w_floor", float(w_floor))

    def build_uncertainty_set(
        self,
        state: Mapping[str, Any],
        reliability_w: float,
        quantile: float = 50.0,
        *,
        cfg: Mapping[str, Any],
        drift_flag: bool | None = None,
        prev_meta: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Build uncertainty set with fixed tube width, ignoring live OQE score."""
        w_floor = object.__getattribute__(self, "_w_floor")
        adapter = object.__getattribute__(self, "_adapter")
        return adapter.build_uncertainty_set(
            state,
            w_floor,  # fixed — does not adapt to telemetry degradation
            quantile,
            cfg=cfg,
            drift_flag=drift_flag,
            prev_meta=prev_meta,
        )

    @property
    def wrapper_name(self) -> str:
        w_floor = object.__getattribute__(self, "_w_floor")
        return f"tube_mpc_w{w_floor}"


class CBFWrapper(_AdapterProxy):
    """Control Barrier Function wrapper (observed-state-only safety).

    Sets ``reliability_w = 1.0`` so the uncertainty set is not inflated
    beyond the nominal conformal width.  This models a CBF that evaluates
    h(x_obs) >= 0 at each step without correcting for observation
    degradation.  Under degraded telemetry (x_obs != x_true) the barrier
    approves actions that violate the true safety constraint.
    """

    def build_uncertainty_set(
        self,
        state: Mapping[str, Any],
        reliability_w: float,
        quantile: float = 50.0,
        *,
        cfg: Mapping[str, Any],
        drift_flag: bool | None = None,
        prev_meta: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Build uncertainty set trusting observed state (reliability_w = 1.0)."""
        adapter = object.__getattribute__(self, "_adapter")
        return adapter.build_uncertainty_set(
            state,
            1.0,  # full trust in observed state — CBF evaluated on x_obs
            quantile,
            cfg=cfg,
            drift_flag=drift_flag,
            prev_meta=prev_meta,
        )

    @property
    def wrapper_name(self) -> str:
        return "cbf_observed_state"


class LagrangianWrapper(_AdapterProxy):
    """Lagrangian / Safe-RL analog wrapper (soft constraint enforcement).

    Models a deployment-time agent trained with a Lagrangian safety penalty.
    Uses observed-state uncertainty (reliability_w = 1.0) and relaxes the
    tightened action set boundaries by ``soft_margin``.  The combined effect
    is that:

    1. The uncertainty set is not inflated for telemetry quality.
    2. The repaired action set is softer than the hard projection in DC3S.

    Under degraded telemetry the soft penalty based on x_obs cannot prevent
    true-state constraint violations — the same OASG failure mode as CBF,
    compounded by the relaxed hard-boundary enforcement.

    Args:
        adapter: Any domain adapter implementing the DomainAdapter interface.
        soft_margin: Fraction by which numeric constraint boundaries in the
            tightened set are relaxed outward.  0.0 = hard DC3S-equivalent
            enforcement; 0.25 = 25% softening of each bound slack.
    """

    def __init__(self, adapter: Any, soft_margin: float = 0.25) -> None:
        super().__init__(adapter)
        object.__setattr__(self, "_soft_margin", float(max(0.0, min(1.0, soft_margin))))

    def build_uncertainty_set(
        self,
        state: Mapping[str, Any],
        reliability_w: float,
        quantile: float = 50.0,
        *,
        cfg: Mapping[str, Any],
        drift_flag: bool | None = None,
        prev_meta: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Build uncertainty set on observed state — no OQE adaptation."""
        adapter = object.__getattribute__(self, "_adapter")
        return adapter.build_uncertainty_set(
            state,
            1.0,  # Lagrangian penalty evaluated on observed state only
            quantile,
            cfg=cfg,
            drift_flag=drift_flag,
            prev_meta=prev_meta,
        )

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Tighten action set with soft boundary relaxation.

        Calls the wrapped adapter's tighten_action_set then relaxes each
        numeric value by soft_margin — representing the residual softness
        of a Lagrangian penalty that was trained to reduce but not hard-clip
        violations.
        """
        adapter = object.__getattribute__(self, "_adapter")
        soft_margin = object.__getattribute__(self, "_soft_margin")
        tightened = dict(adapter.tighten_action_set(uncertainty, constraints, cfg=cfg))

        # Widen numeric bounds by soft_margin fraction of the slack to constraint
        result: dict[str, Any] = {}
        for key, val in tightened.items():
            if isinstance(val, (int, float)) and key in constraints:
                try:
                    c_val = float(constraints[key])
                    t_val = float(val)
                    slack = abs(c_val - t_val)
                    # Move tightened bound toward (and possibly past) constraint bound
                    if t_val < c_val:
                        result[key] = t_val - slack * soft_margin
                    else:
                        result[key] = t_val + slack * soft_margin
                except (TypeError, ValueError):
                    result[key] = val
            else:
                result[key] = val
        return result

    @property
    def wrapper_name(self) -> str:
        soft_margin = object.__getattribute__(self, "_soft_margin")
        return f"lagrangian_sm{soft_margin}"


def wrap_adapter(adapter: Any, strategy: str, **kwargs: Any) -> Any:
    """Wrap a domain adapter with a named SOTA safety strategy.

    Args:
        adapter: Any domain adapter with the DomainAdapter interface.
        strategy: One of 'dc3s', 'tube_mpc', 'cbf', 'lagrangian'.
        **kwargs: Strategy-specific keyword arguments:
            tube_mpc: ``w_floor`` (float, default 0.5)
            lagrangian: ``soft_margin`` (float, default 0.25)

    Returns:
        Wrapped adapter instance, or the original adapter unchanged for 'dc3s'.

    Raises:
        ValueError: If ``strategy`` is not one of the four known strategies.
    """
    if strategy == "dc3s":
        return adapter
    if strategy == "tube_mpc":
        return TubeMPCWrapper(adapter, w_floor=float(kwargs.get("w_floor", 0.5)))
    if strategy == "cbf":
        return CBFWrapper(adapter)
    if strategy == "lagrangian":
        return LagrangianWrapper(adapter, soft_margin=float(kwargs.get("soft_margin", 0.25)))
    raise ValueError(
        f"Unknown SOTA strategy: {strategy!r}. "
        "Choose from: 'dc3s', 'tube_mpc', 'cbf', 'lagrangian'"
    )


STRATEGIES: list[str] = ["dc3s", "tube_mpc", "cbf", "lagrangian"]
"""Ordered list of all supported SOTA comparison strategies."""

STRATEGY_LABELS: dict[str, str] = {
    "dc3s":       "DC3S (ORIUS)",
    "tube_mpc":   "Tube MPC",
    "cbf":        "CBF (observed state)",
    "lagrangian": "Lagrangian Safe-RL",
}
"""Display labels for each strategy, used in tables and figures."""
