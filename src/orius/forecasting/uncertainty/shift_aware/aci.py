from __future__ import annotations

from .state import AdaptiveQuantileState, ShiftAwareConfig


def update_adaptive_quantile(
    state: AdaptiveQuantileState,
    *,
    is_miss: bool,
    config: ShiftAwareConfig,
) -> AdaptiveQuantileState:
    mode = config.aci_mode
    if mode == "fixed":
        return state

    direction = 1.0 if is_miss else -1.0
    step = config.adaptation_step
    if mode == "aci_clipped":
        step = min(step, 0.05)

    prev_alpha = state.effective_alpha
    next_alpha = prev_alpha + direction * step
    next_alpha = min(1.0 - 1e-6, max(1e-6, next_alpha))

    target = max(config.target_alpha, 1e-6)
    quantile_scale = next_alpha / target
    quantile_scale = min(config.max_quantile, max(config.min_quantile, quantile_scale))

    state.effective_alpha = float(next_alpha)
    state.effective_quantile = float(quantile_scale)
    state.updates += 1
    state.instability = float(abs(next_alpha - prev_alpha))
    return state


def make_aci_state(config: ShiftAwareConfig) -> AdaptiveQuantileState:
    return AdaptiveQuantileState(
        mode=config.aci_mode,
        target_alpha=config.target_alpha,
        effective_alpha=config.target_alpha,
        effective_quantile=1.0,
        step_size=config.adaptation_step,
        updates=0,
    )
