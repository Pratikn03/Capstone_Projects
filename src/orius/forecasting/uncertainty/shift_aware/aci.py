from __future__ import annotations

from .state import AdaptiveQuantileState


def update_adaptive_quantile(state: AdaptiveQuantileState, *, miss: bool) -> AdaptiveQuantileState:
    if state.mode == "fixed":
        return state

    step = float(max(state.learning_rate, 0.0))
    target_alpha = float(state.base_alpha)
    eff = float(state.effective_alpha)
    direction = 1.0 if miss else -1.0

    if state.mode == "aci_basic":
        eff = eff + step * direction
    else:  # aci_clipped
        miss_scale = 1.0 + min(state.miss_streak, 5) * 0.2 if miss else 1.0
        eff = eff + step * direction * miss_scale

    eff = float(min(max(eff, state.alpha_min), state.alpha_max))
    state.effective_alpha = eff
    state.updates += 1
    state.miss_streak = state.miss_streak + 1 if miss else 0
    state.base_alpha = target_alpha
    return state
