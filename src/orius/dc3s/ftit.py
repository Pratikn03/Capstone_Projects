"""Fault-tolerant interval tracking (FTIT) state updates for DC3S."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


FTIT_FAULT_KEYS = ("dropout", "stale_sensor", "delay_jitter", "out_of_order", "spikes")


def _cfg_block(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(cfg or {})
    ftit = payload.get("ftit")
    if isinstance(ftit, Mapping):
        return dict(ftit)
    return payload


def _f(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _fault_state(adaptive_state: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = dict(adaptive_state or {})
    ftit_state = root.get("ftit")
    if not isinstance(ftit_state, Mapping):
        ftit_state = {}
    return root, dict(ftit_state)


def _rate_map(raw: Mapping[str, Any] | None) -> dict[str, float]:
    payload = dict(raw or {})
    return {key: _f(payload.get(key), 0.0) for key in FTIT_FAULT_KEYS}


def _alpha_map(cfg: Mapping[str, Any]) -> dict[str, float]:
    return {
        "dropout": _f(cfg.get("alpha_dropout"), 1.0),
        "stale_sensor": _f(cfg.get("alpha_stale_sensor"), 1.0),
        "delay_jitter": _f(cfg.get("alpha_delay_jitter"), 1.0),
        "out_of_order": _f(cfg.get("alpha_out_of_order"), 1.0),
        "spikes": _f(cfg.get("alpha_spikes"), 1.0),
    }


def _bool_flag(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) != 0.0 else 0.0
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "on"}:
            return 1.0
        if low in {"0", "false", "no", "n", "off"}:
            return 0.0
    return 0.0


def preview_fault_state(
    *,
    adaptive_state: Mapping[str, Any] | None,
    fault_flags: Mapping[str, bool],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Preview the next FTIT rolling-fault state without mutating the caller."""
    ftit_cfg = _cfg_block(cfg)
    _root, state = _fault_state(adaptive_state)
    decay = float(np.clip(_f(ftit_cfg.get("decay"), 0.98), 0.0, 1.0))
    n_prev = _f(state.get("n"), 0.0)
    s_prev = _rate_map(state.get("s"))
    n_next = decay * n_prev + 1.0
    s_next: dict[str, float] = {}
    p_next: dict[str, float] = {}
    for key in FTIT_FAULT_KEYS:
        s_val = decay * s_prev[key] + _bool_flag(fault_flags.get(key))
        s_next[key] = float(s_val)
        p_next[key] = float(s_val / max(n_next, 1e-12))

    w_t = 1.0
    for key, alpha in _alpha_map(ftit_cfg).items():
        w_t *= float(np.clip(1.0 - p_next[key], 1e-12, 1.0)) ** float(alpha)
    w_t = float(np.clip(w_t, 0.0, 1.0))

    return {
        "n": float(n_next),
        "s": s_next,
        "p": p_next,
        "w_t": w_t,
    }


def update(
    *,
    adaptive_state: Mapping[str, Any] | None,
    fault_flags: Mapping[str, bool],
    constraints: Mapping[str, Any],
    cfg: Mapping[str, Any],
    stale_tracker: Mapping[str, Any] | None = None,
    sigma2_observation: float | None = None,
) -> dict[str, Any]:
    """Update FTIT state, reliability-derived tube width, and SOC tube bounds."""
    ftit_cfg = _cfg_block(cfg)
    root, state = _fault_state(adaptive_state)
    preview = preview_fault_state(adaptive_state=adaptive_state, fault_flags=fault_flags, cfg=ftit_cfg)

    sigma2_init = _f(ftit_cfg.get("sigma2_init"), 1.0)
    sigma2_decay = float(np.clip(_f(ftit_cfg.get("sigma2_decay"), 0.95), 0.0, 1.0))
    sigma2_floor = max(_f(ftit_cfg.get("sigma2_floor"), 1.0e-6), 1.0e-12)
    sigma2_prev = max(_f(state.get("sigma2"), sigma2_init), sigma2_floor)
    if sigma2_observation is None:
        sigma2_next = sigma2_prev
    else:
        sigma2_next = max(
            sigma2_floor,
            sigma2_decay * sigma2_prev + (1.0 - sigma2_decay) * max(float(sigma2_observation), 0.0),
        )

    max_power = max(_f(constraints.get("max_power_mw"), 0.0), 0.0)
    capacity = max(_f(constraints.get("capacity_mwh"), 0.0), 0.0)
    gamma_min = max(_f(ftit_cfg.get("gamma_min_mw"), 0.0), 0.0)
    gamma_max_cfg = ftit_cfg.get("gamma_max_mw")
    gamma_max = max(_f(gamma_max_cfg, 0.25 * max_power), gamma_min)
    gamma_power = max(_f(ftit_cfg.get("gamma_power"), 1.0), 0.0)
    gamma_mw = gamma_min + ((1.0 - preview["w_t"]) ** gamma_power) * max(0.0, gamma_max - gamma_min)

    decay_e = float(np.clip(_f(ftit_cfg.get("decay_e"), 0.95), 0.0, 1.0))
    dt_hours = max(_f(ftit_cfg.get("dt_hours"), _f(constraints.get("time_step_hours"), 1.0)), 0.0)
    e_prev = max(_f(state.get("e_t_mwh"), _f(ftit_cfg.get("e_min_mwh"), 0.0)), 0.0)
    e_min = max(_f(ftit_cfg.get("e_min_mwh"), 0.0), 0.0)
    e_max_cfg = ftit_cfg.get("e_max_mwh")
    e_max = max(_f(e_max_cfg, 0.5 * capacity), e_min)
    e_next = float(np.clip(decay_e * e_prev + dt_hours * gamma_mw, e_min, e_max))

    soc_min = _f(constraints.get("min_soc_mwh"), 0.0)
    soc_max = _f(constraints.get("max_soc_mwh"), capacity)
    soc_lower = soc_min + e_next
    soc_upper = soc_max - e_next
    if soc_lower > soc_upper:
        midpoint = 0.5 * (soc_lower + soc_upper)
        soc_lower = midpoint
        soc_upper = midpoint

    next_ftit = {
        "n": float(preview["n"]),
        "s": dict(preview["s"]),
        "p": dict(preview["p"]),
        "sigma2": float(sigma2_next),
        "e_t_mwh": float(e_next),
        "gamma_mw": float(gamma_mw),
        "soc_tube_lower_mwh": float(soc_lower),
        "soc_tube_upper_mwh": float(soc_upper),
        "stale_tracker": dict(stale_tracker or state.get("stale_tracker") or {}),
    }
    root["ftit"] = next_ftit

    return {
        "adaptive_state": root,
        "w_t": float(preview["w_t"]),
        "p_drop": float(preview["p"]["dropout"]),
        "p_stale": float(preview["p"]["stale_sensor"]),
        "p_delay": float(preview["p"]["delay_jitter"]),
        "p_ooo": float(preview["p"]["out_of_order"]),
        "p_spike": float(preview["p"]["spikes"]),
        "gamma_mw": float(gamma_mw),
        "e_t_mwh": float(e_next),
        "soc_tube_lower_mwh": float(soc_lower),
        "soc_tube_upper_mwh": float(soc_upper),
        "sigma2": float(sigma2_next),
    }
