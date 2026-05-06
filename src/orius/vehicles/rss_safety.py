"""RSS (Responsibility-Sensitive Safety) longitudinal safe-gap model.

Implements the one-dimensional longitudinal safe-following distance from:

    Shalev-Shwartz, S., Shammah, S., & Shashua, A. (2017).
    "On a Formal Model of Safe and Scalable Self-driving Cars."
    arXiv:1708.06374.  §3.1: Longitudinal safety.

The safe gap is the minimum inter-vehicle distance that guarantees no
rear-end collision if the lead vehicle brakes at its maximum capability:

    gap_safe = v_ego · t_resp
             + v_ego² / (2 · a_min_brake_ego)
             − v_lead² / (2 · a_max_brake_lead)

clamped to max(0, ...).

See also ISO 22179:2009 (ACC forward-collision mitigation) for
parameter calibration context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RssParameters",
    "rss_safe_gap",
    "rss_safe_gap_vec",
    "rss_violation",
    "rss_violation_vec",
]


@dataclass(frozen=True)
class RssParameters:
    """Tuneable RSS constants.

    Default values follow the *highway* profile from
    Shalev-Shwartz et al. 2017, Table 1.
    """

    t_resp: float = 0.75
    """Ego reaction time (seconds).  0.75 s is conservative for
    a software-controlled ADS; 1.5–2.0 s for human drivers."""

    a_min_brake_ego: float = 4.0
    """Ego minimum comfortable braking deceleration (m/s²).
    Represents the weakest braking the ego is *guaranteed* to achieve."""

    a_max_brake_lead: float = 6.0
    """Lead worst-case braking deceleration (m/s²).
    Represents the hardest braking the lead *could* perform."""


# ---------------------------------------------------------------------------
# Scalar API
# ---------------------------------------------------------------------------


def rss_safe_gap(
    v_ego: float,
    v_lead: float,
    t_resp: float = 0.75,
    a_min_brake_ego: float = 4.0,
    a_max_brake_lead: float = 6.0,
) -> float:
    """Compute the RSS longitudinal safe-following gap (metres).

    Parameters
    ----------
    v_ego : float
        Ego speed (m/s), ≥ 0.
    v_lead : float
        Lead vehicle speed (m/s), ≥ 0.
    t_resp, a_min_brake_ego, a_max_brake_lead :
        RSS parameters — see :class:`RssParameters`.

    Returns
    -------
    float
        Safe following distance (metres), ≥ 0.
    """
    reaction_dist = v_ego * t_resp
    ego_brake_dist = (v_ego**2) / (2.0 * a_min_brake_ego)
    lead_brake_dist = (v_lead**2) / (2.0 * a_max_brake_lead)
    return max(0.0, reaction_dist + ego_brake_dist - lead_brake_dist)


def rss_violation(
    gap_actual: float,
    v_ego: float,
    v_lead: float,
    t_resp: float = 0.75,
    a_min_brake_ego: float = 4.0,
    a_max_brake_lead: float = 6.0,
) -> bool:
    """Return ``True`` if the actual gap violates the RSS safe distance."""
    return gap_actual < rss_safe_gap(v_ego, v_lead, t_resp, a_min_brake_ego, a_max_brake_lead)


# ---------------------------------------------------------------------------
# Vectorised (numpy) API — for batch computation over parquet columns
# ---------------------------------------------------------------------------


def rss_safe_gap_vec(
    v_ego: np.ndarray,
    v_lead: np.ndarray,
    params: RssParameters | None = None,
) -> np.ndarray:
    """Vectorised RSS safe gap over arrays.

    Returns an array of the same length as *v_ego* / *v_lead*.
    """
    p = params or RssParameters()
    reaction = v_ego * p.t_resp
    ego_brake = np.square(v_ego) / (2.0 * p.a_min_brake_ego)
    lead_brake = np.square(v_lead) / (2.0 * p.a_max_brake_lead)
    return np.maximum(0.0, reaction + ego_brake - lead_brake)


def rss_violation_vec(
    gap_actual: np.ndarray,
    v_ego: np.ndarray,
    v_lead: np.ndarray,
    params: RssParameters | None = None,
) -> np.ndarray:
    """Vectorised RSS violation check.  Returns a bool array."""
    return gap_actual < rss_safe_gap_vec(v_ego, v_lead, params)
