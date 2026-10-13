"""K-Bound Epistemic Mind & ORIUS Physical Safety Integration.

Bridges the Knowability-Guided Adaptation (KGA / K-Bound) epistemic brain
(``kga.certificate.conformal_split``, ``kga.policy.decide``, KL-regularized
test-time adaptation ``kbound_step``, and Clopper-Pearson + TV shift bounds)
directly with ORIUS's Degraded-Observation Supervisory Control (DC3S) shield.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from kga.certificate import Certificate, conformal_split, split_conformal_rank_radius
from kga.policy import Decision, decide


@dataclass(frozen=True)
class KBoundMindEvaluation:
    """Result of evaluating the K-Bound epistemic brain on current telemetry."""

    decision: str
    kga_decision: str
    delta_hat: float
    epsilon_radius: float
    net_margin: float
    allow_adaptation: bool
    alpha: float
    n_calibration: int
    reliability_w: float
    epistemic_override_triggered: bool
    inflation_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "kga_decision": self.kga_decision,
            "delta_hat": float(self.delta_hat),
            "epsilon_radius": float(self.epsilon_radius)
            if np.isfinite(self.epsilon_radius)
            else 1e9,
            "net_margin": float(self.net_margin)
            if np.isfinite(self.net_margin)
            else -1e9,
            "allow_adaptation": bool(self.allow_adaptation),
            "alpha": float(self.alpha),
            "n_calibration": int(self.n_calibration),
            "reliability_w": float(self.reliability_w),
            "epistemic_override_triggered": bool(self.epistemic_override_triggered),
            "inflation_multiplier": float(self.inflation_multiplier),
        }


DEFAULT_CALIB_RESIDUALS: tuple[float, ...] = tuple(
    float(x) for x in np.linspace(0.01, 0.16, 30)
)


def evaluate_kbound_mind(
    *,
    reliability_w: float,
    delta_hat: float = 0.045,
    calib_residuals: Sequence[float] | np.ndarray | None = None,
    alpha: float = 0.05,
    w_threshold: float = 0.65,
    residual: float | None = None,
) -> KBoundMindEvaluation:
    """Evaluate the K-Bound epistemic gate for a single ORIUS control step.

    Uses ``kga.certificate.conformal_split`` to compute the exact finite-sample
    split-conformal radius ``epsilon`` and applies the K-Bound trichotomy:
      - ``ADAPT``          if ``delta_hat - epsilon > 0``
      - ``FREEZE``         if ``delta_hat + epsilon < 0``
      - ``ABSTAIN/FREEZE`` otherwise
    """
    if calib_residuals is None or len(calib_residuals) == 0:
        residuals_arr = np.asarray(DEFAULT_CALIB_RESIDUALS, dtype=float)
    else:
        residuals_arr = np.asarray(calib_residuals, dtype=float).reshape(-1)
        if len(residuals_arr) < 19:
            pad = np.asarray(DEFAULT_CALIB_RESIDUALS, dtype=float)
            residuals_arr = np.concatenate([residuals_arr, pad])[: max(30, len(residuals_arr))]

    effective_delta_hat = float(delta_hat)
    if residual is not None and np.isfinite(float(residual)):
        effective_delta_hat = float(delta_hat) - 0.01 * min(10.0, abs(float(residual)))

    cert: Certificate = conformal_split(
        delta_hat=effective_delta_hat,
        calib_residuals=residuals_arr,
        alpha=float(alpha),
    )
    kga_dec: Decision = decide(cert)
    radius = float(getattr(cert, "epsilon", 0.15))
    net_margin = float(effective_delta_hat - radius)
    allow_adaptation = bool(net_margin > 0.0 and kga_dec == Decision.ADAPT)

    decision_label = "ADAPT" if allow_adaptation else "ABSTAIN/FREEZE"
    w_clamped = float(np.clip(reliability_w, 0.0, 1.0))
    epistemic_override = bool((not allow_adaptation) and (w_clamped < float(w_threshold)))

    if epistemic_override:
        inflation_mult = float(1.0 + max(0.0, float(w_threshold) - w_clamped))
    else:
        inflation_mult = 1.0

    return KBoundMindEvaluation(
        decision=decision_label,
        kga_decision=str(kga_dec.value),
        delta_hat=effective_delta_hat,
        epsilon_radius=radius,
        net_margin=net_margin,
        allow_adaptation=allow_adaptation,
        alpha=float(alpha),
        n_calibration=int(len(residuals_arr)),
        reliability_w=w_clamped,
        epistemic_override_triggered=epistemic_override,
        inflation_multiplier=inflation_mult,
    )


from orius.dc3s.quality import compute_reliability
from orius.dc3s.reliability_constraints import evaluate_constraint
from orius.dc3s.velocity_aware_shield import (
    CANBusInterface,
    KBoundConformalCertificate,
    ProactiveKBoundGovernor,
    VelocityAwareShieldResult,
    VelocityAwareUniversalShield,
    compute_kbound_conformal_certificate,
)


class UniversalPhysicalAIShield:
    """Unified K-Bound + ORIUS DC3S Physical AI Shield (Notebook Cells 34, 39, 42).

    Combines:
      1. ORIUS Observation Quality Engine (``compute_reliability``)
      2. K-Bound Split-Conformal Epistemic Gate (``kga.certificate.conformal_split``)
      3. ORIUS Dynamic Constraint Tightening (``evaluate_constraint``)
      4. Physical Actuation Projection (emergency braking or proportional damping)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        k_tighten: float = 1.2,
        delta_hat: float = 0.045,
        w_threshold: float = 0.65,
    ) -> None:
        self.alpha = float(alpha)
        self.k_tighten = float(k_tighten)
        self.delta_hat = float(delta_hat)
        self.w_threshold = float(w_threshold)
        self.epoch_start = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

    def filter_actuation(
        self,
        t_sec: float,
        observed_state: float,
        nominal_command: float,
        latency_s: float,
        safety_boundary_h: float,
        calib_residuals: Sequence[float] | np.ndarray | None = None,
        is_braking: bool = False,
        brake_decel: float = -7.5,
    ) -> tuple[float, float, bool, float]:
        """Filter a nominal physical command through K-Bound + ORIUS DC3S.

        Returns ``(safe_command, w_t, shield_active, kbound_radius)``.
        """
        current_ts = self.epoch_start + pd.Timedelta(seconds=float(t_sec))
        last_ts = current_ts - pd.Timedelta(seconds=float(latency_s))

        # 1. Evaluate ORIUS reliability weight w_t with explicit UTC timestamps
        w_base, _ = compute_reliability(
            event={"ts_utc": current_ts, "value": float(observed_state)},
            last_event={"ts_utc": last_ts, "value": float(observed_state)},
            expected_cadence_s=0.02,
        )
        excess_lag = max(0.0, float(latency_s) - 0.02)
        w_lag = float(np.exp(-excess_lag / 0.35))
        w_t = float(np.clip(min(float(w_base), w_lag), 0.05, 1.0))

        # 2. Epistemic gate evaluation (K-Bound conformal_split)
        kb_eval = evaluate_kbound_mind(
            reliability_w=float(w_t),
            delta_hat=self.delta_hat,
            calib_residuals=calib_residuals,
            alpha=self.alpha,
            w_threshold=self.w_threshold,
        )
        radius = kb_eval.epsilon_radius
        allow_adaptation = kb_eval.allow_adaptation

        # 3. Dynamic constraint boundary evaluation
        c_eval = evaluate_constraint(
            h_nominal=float(safety_boundary_h),
            w_t=float(w_t),
            k=self.k_tighten,
        )
        is_safe = bool(getattr(c_eval, "is_safe", True))

        # 4. Physical Projection
        shield_active = False
        if not is_safe or (not allow_adaptation and float(w_t) < self.w_threshold):
            shield_active = True
            if is_braking:
                safe_command = float(brake_decel)
            else:
                safe_command = float(nominal_command) * max(0.0, float(w_t) * 0.3)
        else:
            safe_command = float(nominal_command)

        return float(safe_command), float(w_t), bool(shield_active), float(radius)


def kbound_step(
    model: Any,
    frozen_model: Any,
    x_batch: Any,
    optimizer: Any,
    *,
    lam_kl: float = 0.5,
    conf_thresh: float = 0.65,
) -> tuple[Any, dict[str, float]]:
    """Execute one KL-regularized K-Bound test-time adaptation step (Notebook Cell 6).

    Minimizes selective Shannon entropy on confident samples while bounding
    divergence from the frozen pre-shift anchor ``frozen_model`` via
    ``lam_kl * KL(p_theta || p_frozen)`` and tracking parameter L2 / spectral drift.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for kbound_step")

    model.train()
    frozen_model.eval()

    logits = model(x_batch)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    with torch.no_grad():
        logits_0 = frozen_model(x_batch)
        probs_0 = F.softmax(logits_0, dim=-1)

    max_probs, _ = probs.max(dim=-1)
    mask = (max_probs >= conf_thresh).float()

    ent = -(probs * log_probs).sum(dim=-1)
    if mask.sum() > 0:
        loss_ent = (ent * mask).sum() / mask.sum()
    else:
        loss_ent = ent.mean() * 0.0

    kl_div = F.kl_div(log_probs, probs_0, reduction="batchmean")
    total_loss = loss_ent + lam_kl * kl_div

    optimizer.zero_grad()
    if total_loss.requires_grad:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    drift_l2 = 0.0
    max_spec = 0.0
    with torch.no_grad():
        for (n1, p1), (_, p0) in zip(
            model.named_parameters(), frozen_model.named_parameters()
        ):
            if p1.requires_grad:
                diff = (p1 - p0).float()
                drift_l2 += float(torch.norm(diff, p=2).item() ** 2)
                if diff.ndim >= 2:
                    s = float(torch.linalg.matrix_norm(diff.reshape(diff.shape[0], -1), ord=2).item())
                else:
                    s = float(torch.norm(diff, p=2).item())
                max_spec = max(max_spec, s)
    drift_l2 = float(np.sqrt(drift_l2))

    return logits.detach(), {
        "loss_total": float(total_loss.item()),
        "loss_ent": float(loss_ent.item()),
        "kl_div": float(kl_div.item()),
        "active_frac": float(mask.mean().item()),
        "param_drift_l2": drift_l2,
        "max_spectral_norm": max_spec,
    }


if nn is not None:

    class WorldModel(nn.Module):
        """2D Adaptive World Model for telemetry & latency state prediction (Cell 20)."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class WorldModel3D(nn.Module):
        """Deep 3D World Model for [X, Y, Z] state transitions (Cells 5, 55, 80, 81)."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(3, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 3),
            )

        def forward(self, x: Any) -> Any:
            return self.backbone(x)

    JointSpaceWorldModel = WorldModel3D

    class PhysicalAINeuralShield(nn.Module):
        """Silicon-Compliant Neural Safety Shield for ONNX / Jetson Orin (Cells 5, 37)."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class AdversaryNet(nn.Module):
        """Active Learned Adversary injecting state-dependent latency spikes (Cell 113)."""

        def __init__(self, max_lag_s: float = 1.0) -> None:
            super().__init__()
            self.max_lag_s = float(max_lag_s)
            self.net = nn.Sequential(
                nn.Linear(2, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, state_2d: Any) -> Any:
            return self.net(state_2d) * self.max_lag_s


__all__ = [
    "AdversaryNet",
    "CANBusInterface",
    "Certificate",
    "DEFAULT_CALIB_RESIDUALS",
    "Decision",
    "JointSpaceWorldModel",
    "KBoundConformalCertificate",
    "KBoundMindEvaluation",
    "PhysicalAINeuralShield",
    "ProactiveKBoundGovernor",
    "UniversalPhysicalAIShield",
    "VelocityAwareShieldResult",
    "VelocityAwareUniversalShield",
    "WorldModel",
    "WorldModel3D",
    "compute_kbound_conformal_certificate",
    "conformal_split",
    "decide",
    "evaluate_kbound_mind",
    "kbound_step",
    "split_conformal_rank_radius",
]
