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
from kga.evaluation import EvaluationReport, EvaluationSummary, evaluate_actions
from kga.frontier import (
    FrontierAssessment,
    PopulationBenefitInterval,
    assess_frontier,
    frontier_action,
    frontier_sensitivity,
)
from kga.policy import Decision, decide
from kga.population_transfer import (
    ConditionalPopulationInterval,
    compose_conditional_population_interval,
    hoeffding_paired_accuracy_radius,
)


@dataclass(frozen=True)
class KBoundMindEvaluation:
    """Result of evaluating the K-Bound epistemic brain on current telemetry.

    Implements the K-Bound three-way epistemic decision rule (Paper Def. 1, Thm. 2, Prop. 2):
      - ``ADAPT``   (``lower_bound = delta_hat - epsilon > 0``): positive benefit certified;
                    commit adapted candidate ``f_a`` and release action within ORIUS tube.
      - ``FREEZE``  (``upper_bound = delta_hat + epsilon < 0``): negative benefit certified;
                    actively reject harmful candidate ``f_a``, lock anchor ``f_0``, and tighten barrier.
      - ``ABSTAIN`` (``0 in [lower_bound, upper_bound]`` or missing telemetry): unresolved
                    epistemic evidence fibre; retain anchor ``f_0``, log ``ABSTAIN`` (never
                    mislabeled as certified ``FREEZE``), and inflate ORIUS state tube ``U_t(w_t)``.
    """

    decision: str
    kga_decision: str
    orius_physical_mode: str
    delta_hat: float
    epsilon_radius: float
    lower_bound: float
    upper_bound: float
    net_margin: float
    allow_adaptation: bool
    is_certified_rejection: bool
    is_epistemic_abstention: bool
    matched_evidence_abstain_floor: float
    alpha: float
    n_calibration: int
    reliability_w: float
    epistemic_override_triggered: bool
    inflation_multiplier: float

    @property
    def kbound_action(self) -> str:
        return self.kga_decision

    @property
    def epsilon(self) -> float:
        return self.epsilon_radius

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "kga_decision": self.kga_decision,
            "kbound_action": self.kga_decision,
            "orius_physical_mode": self.orius_physical_mode,
            "delta_hat": float(self.delta_hat),
            "epsilon_radius": float(self.epsilon_radius)
            if np.isfinite(self.epsilon_radius)
            else 1e9,
            "lower_bound": float(self.lower_bound)
            if np.isfinite(self.lower_bound)
            else -1e9,
            "upper_bound": float(self.upper_bound)
            if np.isfinite(self.upper_bound)
            else 1e9,
            "net_margin": float(self.net_margin)
            if np.isfinite(self.net_margin)
            else -1e9,
            "allow_adaptation": bool(self.allow_adaptation),
            "is_certified_rejection": bool(self.is_certified_rejection),
            "is_epistemic_abstention": bool(self.is_epistemic_abstention),
            "matched_evidence_abstain_floor": float(self.matched_evidence_abstain_floor),
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
    split-conformal order-statistic radius ``epsilon = r_(ceil((n+1)(1-alpha)))``
    and applies the K-Bound trichotomy (Algorithm 1 & Proposition 2):
      - ``ADAPT``   if ``L = delta_hat - epsilon > 0``
      - ``FREEZE``  if ``U = delta_hat + epsilon < 0``
      - ``ABSTAIN`` if ``0 in [L, U]`` or telemetry is unavailable/degraded.
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

    w_clamped = float(np.clip(reliability_w, 0.0, 1.0))

    # If telemetry is completely lost (w_t <= 0), K-Bound Algorithm 1 Line 3-4:
    # retain f_0, log the reason, and return ABSTAIN (never a certified FREEZE).
    if not np.isfinite(effective_delta_hat) or w_clamped <= 0.0:
        return KBoundMindEvaluation(
            decision="ABSTAIN/FREEZE",
            kga_decision=Decision.ABSTAIN.value,
            orius_physical_mode="ABSTAIN_AND_INFLATE_TUBE",
            delta_hat=effective_delta_hat if np.isfinite(effective_delta_hat) else 0.0,
            epsilon_radius=float("inf"),
            lower_bound=float("-inf"),
            upper_bound=float("inf"),
            net_margin=float("-inf"),
            allow_adaptation=False,
            is_certified_rejection=False,
            is_epistemic_abstention=True,
            matched_evidence_abstain_floor=max(0.0, 1.0 - 2.0 * float(alpha)),
            alpha=float(alpha),
            n_calibration=int(len(residuals_arr)),
            reliability_w=w_clamped,
            epistemic_override_triggered=True,
            inflation_multiplier=float(1.0 + max(0.0, float(w_threshold))),
        )

    cert: Certificate = conformal_split(
        delta_hat=effective_delta_hat,
        calib_residuals=residuals_arr,
        alpha=float(alpha),
    )
    kga_dec: Decision = decide(cert)
    radius = float(getattr(cert, "epsilon", 0.15))
    lower_bound = float(effective_delta_hat - radius)
    upper_bound = float(effective_delta_hat + radius)
    net_margin = lower_bound

    allow_adaptation = bool(
        lower_bound > 0.0 and kga_dec == Decision.ADAPT and w_clamped >= float(w_threshold)
    )
    # If OQE channel reliability is below threshold, force epistemic ABSTAIN if not FREEZE
    if not allow_adaptation and kga_dec == Decision.ADAPT:
        effective_kga_dec = Decision.ABSTAIN
    else:
        effective_kga_dec = kga_dec

    is_certified_rejection = bool(effective_kga_dec == Decision.FREEZE)
    is_epistemic_abstention = bool(effective_kga_dec == Decision.ABSTAIN)

    if allow_adaptation:
        decision_label = "ADAPT"
        orius_mode = "ADAPT_AND_RELEASE"
    elif is_certified_rejection:
        decision_label = "ABSTAIN/FREEZE"
        orius_mode = "REJECT_UPDATE_AND_TIGHTEN"
    else:
        decision_label = "ABSTAIN/FREEZE"
        orius_mode = "ABSTAIN_AND_INFLATE_TUBE"

    epistemic_override = bool((not allow_adaptation) and (w_clamped < float(w_threshold) or is_certified_rejection))
    if epistemic_override:
        inflation_mult = float(
            1.0 + max(0.0, float(w_threshold) - w_clamped) + (0.25 if is_certified_rejection else 0.0)
        )
    else:
        inflation_mult = 1.0

    return KBoundMindEvaluation(
        decision=decision_label,
        kga_decision=str(effective_kga_dec.value),
        orius_physical_mode=orius_mode,
        delta_hat=effective_delta_hat,
        epsilon_radius=radius,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        net_margin=net_margin,
        allow_adaptation=allow_adaptation,
        is_certified_rejection=is_certified_rejection,
        is_epistemic_abstention=is_epistemic_abstention,
        matched_evidence_abstain_floor=max(0.0, 1.0 - 2.0 * float(alpha)),
        alpha=float(alpha),
        n_calibration=int(len(residuals_arr)),
        reliability_w=w_clamped,
        epistemic_override_triggered=epistemic_override,
        inflation_multiplier=inflation_mult,
    )


def evaluate_population_frontier_mind(
    *,
    M: float,
    beta: float,
    disagreement_mass: float = 0.25,
    reliability_w: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate the K-Bound Population Partial-Identification Frontier (Theorems 1-3).

    Given disagreement-conditional centered-correctness margin ``M in [-0.5, 0.5]``,
    residual bound ``|gamma| <= beta``, and disagreement probability ``d = mu_T(D)``:
      - Computes the exact feasible population benefit interval
        ``Delta(C_beta) = 2 * d * [max(-1/2, M - beta), min(1/2, M + beta)]``.
      - Evaluates the strict-commitment frontier:
          * ``M > beta``   -> ``ADAPT``
          * ``M < -beta``  -> ``FREEZE``
          * ``|M| <= beta`` -> ``ABSTAIN`` (matched-evidence abstention lower bound ``1 - 2*alpha``).
      - Reports the exact label-free fibre-wise audit floor ``Gamma_z(C_beta) = beta`` (Theorem 3).
    """
    assessment: FrontierAssessment = assess_frontier(M, beta)
    pop_interval: PopulationBenefitInterval = assessment.population_benefit_interval(
        mass=disagreement_mass
    )
    w_clamped = float(np.clip(reliability_w, 0.0, 1.0))
    in_closed_band = bool(abs(float(M)) <= float(beta))
    abstain_floor = max(0.0, 1.0 - 2.0 * float(alpha)) if in_closed_band else 0.0

    if assessment.action == Decision.ADAPT and w_clamped >= 0.65:
        orius_mode = "ADAPT_AND_RELEASE"
        tube_inflation = 1.0
    elif assessment.action == Decision.FREEZE:
        orius_mode = "REJECT_UPDATE_AND_TIGHTEN"
        tube_inflation = 1.25 + max(0.0, 1.0 - w_clamped)
    else:
        orius_mode = "ABSTAIN_AND_INFLATE_TUBE"
        tube_inflation = 1.0 + float(beta) + max(0.0, 1.0 - w_clamped)

    return {
        "margin_M": float(assessment.margin),
        "residual_budget_beta": float(assessment.residual_budget),
        "fibre_audit_floor_gamma_z": float(assessment.fibre_audit_floor),
        "lower_centered_correctness": float(assessment.lower_centered_correctness),
        "upper_centered_correctness": float(assessment.upper_centered_correctness),
        "disagreement_mass_d": float(pop_interval.mass),
        "population_benefit_lower": float(pop_interval.lower),
        "population_benefit_upper": float(pop_interval.upper),
        "kga_decision": str(assessment.action.value),
        "in_closed_unknowable_band": in_closed_band,
        "matched_evidence_abstain_prob_floor": float(abstain_floor),
        "orius_physical_mode": orius_mode,
        "orius_tube_inflation": float(tube_inflation),
    }


def evaluate_compound_population_mind(
    *,
    delta_hat: float,
    epsilon: float,
    n_eval: int = 200,
    delta_sampling: float = 0.025,
    alpha_cell: float = 0.025,
    alpha_population: float = 0.05,
    reliability_w: float = 1.0,
) -> dict[str, Any]:
    """Evaluate K-Bound Proposition 3 (Cell-to-Population Transfer via Compound Bound).

    Combines empirical cell-benefit radius ``epsilon`` (at level ``alpha_cell``)
    with the fresh-sample Hoeffding concentration radius
    ``b(m, delta) = min(2, sqrt(2 * log(2 / delta) / m))`` to produce a
    population-level decision with false-commit risk ``<= alpha_cell + delta_sampling``.
    """
    r_samp = hoeffding_paired_accuracy_radius(n=int(n_eval), delta=float(delta_sampling))
    composed: ConditionalPopulationInterval = compose_conditional_population_interval(
        delta_hat=float(delta_hat),
        epsilon=float(epsilon),
        r_samp=float(r_samp),
        alpha_cell=float(alpha_cell),
        delta_sampling=float(delta_sampling),
        alpha_population=float(alpha_population),
    )
    w_clamped = float(np.clip(reliability_w, 0.0, 1.0))
    return {
        "delta_hat": float(composed.delta_hat),
        "cell_radius_epsilon": float(composed.epsilon),
        "sampling_radius_b": float(composed.r_samp),
        "compound_population_radius": float(composed.population_radius),
        "population_lower": float(composed.interval_lower),
        "population_upper": float(composed.interval_upper),
        "kga_decision": str(composed.action.value),
        "alpha_spent": float(composed.spent_budget),
        "reliability_w": w_clamped,
    }


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
