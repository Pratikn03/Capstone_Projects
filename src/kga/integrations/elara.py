"""Claim-safe ELARA-U candidate routing behind the KGA decision layer.

ELARA-U constructs a validation-fitted detector candidate. KGA then certifies
whether that candidate should replace the validation-selected frozen expert.
The integration is optional: importing :mod:`kga` does not import ELARA-U.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from kga.benefit import FrozenLinearBenefitEstimator
from kga.kga import KGA
from kga.policy import Decision


class EvaluationMode(str, Enum):
    """Information boundary used to construct the KGA certificate."""

    RETROSPECTIVE_AUDIT = "retrospective_audit"
    TARGET_LABEL_LIGHT = "target_label_light"
    LABEL_FREE = "label_free"


@dataclass
class ELARAKGAResult:
    """One frozen ELARA candidate and the KGA decision over it."""

    mode: EvaluationMode
    decision: Decision
    deployed_action: str
    router_action: str
    frozen_expert: int
    frozen_scores: np.ndarray
    candidate_scores: np.ndarray
    deployed_scores: np.ndarray
    certificate: dict[str, float | int | str | None]
    evidence: dict[str, float]
    labels_used_for_decision: int
    claim_tier: str
    claim_eligible: bool
    claim_reasons: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe decision record without per-example scores."""

        return {
            "mode": self.mode.value,
            "decision": self.decision.value,
            "deployed_action": self.deployed_action,
            "router_action": self.router_action,
            "frozen_expert": self.frozen_expert,
            "certificate": self.certificate,
            "evidence": self.evidence,
            "labels_used_for_decision": self.labels_used_for_decision,
            "claim_tier": self.claim_tier,
            "claim_eligible": self.claim_eligible,
            "claim_reasons": list(self.claim_reasons),
        }


def _as_scores(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D score array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_binary_labels(value: np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value).ravel()
    if arr.size != n:
        raise ValueError(f"{name} length {arr.size} does not match score rows {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if not set(np.unique(arr)).issubset({0, 1}):
        raise ValueError(f"{name} must contain binary labels 0/1")
    return arr.astype(int)


def _load_router_api():
    try:
        from uais.elara_u.router import RouterPolicy, reliability_features, route
    except ImportError:
        try:
            from src.uais.elara_u.router import RouterPolicy, reliability_features, route
        except ImportError as exc:  # pragma: no cover - exercised in standalone installs
            raise ImportError(
                "The ELARA integration requires the optional `uais.elara_u` package. "
                "Install the full repository package to use ELARAKGAGuard."
            ) from exc
    return RouterPolicy, reliability_features, route


def _brier_benefits(y: np.ndarray, frozen: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.asarray((frozen - y) ** 2 - (candidate - y) ** 2, dtype=float)


def _certificate_record(cert) -> dict[str, float | int | str | None]:
    def finite_or_none(value):
        return float(value) if np.isfinite(value) else None

    return {
        "delta_hat": finite_or_none(cert.delta_hat),
        "epsilon": finite_or_none(cert.epsilon),
        "lower": finite_or_none(cert.lower),
        "upper": finite_or_none(cert.upper),
        "method": str(cert.method),
        "alpha": float(cert.alpha),
        "n": int(cert.n),
    }


@dataclass
class ELARAKGAGuard:
    """Use KGA to guard deployment of an ELARA-U routed candidate."""

    alpha: float = 0.10
    router_action: str = "hybrid"
    probe_seed: int = 20260615
    policy: object | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if self.router_action not in {"select", "fuse", "hybrid"}:
            raise ValueError("router_action must be select, fuse, or hybrid")

    def decide(
        self,
        *,
        s_val: np.ndarray,
        y_val: np.ndarray,
        s_test: np.ndarray,
        mode: EvaluationMode | str,
        y_test: np.ndarray | None = None,
        probe_indices: np.ndarray | None = None,
        estimator: FrozenLinearBenefitEstimator | None = None,
        protocol_sha256: str | None = None,
    ) -> ELARAKGAResult:
        """Build the ELARA candidate and apply KGA under a declared mode."""

        try:
            mode = EvaluationMode(mode)
        except ValueError as exc:
            raise ValueError(f"unknown evaluation mode: {mode!r}") from exc

        if mode is EvaluationMode.LABEL_FREE and protocol_sha256 is None:
            raise ValueError("label_free mode requires an externally supplied protocol_sha256")

        s_val = _as_scores(s_val, "s_val")
        s_test = _as_scores(s_test, "s_test")
        if s_val.shape[1] != s_test.shape[1]:
            raise ValueError("s_val and s_test must have the same number of experts")
        y_val = _as_binary_labels(y_val, s_val.shape[0], "y_val")

        RouterPolicy, reliability_features, route = _load_router_api()
        policy = self.policy if self.policy is not None else RouterPolicy()
        reliability = reliability_features(s_val, y_val)
        val_auc = np.asarray(reliability["val_auc"], dtype=float)
        frozen_expert = int(np.nanargmax(val_auc))
        frozen = np.asarray(s_test[:, frozen_expert], dtype=float)
        candidate, router_action = route(s_val, y_val, s_test, policy, action=self.router_action)
        candidate = np.asarray(candidate, dtype=float).ravel()
        if candidate.shape != frozen.shape or not np.all(np.isfinite(candidate)):
            raise ValueError("ELARA candidate scores must be finite and match test rows")

        kga = KGA(alpha=self.alpha)
        evidence = kga.evidence(
            s_val,
            s_test,
            extra={
                "best_val_auc": float(reliability["best_auc"]),
                "val_gap": float(reliability["gap"]),
                "val_disagreement": float(reliability["disagreement"]),
                "n_experts": int(s_val.shape[1]),
            },
        )
        feature_map = {
            "ks_mean": float(evidence.ks_mean),
            "ks_max": float(evidence.ks_max),
            "disagree": float(evidence.disagree),
            "entropy_shift": float(evidence.entropy_shift),
            "conf_shift": float(evidence.conf_shift),
            "ess_frac": float(evidence.ess_frac),
            "best_val_auc": float(reliability["best_auc"]),
            "val_gap": float(reliability["gap"]),
            "val_disagreement": float(reliability["disagreement"]),
            "n_experts": float(s_val.shape[1]),
        }

        claim_reasons: tuple[str, ...]
        if mode is EvaluationMode.LABEL_FREE:
            if y_test is not None:
                raise ValueError("label_free mode must not receive y_test")
            if probe_indices is not None:
                raise ValueError("label_free mode must not receive probe_indices")
            if estimator is None:
                raise ValueError("label_free mode requires a frozen estimator")
            if protocol_sha256 is None:  # narrowed here for static analysis; validated before router loading above
                raise ValueError("label_free mode requires an externally supplied protocol_sha256")
            cert = kga.certify_evidence(
                estimator,
                protocol_sha256=protocol_sha256,
                features=feature_map,
                evidence_schema_version=estimator.evidence_schema_version,
            )
            labels_used = 0
            claim_tier = "label_free_candidate"
            claim_reasons = ("requires_heldout_aggregate_promotion_check",)
        elif mode is EvaluationMode.TARGET_LABEL_LIGHT:
            if y_test is None:
                raise ValueError("target_label_light mode requires y_test for the fixed probe")
            labels = _as_binary_labels(y_test, s_test.shape[0], "y_test")
            if probe_indices is None:
                raise ValueError("target_label_light mode requires fixed probe_indices")
            idx = np.asarray(probe_indices, dtype=int).ravel()
            if idx.size == 0 or np.unique(idx).size != idx.size:
                raise ValueError("probe_indices must be non-empty and unique")
            if np.any(idx < 0) or np.any(idx >= labels.size):
                raise ValueError("probe_indices are out of range")
            benefits = _brier_benefits(labels[idx], frozen[idx], candidate[idx])
            cert = kga.certify_probe(benefits, k=None, benefit_range=2.0)
            labels_used = int(idx.size)
            claim_tier = "target_label_light_candidate"
            claim_reasons = ("requires_heldout_aggregate_promotion_check",)
        else:
            if y_test is None:
                raise ValueError("retrospective_audit mode requires y_test")
            if probe_indices is not None:
                raise ValueError("retrospective_audit mode does not accept probe_indices")
            labels = _as_binary_labels(y_test, s_test.shape[0], "y_test")
            benefits = _brier_benefits(labels, frozen, candidate)
            cert = kga.certify(scores=benefits, benefit_range=2.0)
            labels_used = int(labels.size)
            claim_tier = "retrospective_only"
            claim_reasons = ("uses_full_target_labels_for_decision", "not_deployment_eligible")

        decision = kga.decide(cert)
        deploy_candidate = decision is Decision.ADAPT
        deployed_scores = candidate if deploy_candidate else frozen
        return ELARAKGAResult(
            mode=mode,
            decision=decision,
            deployed_action=(
                "adapt" if deploy_candidate else "retain_frozen" if decision is Decision.ABSTAIN else "freeze"
            ),
            router_action=str(router_action),
            frozen_expert=frozen_expert,
            frozen_scores=frozen,
            candidate_scores=candidate,
            deployed_scores=deployed_scores,
            certificate=_certificate_record(cert),
            evidence=feature_map,
            labels_used_for_decision=labels_used,
            claim_tier=claim_tier,
            claim_eligible=False,
            claim_reasons=claim_reasons,
        )


def _binary_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    from scipy.stats import rankdata

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate_result(result: ELARAKGAResult, y_test: np.ndarray) -> dict[str, float | bool]:
    """Evaluate a frozen decision; labels cannot alter the stored decision."""

    labels = _as_binary_labels(y_test, result.frozen_scores.size, "y_test")
    auc_frozen = _binary_auroc(labels, result.frozen_scores)
    auc_candidate = _binary_auroc(labels, result.candidate_scores)
    auc_kga = _binary_auroc(labels, result.deployed_scores)
    oracle = max(auc_frozen, auc_candidate)
    benefit = float(np.mean(_brier_benefits(labels, result.frozen_scores, result.candidate_scores)))
    false_adapt = bool(result.decision is Decision.ADAPT and benefit <= 0.0)
    return {
        "auroc_frozen": auc_frozen,
        "auroc_candidate": auc_candidate,
        "auroc_kga": auc_kga,
        "auroc_oracle": oracle,
        "regret_frozen": oracle - auc_frozen,
        "regret_candidate": oracle - auc_candidate,
        "regret_kga": oracle - auc_kga,
        "brier_benefit": benefit,
        "harmful_candidate": benefit <= 0.0,
        "false_adapt": false_adapt,
        "covered": result.decision is not Decision.ABSTAIN,
    }
