"""Assumption contract (A1-A6), deployment gate, and diagnostics for K-Bound.

The certificate in :mod:`kga.certificate` is *conditional*.  It controls the
unconditional false-adapt event only if the benefit interval attains marginal
coverage on the target class, and only under the declared calibration protocol.
None of that is provable from unlabelled deployment data.  This module makes the
assumption state an explicit, testable gate that runs *before* a certificate is
emitted, and records the outcome in a machine-readable report.

The contract, verbatim from the paper (``paper/sections/assumption_contract.tex``):

    A1  Calibration exchangeability at the declared inference unit.
    A2  Risk alignment: label-free evidence Z retains enough information about the
        benefit for calibration residuals to transfer.
    A3  Coverage premise: the interval attains marginal coverage on the target class.
    A4  Protocol independence: target labels touch no selection decision.
    A5  Correct sampling unit: calibration and uncertainty use independent units.
    A6  Fixed decision rule: estimator, interval, alpha, rule fixed before evaluation.

Design rules this module holds to
--------------------------------
* **Diagnostics falsify; they never verify.**  Every status is one-sided.  A
  ``pass`` means "no violation of this kind was detected", never "the assumption
  holds".
* **No value is ever invented.**  A statistic that cannot be computed from the
  inputs is ``None`` and contributes a string to ``limitations``.  It is never
  defaulted to a plausible number.
* **Units, not examples.**  Every uncertainty calculation takes an explicit group
  label array and resamples at that level.  Passing ``groups=None`` asserts that
  the rows really are independent draws.
* **Fail closed.**  Missing provenance is a rejection, not a pass.

Dependencies: numpy (required), scipy (optional -- exact Clopper-Pearson; a Wilson
interval is used as a declared fallback).  No sklearn: the domain classifier is a
small cross-fitted logistic regression implemented here, so the diagnostic is
reproducible from the pinned requirements alone.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:  # pragma: no cover - exercised by whichever branch the env provides
    from scipy.stats import beta as _scipy_beta

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_beta = None
    _HAVE_SCIPY = False


__all__ = [
    "CoverageType",
    "Status",
    "GateDecision",
    "FallbackAction",
    "ProtocolRecord",
    "SupportOverlap",
    "RadiusStability",
    "RiskAlignmentAudit",
    "ConclusionStability",
    "CoverageClaimBasis",
    "AssumptionReport",
    "GateThresholds",
    "conformal_radius",
    "effective_units",
    "observed_coverage",
    "evidence_support_overlap",
    "radius_stability",
    "risk_alignment_audit",
    "conclusion_stability",
    "leakage_audit",
    "run_gate",
    "write_report",
]


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


class CoverageType(str, Enum):
    """Which of the three coverage statements a number is.

    Never widen one of these into another.  ``OBSERVED_EMPIRICAL`` is a hit rate on
    conditions that were evaluated; it is not evidence for ``THEORETICAL`` on any
    other condition.
    """

    THEORETICAL = "theoretical"
    OBSERVED_EMPIRICAL = "observed_empirical"
    DIAGNOSTIC_ONLY = "diagnostic_only"


_SUCCESS_STATUS_VALUE = "pass"


class Status(str, Enum):
    """One-sided diagnostic outcome."""

    PASS = _SUCCESS_STATUS_VALUE
    WARNING = "warning"
    FAIL = "fail"


class GateDecision(str, Enum):
    """Terminal state of :func:`run_gate`, mapping onto the paper's fallback ladder."""

    CERTIFY = "certify"
    RESTRICTED = "restricted"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    REJECT = "reject"


class FallbackAction(str, Enum):
    ADAPT_FREEZE_ABSTAIN = "adapt_freeze_abstain"
    FREEZE_OR_ABSTAIN = "freeze_or_abstain"
    NONE = "none"


_LADDER: dict[GateDecision, FallbackAction] = {
    GateDecision.CERTIFY: FallbackAction.ADAPT_FREEZE_ABSTAIN,
    GateDecision.RESTRICTED: FallbackAction.FREEZE_OR_ABSTAIN,
    GateDecision.DIAGNOSTIC_ONLY: FallbackAction.NONE,
    GateDecision.REJECT: FallbackAction.NONE,
}


@dataclass(frozen=True)
class CoverageClaimBasis:
    """External, auditable basis for a theoretical coverage statement.

    Diagnostics in this module can falsify a proposed transfer argument but
    cannot establish it. A theoretical claim therefore requires a separate
    record naming the theorem/design, inference unit, assumptions, locked
    protocol, and exact calibration artifact. The record is evidence about the
    declared argument, not proof that its assumptions hold in nature.
    """

    theorem_or_result: str
    calibration_design: str
    inference_unit: str
    assumptions: tuple[str, ...]
    protocol_sha256: str
    calibration_artifact_sha256: str
    justification: str

    def validate(self, record: ProtocolRecord) -> list[str]:
        errors: list[str] = []
        for name in ("theorem_or_result", "calibration_design", "inference_unit", "justification"):
            if not str(getattr(self, name)).strip():
                errors.append(f"coverage basis: {name} is required")
        if self.inference_unit != record.inference_unit:
            errors.append("coverage basis: inference unit does not match protocol record")
        if not self.assumptions or any(not str(item).strip() for item in self.assumptions):
            errors.append("coverage basis: assumptions must be a non-empty tuple")
        for name in ("protocol_sha256", "calibration_artifact_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64:
                errors.append(f"coverage basis: {name} must be a SHA-256 digest")
                continue
            try:
                int(value, 16)
            except ValueError:
                errors.append(f"coverage basis: {name} must be a SHA-256 digest")
        return errors


@dataclass(frozen=True)
class GateThresholds:
    """Predeclared gate thresholds.

    These must be fixed *before* the promoted target conditions are evaluated (A6)
    and versioned with the protocol.  Tuning them after seeing the outcome voids the
    certificate; :func:`run_gate` records the values it ran with in the report so
    that this is auditable after the fact.
    """

    min_effective_units: int = 20
    support_frac_outside_warn: float = 0.05
    support_frac_outside_fail: float = 0.20
    domain_auroc_warn: float = 0.75
    domain_auroc_fail: float = 0.90
    radius_cv_warn: float = 0.20
    radius_cv_fail: float = 0.50
    decision_disagreement_warn: float = 0.05
    decision_disagreement_fail: float = 0.20
    conclusion_change_fail: float = 0.0  # any flip under an admissible split is fatal


# --------------------------------------------------------------------------- #
# Small numerics (kept local so the diagnostics are reproducible from numpy alone)
# --------------------------------------------------------------------------- #


def _as_2d(z: Any) -> np.ndarray:
    arr = np.asarray(z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"evidence must be 1-D or 2-D, got shape {arr.shape}")
    return arr


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared.  (scipy.stats.rankdata without the import.)"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or a.size != b.size:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    if denom == 0.0:
        return None
    return float((ra * rb).sum() / denom)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """AUROC via the Mann-Whitney rank statistic.  Ties handled by average ranks."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = _rankdata(scores)
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iters: int = 400, lr: float = 0.5) -> np.ndarray:
    """Deterministic L2 logistic regression by full-batch gradient descent.

    Small and dependency-free on purpose: the separability diagnostic must be
    reproducible from the pinned requirements, and its exact decision boundary does
    not matter -- only the cross-fitted ranking quality does.
    """
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -30, 30)))
        grad = xb.T @ (p - y) / n
        grad[:-1] += l2 * w[:-1] / n
        w -= lr * grad
    return w


def _clopper_pearson(hits: int, n: int, alpha: float = 0.05) -> tuple[float, float, str]:
    """Exact binomial interval; Wilson fallback when scipy is absent (declared)."""
    if n <= 0:
        return (float("nan"), float("nan"), "undefined")
    if _HAVE_SCIPY:
        lo = 0.0 if hits == 0 else float(_scipy_beta.ppf(alpha / 2, hits, n - hits + 1))
        hi = 1.0 if hits == n else float(_scipy_beta.ppf(1 - alpha / 2, hits + 1, n - hits))
        return (lo, hi, "clopper_pearson")
    p = hits / n
    z = 1.959963984540054
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half), "wilson_fallback_no_scipy")


def _cluster_bootstrap_ci(
    hit: np.ndarray, groups: np.ndarray, *, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for a mean, resampling whole groups.

    This is the interval to report when rows are correlated within a domain, an
    episode, a corruption cell, or a seed.  Resampling rows instead would understate
    the width by roughly the square root of the within-group correlation, which is
    exactly the error the A5 gate exists to prevent.
    """
    uniq = np.unique(groups)
    by_group = [hit[groups == g] for g in uniq]
    rng = np.random.default_rng(seed)
    k = len(uniq)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, k, size=k)
        pooled = np.concatenate([by_group[i] for i in idx])
        draws[b] = pooled.mean()
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


# --------------------------------------------------------------------------- #
# A5: the sampling unit
# --------------------------------------------------------------------------- #


def effective_units(groups: Sequence[Any] | np.ndarray | None, n_rows: int) -> int:
    """Number of independent calibration units.

    ``groups=None`` is an assertion that the rows are independent draws.  On every
    track in the K-Bound paper they are not, which is why the gate requires the
    caller to say so explicitly rather than defaulting.
    """
    if groups is None:
        return int(n_rows)
    return int(len(np.unique(np.asarray(groups))))


def conformal_radius(residuals: Sequence[float] | np.ndarray, alpha: float) -> dict[str, Any]:
    """Exact-rank conformal radius and the ceiling its sample size imposes.

    With ``k = ceil((n+1)(1-alpha))``: if ``k > n`` the requested level is
    unattainable at this sample size -- the best attainable coverage is
    ``n/(n+1)`` -- and the radius is ``inf``, i.e. the system abstains.  Returning a
    finite radius here would be the single most dangerous thing this module could do.
    """
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = int(r.size)
    if n == 0:
        return {
            "radius": float("inf"),
            "k": None,
            "n": 0,
            "best_attainable_coverage": None,
            "level_attainable": False,
        }
    k = math.ceil((n + 1) * (1 - alpha))
    attainable = k <= n
    return {
        "radius": float(r[k - 1]) if attainable else float("inf"),
        "k": int(k),
        "n": n,
        "best_attainable_coverage": n / (n + 1),
        "level_attainable": bool(attainable),
    }


# --------------------------------------------------------------------------- #
# Observed empirical coverage (Definition: observed, never theoretical)
# --------------------------------------------------------------------------- #


def observed_coverage(
    delta_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    *,
    groups: Sequence[Any] | np.ndarray | None = None,
    seed: int = 0,
    n_boot: int = 10_000,
) -> dict[str, Any]:
    """Observed interval-hit rate with a dependence-aware interval.

    Returns ``coverage_type='observed_empirical'`` always.  This function has no code
    path that produces a theoretical coverage claim.
    """
    d = np.asarray(delta_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (d.shape == lo.shape == hi.shape):
        raise ValueError("delta_true, lower, upper must have identical shape")
    if d.size == 0:
        return {
            "coverage_type": CoverageType.OBSERVED_EMPIRICAL.value,
            "observed_coverage": None,
            "n_rows": 0,
            "n_units": 0,
            "coverage_interval_95": None,
            "interval_method": "undefined_empty_sample",
        }

    hit = ((d >= lo) & (d <= hi)).astype(float)
    n_rows = int(hit.size)
    cov = float(hit.mean())

    if groups is None:
        n_units = n_rows
        lo_ci, hi_ci, method = _clopper_pearson(int(hit.sum()), n_rows)
    else:
        g = np.asarray(groups)
        n_units = int(len(np.unique(g)))
        lo_ci, hi_ci = _cluster_bootstrap_ci(hit, g, n_boot=n_boot, seed=seed)
        method = f"cluster_bootstrap_{n_boot}_over_{n_units}_units"

    return {
        "coverage_type": CoverageType.OBSERVED_EMPIRICAL.value,
        "observed_coverage": cov,
        "n_rows": n_rows,
        "n_units": n_units,
        "coverage_interval_95": [lo_ci, hi_ci],
        "interval_method": method,
    }


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


@dataclass
class SupportOverlap:
    status: str
    frac_outside_envelope: float | None
    median_nn_distance: float | None
    max_nn_distance: float | None
    max_standardised_deviation: float | None
    domain_classifier_auroc: float | None
    n_cal: int
    n_dep: int
    notes: list[str] = field(default_factory=list)


def evidence_support_overlap(
    z_cal: Any,
    z_dep: Any,
    *,
    thresholds: GateThresholds | None = None,
    n_folds: int = 5,
) -> SupportOverlap:
    """Is the deployment evidence inside the calibrated support?

    High separability is evidence *against* direct transfer.  Low separability is
    **not** evidence for it: two conditions can be indistinguishable in Z and still
    have different benefit -- that is the impossibility configuration, not a corner
    case.  The AUROC below is cross-fitted so it cannot be inflated by in-sample fit.
    """
    th = thresholds or GateThresholds()
    a = _as_2d(z_cal)
    b = _as_2d(z_dep)
    notes: list[str] = []

    if a.shape[1] != b.shape[1]:
        raise ValueError(f"evidence dimensionality differs: cal has {a.shape[1]}, dep has {b.shape[1]}")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return SupportOverlap(
            status=Status.FAIL.value,
            frac_outside_envelope=None,
            median_nn_distance=None,
            max_nn_distance=None,
            max_standardised_deviation=None,
            domain_classifier_auroc=None,
            n_cal=int(a.shape[0]),
            n_dep=int(b.shape[0]),
            notes=["empty calibration or deployment evidence; overlap undefined"],
        )

    mu = a.mean(axis=0)
    sd = a.std(axis=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    if np.any(sd == 0):
        notes.append(
            f"{int((sd == 0).sum())} calibration feature(s) constant; "
            "standardised deviation for those columns is not informative"
        )
    a_s = (a - mu) / sd_safe
    b_s = (b - mu) / sd_safe

    lo, hi = a.min(axis=0), a.max(axis=0)
    outside = np.any((b < lo) | (b > hi), axis=1)
    frac_outside = float(outside.mean())

    # nearest-neighbour distance, chunked so a large deployment batch cannot blow up
    nn = np.empty(b_s.shape[0], dtype=float)
    chunk = max(1, int(2e7 // max(1, a_s.shape[0] * a_s.shape[1])))
    for start in range(0, b_s.shape[0], chunk):
        blk = b_s[start : start + chunk]
        d2 = ((blk[:, None, :] - a_s[None, :, :]) ** 2).sum(axis=2)
        nn[start : start + blk.shape[0]] = np.sqrt(d2.min(axis=1))

    max_std_dev = float(np.abs(b_s).max())

    auroc = _crossfit_domain_auroc(a_s, b_s, n_folds=n_folds)
    if auroc is None:
        notes.append("domain-classifier AUROC not computable at this sample size")

    status = Status.PASS
    if frac_outside >= th.support_frac_outside_fail or (auroc is not None and auroc >= th.domain_auroc_fail):
        status = Status.FAIL
    elif frac_outside >= th.support_frac_outside_warn or (auroc is not None and auroc >= th.domain_auroc_warn):
        status = Status.WARNING

    notes.append(
        "one-sided: a PASS records that no support violation of this kind was "
        "detected, not that calibration transfer holds"
    )
    return SupportOverlap(
        status=status.value,
        frac_outside_envelope=frac_outside,
        median_nn_distance=float(np.median(nn)),
        max_nn_distance=float(nn.max()),
        max_standardised_deviation=max_std_dev,
        domain_classifier_auroc=auroc,
        n_cal=int(a.shape[0]),
        n_dep=int(b.shape[0]),
        notes=notes,
    )


def _crossfit_domain_auroc(a_s: np.ndarray, b_s: np.ndarray, *, n_folds: int = 5) -> float | None:
    n_a, n_b = a_s.shape[0], b_s.shape[0]
    if min(n_a, n_b) < 2:
        return None
    folds = min(n_folds, n_a, n_b)
    if folds < 2:
        return None

    x = np.vstack([a_s, b_s])
    y = np.concatenate([np.zeros(n_a), np.ones(n_b)])
    # deterministic interleaved assignment: no RNG, so the number is reproducible
    fold_id = np.concatenate([np.arange(n_a) % folds, np.arange(n_b) % folds])

    scores = np.empty(x.shape[0], dtype=float)
    for f in range(folds):
        tr, te = fold_id != f, fold_id == f
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            return None
        w = _fit_logistic(x[tr], y[tr])
        scores[te] = np.hstack([x[te], np.ones((int(te.sum()), 1))]) @ w
    return _auroc(scores, y)


@dataclass
class RadiusStability:
    status: str
    mean_radius: float | None
    min_radius: float | None
    max_radius: float | None
    radius_range: float | None
    radius_cv: float | None
    decision_disagreement_rate: float | None
    n_splits: int
    notes: list[str] = field(default_factory=list)


def radius_stability(
    residuals: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    alpha: float,
    *,
    delta_hat: Sequence[float] | np.ndarray | None = None,
    thresholds: GateThresholds | None = None,
) -> RadiusStability:
    """Recompute the radius leave-one-group-out and see whether anything moves.

    A stable radius is not a correct radius -- the whole calibration set can be
    uniformly wrong.  What instability *does* show is that the promoted number
    depends on an arbitrary choice of fold, seed, domain, or block, and that is
    enough to withhold a strict decision.
    """
    th = thresholds or GateThresholds()
    r = np.asarray(residuals, dtype=float)
    g = np.asarray(groups)
    if r.shape[0] != g.shape[0]:
        raise ValueError("residuals and groups must have the same length")

    uniq = np.unique(g)
    notes: list[str] = []
    if len(uniq) < 2:
        return RadiusStability(
            status=Status.FAIL.value,
            mean_radius=None,
            min_radius=None,
            max_radius=None,
            radius_range=None,
            radius_cv=None,
            decision_disagreement_rate=None,
            n_splits=int(len(uniq)),
            notes=["fewer than two independent units; stability is not assessable"],
        )

    radii: list[float] = []
    for held in uniq:
        sub = r[g != held]
        out = conformal_radius(sub, alpha)
        radii.append(out["radius"])

    finite = [x for x in radii if math.isfinite(x)]
    if not finite:
        return RadiusStability(
            status=Status.FAIL.value,
            mean_radius=None,
            min_radius=None,
            max_radius=None,
            radius_range=None,
            radius_cv=None,
            decision_disagreement_rate=None,
            n_splits=int(len(uniq)),
            notes=[
                f"alpha={alpha} unattainable in every leave-one-unit-out fold at this "
                "sample size; the certificate cannot be issued"
            ],
        )
    if len(finite) < len(radii):
        notes.append(
            f"{len(radii) - len(finite)} of {len(radii)} folds could not attain "
            f"alpha={alpha}; those folds abstain by construction and are excluded "
            "from the spread statistics"
        )

    arr = np.asarray(finite, dtype=float)
    mean_r = float(arr.mean())
    cv = float(arr.std(ddof=1) / mean_r) if len(arr) > 1 and mean_r > 0 else None

    disagreement: float | None = None
    if delta_hat is not None:
        dh = np.asarray(delta_hat, dtype=float)
        pooled = conformal_radius(r, alpha)["radius"]
        if math.isfinite(pooled):
            base = _actions(dh, pooled)
            flips = np.zeros(dh.shape[0], dtype=bool)
            for rad in finite:
                flips |= _actions(dh, rad) != base
            disagreement = float(flips.mean())
        else:
            notes.append("pooled radius is infinite (alpha unattainable); decision disagreement not computed")

    status = Status.PASS
    if (cv is not None and cv >= th.radius_cv_fail) or (
        disagreement is not None and disagreement >= th.decision_disagreement_fail
    ):
        status = Status.FAIL
    elif (cv is not None and cv >= th.radius_cv_warn) or (
        disagreement is not None and disagreement >= th.decision_disagreement_warn
    ):
        status = Status.WARNING

    return RadiusStability(
        status=status.value,
        mean_radius=mean_r,
        min_radius=float(arr.min()),
        max_radius=float(arr.max()),
        radius_range=float(arr.max() - arr.min()),
        radius_cv=cv,
        decision_disagreement_rate=disagreement,
        n_splits=int(len(uniq)),
        notes=notes,
    )


def _actions(delta_hat: np.ndarray, radius: float) -> np.ndarray:
    """1 = adapt (L>0), -1 = freeze (U<0), 0 = abstain."""
    lo = delta_hat - radius
    hi = delta_hat + radius
    out = np.zeros(delta_hat.shape[0], dtype=int)
    out[lo > 0] = 1
    out[hi < 0] = -1
    return out


@dataclass
class RiskAlignmentAudit:
    retrospective: bool
    mae: float | None
    rmse: float | None
    spearman: float | None
    sign_agreement: float | None
    false_adapt_by_group: dict[str, float] | None
    coverage_by_evidence_region: dict[str, float] | None
    n_rows: int
    notes: list[str] = field(default_factory=list)


def risk_alignment_audit(
    delta_hat: Sequence[float] | np.ndarray,
    delta_true: Sequence[float] | np.ndarray,
    *,
    groups: Sequence[Any] | np.ndarray | None = None,
    radius: float | None = None,
    evidence_region: Sequence[Any] | np.ndarray | None = None,
) -> RiskAlignmentAudit:
    """Retrospective audit of A2.  Requires labels, so it is never a deployment test.

    This is the honest status of the risk-alignment evidence: it says something about
    the tracks that were labelled, and nothing about the one being deployed.
    """
    dh = np.asarray(delta_hat, dtype=float)
    dt = np.asarray(delta_true, dtype=float)
    if dh.shape != dt.shape:
        raise ValueError("delta_hat and delta_true must have identical shape")
    notes = [
        "retrospective: computed from evaluation labels, unavailable at deployment "
        "time; it audits A2 on evaluated conditions and does not license A2 elsewhere"
    ]
    if dh.size == 0:
        return RiskAlignmentAudit(True, None, None, None, None, None, None, 0, notes + ["empty sample"])

    err = dh - dt
    sign_agree = float((np.sign(dh) == np.sign(dt)).mean())

    fa_by_group: dict[str, float] | None = None
    if radius is not None and math.isfinite(radius):
        acts = _actions(dh, radius)
        harmful = dt <= 0
        if groups is not None:
            g = np.asarray(groups)
            fa_by_group = {str(u): float(((acts == 1) & harmful)[g == u].mean()) for u in np.unique(g)}
        else:
            fa_by_group = {"__pooled__": float(((acts == 1) & harmful).mean())}
    elif radius is not None:
        notes.append("radius is infinite; false-adapt by group not computed")

    cov_by_region: dict[str, float] | None = None
    if evidence_region is not None and radius is not None and math.isfinite(radius):
        reg = np.asarray(evidence_region)
        inside = (dt >= dh - radius) & (dt <= dh + radius)
        cov_by_region = {str(u): float(inside[reg == u].mean()) for u in np.unique(reg)}

    return RiskAlignmentAudit(
        retrospective=True,
        mae=float(np.abs(err).mean()),
        rmse=float(np.sqrt((err**2).mean())),
        spearman=_spearman(dh, dt),
        sign_agreement=sign_agree,
        false_adapt_by_group=fa_by_group,
        coverage_by_evidence_region=cov_by_region,
        n_rows=int(dh.size),
        notes=notes,
    )


@dataclass
class ConclusionStability:
    status: str
    n_alternatives: int
    n_changed: int
    frac_changed: float | None
    changed_under: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def conclusion_stability(
    promoted_conclusion: Any,
    alternatives: Mapping[str, Callable[[], Any]],
    *,
    thresholds: GateThresholds | None = None,
) -> ConclusionStability:
    """Does the promoted conclusion survive every *equally valid* analysis choice?

    ``alternatives`` maps a description ("cluster by corruption family", "split by
    seed") to a zero-argument callable that recomputes the conclusion under that
    choice.  A conclusion that survives only one admissible clustering is a
    conclusion about the clustering, and the gate downgrades it to diagnostic.
    """
    th = thresholds or GateThresholds()
    if not alternatives:
        return ConclusionStability(
            status=Status.FAIL.value,
            n_alternatives=0,
            n_changed=0,
            frac_changed=None,
            notes=[
                "no alternative analysis supplied; sensitivity to the calibration "
                "construction is untested, which is a failure of the check, not a pass"
            ],
        )

    changed: list[str] = []
    notes: list[str] = []
    for name, fn in alternatives.items():
        try:
            if fn() != promoted_conclusion:
                changed.append(name)
        except Exception as exc:  # a broken alternative is not a passing alternative
            changed.append(f"{name} (raised {type(exc).__name__})")
            notes.append(f"alternative {name!r} failed to evaluate: {exc}")

    frac = len(changed) / len(alternatives)
    status = Status.FAIL.value if frac > th.conclusion_change_fail else Status.PASS.value
    return ConclusionStability(
        status=status,
        n_alternatives=len(alternatives),
        n_changed=len(changed),
        frac_changed=frac,
        changed_under=changed,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# A4 / A6: protocol provenance
# --------------------------------------------------------------------------- #


@dataclass
class ProtocolRecord:
    """Chronology and provenance for one promoted claim.

    Timestamps are ISO-8601 strings.  Absence is not neutral: a claim whose lock
    identifier or fix-time is unknown fails the leakage audit, because an assertion
    of pre-registration without a timestamped lock is not evidence of it.
    """

    protocol: str
    dataset: str
    inference_unit: str
    candidate_fixed_at: str | None = None
    calibration_design_fixed_at: str | None = None
    target_evaluated_at: str | None = None
    target_labels_accessed: bool | None = None
    target_labels_used_for_routing: bool | None = None
    test_set_influenced_hparams: bool | None = None
    calibration_test_separated: bool | None = None
    protocol_lock_id: str | None = None
    failed_runs_retained: bool | None = None


def leakage_audit(record: ProtocolRecord) -> tuple[Status, list[str]]:
    """A4 and A6 as a check on the record, failing closed on missing provenance."""
    reasons: list[str] = []

    if record.target_labels_used_for_routing is True:
        reasons.append("A4 violated: target labels used for routing/selection")
    if record.test_set_influenced_hparams is True:
        reasons.append("A4 violated: test set influenced hyperparameters")
    if record.calibration_test_separated is False:
        reasons.append("A4 violated: calibration and evaluation not separated")

    for name, value in (
        ("target_labels_used_for_routing", record.target_labels_used_for_routing),
        ("test_set_influenced_hparams", record.test_set_influenced_hparams),
        ("calibration_test_separated", record.calibration_test_separated),
    ):
        if value is None:
            reasons.append(f"provenance unknown: {name} not recorded")

    if not record.protocol_lock_id:
        reasons.append("no protocol lock identifier: A6 chronology unverifiable")

    fixed = record.candidate_fixed_at
    evaluated = record.target_evaluated_at
    if fixed is None or evaluated is None:
        reasons.append("candidate fix time or evaluation time not recorded (A6)")
    elif fixed > evaluated:
        reasons.append(f"A6 violated: candidate fixed at {fixed}, after target evaluation at {evaluated}")

    design = record.calibration_design_fixed_at
    if design is not None and evaluated is not None and design > evaluated:
        reasons.append(f"A6 violated: calibration design fixed at {design}, after target evaluation at {evaluated}")

    if record.failed_runs_retained is False:
        reasons.append(
            "failed runs not retained: the promoted result cannot be placed in the "
            "population of attempts it was selected from"
        )

    return (Status.PASS if not reasons else Status.FAIL), reasons


# --------------------------------------------------------------------------- #
# The report and the gate
# --------------------------------------------------------------------------- #


@dataclass
class AssumptionReport:
    """Machine-readable assumption state, emitted alongside every result.

    ``theoretical_coverage_claimed`` is False unless a caller supplies a valid
    :class:`CoverageClaimBasis` and explicitly requests the claim. There is no
    code path in which a diagnostic result alone flips it to True.
    """

    dataset: str
    protocol: str
    inference_unit: str
    calibration_test_separated: bool | None
    candidate_fixed_before_test: bool | None
    target_labels_used_for_routing: bool | None
    coverage_type: str
    theoretical_coverage_claimed: bool
    coverage_claim_basis: dict[str, Any] | None
    observed_coverage: float | None
    n_rows: int | None
    n_units: int | None
    coverage_interval_95: list[float] | None
    coverage_interval_method: str | None
    support_overlap_status: str
    radius_stability_status: str
    conclusion_stability_status: str
    leakage_status: str
    deployment_gate: str
    fallback_action: str
    alpha: float
    thresholds: dict[str, Any]
    diagnostics: dict[str, Any]
    limitations: list[str] = field(default_factory=list)
    schema_version: str = "kbound-assumption-report/2"

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False, **kwargs)


def run_gate(
    *,
    record: ProtocolRecord,
    alpha: float,
    residuals: Sequence[float] | np.ndarray | None = None,
    calibration_groups: Sequence[Any] | np.ndarray | None = None,
    z_cal: Any | None = None,
    z_dep: Any | None = None,
    delta_hat: Sequence[float] | np.ndarray | None = None,
    delta_true: Sequence[float] | np.ndarray | None = None,
    interval_lower: Sequence[float] | np.ndarray | None = None,
    interval_upper: Sequence[float] | np.ndarray | None = None,
    evaluation_groups: Sequence[Any] | np.ndarray | None = None,
    evidence_region: Sequence[Any] | np.ndarray | None = None,
    conclusion: Any | None = None,
    alternatives: Mapping[str, Callable[[], Any]] | None = None,
    thresholds: GateThresholds | None = None,
    claim_theoretical_coverage: bool = False,
    coverage_claim_basis: CoverageClaimBasis | None = None,
    seed: int = 0,
) -> AssumptionReport:
    """Run the seven-step assumption gate and return the report.

    Steps 1-2 are fatal protocol checks; 3-5 are statistical diagnostics that degrade
    the permitted behaviour rather than abort; step 6 is the certified rule; step 7
    (sequential monitoring) is the deployment's responsibility and is recorded here
    only as a requirement.  Diagnostics still run after a fatal check so the report is
    informative about *why*, but the decision cannot be upgraded afterwards.
    """
    th = thresholds or GateThresholds()
    limitations: list[str] = []
    diagnostics: dict[str, Any] = {}

    # -- Step 1: protocol separation (A4, A6) -- fatal ---------------------- #
    leak_status, leak_reasons = leakage_audit(record)
    diagnostics["leakage"] = {"status": leak_status.value, "reasons": leak_reasons}
    limitations.extend(leak_reasons)
    decision = GateDecision.REJECT if leak_status is Status.FAIL else GateDecision.CERTIFY

    # -- Step 2: inference-unit adequacy (A5) ------------------------------- #
    n_rows_cal = 0 if residuals is None else int(np.asarray(residuals).size)
    n_eff = effective_units(calibration_groups, n_rows_cal)
    diagnostics["effective_units"] = {
        "n_effective_units": n_eff,
        "n_calibration_rows": n_rows_cal,
        "minimum_required": th.min_effective_units,
        "unit": record.inference_unit,
        "groups_declared": calibration_groups is not None,
    }
    if calibration_groups is None and n_rows_cal > 0:
        limitations.append(
            "calibration groups not declared: rows are being treated as independent "
            "draws, which A5 requires the caller to justify"
        )
    if n_eff < th.min_effective_units:
        limitations.append(
            f"only {n_eff} effective calibration units against a declared minimum of {th.min_effective_units}; A5 unmet"
        )
        decision = _downgrade(decision, GateDecision.DIAGNOSTIC_ONLY)

    if residuals is not None:
        rad = conformal_radius(residuals, alpha)
        diagnostics["conformal_radius"] = rad
        if not rad["level_attainable"]:
            limitations.append(
                f"alpha={alpha} is unattainable at n={rad['n']}: best attainable "
                f"coverage is {rad['best_attainable_coverage']:.4f}; the radius is "
                "infinite and the system abstains"
            )
            decision = _downgrade(decision, GateDecision.DIAGNOSTIC_ONLY)

    # -- Step 3: evidence-support overlap (A1, A2) -------------------------- #
    if z_cal is not None and z_dep is not None:
        so = evidence_support_overlap(z_cal, z_dep, thresholds=th)
        diagnostics["support_overlap"] = asdict(so)
        support_status = so.status
        if support_status == Status.FAIL.value:
            limitations.append("deployment evidence outside the calibrated support; A1/A2 suspect")
            decision = _downgrade(decision, GateDecision.RESTRICTED)
        elif support_status == Status.WARNING.value:
            limitations.append("support-overlap warning: strict adapt is withheld, freeze/abstain only")
            decision = _downgrade(decision, GateDecision.RESTRICTED)
    else:
        support_status = Status.FAIL.value
        limitations.append(
            "support overlap not evaluated (calibration or deployment evidence not "
            "supplied); an unevaluated check is a failed check"
        )
        decision = _downgrade(decision, GateDecision.RESTRICTED)

    # -- Step 4: calibration stability -------------------------------------- #
    if residuals is not None and calibration_groups is not None:
        rs = radius_stability(residuals, calibration_groups, alpha, delta_hat=delta_hat, thresholds=th)
        diagnostics["radius_stability"] = asdict(rs)
        radius_status = rs.status
        if radius_status == Status.FAIL.value:
            limitations.append("radius or action mix unstable across admissible splits; strict decisions withheld")
            decision = _downgrade(decision, GateDecision.RESTRICTED)
        elif radius_status == Status.WARNING.value:
            limitations.append("radius stability warning; freeze/abstain only")
            decision = _downgrade(decision, GateDecision.RESTRICTED)
    else:
        radius_status = Status.FAIL.value
        limitations.append("radius stability not evaluated (residuals or unit labels missing)")
        decision = _downgrade(decision, GateDecision.RESTRICTED)

    # -- Step 5: sensitivity of the promoted conclusion --------------------- #
    if conclusion is not None:
        cs = conclusion_stability(conclusion, alternatives or {}, thresholds=th)
        diagnostics["conclusion_stability"] = asdict(cs)
        conclusion_status = cs.status
        if conclusion_status == Status.FAIL.value:
            if cs.changed_under:
                limitations.append(
                    "promoted conclusion changes under an equally valid analysis "
                    f"choice ({', '.join(cs.changed_under)}); result is diagnostic only"
                )
            else:
                limitations.append("sensitivity to the calibration construction untested; result is diagnostic only")
            decision = _downgrade(decision, GateDecision.DIAGNOSTIC_ONLY)
    else:
        conclusion_status = Status.FAIL.value
        limitations.append("no promoted conclusion supplied; sensitivity untested")
        decision = _downgrade(decision, GateDecision.DIAGNOSTIC_ONLY)

    # -- Observed coverage (a report field, never a gate input) ------------- #
    cov: dict[str, Any] = {
        "coverage_type": CoverageType.DIAGNOSTIC_ONLY.value,
        "observed_coverage": None,
        "n_rows": None,
        "n_units": None,
        "coverage_interval_95": None,
        "interval_method": None,
    }
    if delta_true is not None and interval_lower is not None and interval_upper is not None:
        cov = observed_coverage(
            delta_true,
            interval_lower,
            interval_upper,
            groups=evaluation_groups,
            seed=seed,
        )
        if evaluation_groups is None:
            limitations.append(
                "observed coverage interval computed without unit labels; if the "
                "evaluation rows are correlated this interval is too narrow"
            )
    else:
        limitations.append("observed coverage not computable: labelled benefits or interval endpoints not supplied")

    if delta_hat is not None and delta_true is not None:
        radius_for_audit = None
        if "conformal_radius" in diagnostics:
            radius_for_audit = diagnostics["conformal_radius"]["radius"]
        ra = risk_alignment_audit(
            delta_hat,
            delta_true,
            groups=evaluation_groups,
            radius=radius_for_audit,
            evidence_region=evidence_region,
        )
        diagnostics["risk_alignment"] = asdict(ra)
        limitations.append(
            "risk alignment (A2) is audited retrospectively on labelled tracks only; "
            "it is not established for any unlabelled deployment condition"
        )

    # -- Coverage-claim discipline ------------------------------------------ #
    basis_payload: dict[str, Any] | None = None
    if coverage_claim_basis is not None:
        basis_errors = coverage_claim_basis.validate(record)
        diagnostics["coverage_claim_basis"] = {
            "status": Status.PASS.value if not basis_errors else Status.FAIL.value,
            "errors": basis_errors,
        }
        basis_payload = asdict(coverage_claim_basis)
        if basis_errors:
            limitations.extend(basis_errors)
            claim_theoretical_coverage = False
    elif claim_theoretical_coverage:
        claim_theoretical_coverage = False
        limitations.append("theoretical coverage claim refused: no auditable CoverageClaimBasis supplied")

    if claim_theoretical_coverage and decision is not GateDecision.CERTIFY:
        claim_theoretical_coverage = False
        limitations.append("theoretical coverage claim withdrawn: the gate did not return CERTIFY")
    coverage_type = CoverageType.THEORETICAL.value if claim_theoretical_coverage else cov["coverage_type"]
    if not claim_theoretical_coverage:
        limitations.append(
            "no theoretical coverage claim: A1-A3 are not checkable from label-free "
            "deployment evidence, so any coverage figure here is an observed hit rate"
        )

    candidate_fixed_before_test = None
    if record.candidate_fixed_at and record.target_evaluated_at:
        candidate_fixed_before_test = record.candidate_fixed_at <= record.target_evaluated_at

    return AssumptionReport(
        dataset=record.dataset,
        protocol=record.protocol,
        inference_unit=record.inference_unit,
        calibration_test_separated=record.calibration_test_separated,
        candidate_fixed_before_test=candidate_fixed_before_test,
        target_labels_used_for_routing=record.target_labels_used_for_routing,
        coverage_type=coverage_type,
        theoretical_coverage_claimed=bool(claim_theoretical_coverage),
        coverage_claim_basis=basis_payload,
        observed_coverage=cov["observed_coverage"],
        n_rows=cov["n_rows"],
        n_units=cov["n_units"],
        coverage_interval_95=cov["coverage_interval_95"],
        coverage_interval_method=cov["interval_method"],
        support_overlap_status=support_status,
        radius_stability_status=radius_status,
        conclusion_stability_status=conclusion_status,
        leakage_status=leak_status.value,
        deployment_gate=decision.value,
        fallback_action=_LADDER[decision].value,
        alpha=float(alpha),
        thresholds=asdict(th),
        diagnostics=diagnostics,
        limitations=_dedupe(limitations),
    )


_SEVERITY = {
    GateDecision.CERTIFY: 0,
    GateDecision.RESTRICTED: 1,
    GateDecision.DIAGNOSTIC_ONLY: 2,
    GateDecision.REJECT: 3,
}


def _downgrade(current: GateDecision, proposed: GateDecision) -> GateDecision:
    """Monotone: the gate can only ever move down the ladder, never back up."""
    return current if _SEVERITY[current] >= _SEVERITY[proposed] else proposed


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def write_report(report: AssumptionReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.to_json() + "\n", encoding="utf-8")
    return p
