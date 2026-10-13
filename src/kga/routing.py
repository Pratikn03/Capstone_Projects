"""kga.routing -- Multicandidate and multiclass Bonferroni routing (Wave 4).

Implements the selection-proof guarantees validated in:

* ``theory_v2/val_multicandidate.py`` (``thm:multicand``)
* ``theory_v2/val_multiclass_multicandidate.py`` (``thm:multiclass-multicand``)
* ``theory_v2/val_anytime_multicandidate.py`` (``thm:anytime-multicand``)

The batch path uses per-candidate split-conformal lower confidence bounds with
Bonferroni calibration level ``alpha / K``.  The anytime path runs ``K``
parallel betting e-processes with wealth threshold ``K / alpha``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from kga._validation import as_float_array
from kga.certificate import min_calibration_size as _min_calibration_size
from kga.certificate import split_conformal_rank_radius as _split_conformal_rank_radius

Selector = Literal["argmax_lcb", "first_positive"]


@dataclass(frozen=True)
class CandidateCertificate:
    """Per-candidate benefit certificate on the disagreement region."""

    index: int
    delta_hat: float
    epsilon: float

    @property
    def available(self) -> bool:
        """Whether this candidate has a finite estimate and usable radius."""
        return math.isfinite(self.delta_hat) and math.isfinite(self.epsilon) and self.epsilon >= 0.0

    @property
    def lcb(self) -> float:
        if not self.available:
            return -math.inf
        return self.delta_hat - self.epsilon


@dataclass(frozen=True)
class RoutingDecision:
    """Bonferroni multicandidate / multiclass routing outcome.

    ``feasible`` is ``False`` when no candidate has a usable certificate,
    including when the calibration set is too small for the Bonferroni level
    ``alpha / K`` to be attainable (see :func:`route_panel`);
    in that case ``selected is None`` and ``committed is False`` regardless of
    the point estimates.
    """

    selected: int | None
    certificates: tuple[CandidateCertificate, ...]
    alpha: float
    bonferroni_alpha: float
    committed: bool
    feasible: bool = True
    min_n_cal: int | None = None

    @property
    def decision(self) -> str:
        if self.selected is None:
            return "abstain"
        return "adapt"


def split_conformal_rank_radius(cal_errors: np.ndarray, level: float) -> float:
    """Exact rank radius for signed or absolute calibration errors.

    Thin wrapper over :func:`kga.certificate.split_conformal_rank_radius`; it
    takes absolute values first so that signed calibration errors may be passed.
    Inherits the small-``n`` behaviour: ``+inf`` when
    ``ceil((n + 1)(1 - level)) > n``.
    """
    return _split_conformal_rank_radius(np.abs(as_float_array(cal_errors).ravel()), level)


def candidate_lcb_from_calibration(
    deploy_score: float,
    cal_scores: np.ndarray,
    cal_truth: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float, float]:
    """LOO-style LCB: deploy score with conformal radius from calibration residuals.

    When ``alpha`` is too small for ``len(cal_scores)`` the radius is ``+inf``
    and the returned LCB is ``-inf``, so the candidate can never be selected.
    That is the intended behaviour: at a Bonferroni level of ``alpha / K`` the
    per-candidate calibration requirement is ``K`` times stricter than it looks
    (panel finding F2-7).
    """
    cal_scores = as_float_array(cal_scores).ravel()
    cal_truth = as_float_array(cal_truth).ravel()
    if cal_scores.shape != cal_truth.shape:
        raise ValueError("cal_scores and cal_truth must have the same shape")
    if not math.isfinite(float(deploy_score)):
        raise ValueError("deploy_score must be finite")
    residuals = np.abs(cal_scores - cal_truth)
    eps = split_conformal_rank_radius(residuals, alpha)
    delta_hat = float(deploy_score)
    return delta_hat, eps, delta_hat - eps


def multiclass_benefit(mu_d: float, pa: float, p0: float) -> float:
    """Multiclass 0/1 benefit on D: Delta = mu_D * (p_a - p_0)."""
    return float(mu_d) * (float(pa) - float(p0))


def multiclass_harmful(delta: float, pa: float, p0: float, mu_d: float) -> bool:
    """False-harm event: Delta <= 0 iff p_a <= p_0 when mu_D > 0."""
    if mu_d > 0:
        return float(pa) <= float(p0)
    return float(delta) <= 0.0


def bonferroni_multicandidate_route(
    lcbs: Sequence[float],
    *,
    alpha: float | None = None,
    selector: Selector = "argmax_lcb",
) -> int | None:
    """Return selected candidate index, or None to abstain.

    ``alpha`` is **not used** and is retained only so existing call sites keep
    working: the Bonferroni correction has already been spent when the LCBs were
    built at level ``alpha / K``, and applying it again here would be
    double-counting.  Panel finding F2-15 flagged the dead parameter; it is
    documented rather than deleted because it is part of the published API.
    """
    if selector not in ("argmax_lcb", "first_positive"):
        raise ValueError(f"unknown selector: {selector!r}")
    arr = as_float_array(lcbs)
    if arr.ndim != 1:
        raise ValueError("lcbs must be a one-dimensional sequence")
    if arr.size == 0:
        return None
    # Overflow/NaN is unavailable evidence, never an exceptionally strong LCB.
    positive = np.where(np.isfinite(arr) & (arr > 0.0))[0]
    if positive.size == 0:
        return None
    if selector == "first_positive":
        return int(positive[0])
    return int(positive[np.argmax(arr[positive])])


def route_panel(
    deploy_scores: np.ndarray,
    cal_scores: np.ndarray,
    cal_truth: np.ndarray,
    *,
    alpha: float,
    selector: Selector = "argmax_lcb",
) -> RoutingDecision:
    """Bonferroni FWER routing over K candidates (batch certificates).

    Parameters
    ----------
    deploy_scores : (K,)
        Deploy-time benefit point estimates per candidate.
    cal_scores : (K, n_cal)
        Calibration benefit estimates.
    cal_truth : (K, n_cal)
        Calibration ground-truth benefits (for radius only; not used at deploy).

    Feasibility (panel finding F2-7 / fix-queue item 25)
    ---------------------------------------------------
    The per-candidate level is ``alpha / K``, so the exact-rank radius needs
    ``n_cal >= min_calibration_size(alpha / K)`` residuals: at ``alpha = 0.1``
    with ``K = 5`` that is ``n_cal >= 49``, not ``n_cal >= 9``.  Below that
    threshold no finite radius attains the corrected level, the radii come back
    ``+inf``, and this function returns ``feasible=False``, ``committed=False``.
    It used to return ``committed=True`` at an unattainable level.
    """
    deploy_scores = as_float_array(deploy_scores).ravel()
    cal_scores = as_float_array(cal_scores)
    cal_truth = as_float_array(cal_truth)
    k = deploy_scores.size
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if selector not in ("argmax_lcb", "first_positive"):
        raise ValueError(f"unknown selector: {selector!r}")
    if k == 0:
        raise ValueError("route_panel requires at least one candidate")
    if cal_scores.ndim != 2 or cal_truth.ndim != 2:
        raise ValueError("cal_scores and cal_truth must be two-dimensional")
    if cal_scores.shape != cal_truth.shape or cal_scores.shape[0] != k:
        raise ValueError("deploy_scores length must match cal_scores/cal_truth rows")
    bonf = alpha / k
    n_min = _min_calibration_size(bonf)
    n_cal = int(cal_scores.shape[1])
    feasible = n_cal >= n_min
    certs: list[CandidateCertificate] = []
    lcbs: list[float] = []
    for i in range(k):
        try:
            dh, eps, _ = candidate_lcb_from_calibration(deploy_scores[i], cal_scores[i], cal_truth[i], alpha=bonf)
        except ValueError:
            # Do not impute missing/nonfinite evidence or drop residuals from
            # the locked pool. This candidate has no usable certificate.
            dh, eps = float(deploy_scores[i]), math.inf
        candidate = CandidateCertificate(index=i, delta_hat=dh, epsilon=eps)
        certs.append(candidate)
        lcbs.append(candidate.lcb)
    feasible = feasible and any(candidate.available for candidate in certs)
    selected = None if not feasible else bonferroni_multicandidate_route(lcbs, alpha=alpha, selector=selector)
    return RoutingDecision(
        selected=selected,
        certificates=tuple(certs),
        alpha=float(alpha),
        bonferroni_alpha=float(bonf),
        committed=selected is not None,
        feasible=feasible,
        min_n_cal=n_min,
    )


class _BettingEProcess:
    """One-sided betting e-process (matches val_anytime_multicandidate)."""

    def __init__(
        self,
        a: float = -1.0,
        b: float = 1.0,
        cap: float = 0.5,
    ) -> None:
        # NOTE (F2-15): this class used to store an unused ``self.alpha``.  The
        # threshold is supplied by ``rejected_null(global_alpha, k)``, which is
        # the only level that matters, so the stale copy has been removed.
        if not math.isfinite(a) or not math.isfinite(b) or not a < 0.0 < b:
            raise ValueError("finite a < 0 < b is required")
        if not 0.0 < cap < 1.0:
            raise ValueError("cap must be in (0, 1)")
        self.a, self.b = a, b
        self.lam_max = cap / (-a)
        self.s1 = 0.0
        self.s2 = 0.25
        self.cnt = 0.0
        self.logw = 0.0

    def update(self, x: float) -> float:
        x = float(x)
        if not math.isfinite(x) or x < self.a or x > self.b:
            raise ValueError(f"benefit must be finite and in the predeclared support [{self.a}, {self.b}]")
        mu = self.s1 / self.cnt if self.cnt > 0 else 0.0
        s2 = self.s2 / max(self.cnt, 1.0)
        lam = float(np.clip(mu / s2 if s2 > 0 else 0.0, 0.0, self.lam_max))
        self.logw += math.log(max(1.0 + lam * x, 1e-300))
        self.s1 += x
        self.s2 += x * x
        self.cnt += 1.0
        return self.logw

    def rejected_null(self, global_alpha: float, k: int) -> bool:
        return self.logw >= math.log(k / global_alpha)


class AnytimeMulticandidatePanel:
    """K parallel e-processes with Bonferroni anytime threshold (thm:anytime-multicand)."""

    def __init__(self, k: int, alpha: float = 0.1) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.k = k
        self.alpha = float(alpha)
        self._procs = [_BettingEProcess() for _ in range(k)]
        self._steps = 0

    def update(self, benefits: Sequence[float]) -> int | None:
        """Ingest one vector of per-candidate benefits; return the adapt index.

        **All** ``K`` e-processes ingest the step before any rejection is
        checked.  The previous implementation returned as soon as the first
        candidate crossed ``log(K / alpha)``, which left candidates
        ``i + 1 ... K - 1`` with a hole in their observation sequence -- their
        wealth was then no longer a function of the full stream, and the
        returned index was biased toward low candidate indices independently of
        evidence strength (panel finding F2-15).

        Returns
        -------
        int or None
            The index of the crossed candidate with the largest wealth, or
            ``None`` if none crossed.  Ties break to the lowest index.
        """
        if len(benefits) != self.k:
            raise ValueError(f"expected {self.k} benefits, got {len(benefits)}")
        values = tuple(float(x) for x in as_float_array(benefits))
        # Validate the complete step before advancing any process. Otherwise
        # a later invalid candidate leaves earlier candidates partially updated.
        for x, process in zip(values, self._procs, strict=True):
            if not math.isfinite(x) or x < process.a or x > process.b:
                raise ValueError(f"benefit must be finite and in the predeclared support [{process.a}, {process.b}]")
        self._steps += 1
        # 1. every process ingests the step ...
        for i, x in enumerate(values):
            self._procs[i].update(x)
        # 2. ... and only then do we look for rejections.
        crossed = [i for i in range(self.k) if self._procs[i].rejected_null(self.alpha, self.k)]
        if not crossed:
            return None
        return max(crossed, key=lambda i: (self._procs[i].logw, -i))

    def crossed(self) -> tuple[int, ...]:
        """Indices of every candidate whose e-process has crossed ``K / alpha``."""
        return tuple(i for i in range(self.k) if self._procs[i].rejected_null(self.alpha, self.k))

    @property
    def steps(self) -> int:
        return self._steps
