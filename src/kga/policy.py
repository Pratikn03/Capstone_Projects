"""kga.policy -- Trichotomy with conditional certificate-based error control.

The Knowability-Guided Adaptation (KGA) gate turns a finite-sample certificate
``Delta_hat +/- epsilon`` into one of three actions:

* ``ADAPT``   -- the certified lower bound is strictly positive,
                 ``Delta_hat - epsilon > 0``;
* ``FREEZE``  -- the certified upper bound is strictly negative,
                 ``Delta_hat + epsilon < 0``;
* ``ABSTAIN`` -- the certificate representation does not support a strict
                 commitment. This may reflect finite sample size, estimator
                 error, transfer failure, or structural ambiguity.

This is the maintained public KGA rule. Historical experiment artifacts retain
their own declared protocols; this implementation does not retroactively alter
their calibration designs or validate their statistical assumptions.

Conditional certificate criterion.
    Let ``B`` be the certificate's declared scalar target. If its lower bound
    satisfies ``P(B >= delta_hat - epsilon) >= 1 - alpha`` under the stated
    sampling/calibration/transfer assumptions, then ADAPT implies ``B > 0`` on
    that coverage event. Consequently

        P( ADAPT  and  B <= 0 )  <=  alpha.

    The symmetric statement requires a valid upper bound for that same target.
    Coverage for a measured cell does not, by itself, cover population benefit.
    These are unconditional joint error-event bounds, not bounds conditional
    on commitment or a claim of simultaneous two-sided coverage for every
    certificate method.

    The specified ``evalue_anytime`` e-process instead controls directional
    rejection over time for its fixed declared nulls and bounded stream, with
    predictable bets. Ville's argument does not authorize arbitrary repeated
    deployment decisions, candidate changes, or an adaptively changed testing
    setup. Its common certificate container is not a confidence interval.

The threshold function cannot verify the target, coverage, or transfer
assumptions. Its conditional guarantee is inherited from a valid certificate,
not created by choosing a numerical radius.

Which function should I call?
-----------------------------
* :func:`decide`      -- one :class:`~kga.certificate.Certificate` -> one
  :class:`Decision`.  The primitive.
* :func:`decide_batch` -- vectorised :func:`decide` over arrays of
  ``delta_hat`` / ``epsilon`` (``epsilon`` may be scalar or per-cell).
* :func:`decide_kga`  -- the historical controlled-grid replay compatibility
  rule: stored per-cell benefit estimates and realised benefits in, decisions
  out, with a leave-one-out-of-pool exact-rank residual radius. The canonical
  refitted controlled-grid path is :func:`kga.crossfit.controlled_grid_crossfit`.
  The seven
  copy-pasted ``decide_kga`` forks that produced the published numbers used an
  interpolated ``np.quantile`` over a pool that included the scored cell, and
  they are gone.

Tie-breaking
------------
Every comparison is **strict**, and this is deliberate: it is the same
convention as the knowability frontier, where a cell is committed only when
``|M| > beta`` and ``|M| == beta`` is unknowable
(``experiments/kbound/theory_validation/val_frontier.py:67,72``;
``val_benefit_frontier.py:50`` takes the ``abs(M) <= beta`` branch as
"unknowable").  So a certificate whose lower bound is exactly zero ABSTAINs, and
a certificate with ``epsilon = +inf`` (the small-``n`` infeasible conformal case)
always ABSTAINs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from kga._validation import as_float_array

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from kga.certificate import Certificate


class Decision(str, Enum):
    """Trichotomy outcome of the KGA gate.

    Subclasses :class:`str` so that members compare equal to their string value
    (``Decision.ADAPT == "ADAPT"``) and serialise cleanly to JSON.
    """

    ADAPT = "ADAPT"
    FREEZE = "FREEZE"
    ABSTAIN = "ABSTAIN"


def decide(certificate: Certificate, alpha: float | None = None) -> Decision:
    """Apply the K-Bound trichotomy to a finite-sample certificate.

    Parameters
    ----------
    certificate : Certificate
        A certificate carrying ``delta_hat`` (estimated benefit of adapting over
        freezing) and ``epsilon`` (the one-sided/symmetric confidence radius at
        level ``certificate.alpha``).
    alpha : float, optional
        If given, asserts that the certificate was computed at this level.  The
        decision rule itself does not re-use ``alpha`` (the guarantee is baked
        into ``epsilon``); this argument exists only to make the operating level
        explicit at the call site and to catch accidental mismatches.

    Returns
    -------
    Decision
        ``Decision.ADAPT``  if ``delta_hat - epsilon > 0``,
        ``Decision.FREEZE`` if ``delta_hat + epsilon < 0``,
        ``Decision.ABSTAIN`` otherwise.

    Raises
    ------
    ValueError
        If ``epsilon`` is negative or NaN, if ``delta_hat`` is nonfinite,
        if the level or sample count is invalid, or if ``alpha`` is supplied
        and disagrees with ``certificate.alpha``. A ``+inf`` radius represents
        an unavailable finite certificate and returns ABSTAIN.

    Notes
    -----
    Boundaries are decided with *strict* inequalities, so a certificate whose
    lower bound is exactly zero (``delta_hat == epsilon``) ABSTAINs rather than
    ADAPTs.  This is the conservative choice and matches the reference scripts.

    Examples
    --------
    >>> from kga.certificate import Certificate
    >>> decide(Certificate(delta_hat=0.15, epsilon=0.05, method="ebern",
    ...                     alpha=0.1, n=100))
    <Decision.ADAPT: 'ADAPT'>
    >>> decide(Certificate(delta_hat=-0.15, epsilon=0.05, method="ebern",
    ...                     alpha=0.1, n=100))
    <Decision.FREEZE: 'FREEZE'>
    >>> decide(Certificate(delta_hat=0.03, epsilon=0.10, method="ebern",
    ...                     alpha=0.1, n=100))
    <Decision.ABSTAIN: 'ABSTAIN'>
    """
    delta_hat = float(certificate.delta_hat)
    epsilon = float(certificate.epsilon)
    if not math.isfinite(delta_hat):
        raise ValueError(f"delta_hat must be finite, got {delta_hat}")
    if math.isnan(epsilon):
        raise ValueError("epsilon must not be NaN")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if not (0.0 < float(certificate.alpha) < 1.0):
        raise ValueError("certificate.alpha must be finite and in (0, 1)")
    if not isinstance(certificate.n, (int, np.integer)) or isinstance(certificate.n, bool) or certificate.n < 1:
        raise ValueError("certificate.n must be a positive integer")
    if alpha is not None and not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be finite and in (0, 1)")
    if alpha is not None and abs(alpha - float(certificate.alpha)) > 1e-12:
        raise ValueError(
            f"alpha={alpha} does not match certificate.alpha={certificate.alpha}; "
            "compute the certificate at the level you intend to decide at"
        )

    # An infinite radius is the maximally-uncertain certificate (e.g. a single
    # sample with no variance estimate): it brackets all of R, so ABSTAIN.
    if math.isinf(epsilon):
        return Decision.ABSTAIN
    if delta_hat - epsilon > 0.0:
        return Decision.ADAPT
    if delta_hat + epsilon < 0.0:
        return Decision.FREEZE
    return Decision.ABSTAIN


# ---------------------------------------------------------------------------
# Historical stored-prediction controlled-grid replay compatibility
# ---------------------------------------------------------------------------
#: Calibration conventions accepted by :func:`decide_kga`.
CALIBRATIONS = ("loo", "in_pool")


def decide_batch(delta_hat, epsilon, *, alpha: float = 0.1) -> np.ndarray:
    """Vectorised :func:`decide`: arrays in, array of decision strings out.

    Parameters
    ----------
    delta_hat : array-like
        Per-cell benefit point estimates ``Delta_hat_i``.
    epsilon : float or array-like
        Confidence radius.  A scalar is broadcast; a per-cell array (what
        :func:`kga.certificate.conformal_radii_loo` returns) is used elementwise.
        Non-finite or masked entries yield ABSTAIN.
    alpha : float, default=0.1
        The level ``epsilon`` was computed at.  Recorded on the intermediate
        certificates so a mismatched level is caught, exactly as in
        :func:`decide`.

    Returns
    -------
    numpy.ndarray of str
        ``'ADAPT'`` / ``'FREEZE'`` / ``'ABSTAIN'``, same shape as ``delta_hat``.
        Plain ``str`` rather than :class:`Decision` because that is what every
        artifact schema in this repository stores.

    Examples
    --------
    >>> decide_batch([0.2, -0.2, 0.0], 0.1).tolist()
    ['ADAPT', 'FREEZE', 'ABSTAIN']
    >>> decide_batch([0.2], float('inf')).tolist()
    ['ABSTAIN']
    """
    from kga.certificate import Certificate

    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    bh = as_float_array(delta_hat)
    eps = np.broadcast_to(as_float_array(epsilon), bh.shape)
    flat = np.empty(bh.size, dtype=object)
    for i in range(bh.size):
        # A missing estimate/radius is unavailable evidence, not a negative
        # interval. Keep scalar validation strict while honoring this batch
        # API's documented per-cell ABSTAIN contract.
        if not math.isfinite(float(bh.flat[i])) or not math.isfinite(float(eps.flat[i])):
            flat[i] = Decision.ABSTAIN.value
            continue
        cert = Certificate(
            delta_hat=float(bh.flat[i]),
            epsilon=float(eps.flat[i]),
            method="conformal",
            alpha=float(alpha),
            n=int(bh.size),
        )
        flat[i] = decide(cert, alpha=alpha).value
    return flat.reshape(bh.shape)


def decide_kga(
    delta_hat,
    benefit,
    *,
    alpha: float = 0.1,
    calibration: str = "loo",
):
    """Historical controlled-grid replay: stored ``b_hat``/``B`` in, decisions out.

    New controlled-grid results must use
    :func:`kga.crossfit.controlled_grid_crossfit`, which refits from ``Z`` and
    ``B`` with the scored outcome absent from both estimator fitting and
    residual calibration.  This compatibility function implements exactly one
    stored-prediction radius rule.  It has one switch, ``calibration``, and
    that switch is **not** a statistical option: its non-default value replays an
    archived pre-fix artifact and is forbidden for a new number (see below).
    There is no clamp mode and no interpolated mode, at any setting:

    1. **Radius** -- the exact split-conformal *rank* quantile
       ``eps = r_(k)``, ``k = ceil((n + 1)(1 - alpha))``, over the absolute
       residuals ``r_i = |delta_hat_i - benefit_i|``
       (:func:`kga.certificate.split_conformal_rank_radius`).  Interpolated
       ``np.quantile`` is never used.  When ``k > n`` the radius is ``+inf``
       and every cell ABSTAINs (fix-queue item 25).
    2. **Pool** -- cell ``i``'s radius is calibrated on the *other* ``n - 1``
       residuals only (:func:`kga.certificate.conformal_radii_loo`), removing
       direct self-inclusion (fix-queue item 4). This does not prove residual
       exchangeability across correlated or heterogeneous conditions.
    3. **Trichotomy** -- ADAPT iff ``delta_hat_i - eps_i > 0``; FREEZE iff
       ``delta_hat_i + eps_i < 0``; otherwise ABSTAIN.  Strict inequalities, so
       a lower bound of exactly zero ABSTAINs, matching the ``|M| > beta``
       commitment rule of the knowability frontier.

    The certificate criterion applies conditionally: if the interval has the stated
    marginal coverage, ADAPT implies positive benefit on the coverage event.
    The leave-one-condition-out grid construction itself is an empirical
    calibration design, not a proof of exact split-conformal coverage.

    Parameters
    ----------
    delta_hat : array-like of shape (n,)
        Per-cell benefit point estimates, e.g. the leave-one-cell-out
        gradient-boosted ``b_hat`` stored in the per-condition dumps.
    benefit : array-like of shape (n,)
        The realised per-cell benefits ``B_i = acc(f_a) - acc(f_0)`` used to form
        calibration residuals.  Used for the radius only, never for the decision.
    alpha : float, default=0.1
        Miscoverage level.
    calibration : {'loo', 'in_pool'}, default='loo'
        ``'loo'`` is the rule.  ``'in_pool'`` reproduces the archived, leaky
        pre-fix radius (one radius over all ``n`` residuals, including the scored
        cell's own) and exists only so a historical artifact can be replayed;
        it must not be used for a new number.  It changes the *pool*, never the
        rank rule.

    There is deliberately no ``on_infeasible`` argument.  When the calibration
    pool is too small for ``alpha`` (``k > n``, i.e. ``n <= 8`` at
    ``alpha = 0.10``; under ``'loo'`` that means ``n <= 9`` cells) the radius is
    ``+inf``, a :class:`UserWarning` is emitted, and every cell ABSTAINs.  The
    superseded clamped radius is not reachable from here at any setting.

    Returns
    -------
    (epsilon, decisions) : (numpy.ndarray, numpy.ndarray)
        ``epsilon`` has one radius **per cell** (it is not a scalar under
        ``'loo'``); ``decisions`` are ``'ADAPT'``/``'FREEZE'``/``'ABSTAIN'``
        strings.  A caller that needs one number for a table must report
        ``float(np.mean(epsilon))`` and label it a mean.

    Raises
    ------
    ValueError
        If the two arrays differ in length, or ``calibration`` is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> b = np.linspace(-0.2, 0.2, 40)
    >>> eps, dec = decide_kga(b, b)          # perfect estimator -> zero residuals
    >>> set(dec.tolist()) <= {'ADAPT', 'FREEZE', 'ABSTAIN'}
    True
    """
    from kga.certificate import conformal_radii_loo, split_conformal_rank_radius

    if calibration not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {sorted(CALIBRATIONS)}, got {calibration!r}")
    bh = as_float_array(delta_hat).ravel()
    bt = as_float_array(benefit).ravel()
    if bh.shape != bt.shape:
        raise ValueError(f"delta_hat and benefit must have the same length, got {bh.size} and {bt.size}")
    residuals = np.abs(bh - bt)
    if calibration == "loo":
        eps = conformal_radii_loo(residuals, alpha)
    else:
        eps = np.full(bh.size, split_conformal_rank_radius(residuals, alpha), dtype=float)
    return eps, decide_batch(bh, eps, alpha=alpha)


@dataclass(frozen=True)
class HierarchicalSelection:
    """Outcome of evaluating multiple adaptation candidates (e.g. norm, head, full)."""

    decision: Decision
    selected_candidate: str | None
    certified_lower_bound: float
    margin_gap: float
    candidate_bounds: dict[str, tuple[float, float]]


def decide_hierarchical_candidates(
    candidates: dict[str, Certificate],
    alpha: float = 0.10,
    correction: str = "bonferroni",
) -> HierarchicalSelection:
    """Evaluate multiple adaptation candidates with family-wise Type-I error control.

    Under Bonferroni correction over K candidates, each certificate is evaluated
    at effective level alpha / K, ensuring overall false-adapt rate <= alpha.
    Selects the candidate with the greatest certified positive lower bound.
    """
    if not candidates:
        return HierarchicalSelection(
            decision=Decision.FREEZE,
            selected_candidate=None,
            certified_lower_bound=0.0,
            margin_gap=0.0,
            candidate_bounds={},
        )

    k = len(candidates)
    effective_alpha = alpha / k if correction == "bonferroni" else alpha

    bounds: dict[str, tuple[float, float]] = {}
    beneficial_candidates: list[tuple[str, float, float]] = []

    for name, cert in candidates.items():
        delta_hat = float(cert.delta_hat)
        eps = float(cert.epsilon)
        lower = delta_hat - eps
        upper = delta_hat + eps
        bounds[name] = (lower, upper)
        if lower > 0.0:
            beneficial_candidates.append((name, lower, delta_hat))

    if beneficial_candidates:
        beneficial_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_name, best_lower, best_delta = beneficial_candidates[0]
        return HierarchicalSelection(
            decision=Decision.ADAPT,
            selected_candidate=best_name,
            certified_lower_bound=best_lower,
            margin_gap=best_lower,
            candidate_bounds=bounds,
        )

    all_strictly_harmful = all(up < 0.0 for _, up in bounds.values())
    if all_strictly_harmful:
        return HierarchicalSelection(
            decision=Decision.FREEZE,
            selected_candidate=None,
            certified_lower_bound=max(low for low, _ in bounds.values()),
            margin_gap=min(abs(up) for _, up in bounds.values()),
            candidate_bounds=bounds,
        )

    return HierarchicalSelection(
        decision=Decision.ABSTAIN,
        selected_candidate=None,
        certified_lower_bound=0.0,
        margin_gap=0.0,
        candidate_bounds=bounds,
    )
