"""kga.certificate -- finite-sample certificates Delta_hat +/- epsilon for KGA.

A *certificate* answers the question "by how much does adapting beat freezing,
and how sure are we?".  Concretely it carries

    Delta_hat   estimated benefit  Delta = R(f0) - R(fa)  (>0 means adapt helps),
    epsilon     a confidence radius at level ``alpha``,

Feeding the certificate to :func:`kga.policy.decide` yields the
ADAPT/FREEZE/ABSTAIN trichotomy. The conditional certificate criterion gives
false-adapt ``<= alpha`` when the bound covers its declared target under the
stated assumptions.

Sidedness of the radius (read this before quoting an interval)
--------------------------------------------------------------
:func:`empirical_bernstein` and :func:`hoeffding` return **one-sided** radii:
each satisfies, separately,

    Pr[ Delta >= Delta_hat - epsilon ] >= 1 - alpha        (lower side)
    Pr[ Delta <= Delta_hat + epsilon ] >= 1 - alpha        (upper side)

but *not* both simultaneously at ``1 - alpha``.  By a union bound the two-sided
event ``|Delta - Delta_hat| <= epsilon`` holds only at ``1 - 2 alpha``.  This is
exactly what the trichotomy needs and no more: the ADAPT branch only ever uses
the lower side and the FREEZE branch only ever uses the upper side, so
``FA_u <= alpha`` and ``FF_u <= alpha`` each follow from a *single* one-sided
statement.  Any figure, table or sentence that presents ``[lower, upper]`` as a
simultaneous interval must label it ``1 - 2 alpha``.  (Panel finding F2-12 /
F1-12; the historical companion fix restates the certificate criterion as two
one-sided coverage conditions rather than one two-sided one.)

:func:`conformal_split` is the exception: its radius is a two-sided
order statistic of ``|Delta_hat - Delta|``, so ``Pr[|Delta - Delta_hat| <=
epsilon] >= 1 - alpha`` holds simultaneously for that estimator.

This module provides four estimators, each mirroring a piece of the paper's
canonical code:

* :func:`empirical_bernstein` -- Maurer & Pontil (2009) empirical-Bernstein LCB,
  the default batch certificate, identical to
  ``vendored_from_elara/certification/switching_certificate.py::
  empirical_bernstein_lcb`` and ``kbound_pkg/kbound/certificate.py``.
* :func:`hoeffding` -- the distribution-free Hoeffding LCB (looser baseline).
* :func:`conformal_split` -- the exact split-conformal order-statistic radius
  over ``|Delta_hat - Delta|`` calibration residuals, the
  cross-task estimator used in ``knowability_experiment.py`` /
  ``mixed_regime_experiment.py``.
* :func:`evalue_anytime` -- the specified bounded-stream testing-by-betting
  e-process (Ville's inequality), mirroring the historical validation in
  ``experiments/kbound/theory_validation/val_thm3_evalue.py`` and
  ``kbound_pkg/kbound/eprocess.py``.

All estimators are pure ``numpy``/``math``, deterministic, and torch-free.

Provenance / attribution
------------------------
The empirical-Bernstein switching certificate (:func:`empirical_bernstein`,
Maurer & Pontil 2009) is **shared with the companion ELARA work**: the identical
Maurer-Pontil lower-confidence-bound underlies ELARA's Phase-2
``switching_certificate``, which now *delegates* to this ``kga`` implementation as
the single source of truth.  It is reproduced in the K-Bound tree so the package
is fully self-contained, and the shared lineage is acknowledged openly rather than
hidden.  K-Bound imports nothing from ``src/elara``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from kga._validation import as_float_array

#: Allowed estimator identifiers (the ``method`` field of a :class:`Certificate`).
METHODS = ("ebern", "hoeffding", "conformal", "evalue")


class InsufficientCalibrationError(ValueError):
    """Raised when no finite conformal radius can attain the requested level.

    The exact split-conformal rank rule needs ``k = ceil((n + 1) * (1 - alpha))``
    calibration residuals and only has ``n``.  When ``k > n`` -- equivalently
    ``n < 1/alpha - 1`` -- *every* finite radius under-covers, and the only
    honest output is ``+inf`` (forced ABSTAIN).  See
    :func:`min_calibration_size`.
    """


@dataclass(frozen=True)
class Certificate:
    """A finite-sample certificate for the benefit of adapting over freezing.

    Attributes
    ----------
    delta_hat : float
        Point estimate of the benefit ``Delta = R(f0) - R(fa)``.  Positive means
        adapting reduces risk (helps); negative means it hurts.
    epsilon : float
        Confidence radius at level ``alpha``.  The certified lower bound is
        ``delta_hat - epsilon`` and the certified upper bound is
        ``delta_hat + epsilon``.  For ``method in {'ebern', 'hoeffding'}`` each
        bound holds *separately* at ``1 - alpha``; the two together hold only at
        ``1 - 2 alpha`` (see the module docstring).
    method : str
        Estimator identifier, one of :data:`METHODS`.
    alpha : float
        Miscoverage level in ``(0, 1)`` at which ``epsilon`` was computed.
    n : int
        Effective sample size the certificate was built from (number of paired
        benefits, calibration residuals, or observed e-process steps).
    """

    delta_hat: float
    epsilon: float
    method: str
    alpha: float
    n: int

    @property
    def lower(self) -> float:
        """One-sided certified lower bound ``delta_hat - epsilon`` at ``1 - alpha``.

        This is the only quantity the ADAPT branch consults.  Do not pair it
        with :attr:`upper` and call the result a ``1 - alpha`` interval for the
        ``ebern``/``hoeffding`` methods -- that pairing is ``1 - 2 alpha``.
        """
        return self.delta_hat - self.epsilon

    @property
    def upper(self) -> float:
        """One-sided certified upper bound ``delta_hat + epsilon`` at ``1 - alpha``.

        This is the only quantity the FREEZE branch consults.  See :attr:`lower`
        for the sidedness caveat.
        """
        return self.delta_hat + self.epsilon

    @property
    def interval_level(self) -> float | None:
        """Simultaneous interval coverage, when this is an interval object.

        The e-value path encodes a directional threshold crossing in the common
        certificate container; it is not a confidence interval.
        """
        if self.method == "conformal":
            return 1.0 - self.alpha
        if self.method == "evalue":
            return None
        return 1.0 - 2.0 * self.alpha


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _check_benefit_range(benefit_range: float, caller: str) -> float:
    """Validate the *a priori* benefit range ``R = b - a``.

    ``benefit_range`` is required (no data-estimated fallback): Maurer-Pontil and
    Hoeffding both need ``R`` to be fixed before the sample is seen, and the
    observed ``max - min`` is a downward-biased estimate of it, which makes the
    radius anti-conservative.  Panel findings F1-12 / F2-12 and the verifier's
    note that the identical defect sat in :func:`hoeffding`.
    """
    if benefit_range is None:
        raise ValueError(
            f"{caller}: benefit_range is required and must be an a-priori bound on "
            "the benefit support (b - a). It used to default to the observed "
            "max - min, which is data-dependent and voids the finite-sample "
            "guarantee. For |p - y| paired 0/1 losses pass benefit_range=2.0."
        )
    rng = float(benefit_range)
    if not math.isfinite(rng) or rng <= 0.0:
        raise ValueError(f"{caller}: benefit_range must be finite and > 0, got {benefit_range!r}")
    return rng


def _as_1d(x: np.ndarray, name: str, *, allow_empty: bool = False) -> np.ndarray:
    arr = as_float_array(x).ravel()
    if arr.size == 0 and not allow_empty:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


# ---------------------------------------------------------------------------
# (1) Empirical-Bernstein  (the default batch certificate)
# ---------------------------------------------------------------------------
# PROVENANCE (integrity pass 2026-06-20): the empirical-Bernstein certificate below
# is shared with the ELARA companion work (same Maurer-Pontil 2009 LCB). ELARA's
# certification.switching_certificate delegates to this function; it is vendored here
# so the K-Bound package runs with zero dependency on src/elara. Honest attribution.
def empirical_bernstein(
    paired_benefits: np.ndarray,
    *,
    alpha: float = 0.1,
    benefit_range: float,
) -> Certificate:
    """Maurer-Pontil (2009) empirical-Bernstein batch certificate.

    For i.i.d. paired benefits ``X_i = loss(f0_i) - loss(fa_i)`` in ``[a, b]``
    with empirical mean ``mu_hat`` and unbiased sample variance ``V_hat``, with
    probability at least ``1 - alpha``::

        E[X] >= mu_hat - sqrt( 2 V_hat ln(2/alpha) / n )
                       - 7 R ln(2/alpha) / ( 3 (n - 1) ),     R = b - a.

    The estimate is ``delta_hat = mu_hat`` and the radius is the sum of the two
    subtracted terms, so ``delta_hat - epsilon`` reproduces the LCB above.  This
    is tighter than Hoeffding whenever ``V_hat << R^2`` (the usual regime, where
    most paired benefits agree in sign), and is deterministic, streamable, and
    finite-sample valid.

    The radius is **one-sided at ``alpha``** (note the ``ln(2/alpha)``: the 2 pays
    for the union of the variance and range deviations *within* one side, not for
    two sides).  ``[lower, upper]`` is therefore a ``1 - 2 alpha`` interval; see
    the module docstring.

    Parameters
    ----------
    paired_benefits : array-like of shape (n,)
        Per-sample benefits ``X_i`` in a bounded interval.
    alpha : float, default=0.1
        Miscoverage level.  ``P(E[X] >= delta_hat - epsilon) >= 1 - alpha``.
    benefit_range : float, **required**, keyword-only
        The *a priori* range ``R = b - a`` of the benefit support.  Maurer-Pontil
        requires ``R`` to be known before seeing the data; substituting the
        observed ``max - min`` (which this function used to do by default) makes
        the range term data-dependent and under-estimates ``R``, so the stated
        ``1 - alpha`` guarantee no longer holds.  There is no safe default, so
        there is no default: for ``|p - y|`` paired 0/1 losses pass ``2.0``.
        Panel findings F1-12 / F2-12.

    Returns
    -------
    Certificate
        With ``method='ebern'``.

    Raises
    ------
    TypeError
        If ``benefit_range`` is omitted.
    ValueError
        If ``benefit_range`` is not a strictly positive finite float.

    References
    ----------
    Maurer & Pontil, "Empirical Bernstein Bounds and Sample-Variance
    Penalization" (COLT 2009).  Mirrors
    ``switching_certificate.py::empirical_bernstein_lcb``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.full(500, 0.2)
    >>> cert = empirical_bernstein(x, alpha=0.05, benefit_range=2.0)
    >>> cert.lower < cert.delta_hat
    True
    """
    _check_alpha(alpha)
    rng = _check_benefit_range(benefit_range, "empirical_bernstein")
    arr = _as_1d(paired_benefits, "paired_benefits")
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        # No variance estimate; degenerate radius (infinite -> always ABSTAIN).
        return Certificate(delta_hat=mean, epsilon=float("inf"), method="ebern", alpha=alpha, n=n)
    var = float(arr.var(ddof=1))

    ln_term = math.log(2.0 / alpha)
    var_term = math.sqrt(2.0 * var * ln_term / n)
    range_term = 7.0 * rng * ln_term / (3.0 * (n - 1))
    epsilon = var_term + range_term
    return Certificate(delta_hat=mean, epsilon=epsilon, method="ebern", alpha=alpha, n=n)


# ---------------------------------------------------------------------------
# (2) Hoeffding  (distribution-free baseline)
# ---------------------------------------------------------------------------
def hoeffding(
    paired_benefits: np.ndarray,
    *,
    alpha: float = 0.1,
    benefit_range: float,
) -> Certificate:
    """Hoeffding certificate -- distribution-free LCB on the mean benefit.

    For i.i.d. ``X_i`` in ``[a, b]``, Hoeffding's inequality gives, with
    probability at least ``1 - alpha`` (one-sided)::

        E[X] >= mu_hat - R sqrt( ln(1/alpha) / (2 n) ),     R = b - a.

    This variance-free radius is always at least as large as the
    empirical-Bernstein one in the small-variance regime, so it is provided
    mainly as a conservative baseline / sanity comparator.

    The ``ln(1/alpha)`` makes this an explicitly **one-sided** radius at
    ``alpha``; pairing ``lower`` and ``upper`` gives a ``1 - 2 alpha`` interval.

    Parameters
    ----------
    paired_benefits : array-like of shape (n,)
    alpha : float, default=0.1
    benefit_range : float, **required**, keyword-only
        The *a priori* range ``R = b - a``.  As in
        :func:`empirical_bernstein`, the data-estimated fallback that used to be
        the default voids the guarantee, so it has been removed.  Pass ``2.0``
        for ``|p - y|`` paired 0/1 losses.

    Returns
    -------
    Certificate
        With ``method='hoeffding'``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0.0, 0.4, 200)
    >>> cert = hoeffding(x, alpha=0.1, benefit_range=2.0)
    >>> cert.epsilon > 0
    True
    """
    _check_alpha(alpha)
    rng = _check_benefit_range(benefit_range, "hoeffding")
    arr = _as_1d(paired_benefits, "paired_benefits")
    n = arr.size
    mean = float(arr.mean())
    epsilon = rng * math.sqrt(math.log(1.0 / alpha) / (2.0 * n))
    return Certificate(delta_hat=mean, epsilon=epsilon, method="hoeffding", alpha=alpha, n=n)


# ---------------------------------------------------------------------------
# (3) Split-conformal  (cross-task estimator used in the main experiments)
# ---------------------------------------------------------------------------
def min_calibration_size(alpha: float) -> int:
    """Smallest ``n`` for which a *finite* conformal radius attains ``1 - alpha``.

    The exact-rank radius is ``r_(k)`` with ``k = ceil((n + 1) * (1 - alpha))``.
    A finite radius exists iff ``k <= n``, i.e. iff ``n + 1 >= 1/alpha``.  So the
    feasibility threshold is ``n >= ceil(1/alpha) - 1``; at ``alpha = 0.10`` that
    is ``n >= 9``, and at ``alpha = 0.02`` (the Bonferroni level of a 5-candidate
    panel at ``alpha = 0.10``) it is ``n >= 49``.

    Examples
    --------
    >>> min_calibration_size(0.1)
    9
    >>> min_calibration_size(0.02)
    49
    """
    _check_alpha(alpha)
    return int(math.ceil(1.0 / alpha)) - 1


def conformal_attained_level(n: int, alpha: float) -> float:
    """Coverage attainable at pool size ``n`` -- the *diagnostic*, not the rule.

    Returns ``min(n, k) / (n + 1)`` with ``k = ceil((n + 1)(1 - alpha))``.  For
    ``n >= min_calibration_size(alpha)`` this is ``k / (n + 1) >= 1 - alpha``, the
    level the shipped rule actually attains.  For smaller ``n`` it is
    ``n / (n + 1) < 1 - alpha``: the *best any finite radius could do*, which is
    why :func:`split_conformal_rank_radius` refuses to return one and yields
    ``+inf`` instead (panel finding F1-5 / F2-7).

    The ``min(n, .)`` here is a **ceiling on attainable coverage**, not a radius
    clamp: nothing in this module selects an order statistic with it.  See
    :func:`legacy_clamped_radius` for the superseded rule that did.
    """
    _check_alpha(alpha)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    return k / (n + 1)


def legacy_clamped_radius(calib_residuals: np.ndarray, alpha: float = 0.1) -> float:
    """SUPERSEDED, under-covering rule ``r_(min(n, k))``. Never call for a new number.

    This reproduces the pre-fix radius that shipped before fix-queue item 25:
    ``k = min(n, ceil((n + 1)(1 - alpha)))``, i.e. the maximum residual whenever
    ``k > n``.  Its attained coverage is ``n / (n + 1) < 1 - alpha``, so the
    ``1 - alpha`` promise in the old docstring was false.

    It is kept as a *separate, explicitly named* function -- not as a mode of
    :func:`split_conformal_rank_radius` -- for exactly two purposes:

    1. replaying an archived pre-fix artifact byte-for-byte, and
    2. keeping the Monte-Carlo under-coverage regression test honest
       (``tests/test_kga_canonical_rule.py::TestSmallNInfeasible``).

    It is unreachable from :func:`kga.policy.decide_kga`, from
    :func:`conformal_split` and from every driver: the canonical path has no
    argument that can select it.

    Examples
    --------
    >>> import numpy as np
    >>> legacy_clamped_radius(np.arange(1.0, 9.0), 0.1)   # n = 8, k = 9 -> r_(8)
    8.0
    """
    _check_alpha(alpha)
    arr = _as_1d(calib_residuals, "calib_residuals")
    if np.any(arr < 0.0):
        raise ValueError("calib_residuals must be non-negative (they are |Delta_hat - Delta|)")
    n = arr.size
    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    return float(np.sort(arr)[k - 1])


def split_conformal_rank_radius(
    calib_residuals: np.ndarray,
    alpha: float = 0.1,
    *,
    on_infeasible: str = "inf",
) -> float:
    """Return the exact finite-sample split-conformal residual radius.

    For sorted residuals ``r_(1) <= ... <= r_(n)`` this uses the **exact rank**

        k = ceil((n + 1) * (1 - alpha)),      epsilon = r_(k)

    and returns ``r_(k)``.  ``numpy.quantile``'s default linear interpolation is
    never used: it is not an observed order statistic and does not satisfy the
    finite-sample rank argument, and mixing the two rules inside one claim is
    panel finding F1-2 / F4-10 / F5-2.

    Small ``n`` (panel finding F1-5 / F2-7)
    ---------------------------------------
    When ``k > n`` -- equivalently ``n < min_calibration_size(alpha)``, i.e.
    ``n <= 8`` at ``alpha = 0.10`` -- **no finite radius attains ``1 - alpha``**.
    The maximum residual, which this function used to return via a
    ``k = min(n, ...)`` clamp, attains only ``n / (n + 1) < 1 - alpha`` while the
    docstring promised ``1 - alpha``.  **There is no clamp mode any more**: the
    function returns ``+inf`` (or raises), which :func:`kga.policy.decide` turns
    into ABSTAIN -- the honest answer, and the one three sibling scripts in this
    repo (``ablation_exactrank.py`` and ``official_baselines_headtohead.py``)
    and the project's maintained claim validator
    (``src/scripts/validate_manuscript_claims.py``) and theory validator
    (``theory_v2/val_multicandidate.py:93-98``) already give.  The superseded
    clamped value is available only from the separately named
    :func:`legacy_clamped_radius`, which no decision path calls.

    Parameters
    ----------
    calib_residuals : array-like of shape (n,)
        Non-negative residuals ``|Delta_hat_i - Delta_i|``.
    alpha : float, default=0.1
        Miscoverage level.
    on_infeasible : {'inf', 'raise'}, default='inf'
        What to do when ``k > n``.  Both branches refuse to certify; they differ
        only in whether the refusal is a value or an exception.

        * ``'inf'``   -- return ``+inf`` (forced ABSTAIN) and emit a
          :class:`UserWarning`.  This is the default and the correct behaviour.
        * ``'raise'`` -- raise :class:`InsufficientCalibrationError`.

    Examples
    --------
    >>> import numpy as np
    >>> split_conformal_rank_radius(np.arange(1.0, 10.0), 0.1)   # n = 9, k = 9
    9.0
    """
    _check_alpha(alpha)
    if on_infeasible not in ("inf", "raise"):
        raise ValueError(
            f"on_infeasible must be 'inf' or 'raise', got {on_infeasible!r}. "
            "The 'clamp' mode was removed (fix-queue item 25): the clamped radius "
            "under-covers, and it is now only reachable through the explicitly named "
            "kga.certificate.legacy_clamped_radius()."
        )
    # Zero residuals is also an infeasible rank. In particular this occurs in
    # one-cell LOO replay; it must not select a finite radius or a strict sign.
    arr = _as_1d(calib_residuals, "calib_residuals", allow_empty=True)
    if np.any(arr < 0.0):
        raise ValueError("calib_residuals must be non-negative (they are |Delta_hat - Delta|)")
    n = arr.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        n_min = min_calibration_size(alpha)
        msg = (
            f"split-conformal at alpha={alpha} needs n >= {n_min} calibration "
            f"residuals but got n={n}: the exact rank k={k} exceeds n, so no "
            f"finite radius attains 1-alpha (the best attainable coverage is "
            f"n/(n+1) = {n / (n + 1):.4f}). Returning +inf => ABSTAIN."
        )
        if on_infeasible == "raise":
            raise InsufficientCalibrationError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return float("inf")
    return float(np.sort(arr)[k - 1])


def conformal_radii_loo(
    calib_residuals: np.ndarray,
    alpha: float = 0.1,
    *,
    on_infeasible: str = "inf",
) -> np.ndarray:
    """Leave-one-out-of-pool radii: cell ``i``'s radius excludes cell ``i``.

    Returns an array ``eps`` of the same length as ``calib_residuals``, where
    ``eps[i] = split_conformal_rank_radius(delete(residuals, i), alpha)``.

    This is fix-queue item 4.  Every shipped runner used to compute one radius
    over *all* ``N`` residuals -- including the residual of the very cell being
    scored -- which makes ``epsilon`` a function of the test labels that the
    ``FA_u <= alpha`` guarantee attaches to. Excluding the scored index removes
    direct self-inclusion. It does **not** establish exchangeability when cells
    are correlated, condition dependent, or scored by different fitted models.
    Controlled-grid use is therefore empirical leave-one-condition-out residual
    calibration, not exact split conformal or jackknife+.

    Note the knock-on effect at small ``n``: a pool of ``n`` residuals yields
    LOO pools of ``n - 1``, so feasibility now requires
    ``n >= min_calibration_size(alpha) + 1`` (``n >= 10`` at ``alpha = 0.10``).
    """
    arr = _as_1d(calib_residuals, "calib_residuals")
    n = arr.size
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = split_conformal_rank_radius(np.delete(arr, i), alpha, on_infeasible=on_infeasible)
    return out


def conformal_split(
    delta_hat: float,
    calib_residuals: np.ndarray,
    *,
    alpha: float = 0.1,
    on_infeasible: str = "inf",
) -> Certificate:
    """Split-conformal certificate around a benefit point estimate.

    Given calibration residuals ``r_i = |Delta_hat_i - Delta_i|`` from held-out
    tasks/instances -- the query instance's own residual **must not** be in the
    pool -- the radius

        k = ceil((n + 1) * (1 - alpha))
        epsilon = r_(k)                    (+inf when k > n)

    guarantees that a fresh ``Delta_hat`` deviates from the true ``Delta`` by at
    most ``epsilon`` with probability at least ``1 - alpha`` over the exchangeable
    calibration split (split-conformal coverage). The returned radius is an
    observed residual order statistic; it does not use interpolated quantiles.
    Because it bounds ``|Delta_hat - Delta|``, this is the one *genuinely
    two-sided* estimator in the module.

    Parameters
    ----------
    delta_hat : float
        Point estimate of the benefit for the query instance (e.g. from a
        leave-one-out / cross-fitted benefit regressor).
    calib_residuals : array-like of shape (n,)
        Absolute prediction errors ``|Delta_hat_i - Delta_i|`` on the
        calibration set.  Must be non-negative and must exclude the query.
    alpha : float, default=0.1
        Miscoverage level.
    on_infeasible : {'inf', 'raise'}, default='inf'
        Behaviour when ``n < min_calibration_size(alpha)``; see
        :func:`split_conformal_rank_radius`.  There is no clamp mode.

    Returns
    -------
    Certificate
        With ``method='conformal'`` and ``n`` = number of calibration residuals.
        ``epsilon`` is ``+inf`` (hence ABSTAIN) when ``n`` is too small.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.abs(np.random.default_rng(0).standard_normal(200))
    >>> cert = conformal_split(0.2, r, alpha=0.1)
    >>> k = int(np.ceil((len(r) + 1) * 0.9))
    >>> cert.epsilon == float(np.sort(r)[k - 1])
    True
    """
    _check_alpha(alpha)
    if not math.isfinite(float(delta_hat)):
        raise ValueError(f"delta_hat must be finite, got {delta_hat}")
    arr = _as_1d(calib_residuals, "calib_residuals")
    epsilon = split_conformal_rank_radius(arr, alpha, on_infeasible=on_infeasible)
    return Certificate(delta_hat=float(delta_hat), epsilon=epsilon, method="conformal", alpha=alpha, n=arr.size)


# ---------------------------------------------------------------------------
# (4) Anytime-valid e-value  (Ville / testing-by-betting)
# ---------------------------------------------------------------------------
def evalue_anytime(
    paired_benefits: np.ndarray,
    *,
    alpha: float = 0.1,
    a: float = -1.0,
    b: float = 1.0,
    bet_cap_frac: float = 0.5,
    prior_var: float = 0.25,
    prior_weight: float = 1.0,
) -> Certificate:
    """Bounded-stream e-value certificate using testing-by-betting.

    Runs the two one-sided betting e-processes of the historical validation
    ``val_thm3_evalue.py`` over the supplied benefit stream::

        E_t^+ = prod_{i<=t} (1 + lam_i  X_i)       tests H0 : Delta <= 0,
        E_t^- = prod_{i<=t} (1 + nu_i (-X_i))      tests H0': Delta >= 0,

    with predictable truncated-aGRAPA bets ``lam_i = clip(mu_hat/sigma2_hat, 0,
    bet_cap_frac/|a|)`` (and symmetrically ``nu_i``).  Each process is a
    nonnegative supermartingale under its null, so by **Ville's inequality**
    ``P(exists t : E_t >= 1/alpha) <= alpha`` -- simultaneously over all sample
    sizes, with no multiplicity correction for repeated looks.

    The returned certificate encodes the *current* anytime decision in the same
    ``delta_hat +/- epsilon`` form consumed by :func:`kga.policy.decide`:

    * if ``E_t^+ >= 1/alpha`` (reject ``Delta <= 0``) we emit a positive
      certified lower bound, so ``decide`` returns ADAPT;
    * if ``E_t^- >= 1/alpha`` (reject ``Delta >= 0``) we emit a negative
      certified upper bound, so ``decide`` returns FREEZE;
    * otherwise the radius covers the running mean and ``decide`` ABSTAINs,
      i.e. "keep sampling".

    ``delta_hat`` is the running sample mean of the benefits (a consistent point
    estimate of ``Delta``); only the *sign of the certified bound* carries the
    anytime guarantee.

    Parameters
    ----------
    paired_benefits : array-like of shape (n,)
        Ordered stream of per-sample benefits ``X_i`` in ``[a, b]``. Values
        outside the predeclared support raise; they are never silently clipped.
    alpha : float, default=0.1
        Anytime type-I error budget.
    a, b : float, default=-1.0, 1.0
        Lower/upper bounds of the benefit range.  ``a`` must be ``< 0`` and
        ``b`` must be ``> 0`` for the two one-sided tests to be well posed.
    bet_cap_frac : float, default=0.5
        Bet cap: ``lam in [0, bet_cap_frac/|a|]``; ``< 1`` guarantees the wealth
        factors stay strictly positive.
    prior_var, prior_weight : float
        Prior pseudo-variance and pseudo-count for the running second moment.

    Returns
    -------
    Certificate
        With ``method='evalue'`` and ``n`` = stream length.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.full(2000, 0.3)            # strongly positive benefit
    >>> cert = evalue_anytime(x, alpha=0.1)
    >>> cert.lower > 0                    # ADAPT certified
    True
    """
    _check_alpha(alpha)
    if not math.isfinite(a) or a >= 0.0:
        raise ValueError(f"a must be finite and < 0 for the one-sided test on H0: Delta<=0; got {a}")
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError(f"b must be finite and > 0 for the one-sided test on H0': Delta>=0; got {b}")
    if not 0.0 < bet_cap_frac < 1.0:
        raise ValueError("bet_cap_frac must be in (0, 1)")
    if not math.isfinite(prior_var) or prior_var <= 0.0:
        raise ValueError("prior_var must be finite and > 0")
    if not math.isfinite(prior_weight) or prior_weight <= 0.0:
        raise ValueError("prior_weight must be finite and > 0")
    arr = _as_1d(paired_benefits, "paired_benefits")
    if np.any(arr < a) or np.any(arr > b):
        raise ValueError(
            f"paired_benefits must lie in the predeclared support [{a}, {b}]; clipping would change the tested mean"
        )
    n = arr.size

    lam_max_plus = bet_cap_frac / (-a)
    lam_max_minus = bet_cap_frac / b
    log_thr = math.log(1.0 / alpha)

    s1 = 0.0  # running sum of X_j (for E^+)
    s1m = 0.0  # running sum of -X_j (for E^-)
    s2 = prior_weight * prior_var  # running second moment + prior (E^+)
    s2m = prior_weight * prior_var  # running second moment + prior (E^-)
    cnt = 0.0
    cnt_var = float(prior_weight)

    log_w_plus = 0.0
    log_w_minus = 0.0

    for x in arr:
        x_value = float(x)
        # Predictable bets from PAST statistics only.
        mu_hat = s1 / cnt if cnt > 0 else 0.0
        mu_hat_m = s1m / cnt if cnt > 0 else 0.0
        sig2 = s2 / cnt_var
        sig2m = s2m / cnt_var
        lam_plus = float(np.clip(mu_hat / sig2 if sig2 > 0 else 0.0, 0.0, lam_max_plus))
        lam_minus = float(np.clip(mu_hat_m / sig2m if sig2m > 0 else 0.0, 0.0, lam_max_minus))

        log_w_plus += math.log(max(1.0 + lam_plus * x_value, 1e-300))
        log_w_minus += math.log(max(1.0 + lam_minus * (-x_value), 1e-300))

        # Update predictable stats AFTER betting (so they remain F_{i-1}-measurable).
        s1 += x_value
        s1m += -x_value
        s2 += x_value**2
        s2m += x_value**2
        cnt += 1.0
        cnt_var += 1.0

    mean = float(arr.mean())
    # Encode the *current* anytime decision in delta_hat +/- epsilon form, so the
    # same kga.policy.decide trichotomy reads it. With epsilon == 0 the decision
    # reduces to sign(delta_hat); we set delta_hat to a certified-sign margin.
    if log_w_plus >= log_thr:
        # E_t^+ crossed: reject H0: Delta <= 0  =>  certified positive lower
        # bound  =>  ADAPT. The running mean may not yet be positive, but the
        # e-process certifies Delta > 0 at anytime level alpha; report a strictly
        # positive certified margin (use the mean if it is already positive).
        delta_hat = mean if mean > 0.0 else 1e-12
        return Certificate(delta_hat=delta_hat, epsilon=0.0, method="evalue", alpha=alpha, n=n)
    if log_w_minus >= log_thr:
        # E_t^- crossed: reject H0': Delta >= 0  =>  certified negative upper
        # bound  =>  FREEZE.
        delta_hat = mean if mean < 0.0 else -1e-12
        return Certificate(delta_hat=delta_hat, epsilon=0.0, method="evalue", alpha=alpha, n=n)

    # Neither threshold crossed: ABSTAIN ("keep sampling"). The radius brackets
    # zero around the running mean so that delta_hat - epsilon <= 0 <= delta_hat
    # + epsilon.
    epsilon = abs(mean) + 1e-12
    return Certificate(delta_hat=mean, epsilon=epsilon, method="evalue", alpha=alpha, n=n)


def worst_group_conformal_radius(group_residuals: list[np.ndarray], alpha: float) -> float:
    """Compute the maximum exact-rank radius across calibration groups.

    eps_robust = max_{g} split_conformal_rank_radius(R_g, alpha)

    This is conservative for the supplied groups. It protects a new group only
    under an additional dominance/transfer assumption that its residual law is
    no heavier-tailed than the worst calibration group.
    """
    if not group_residuals:
        raise ValueError("group_residuals must contain at least one group")
    radii = [split_conformal_rank_radius(res, alpha) for res in group_residuals]
    return float(max(radii))
