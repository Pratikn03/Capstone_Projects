"""kga.evidence -- label-free evidence Z for the knowability gate.

The KGA gate must decide ADAPT/FREEZE/ABSTAIN **without target labels**.  All it
may look at is the label-free evidence ``Z``: statistics computed from the
calibration (source/validation) scores and the unlabelled test (target) scores.
These are exactly the observable quantities used throughout the K-Bound
experiments -- see ``src/scripts/kbound/knowability_experiment.py`` (the ``Z =
dict(...)`` block, lines 74-96) and ``mixed_regime_experiment.py`` (the
``evidence`` helper, lines 47-54).

The signals fall into four families motivated by the identifiability and
coverage arguments:

* **Drift (KS).**  Per-detector Kolmogorov-Smirnov distance between the
  calibration and test score marginals (``scipy.stats.ks_2samp``).  Large drift
  is the *observable* signature of a covariate shift the gate can react to
  (the covariate-shift positive result and disagreement-region analysis).
  When the marginal is preserved but the signal is destroyed (the
  "covert" failure of ``mixed_regime_experiment.py``), KS ~ 0 and the regime is
  unknowable (the hidden-drift construction and observable-fiber criterion).
* **Disagreement.**  ``1 - mean pairwise rank correlation`` of the test scores
  across detectors -- a label-free proxy for the size/heterogeneity of the
  disagreement region ``D`` on which sign-of-difference lives (the
  prediction-disagreement decomposition).
* **Entropy / confidence shift.**  Mean Shannon entropy and mean confidence of
  the score distributions, and their calibration->test drops.  These mirror the
  ``pre/post`` entropy & confidence signals of the deep-TTA evidence vector in
  ``docs/research/kbound/kbound_pkg/kbound/evidence.py``; a large confidence
  *increase* with a collapsing marginal is the classic entropy-collapse warning.
* **Importance weight / ESS.**  A Gaussian-ratio importance weight between the
  calibration and test score laws and its effective sample size
  ``ESS = (sum w)^2 / sum w^2``.  Low ESS means the target is poorly covered by
  the source, so importance-weighted certificates are unreliable (the
  covariate-shift support-overlap condition; cf. the weight-variance/overlap features in
  ``kbound_full_experiments.py::run_regression``).

All functions are pure ``numpy``/``scipy`` and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp, rankdata

from kga._validation import as_float_array

EVIDENCE_SCHEMA_VERSION = "kga-generic-score-evidence/1"
EVIDENCE_FEATURE_NAMES = (
    "ks_mean",
    "ks_max",
    "disagree",
    "entropy_shift",
    "conf_shift",
    "calib_entropy",
    "test_entropy",
    "calib_conf",
    "test_conf",
    "ess_frac",
)


@dataclass(frozen=True)
class Evidence:
    """Container for the label-free evidence ``Z``.

    Attributes
    ----------
    ks_mean, ks_max : float
        Mean and max per-detector Kolmogorov-Smirnov drift between calibration
        and test score marginals.  ``0`` means no observable distribution shift.
    disagree : float
        ``1 - mean pairwise Spearman (rank) correlation`` of the test scores
        across detectors.  ``0`` for a single detector or perfectly concordant
        detectors; larger means a bigger disagreement region.
    entropy_shift : float
        ``test_entropy - calib_entropy`` of the (binned) score distributions.
    conf_shift : float
        ``test_confidence - calib_confidence`` where confidence is the mean
        distance of a (min-max normalised) score from the indecision point 0.5,
        i.e. how "sharp"/decisive the scores are.
    calib_entropy, test_entropy : float
        Shannon entropy (nats) of the calibration and test score histograms.
    calib_conf, test_conf : float
        Mean decisiveness of the calibration and test scores (in ``[0, 1]``).
    ess : float
        Effective sample size of the Gaussian-ratio importance weights, in
        ``(0, n_test]``.  Low values flag poor source->target support overlap.
    ess_frac : float
        ``ess / n_test`` in ``(0, 1]`` -- a scale-free overlap quality.
    n_calib, n_test : int
        Sample sizes of the calibration and test score sets.
    n_detectors : int
        Number of detector columns (1 for a single-column score vector).
    extra : dict
        Optional caller-supplied annotations (kept for explainability).
    """

    ks_mean: float
    ks_max: float
    disagree: float
    entropy_shift: float
    conf_shift: float
    calib_entropy: float
    test_entropy: float
    calib_conf: float
    test_conf: float
    ess: float
    ess_frac: float
    n_calib: int
    n_test: int
    n_detectors: int
    extra: dict = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        """Immutable schema identifier required by frozen benefit estimators."""

        return EVIDENCE_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, float]:
        """Return the exact deployment feature map in schema order."""

        return dict(zip(EVIDENCE_FEATURE_NAMES, self.to_vector().tolist(), strict=True))

    def to_vector(self) -> np.ndarray:
        """Return the core scalar signals as a fixed-order feature vector.

        Order: ``[ks_mean, ks_max, disagree, entropy_shift, conf_shift,
        calib_entropy, test_entropy, calib_conf, test_conf, ess_frac]``.
        Useful as the input ``Z`` to a downstream benefit regressor.
        """
        return as_float_array([getattr(self, name) for name in EVIDENCE_FEATURE_NAMES])


def _as_2d(scores: np.ndarray, name: str) -> np.ndarray:
    """Coerce a score array to shape ``(n_samples, n_detectors)``."""
    arr = as_float_array(scores)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must have at least one sample")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one detector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _rank_norm(scores: np.ndarray) -> np.ndarray:
    """Column-wise rank-normalise scores to ``[0, 1]`` (matches the K-Bound code).

    Mirrors ``rank_norm`` in ``knowability_experiment.py``: each column becomes
    ``(rank - 1) / (n - 1)``.  For ``n == 1`` the single row maps to ``0.5``.
    """
    n = scores.shape[0]
    out = np.empty_like(scores, dtype=float)
    if n == 1:
        out[:] = 0.5
        return out
    for j in range(scores.shape[1]):
        out[:, j] = (rankdata(scores[:, j]) - 1.0) / (n - 1.0)
    return out


def _hist_entropy(x: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy (nats) of a 1-D sample's normalised histogram.

    A label-free measure of how spread-out the scores are.  Robust to the score
    scale (the histogram is taken over the observed range).
    """
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return 0.0
    counts, _ = np.histogram(x, bins=bins, range=(lo, hi))
    p = counts.astype(float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _decisiveness(x: np.ndarray) -> float:
    """Mean distance from the indecision point 0.5 after min-max normalisation.

    Returns a value in ``[0, 1]``: ``0`` for scores piled at the midpoint,
    approaching ``1`` for scores piled at the extremes.  Label-free analogue of
    "mean confidence" / ``conf`` in the deep-TTA evidence vector.
    """
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return 0.0
    xn = (x - lo) / (hi - lo)
    return float(np.mean(np.abs(xn - 0.5)) * 2.0)


def _importance_ess(calib: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Gaussian-ratio importance weights of test under calib, and their ESS.

    Models the calibration and test score laws as 1-D Gaussians and forms the
    self-normalised density-ratio weights ``w_i = N(test_i; mu_c, s_c) /
    N(test_i; mu_t, s_t)`` (the weight that re-expresses a *test* expectation as a
    *calibration-distributed* one).  The effective sample size

        ESS = (sum w)^2 / sum w^2

    measures how many test points are "effectively" usable; ``ESS == n_test``
    iff the weights are uniform (no shift), and ``ESS -> 1`` as the two laws
    separate.  This is the same support-overlap diagnostic used for the
    covariate-shift certificate in ``kbound_full_experiments.py``.

    Direction (panel finding F2-11)
    -------------------------------
    The implementation used to compute ``log w = log p_test - log p_calib``, the
    **reciprocal** of the weight documented above.  ESS is not invariant under
    ``w -> 1/w``: on a variance-*shrinking* shift (calib ~ N(0, 3),
    test ~ N(0, 1)) the reciprocal reports ``ess_frac ~ 0.88`` ("good overlap")
    where the documented weight reports ``~ 0.03`` ("severe overlap failure").
    The documented direction is the correct one for this diagnostic -- the
    question ``ess_frac`` answers is "how well does the calibration set cover the
    test set", i.e. how badly does re-weighting test points back onto the
    calibration law degrade the sample -- so the code now matches the docstring.

    Returns
    -------
    (ess, ess_frac) : tuple of float
        ``ess`` in ``(0, n_test]`` and ``ess_frac = ess / n_test`` in ``(0, 1]``.
    """
    c = np.asarray(calib, dtype=float).ravel()
    t = np.asarray(test, dtype=float).ravel()
    n_t = t.size
    mu_c, s_c = float(c.mean()), float(c.std())
    mu_t, s_t = float(t.mean()), float(t.std())
    # Degenerate spreads: treat as no usable shift information -> uniform weights.
    if s_c <= 1e-12 or s_t <= 1e-12:
        return float(n_t), 1.0
    # log w = log N(t; mu_c, s_c) - log N(t; mu_t, s_t)   (matches the docstring)
    log_w = (-0.5 * ((t - mu_c) / s_c) ** 2 - math.log(s_c)) - (-0.5 * ((t - mu_t) / s_t) ** 2 - math.log(s_t))
    log_w = np.clip(log_w, -50.0, 50.0)
    w = np.exp(log_w - log_w.max())  # stabilise; self-normalisation cancels the offset
    sw = float(w.sum())
    sw2 = float((w**2).sum())
    if sw2 <= 0.0:
        return float(n_t), 1.0
    ess = (sw * sw) / sw2
    ess = float(min(max(ess, 1.0), float(n_t)))
    return ess, ess / float(n_t)


def compute_evidence(
    calib_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    bins: int = 20,
    extra: dict | None = None,
) -> Evidence:
    """Compute the label-free evidence ``Z`` from calibration and test scores.

    No target labels are used anywhere in this function -- every quantity is a
    statistic of the (unlabelled) score arrays, exactly as in the K-Bound
    experiment scripts.

    Parameters
    ----------
    calib_scores : array-like, shape (n_calib,) or (n_calib, n_detectors)
        Detector scores on the calibration / source / validation split.
    test_scores : array-like, shape (n_test,) or (n_test, n_detectors)
        Detector scores on the unlabelled test / target split.  Must have the
        same number of detector columns as ``calib_scores``.
    bins : int, default=20
        Number of histogram bins for the entropy signals.
    extra : dict, optional
        Caller annotations to carry through into :attr:`Evidence.extra`.

    Returns
    -------
    Evidence
        The populated evidence container.

    Raises
    ------
    ValueError
        If the arrays are empty, non-finite, or have mismatched detector counts.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> calib = rng.normal(0, 1, size=(500, 3))
    >>> test = rng.normal(0, 1, size=(500, 3))   # same law -> ~no drift
    >>> z = compute_evidence(calib, test)
    >>> z.ks_mean < 0.1
    True
    """
    c = _as_2d(calib_scores, "calib_scores")
    t = _as_2d(test_scores, "test_scores")
    if c.shape[1] != t.shape[1]:
        raise ValueError(
            f"calib_scores and test_scores must have the same number of detector "
            f"columns; got {c.shape[1]} vs {t.shape[1]}"
        )
    n_detectors = c.shape[1]

    # --- Drift: per-detector KS distance between calib and test marginals. ---
    ks_vals = [float(ks_2samp(c[:, j], t[:, j]).statistic) for j in range(n_detectors)]
    ks_mean = float(np.mean(ks_vals))
    ks_max = float(np.max(ks_vals))

    # --- Disagreement: 1 - mean pairwise rank correlation of test scores. ---
    if n_detectors >= 2:
        rt = _rank_norm(t)
        correlations = []
        for i in range(n_detectors):
            for j in range(i + 1, n_detectors):
                std_i = float(np.std(rt[:, i]))
                std_j = float(np.std(rt[:, j]))
                if std_i <= 1e-12 and std_j <= 1e-12:
                    correlation = 1.0 if np.allclose(rt[:, i], rt[:, j]) else 0.0
                elif std_i <= 1e-12 or std_j <= 1e-12:
                    correlation = 0.0
                else:
                    correlation = float(np.corrcoef(rt[:, i], rt[:, j])[0, 1])
                    if not np.isfinite(correlation):
                        correlation = 0.0
                correlations.append(correlation)
        mean_corr = float(np.mean(correlations)) if correlations else 0.0
        disagree = float(1.0 - mean_corr)
    else:
        disagree = 0.0

    # --- Entropy / confidence shift (pooled across detectors). ---
    calib_entropy = _hist_entropy(c.ravel(), bins=bins)
    test_entropy = _hist_entropy(t.ravel(), bins=bins)
    calib_conf = _decisiveness(c.ravel())
    test_conf = _decisiveness(t.ravel())
    entropy_shift = test_entropy - calib_entropy
    conf_shift = test_conf - calib_conf

    # --- Importance weight / ESS (pooled across detectors). ---
    ess, ess_frac = _importance_ess(c.ravel(), t.ravel())

    return Evidence(
        ks_mean=ks_mean,
        ks_max=ks_max,
        disagree=disagree,
        entropy_shift=entropy_shift,
        conf_shift=conf_shift,
        calib_entropy=calib_entropy,
        test_entropy=test_entropy,
        calib_conf=calib_conf,
        test_conf=test_conf,
        ess=ess,
        ess_frac=ess_frac,
        n_calib=int(c.shape[0]),
        n_test=int(t.shape[0]),
        n_detectors=int(n_detectors),
        extra=dict(extra) if extra else {},
    )
