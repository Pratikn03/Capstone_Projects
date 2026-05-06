"""Forecast comparison statistics: Diebold-Mariano, BCa bootstrap, Holm-Bonferroni.

Centralizes the statistical tests required to defend a "ORIUS beats X" claim
in peer review: Newey-West-corrected DM test for equal predictive accuracy,
paired bias-corrected accelerated bootstrap CIs on the metric difference,
Cohen's d on per-seed metric vectors, and Holm step-down correction across the
full comparison family.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as _scipy_stats


@dataclass(frozen=True)
class DieboldMarianoResult:
    statistic: float
    p_value: float
    horizon: int
    loss: str
    n: int
    mean_diff: float
    long_run_variance: float


@dataclass(frozen=True)
class BootstrapInterval:
    delta: float
    ci_low: float
    ci_high: float
    n_resamples: int
    method: str


def _to_array(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("input array is empty")
    return arr


def _loss_series(y_true: np.ndarray, y_pred: np.ndarray, loss: str) -> np.ndarray:
    err = y_true - y_pred
    if loss == "se":
        return err * err
    if loss == "ae":
        return np.abs(err)
    if loss == "ape":
        denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
        return np.abs(err / denom)
    raise ValueError(f"unknown loss: {loss}; expected 'se', 'ae', or 'ape'")


def _newey_west_lrv(d: np.ndarray, lag: int) -> float:
    n = d.size
    mean = d.mean()
    centered = d - mean
    gamma0 = float(np.dot(centered, centered) / n)
    if lag <= 0:
        return gamma0
    total = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        cov = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        total += 2.0 * weight * cov
    return max(total, 1e-12)


def diebold_mariano(
    y_true: Sequence[float] | np.ndarray,
    y_pred_a: Sequence[float] | np.ndarray,
    y_pred_b: Sequence[float] | np.ndarray,
    *,
    horizon: int = 1,
    loss: str = "se",
    small_sample_correction: bool = True,
) -> DieboldMarianoResult:
    """Two-sided Diebold-Mariano test of equal predictive accuracy.

    Positive ``statistic`` and small ``p_value`` mean model B beats model A on
    the chosen loss (because ``d_t = L(A) - L(B)`` is positive). Long-run
    variance uses the Newey-West kernel with bandwidth ``horizon - 1`` per
    Diebold & Mariano (1995); Harvey-Leybourne-Newbold (1997) small-sample
    correction is applied by default.
    """
    yt = _to_array(y_true)
    pa = _to_array(y_pred_a)
    pb = _to_array(y_pred_b)
    n = min(len(yt), len(pa), len(pb))
    if n < 4:
        raise ValueError("Diebold-Mariano requires at least 4 paired observations")
    yt, pa, pb = yt[:n], pa[:n], pb[:n]
    la = _loss_series(yt, pa, loss)
    lb = _loss_series(yt, pb, loss)
    d = la - lb
    valid = np.isfinite(d)
    d = d[valid]
    n_eff = d.size
    if n_eff < 4:
        raise ValueError("Diebold-Mariano: too few finite paired losses after filtering")

    lag = max(int(horizon) - 1, 0)
    lrv = _newey_west_lrv(d, lag)
    dm_stat = float(d.mean() / math.sqrt(lrv / n_eff))
    if small_sample_correction:
        h = int(horizon)
        correction = math.sqrt(max((n_eff + 1 - 2 * h + h * (h - 1) / n_eff) / n_eff, 1e-12))
        dm_stat = dm_stat * correction
        df = max(n_eff - 1, 1)
        p_value = float(2.0 * (1.0 - _scipy_stats.t.cdf(abs(dm_stat), df=df)))
    else:
        p_value = float(2.0 * (1.0 - _scipy_stats.norm.cdf(abs(dm_stat))))
    return DieboldMarianoResult(
        statistic=float(dm_stat),
        p_value=float(p_value),
        horizon=int(horizon),
        loss=str(loss),
        n=int(n_eff),
        mean_diff=float(d.mean()),
        long_run_variance=float(lrv),
    )


def _moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = math.ceil(n / block_size)
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    indices = np.concatenate([np.arange(s, s + block_size) for s in starts])
    return indices[:n]


def paired_block_bootstrap(
    y_true: Sequence[float] | np.ndarray,
    y_pred_a: Sequence[float] | np.ndarray,
    y_pred_b: Sequence[float] | np.ndarray,
    *,
    metric: str = "rmse",
    n_resamples: int = 10_000,
    block_size: int | None = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Paired moving-block bootstrap CI on (metric(B) - metric(A)).

    Negative interval means model B has a smaller error than model A. The
    block size defaults to ``ceil(n^{1/3})`` to preserve serial dependence
    (Politis & Romano, 1994). BCa correction is applied so the interval is
    valid even when the difference distribution is skewed.
    """
    yt = _to_array(y_true)
    pa = _to_array(y_pred_a)
    pb = _to_array(y_pred_b)
    n = min(len(yt), len(pa), len(pb))
    if n < 8:
        raise ValueError("paired bootstrap requires at least 8 paired observations")
    yt, pa, pb = yt[:n], pa[:n], pb[:n]
    block = int(block_size) if block_size else max(2, int(round(n ** (1.0 / 3.0))))
    rng = np.random.default_rng(int(seed))

    def metric_value(yt_arr: np.ndarray, pa_arr: np.ndarray, pb_arr: np.ndarray) -> float:
        if metric == "rmse":
            ea = float(np.sqrt(np.mean((yt_arr - pa_arr) ** 2)))
            eb = float(np.sqrt(np.mean((yt_arr - pb_arr) ** 2)))
        elif metric == "mae":
            ea = float(np.mean(np.abs(yt_arr - pa_arr)))
            eb = float(np.mean(np.abs(yt_arr - pb_arr)))
        else:
            raise ValueError(f"unknown metric: {metric}")
        return eb - ea

    point = metric_value(yt, pa, pb)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = _moving_block_indices(n, block, rng)
        boot[i] = metric_value(yt[idx], pa[idx], pb[idx])

    n_less = float(np.sum(boot < point))
    if n_less in (0.0, float(n_resamples)):
        z0 = 0.0
    else:
        z0 = float(_scipy_stats.norm.ppf(n_less / n_resamples))

    jack = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack[i] = metric_value(yt[mask], pa[mask], pb[mask])
    jack_mean = jack.mean()
    num = float(np.sum((jack_mean - jack) ** 3))
    den = float(6.0 * (np.sum((jack_mean - jack) ** 2)) ** 1.5 + 1e-12)
    a_hat = num / den

    alpha = 1.0 - confidence
    z_lo = _scipy_stats.norm.ppf(alpha / 2.0)
    z_hi = _scipy_stats.norm.ppf(1.0 - alpha / 2.0)

    def adjust(z: float) -> float:
        denom = 1.0 - a_hat * (z0 + z)
        return float(_scipy_stats.norm.cdf(z0 + (z0 + z) / max(denom, 1e-12)))

    p_lo = adjust(z_lo)
    p_hi = adjust(z_hi)
    ci_low = float(np.quantile(boot, max(min(p_lo, 1.0), 0.0)))
    ci_high = float(np.quantile(boot, max(min(p_hi, 1.0), 0.0)))
    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low

    return BootstrapInterval(
        delta=float(point),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_resamples=int(n_resamples),
        method="moving_block_bca",
    )


def cohens_d(values_a: Sequence[float] | np.ndarray, values_b: Sequence[float] | np.ndarray) -> float:
    """Cohen's d on two paired-sample metric vectors (typically per-seed)."""
    a = _to_array(values_a)
    b = _to_array(values_b)
    if a.size < 2 or b.size < 2:
        return float("nan")
    diff = b - a
    if diff.size < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        return float("inf") if abs(float(diff.mean())) > 0 else 0.0
    return float(diff.mean() / sd)


def holm_bonferroni(p_values: Iterable[float], alpha: float = 0.05) -> list[tuple[float, float, bool]]:
    """Holm step-down correction.

    Returns ``[(p_raw, p_adjusted, reject)]`` in the same order as input.
    """
    raw = [float(p) for p in p_values]
    m = len(raw)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: raw[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * raw[idx]
        running_max = max(running_max, min(adj, 1.0))
        adjusted[idx] = running_max
    return [(raw[i], adjusted[i], adjusted[i] < alpha) for i in range(m)]
