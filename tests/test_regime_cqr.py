from __future__ import annotations

import numpy as np

from orius.forecasting.uncertainty.cqr import (
    RegimeCQR,
    RegimeCQRConfig,
    assign_bins,
    cqr_scores,
    rolling_volatility,
)


def test_regime_cqr_fit_predict_shapes_and_bins() -> None:
    y_cal = np.linspace(100.0, 200.0, 120)
    q_lo_cal = y_cal - 8.0
    q_hi_cal = y_cal + 8.0

    model = RegimeCQR(RegimeCQRConfig(alpha=0.10, n_bins=3, vol_window=12))
    meta = model.fit(y_cal=y_cal, q_lo_cal=q_lo_cal, q_hi_cal=q_hi_cal)

    assert meta["n_bins"] == 3
    assert len(meta["qhat_by_bin"]) == 3

    y_test = np.linspace(120.0, 210.0, 48)
    q_lo_test = y_test - 7.0
    q_hi_test = y_test + 7.0
    lo, hi, bins = model.predict_interval(y_context=y_test, q_lo=q_lo_test, q_hi=q_hi_test)

    assert lo.shape == y_test.shape
    assert hi.shape == y_test.shape
    assert bins.shape == y_test.shape
    assert np.all((bins >= 0) & (bins <= 2))
    assert np.all(hi >= lo)


def test_regime_cqr_helpers() -> None:
    y = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    vol = rolling_volatility(y, window=3)
    assert vol.shape == y.shape
    assert np.all(vol >= 0.0)

    bins, edges = assign_bins(vol, n_bins=3)
    assert bins.shape == vol.shape
    assert edges.shape[0] == 4

    scores = cqr_scores(y_true=y, q_lo=y - 0.5, q_hi=y + 0.5)
    assert scores.shape == y.shape
    assert np.all(scores >= 0.0)
