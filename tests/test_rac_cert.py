from __future__ import annotations

import numpy as np

from orius.dc3s.rac_cert import RACCertConfig, RACCertModel, compute_q_multiplier, normalize_sensitivity


def test_rac_cert_fit_and_predict_shapes() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(1000.0, 50.0, size=256)
    q_lo = y - 40.0
    q_hi = y + 40.0

    model = RACCertModel(cfg=RACCertConfig(n_vol_bins=3, vol_window=24, qhat_shrink_tau=20.0))
    meta = model.fit(y_cal=y, q_lo_cal=q_lo, q_hi_cal=q_hi)
    assert "qhat_by_vol_bin" in meta
    qhat = model.qhat_for_context(y_context=y[:48], horizon=48)
    assert qhat.shape == (48,)
    assert np.all(qhat >= 0.0)


def test_rac_q_multiplier_monotonicity() -> None:
    cfg = RACCertConfig(beta_reliability=0.7, beta_sensitivity=0.5, max_q_multiplier=3.0)
    low_sens = normalize_sensitivity(0.1, norm_ref=0.5)
    high_sens = normalize_sensitivity(0.5, norm_ref=0.5)
    q_good, _ = compute_q_multiplier(w_t=0.95, sensitivity_norm=low_sens, cfg=cfg)
    q_bad, _ = compute_q_multiplier(w_t=0.50, sensitivity_norm=high_sens, cfg=cfg)
    assert q_bad >= q_good
