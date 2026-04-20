from __future__ import annotations

import math

import numpy as np
import pytest

from orius.universal_theory.battery_instantiation import certificate_expiration_bound


def test_t6_bound_is_monotone_in_delta_bnd_sigma_and_delta() -> None:
    baseline = certificate_expiration_bound(
        interval_lower_mwh=40.0,
        interval_upper_mwh=60.0,
        soc_min_mwh=0.0,
        soc_max_mwh=100.0,
        sigma_d=5.0,
        delta=0.05,
    )
    wider_margin = certificate_expiration_bound(
        interval_lower_mwh=35.0,
        interval_upper_mwh=65.0,
        soc_min_mwh=0.0,
        soc_max_mwh=100.0,
        sigma_d=5.0,
        delta=0.05,
    )
    lower_sigma = certificate_expiration_bound(
        interval_lower_mwh=40.0,
        interval_upper_mwh=60.0,
        soc_min_mwh=0.0,
        soc_max_mwh=100.0,
        sigma_d=4.0,
        delta=0.05,
    )
    lower_confidence = certificate_expiration_bound(
        interval_lower_mwh=40.0,
        interval_upper_mwh=60.0,
        soc_min_mwh=0.0,
        soc_max_mwh=100.0,
        sigma_d=5.0,
        delta=0.10,
    )

    assert wider_margin["tau_expire_lb"] < baseline["tau_expire_lb"]
    assert lower_sigma["tau_expire_lb"] > baseline["tau_expire_lb"]
    assert lower_confidence["tau_expire_lb"] > baseline["tau_expire_lb"]


def test_t6_bound_matches_closed_form() -> None:
    result = certificate_expiration_bound(
        interval_lower_mwh=45.0,
        interval_upper_mwh=55.0,
        soc_min_mwh=0.0,
        soc_max_mwh=100.0,
        sigma_d=5.0,
        delta=0.05,
    )
    expected = math.floor((45.0**2) / (2.0 * (5.0**2) * math.log(2.0 / 0.05)))
    assert result["tau_expire_lb"] == expected
    assert result["denominator"] == pytest.approx(2.0 * (5.0**2) * math.log(2.0 / 0.05))
    assert result["theorem_id"] == "T6"
    assert result["requires_delta"] is True
    assert result["legacy_surface_allowed"] is False
    assert result["theorem_contract"]["status"] == "runtime_linked"


def test_t6_empirical_subgaussian_bound() -> None:
    sigma_d = 1.0
    delta = 0.10
    delta_bnd = 6.0
    result = certificate_expiration_bound(
        interval_lower_mwh=10.0 + delta_bnd,
        interval_upper_mwh=90.0 - delta_bnd,
        soc_min_mwh=10.0,
        soc_max_mwh=90.0,
        sigma_d=sigma_d,
        delta=delta,
    )
    tau = int(result["tau_expire_lb"])
    assert tau > 0

    rng = np.random.default_rng(0)
    episodes = 10_000
    steps = rng.normal(loc=0.0, scale=sigma_d, size=(episodes, tau))
    walks = np.cumsum(steps, axis=1)
    max_abs = np.max(np.abs(walks), axis=1)
    survived = float(np.mean(max_abs <= delta_bnd))

    assert survived >= 1.0 - delta - 0.02
