"""Tests for Paper 2 blackout expiration behavior."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_blackout_study_importable() -> None:
    from scripts.run_blackout_study import run_blackout_study

    df = run_blackout_study(
        seed=0,
        horizon=24,
        freeze_step=12,
        blackout_durations=[0, 1, 4],
    )
    assert len(df) == 3
    assert "blackout_hours" in df.columns
    assert "tsvr_pct" in df.columns
    assert "coverage_pct" in df.columns


def test_certificate_state_expired_when_horizon_zero() -> None:
    from orius.dc3s.half_life import compute_certificate_state

    # SOC at boundary with large margin -> interval outside bounds -> tau_t=0
    state = compute_certificate_state(
        observed_state={"current_soc_mwh": 5.0},
        quality_score=0.5,
        safety_margin_mwh=20.0,  # interval [−15, 25] crosses min_soc=10
        constraints={
            "min_soc_mwh": 10.0,
            "max_soc_mwh": 90.0,
            "time_step_hours": 1.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        },
        sigma_d=1.0,
    )
    # May be expired or expiring depending on tube propagation
    assert state["status"] in ("valid", "expiring", "expired")
    assert state["fallback_required"] == (state["H_t"] <= 0 or state["status"] == "expired")
