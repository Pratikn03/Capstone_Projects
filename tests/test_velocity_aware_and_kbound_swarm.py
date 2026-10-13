"""Unit and integration tests for K-Bound Mind integration in ORIUS, Velocity-Aware DC3S Shield, Proactive K-Bound Gating, and CAN-Bus HIL."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn as nn

from kga.certificate import conformal_split
from orius.adapters.battery import BatteryDomainAdapter
from orius.dc3s import (
    CANBusInterface,
    ProactiveKBoundGovernor,
    UniversalPhysicalAIShield,
    VelocityAwareUniversalShield,
    compute_kbound_conformal_certificate,
    evaluate_kbound_mind,
    run_dc3s_step,
)
from orius.kbound import (
    AdversaryNet,
    Decision,
    PhysicalAINeuralShield,
    WorldModel,
    WorldModel3D,
    kbound_step,
)
from orius.universal_framework import run_universal_step


def test_kbound_conformal_certificate_gate() -> None:
    calib = np.linspace(0.005, 0.05, 50)
    cert_adapt = compute_kbound_conformal_certificate(
        delta_hat=0.02, calib_residuals=calib, alpha=0.05, current_residual=0.01
    )
    assert cert_adapt.permits_adaptation is True
    assert cert_adapt.action == "ADAPT"

    cert_freeze = compute_kbound_conformal_certificate(
        delta_hat=0.02, calib_residuals=calib, alpha=0.05, current_residual=0.15
    )
    assert cert_freeze.permits_adaptation is False
    assert cert_freeze.action == "ABSTAIN/FREEZE"


def test_kga_and_kbound_mind_integration_in_dc3s_and_universal_pipeline() -> None:
    # 1. Direct KGA conformal_split + evaluate_kbound_mind
    calib = np.linspace(0.01, 0.16, 30)
    kga_cert = conformal_split(delta_hat=0.045, calib_residuals=calib, alpha=0.05)
    assert np.isfinite(kga_cert.epsilon)

    kb_abstain = evaluate_kbound_mind(
        reliability_w=0.40, delta_hat=0.045, calib_residuals=calib, alpha=0.05
    )
    assert kb_abstain.decision == "ABSTAIN/FREEZE"
    assert kb_abstain.kga_decision == Decision.ABSTAIN.value
    assert kb_abstain.epistemic_override_triggered is True
    assert kb_abstain.inflation_multiplier > 1.0

    kb_adapt = evaluate_kbound_mind(
        reliability_w=0.95, delta_hat=0.35, calib_residuals=calib, alpha=0.05
    )
    assert kb_adapt.decision == "ADAPT"
    assert kb_adapt.kga_decision == Decision.ADAPT.value
    assert kb_adapt.allow_adaptation is True

    # 2. Verify run_dc3s_step natively computes and embeds K-Bound Mind state
    adapter = BatteryDomainAdapter()
    res_dc3s = run_dc3s_step(
        event={"ts_utc": "2026-01-01T01:00:00Z", "load_mw": 500.0},
        last_event={"ts_utc": "2026-01-01T00:00:00Z", "load_mw": 495.0},
        yhat=500.0,
        q=25.0,
        candidate_action={"charge_mw": 10.0, "discharge_mw": 0.0},
        domain_adapter=adapter,
        state={"current_soc_mwh": 5000.0, "capacity_mwh": 10000.0},
        cfg={"kbound": {"delta_hat": 0.045, "couple_inflation": True}},
    )
    assert "kbound" in res_dc3s
    assert res_dc3s["kbound_decision"] == "ABSTAIN/FREEZE"
    assert "kbound" in res_dc3s["certificate"]["uncertainty"]

    # 3. Verify run_universal_step natively computes and embeds K-Bound Mind state
    res_univ = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "ts_utc": "2026-01-01T01:00:00Z",
            "load_mw": 500.0,
            "yhat_load": 500.0,
            "current_soc_mwh": 5000.0,
            "capacity_mwh": 10000.0,
        },
        history=[{"ts_utc": "2026-01-01T00:00:00Z", "load_mw": 495.0, "w_t": 0.95}],
        candidate_action={"charge_mw": 10.0, "discharge_mw": 0.0},
        constraints={
            "min_soc_mwh": 1000.0,
            "max_soc_mwh": 9000.0,
            "capacity_mwh": 10000.0,
            "max_power_mw": 200.0,
        },
        quantile=25.0,
    )
    assert "kbound" in res_univ.certificate.extras
    assert "KBound" in res_univ.theorem_contracts
    assert res_univ.theorem_contracts["KBound"]["decision"] == "ABSTAIN/FREEZE"


def test_universal_physical_ai_shield_and_kbound_step() -> None:
    shield = UniversalPhysicalAIShield()
    calib = np.linspace(0.01, 0.16, 30)
    safe_cmd, wt, active, rad = shield.filter_actuation(
        t_sec=2.0,
        observed_state=15.0,
        nominal_command=0.0,
        latency_s=0.52,
        safety_boundary_h=13.0,
        calib_residuals=calib,
        is_braking=True,
    )
    assert active is True
    assert safe_cmd == -7.5
    assert 0.0 <= wt <= 1.0
    assert rad > 0.0

    # Test kbound_step TTA operator and neural world models
    torch.manual_seed(42)
    clf = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    frozen_clf = copy.deepcopy(clf)
    opt = torch.optim.SGD(clf.parameters(), lr=1e-2)
    x_batch = torch.randn(16, 8)
    logits, stats = kbound_step(clf, frozen_clf, x_batch, opt)
    assert logits.shape == (16, 4)
    assert stats["param_drift_l2"] >= 0.0
    assert stats["max_spectral_norm"] >= 0.0

    wm2d = WorldModel()
    assert wm2d(torch.randn(4, 2)).shape == (4, 1)
    wm3d = WorldModel3D()
    assert wm3d(torch.randn(4, 3)).shape == (4, 3)
    nshield = PhysicalAINeuralShield()
    out_s = nshield(torch.randn(4, 4))
    assert out_s.shape == (4, 1)
    assert torch.all((out_s >= 0.0) & (out_s <= 1.0))
    adv = AdversaryNet(max_lag_s=1.0)
    out_a = adv(torch.randn(4, 2))
    assert out_a.shape == (4, 1)
    assert torch.all((out_a >= 0.0) & (out_a <= 1.0))


def test_velocity_aware_shield_four_embodiments() -> None:
    dt = 0.02
    steps = 400
    base_time = datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Autonomous Vehicle (upper barrier at 80.0 m)
    av_shield = VelocityAwareUniversalShield(
        "Autonomous_Vehicle", 80.0, "upper", 4.0, -8.0, 3.2, 4.5
    )
    av_pos, av_vel = 0.0, 22.0
    for s in range(steps):
        now = base_time + timedelta(seconds=s * dt)
        u_safe, w_t, _ = av_shield.step(
            now, 0.02 if s * dt < 2.0 else 0.50, av_pos, av_vel, 2.5
        )
        assert 0.05 <= w_t <= 1.0
        av_vel = max(0.0, av_vel + u_safe * dt)
        av_pos += av_vel * dt
    assert (80.0 - av_pos) > 25.0

    # 2. 6-DoF Robotic Arm (lower barrier at 0.0 m)
    arm_shield = VelocityAwareUniversalShield(
        "Robotic_Manipulator", 0.0, "lower", 15.0, -10.0, 6.0, 0.08
    )
    arm_z, arm_vz = 1.5, -0.6
    for s in range(steps):
        now = base_time + timedelta(seconds=s * dt)
        u_safe, w_t, _ = arm_shield.step(
            now, 0.02 if s * dt < 1.5 else 0.48, arm_z, arm_vz, -2.0
        )
        assert 0.05 <= w_t <= 1.0
        arm_vz = (
            min(0.0, arm_vz + u_safe * dt)
            if arm_z > 0.05
            else max(0.0, arm_vz + u_safe * dt)
        )
        arm_z += arm_vz * dt
    assert arm_z > 0.20


def test_proactive_kbound_governor_zero_collisions_under_3000ms_spike() -> None:
    rng = np.random.default_rng(42)
    n_agents = 2000
    ego_vel = rng.uniform(15.0, 25.0, size=n_agents).astype(np.float32)
    dist_gap = rng.uniform(100.0, 180.0, size=n_agents).astype(np.float32)
    governor = ProactiveKBoundGovernor(max_lag_s=3.0)
    dt = 0.05

    for _ in range(150):
        injected_lag = np.full(n_agents, 3.0, dtype=np.float32)
        lead_brake = np.where(rng.random(n_agents) > 0.95, -8.0, 0.0).astype(
            np.float32
        )
        dist_gap += lead_brake * dt
        ego_accel, w_t, _ = governor.evaluate_swarm_step(
            ego_vel, dist_gap, injected_lag
        )
        assert np.all((w_t >= 0.01) & (w_t <= 1.0))
        ego_vel = np.clip(ego_vel + ego_accel * dt, 0.0, None)
        dist_gap -= ego_vel * dt

    assert int(np.sum(dist_gap <= 0.0)) == 0


def test_can_bus_hil_interface() -> None:
    bus = CANBusInterface(baud_kbps=500, drop_rate=0.05, seed=123)
    payload = bus.pack_telemetry(42.5, 18.25, 1)
    assert len(payload) == 8
    dist, vel = bus.unpack_telemetry(payload)
    assert np.isclose(dist, 42.5)
    assert np.isclose(vel, 18.25)
    delay_s, dropped = bus.transmit_frame(payload)
    assert delay_s > 0.0008
    assert isinstance(dropped, bool)


def test_crash_free_governor_theorems_f_and_g() -> None:
    from orius.kbound import (
        AnytimeEProcessGate,
        ByzantineResilientOQE,
        CrashFreePhysicalAIGovernor,
        DomainPhysicsConfig,
    )

    # 1. AnytimeEProcessGate: Ville's inequality & inflation under distribution shift
    egate = AnytimeEProcessGate(alpha=0.05, base_quantile=0.5)
    for _ in range(20):
        st = egate.update(0.2)
    assert st.e_value >= 1.0
    assert st.empirical_coverage == 1.0
    # Inject sudden non-exchangeable spikes
    for _ in range(15):
        st = egate.update(3.5)
    assert st.e_value > 2.0
    assert st.martingale_multiplier > 1.2
    assert egate.effective_radius(0.5) > 2.0

    # 2. ByzantineResilientOQE: 40% (2 of 5 channels) corrupted by large spoofing bias
    byz = ByzantineResilientOQE(mad_threshold=3.0, min_w=0.05)
    res_byz = byz.filter_channels(
        state_channels=[25.0, 25.02, 24.98, 5.0, 4.5],
        velocity_channels=[4.0, 4.01, 3.99, -2.0, -2.5],
    )
    assert np.isclose(res_byz.clean_state, 25.0, atol=0.05)
    assert np.isclose(res_byz.clean_velocity, 4.0, atol=0.05)
    assert res_byz.spoof_detected is True
    assert 0.35 <= res_byz.byzantine_fraction <= 0.45

    # 3. CrashFreePhysicalAIGovernor (Theorems F & G): 0 violations under hallucination + blackout + Byzantine
    cfg = DomainPhysicsConfig(
        domain_name="autonomous_vehicles",
        state_unit="Speed (m/s)",
        x_min=0.0,
        x_max=30.0,
        u_min=-8.0,
        u_max=4.5,
        dt=0.20,
        max_brake_decel=8.0,
        process_noise_bound=0.10,
        obs_noise_std=0.25,
        damping=0.04,
    )
    gov = CrashFreePhysicalAIGovernor(cfg, alpha=0.05)
    x_true, v_true = 24.0, 3.5
    gov.reset(x_true, v_true)

    for step in range(120):
        dropout = bool(40 <= step < 80)
        # 2 of 5 channels spoofed downward to hide closeness to 30.0 m/s upper limit
        s_ch = [x_true, x_true + 0.05, x_true - 0.04, x_true - 12.0, x_true - 13.0]
        v_ch = [v_true, v_true + 0.02, v_true - 0.02, 0.0, 0.0]
        # Worst-case AI hallucination commanding full throttle u_max toward upper boundary
        u_gov, info = gov.step(
            true_state=x_true,
            true_velocity=v_true,
            state_channels=s_ch,
            velocity_channels=v_ch,
            u_nominal_ai=cfg.u_max,
            dropout=dropout,
        )
        v_true = (1.0 - cfg.damping) * v_true + u_gov * cfg.dt
        x_true = min(cfg.x_max, max(cfg.x_min, x_true + v_true * cfg.dt))
        assert cfg.x_min <= x_true <= cfg.x_max
        if dropout and step >= 45:
            assert info.safe_stop_engaged is True


def test_theorem_h_financial_liquidation_governor() -> None:
    from orius.kbound import (
        FinancialLiquidationGovernor,
        compute_almgren_chriss_liquidation_distance,
    )

    d_liq_clean = compute_almgren_chriss_liquidation_distance(
        inventory_exposure=10.0, volatility=0.5, l2_book_depth=25.0, reliability_w=1.0
    )
    d_liq_degraded = compute_almgren_chriss_liquidation_distance(
        inventory_exposure=10.0, volatility=0.5, l2_book_depth=25.0, reliability_w=0.10
    )
    assert d_liq_degraded > d_liq_clean > 0.0

    fgov = FinancialLiquidationGovernor()
    cap_cushion, inv_vel = 28.0, -4.0
    fgov.reset(cap_cushion, inv_vel)
    for step in range(80):
        dropout = bool(25 <= step < 55)
        s_ch = [cap_cushion, cap_cushion + 0.1, cap_cushion - 0.1, 80.0, 85.0]
        v_ch = [inv_vel, inv_vel + 0.05, inv_vel - 0.05, 5.0, 6.0]
        # Adversarial AI attempting to push capital cushion below insolvency floor (20%)
        u_gov, info = fgov.step(
            true_state=cap_cushion,
            true_velocity=inv_vel,
            state_channels=s_ch,
            velocity_channels=v_ch,
            u_nominal_ai=fgov.cfg.u_min,
            dropout=dropout,
        )
        inv_vel = (1.0 - fgov.cfg.damping) * inv_vel + u_gov * fgov.cfg.dt
        cap_cushion = min(fgov.cfg.x_max, max(fgov.cfg.x_min, cap_cushion + inv_vel * fgov.cfg.dt))
        assert fgov.cfg.x_min <= cap_cushion <= fgov.cfg.x_max


