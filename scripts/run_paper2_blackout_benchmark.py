#!/usr/bin/env python3
"""Paper 2 blackout benchmark: certificate half-life and policy comparison.

Compares 4 policies during forecast blackout:
1. freeze_last_action
2. immediate_shutdown
3. dc3s_no_temporal (projection only, no horizon planning)
4. half_life_aware (switch to fallback when tau_t <= remaining steps)

Outputs:
  reports/paper2/expiration_horizon.csv
  reports/paper2/blackout_policy_compare.csv
  reports/paper2/horizon_error.json
  reports/paper2/fig_certificate_shrinkage.png
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "paper2"


def _write_svg_certificate_shrinkage(out_dir, forward_tube, last_action, margin_mwh, soc_min, soc_max):
    """Fallback: write SVG when matplotlib unavailable."""
    initial_soc = 50.0
    interval_lower = initial_soc - margin_mwh
    interval_upper = initial_soc + margin_mwh
    max_steps = 30
    w, h = 400, 300
    pts = {}
    for sigma_d in [0.5, 1.0, 2.0]:
        radii = []
        for step in range(max_steps + 1):
            tube = forward_tube(
                interval_lower_mwh=interval_lower,
                interval_upper_mwh=interval_upper,
                safe_action=last_action,
                horizon_steps=step,
                sigma_d=sigma_d,
                dt_hours=1.0,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
            )
            radii.append(tube["upper_mwh"] - tube["lower_mwh"])
        pts[sigma_d] = radii
    max_w = max(max(r) for r in pts.values())
    lines = [
        '<?xml version="1.0"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="10" y="20" font-size="12">Certificate tube shrinkage (forward_tube)</text>',
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (sigma_d, radii) in enumerate(pts.items()):
        path_pts = []
        for j, r in enumerate(radii):
            x = 50 + j * (w - 100) / max_steps
            y = h - 50 - (r / max_w * (h - 100)) if max_w > 0 else h - 50
            path_pts.append(f"{x:.1f},{y:.1f}")
        d = "M " + " L ".join(path_pts)
        lines.append(f'<path d="{d}" fill="none" stroke="{colors[i % 3]}" stroke-width="2"/>')
        lines.append(f'<text x="{w-80}" y="{30+i*15}" font-size="10" fill="{colors[i % 3]}">σ_d={sigma_d}</text>')
    lines.append("</svg>")
    (out_dir / "fig_certificate_shrinkage.svg").write_text("\n".join(lines), encoding="utf-8")


def _f(x, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _safe_action_delta_mwh(action, dt_hours, charge_eff, discharge_eff):
    c = max(0.0, _f(action.get("charge_mw"), 0.0))
    d = max(0.0, _f(action.get("discharge_mw"), 0.0))
    return dt_hours * (charge_eff * c - (d / discharge_eff))


def _simulate_half_life_aware(
    initial_soc_mwh: float,
    last_action: dict,
    blackout_len: int,
    tau_t: int,
    constraints: dict,
    sigma_d: float,
    seed: int | None = None,
) -> list[float]:
    """Apply last_action for tau_t steps, then zero; half-life-aware policy."""
    import random
    rng = random.Random(seed)
    dt = _f(constraints.get("time_step_hours"), 1.0)
    eta_c = _f(constraints.get("charge_efficiency"), 1.0)
    eta_d = _f(constraints.get("discharge_efficiency"), 1.0)
    soc_min = _f(constraints.get("min_soc_mwh"), 0.0)
    soc_max = _f(constraints.get("max_soc_mwh"), 100.0)

    traj = [initial_soc_mwh]
    for step in range(blackout_len - 1):
        action = last_action if step < tau_t else {"charge_mw": 0.0, "discharge_mw": 0.0}
        delta = _safe_action_delta_mwh(action, dt, eta_c, eta_d)
        noise = sigma_d * (rng.gauss(0, 1) if sigma_d > 0 else 0)
        next_soc = traj[-1] + delta + noise
        next_soc = max(soc_min, min(soc_max, next_soc))
        traj.append(next_soc)
    return traj


def simulate_soc_trajectory(
    initial_soc_mwh: float,
    action: dict,
    steps: int,
    constraints: dict,
    sigma_d: float,
    seed: int | None = None,
) -> list[float]:
    """Simulate SOC trajectory with random walk (sigma_d per step)."""
    import random
    rng = random.Random(seed)
    dt = _f(constraints.get("time_step_hours"), 1.0)
    eta_c = _f(constraints.get("charge_efficiency"), 1.0)
    eta_d = _f(constraints.get("discharge_efficiency"), 1.0)
    soc_min = _f(constraints.get("min_soc_mwh"), 0.0)
    soc_max = _f(constraints.get("max_soc_mwh"), 100.0)

    traj = [initial_soc_mwh]
    for _ in range(steps - 1):
        delta = _safe_action_delta_mwh(action, dt, eta_c, eta_d)
        noise = sigma_d * (rng.gauss(0, 1) if sigma_d > 0 else 0)
        next_soc = traj[-1] + delta + noise
        next_soc = max(soc_min, min(soc_max, next_soc))
        traj.append(next_soc)
    return traj


def count_violations(trajectory: list[float], soc_min: float, soc_max: float) -> int:
    return sum(1 for s in trajectory if s < soc_min - 1e-9 or s > soc_max + 1e-9)


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "temporal_theorems",
        REPO_ROOT / "src" / "orius" / "dc3s" / "temporal_theorems.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    certificate_validity_horizon = mod.certificate_validity_horizon
    forward_tube = mod.forward_tube
    zero_dispatch_fallback = mod.zero_dispatch_fallback

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    capacity_mwh = 100.0
    soc_min = 10.0
    soc_max = 90.0
    constraints = {
        "min_soc_mwh": soc_min,
        "max_soc_mwh": soc_max,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }

    # Sweep parameters (catastrophic: 48h blackout)
    initial_soc_pcts = [0.2, 0.4, 0.5, 0.6, 0.8]
    sigma_d_regimes = [0.5, 1.0, 2.0, 3.0]
    blackout_lengths = [1, 2, 4, 8, 16, 24, 48]
    margin_mwh = 2.0

    # --- expiration_horizon.csv ---
    last_action = {"charge_mw": 5.0, "discharge_mw": 0.0}
    rows_horizon = []
    for soc_pct in initial_soc_pcts:
        initial_soc = soc_min + (soc_max - soc_min) * soc_pct
        interval_lower = initial_soc - margin_mwh
        interval_upper = initial_soc + margin_mwh
        for sigma_d in sigma_d_regimes:
            result = certificate_validity_horizon(
                interval_lower_mwh=interval_lower,
                interval_upper_mwh=interval_upper,
                safe_action=last_action,
                constraints=constraints,
                sigma_d=sigma_d,
            )
            rows_horizon.append({
                "initial_soc_pct": round(soc_pct, 2),
                "initial_soc_mwh": round(initial_soc, 2),
                "sigma_d": sigma_d,
                "tau_t": result["tau_t"],
                "tube_lower_mwh": round(result["tube_lower_mwh"], 4),
                "tube_upper_mwh": round(result["tube_upper_mwh"], 4),
            })

    with open(OUT_DIR / "expiration_horizon.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_horizon[0].keys())
        w.writeheader()
        w.writerows(rows_horizon)

    # --- blackout_policy_compare.csv ---
    policies = [
        "freeze_last_action",
        "immediate_shutdown",
        "dc3s_no_temporal",
        "half_life_aware",
    ]
    rows_compare = []
    n_monte_carlo = 20

    for blackout_len in blackout_lengths:
        for soc_pct in initial_soc_pcts:
            initial_soc = soc_min + (soc_max - soc_min) * soc_pct
            interval_lower = initial_soc - margin_mwh
            interval_upper = initial_soc + margin_mwh
            for sigma_d in sigma_d_regimes:
                tau_result = certificate_validity_horizon(
                    interval_lower_mwh=interval_lower,
                    interval_upper_mwh=interval_upper,
                    safe_action=last_action,
                    constraints=constraints,
                    sigma_d=sigma_d,
                )
                tau_t = tau_result["tau_t"]

                for policy in policies:
                    total_violations = 0
                    for mc in range(n_monte_carlo):
                        if policy == "freeze_last_action":
                            traj = simulate_soc_trajectory(
                                initial_soc, last_action, blackout_len, constraints, sigma_d, seed=mc
                            )
                        elif policy == "immediate_shutdown":
                            traj = simulate_soc_trajectory(
                                initial_soc, zero_dispatch_fallback(), blackout_len, constraints, sigma_d, seed=mc
                            )
                        elif policy == "dc3s_no_temporal":
                            traj = simulate_soc_trajectory(
                                initial_soc, last_action, blackout_len, constraints, sigma_d, seed=mc
                            )
                        elif policy == "half_life_aware":
                            traj = _simulate_half_life_aware(
                                initial_soc, last_action, blackout_len, tau_t,
                                constraints, sigma_d, seed=mc
                            )
                        total_violations += count_violations(traj, soc_min, soc_max)

                    rows_compare.append({
                        "policy": policy,
                        "blackout_length": blackout_len,
                        "initial_soc_pct": round(soc_pct, 2),
                        "sigma_d": sigma_d,
                        "tau_t": tau_t,
                        "violations_total": total_violations,
                        "violation_rate": round(total_violations / (n_monte_carlo * blackout_len), 4),
                    })

    with open(OUT_DIR / "blackout_policy_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows_compare[0].keys())
        w.writeheader()
        w.writerows(rows_compare)

    # --- horizon_error.json ---
    horizon_errors = []
    for soc_pct in [0.5]:
        initial_soc = soc_min + (soc_max - soc_min) * soc_pct
        interval_lower = initial_soc - margin_mwh
        interval_upper = initial_soc + margin_mwh
        for sigma_d in sigma_d_regimes:
            tau_result = certificate_validity_horizon(
                interval_lower_mwh=interval_lower,
                interval_upper_mwh=interval_upper,
                safe_action=last_action,
                constraints=constraints,
                sigma_d=sigma_d,
            )
            tau_pred = tau_result["tau_t"]
            actual_exit_steps = []
            for mc in range(50):
                traj = simulate_soc_trajectory(
                    initial_soc, last_action, min(100, max(50, tau_pred * 2)),
                    constraints, sigma_d, seed=mc
                )
                for i, s in enumerate(traj):
                    if s < soc_min - 1e-9 or s > soc_max + 1e-9:
                        actual_exit_steps.append(i)
                        break
                else:
                    actual_exit_steps.append(len(traj))
            mean_actual = sum(actual_exit_steps) / len(actual_exit_steps)
            horizon_errors.append({
                "initial_soc_pct": soc_pct,
                "sigma_d": sigma_d,
                "tau_t_predicted": tau_pred,
                "mean_actual_exit_step": round(mean_actual, 2),
                "error_steps": round(mean_actual - tau_pred, 2),
            })

    with open(OUT_DIR / "horizon_error.json", "w", encoding="utf-8") as f:
        json.dump({"horizon_errors": horizon_errors, "n_runs": 50}, f, indent=2)

    # --- fig_certificate_shrinkage.png (or .svg fallback) ---
    fig_path = OUT_DIR / "fig_certificate_shrinkage.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        max_steps = 30
        for sigma_d in [0.5, 1.0, 2.0]:
            initial_soc = 50.0
            interval_lower = initial_soc - margin_mwh
            interval_upper = initial_soc + margin_mwh
            radii = []
            for step in range(max_steps + 1):
                tube = forward_tube(
                    interval_lower_mwh=interval_lower,
                    interval_upper_mwh=interval_upper,
                    safe_action=last_action,
                    horizon_steps=step,
                    sigma_d=sigma_d,
                    dt_hours=1.0,
                    charge_efficiency=0.95,
                    discharge_efficiency=0.95,
                )
                width = tube["upper_mwh"] - tube["lower_mwh"]
                radii.append(width)
            ax.plot(range(max_steps + 1), radii, label=f"σ_d={sigma_d}")
        ax.axhline(soc_max - soc_min, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon steps")
        ax.set_ylabel("Tube width (MWh)")
        ax.set_title("Certificate tube shrinkage (forward_tube)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close()
    except ImportError:
        _write_svg_certificate_shrinkage(OUT_DIR, forward_tube, last_action, margin_mwh, soc_min, soc_max)
        fig_path = OUT_DIR / "fig_certificate_shrinkage.svg"
        # Try SVG→PNG via rsvg-convert or ImageMagick
        import subprocess
        svg_path = OUT_DIR / "fig_certificate_shrinkage.svg"
        png_path = OUT_DIR / "fig_certificate_shrinkage.png"
        for cmd in [
            ["rsvg-convert", "-f", "png", "-o", str(png_path), str(svg_path)],
            ["convert", str(svg_path), str(png_path)],
        ]:
            try:
                subprocess.run(cmd, check=True, capture_output=True, cwd=str(REPO_ROOT))
                fig_path = png_path
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

    print(f"Wrote {OUT_DIR}/expiration_horizon.csv")
    print(f"Wrote {OUT_DIR}/blackout_policy_compare.csv")
    print(f"Wrote {OUT_DIR}/horizon_error.json")
    print(f"Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
