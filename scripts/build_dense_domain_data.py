#!/usr/bin/env python3
"""Build dense ORIUS-format CSVs from raw ZeMA and BIDMC datasets.

Produces:
  - data/industrial/processed/industrial_zema_orius.csv  (ZeMA hydraulic → ORIUS format)
  - data/healthcare/processed/healthcare_bidmc_orius.csv  (BIDMC 53-patient → ORIUS format)

These are dense analogs of the battery reference row, structured for the
ORIUS DC³S pipeline with columns:
  timestamp, target, forecast, reliability, domain_label
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def build_zema_orius(out_dir: Path) -> int:
    """Convert ZeMA hydraulic system data to ORIUS format.

    Strategy:
      - Each cycle = one ORIUS timestep
      - target = mean pressure (PS1 mean across 6000 samples)
      - forecast = target + noise (simulating a simple model)
      - reliability = derived from fault profile (worse faults → lower reliability)
    """
    zema_dir = REPO / "data" / "industrial" / "raw" / "zema_hydraulic"
    if not zema_dir.exists():
        print(f"  SKIP: {zema_dir} not found")
        return 0

    print("  Loading ZeMA PS1 (pressure sensor 1)...")
    ps1 = np.loadtxt(zema_dir / "PS1.txt", delimiter="\t")  # (2205, 6000)
    print(f"    PS1 shape: {ps1.shape}")

    print("  Loading ZeMA TS1 (temperature sensor 1)...")
    ts1 = np.loadtxt(zema_dir / "TS1.txt", delimiter="\t")  # (2205, 60)
    print(f"    TS1 shape: {ts1.shape}")

    print("  Loading ZeMA profile (fault labels)...")
    profile = np.loadtxt(zema_dir / "profile.txt", delimiter="\t")  # (2205, 5)
    # Columns: cooler_condition, valve_condition, pump_leakage, accumulator_pressure, stable_flag
    # cooler: 3=close to total failure, 20=reduced efficiency, 100=full efficiency
    # valve: 100=optimal, 90=small lag, 80=severe lag, 73=close to total failure
    # pump: 0=no leakage, 1=weak leakage, 2=severe leakage
    # accumulator: 130=optimal, 115=slightly reduced, 100=severely reduced, 90=close to failure
    # stable: 1=stable, 0=not yet stable (conditions may not have reached steady state)
    print(f"    Profile shape: {profile.shape}")

    n_cycles = ps1.shape[0]
    rng = np.random.RandomState(42)

    # Compute per-cycle features
    ps1_mean = ps1.mean(axis=1)  # mean pressure per cycle
    ps1_std = ps1.std(axis=1)  # pressure variability
    ts1_mean = ts1.mean(axis=1)  # mean temperature per cycle

    # Reliability: combine fault conditions into a [0, 1] score
    # Higher = healthier. Normalize each fault indicator to [0, 1] and average.
    cooler_rel = profile[:, 0] / 100.0  # 3→0.03, 100→1.0
    valve_rel = profile[:, 1] / 100.0  # 73→0.73, 100→1.0
    pump_rel = 1.0 - profile[:, 2] / 2.0  # 0→1.0, 2→0.0
    accum_rel = (profile[:, 3] - 90.0) / 40.0  # 90→0.0, 130→1.0
    accum_rel = np.clip(accum_rel, 0, 1)
    stable = profile[:, 4]

    reliability = (cooler_rel + valve_rel + pump_rel + accum_rel + stable) / 5.0
    reliability = np.clip(reliability, 0.01, 1.0)

    # target = actual pressure reading
    # forecast = simple EMA-based prediction + noise
    target = ps1_mean
    forecast = np.zeros_like(target)
    forecast[0] = target[0]
    for i in range(1, len(target)):
        forecast[i] = 0.8 * forecast[i - 1] + 0.2 * target[i - 1]
    # Add realistic noise
    noise_scale = ps1_std * 0.3
    forecast += rng.normal(0, noise_scale)

    # Write ORIUS CSV
    out_path = out_dir / "industrial_zema_orius.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "target",
                "forecast",
                "reliability",
                "ps1_mean",
                "ps1_std",
                "ts1_mean",
                "cooler_cond",
                "valve_cond",
                "pump_leak",
                "accum_press",
                "domain_label",
            ]
        )
        for i in range(n_cycles):
            writer.writerow(
                [
                    f"2015-01-01T{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}",
                    f"{target[i]:.4f}",
                    f"{forecast[i]:.4f}",
                    f"{reliability[i]:.4f}",
                    f"{ps1_mean[i]:.4f}",
                    f"{ps1_std[i]:.4f}",
                    f"{ts1_mean[i]:.4f}",
                    f"{profile[i, 0]:.0f}",
                    f"{profile[i, 1]:.0f}",
                    f"{profile[i, 2]:.0f}",
                    f"{profile[i, 3]:.0f}",
                    "industrial",
                ]
            )

    print(f"  → {out_path} ({n_cycles} rows)")

    # Also expand to per-sample resolution for density parity
    # Use 100 samples per cycle (subsample from 6000)
    expanded_path = out_dir / "industrial_zema_dense_orius.csv"
    step = 60  # 6000 / 100
    with open(expanded_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "target", "forecast", "reliability", "cycle_id", "sample_id", "domain_label"]
        )
        row_count = 0
        for i in range(n_cycles):
            for j in range(0, 6000, step):
                t_val = ps1[i, j]
                f_val = t_val + rng.normal(0, ps1_std[i] * 0.2)
                writer.writerow(
                    [
                        f"2015-01-01T{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}.{j:04d}",
                        f"{t_val:.4f}",
                        f"{f_val:.4f}",
                        f"{reliability[i]:.4f}",
                        i,
                        j,
                        "industrial",
                    ]
                )
                row_count += 1

    print(f"  → {expanded_path} ({row_count} rows — dense)")
    return n_cycles


def build_bidmc_orius(out_dir: Path) -> int:
    """Convert BIDMC 53-patient vital signs to ORIUS format.

    Strategy:
      - Each patient's Numerics file gives HR, Pulse, Resp, SpO2 at 1 Hz
      - target = SpO2 (blood oxygen saturation — primary safety-critical signal)
      - forecast = simple EMA prediction of SpO2
      - reliability = derived from signal quality (agreement between HR and Pulse)
    """
    bidmc_dir = REPO / "data" / "healthcare" / "raw" / "bidmc_csv"
    if not bidmc_dir.exists():
        print(f"  SKIP: {bidmc_dir} not found")
        return 0

    all_rows: list[list] = []

    for patient_id in range(1, 54):
        num_path = bidmc_dir / f"bidmc_{patient_id:02d}_Numerics.csv"
        if not num_path.exists():
            continue

        with open(num_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        # Parse vital signs
        times = []
        hrs = []
        pulses = []
        resps = []
        spo2s = []

        for row in rows:
            try:
                t = float(row.get("Time [s]", row.get(" Time [s]", "0")).strip())
                hr = float(row.get(" HR", row.get("HR", "0")).strip())
                pulse = float(row.get(" PULSE", row.get("PULSE", "0")).strip())
                resp = float(row.get(" RESP", row.get("RESP", "0")).strip())
                spo2 = float(row.get(" SpO2", row.get("SpO2", "0")).strip())
                times.append(t)
                hrs.append(hr)
                pulses.append(pulse)
                resps.append(resp)
                spo2s.append(spo2)
            except (ValueError, KeyError):
                continue

        if len(times) < 10:
            continue

        spo2_arr = np.array(spo2s, dtype=float)
        hr_arr = np.array(hrs, dtype=float)
        pulse_arr = np.array(pulses, dtype=float)

        # Forecast: EMA of SpO2
        forecast = np.zeros_like(spo2_arr)
        forecast[0] = spo2_arr[0]
        for i in range(1, len(spo2_arr)):
            forecast[i] = 0.85 * forecast[i - 1] + 0.15 * spo2_arr[i - 1]

        # Reliability: based on HR-Pulse agreement (should be similar)
        # Large discrepancy → sensor degradation
        hr_pulse_gap = np.abs(hr_arr - pulse_arr)
        max_gap = max(hr_pulse_gap.max(), 1.0)
        reliability = 1.0 - (hr_pulse_gap / max_gap) * 0.5
        reliability = np.clip(reliability, 0.1, 1.0)

        for i in range(len(times)):
            all_rows.append(
                [
                    f"patient_{patient_id:02d}_t{times[i]:.0f}",
                    f"{spo2_arr[i]:.2f}",
                    f"{forecast[i]:.4f}",
                    f"{reliability[i]:.4f}",
                    f"{hr_arr[i]:.1f}",
                    f"{pulse_arr[i]:.1f}",
                    f"{resps[i]:.1f}",
                    patient_id,
                    "healthcare",
                ]
            )

    # Write ORIUS CSV
    out_path = out_dir / "healthcare_bidmc_orius.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "target",
                "forecast",
                "reliability",
                "hr",
                "pulse",
                "resp",
                "patient_id",
                "domain_label",
            ]
        )
        writer.writerows(all_rows)

    print(f"  → {out_path} ({len(all_rows)} rows)")

    # Also build from Signal files (125 Hz) for maximum density
    sig_rows = 0
    expanded_path = out_dir / "healthcare_bidmc_dense_orius.csv"

    with open(expanded_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "target", "forecast", "reliability", "patient_id", "sample_id", "domain_label"]
        )

        for patient_id in range(1, 54):
            sig_path = bidmc_dir / f"bidmc_{patient_id:02d}_Signals.csv"
            if not sig_path.exists():
                continue

            with open(sig_path) as sf:
                reader = csv.DictReader(sf)
                pleth_vals: list[float] = []
                for row in reader:
                    try:
                        # PLETH = photoplethysmogram — the PPG signal
                        pleth = float(row.get(" PLETH", row.get("PLETH", "0")).strip())
                        pleth_vals.append(pleth)
                    except (ValueError, KeyError):
                        continue

            if len(pleth_vals) < 100:
                continue

            pleth = np.array(pleth_vals)
            # Subsample: take every 10th sample (12.5 Hz instead of 125 Hz)
            pleth_sub = pleth[::10]
            n = len(pleth_sub)

            # Forecast: EMA
            fc = np.zeros(n)
            fc[0] = pleth_sub[0]
            for i in range(1, n):
                fc[i] = 0.9 * fc[i - 1] + 0.1 * pleth_sub[i - 1]

            # Reliability: based on signal variance in local window
            rel = np.ones(n)
            window = 50
            for i in range(window, n):
                local_std = np.std(pleth_sub[i - window : i])
                rel[i] = max(0.1, 1.0 - local_std * 2.0)

            for i in range(n):
                writer.writerow(
                    [
                        f"patient_{patient_id:02d}_s{i * 10}",
                        f"{pleth_sub[i]:.6f}",
                        f"{fc[i]:.6f}",
                        f"{rel[i]:.4f}",
                        patient_id,
                        i * 10,
                        "healthcare",
                    ]
                )
                sig_rows += 1

    print(f"  → {expanded_path} ({sig_rows} rows — dense from PPG signals)")
    return len(all_rows)


def main() -> None:
    print("=== Building Dense ORIUS-Format Datasets ===\n")

    ind_out = REPO / "data" / "industrial" / "processed"
    hc_out = REPO / "data" / "healthcare" / "processed"
    ind_out.mkdir(parents=True, exist_ok=True)
    hc_out.mkdir(parents=True, exist_ok=True)

    print("1. ZeMA Hydraulic → Industrial ORIUS...")
    zema_rows = build_zema_orius(ind_out)

    print("\n2. BIDMC → Healthcare ORIUS...")
    bidmc_rows = build_bidmc_orius(hc_out)

    print("\n=== Summary ===")
    print(f"  Industrial (ZeMA):  {zema_rows} cycles → ORIUS format")
    print(f"  Healthcare (BIDMC): {bidmc_rows} seconds → ORIUS format")

    # Density comparison
    battery_rows = 50402
    print(f"\n=== Density vs. Battery ({battery_rows:,} rows) ===")
    if zema_rows:
        print(f"  Industrial cycle-level:  {zema_rows:,}x — 0.04× battery")
        dense_rows = zema_rows * 100
        print(f"  Industrial dense:       {dense_rows:,}x — {dense_rows / battery_rows:.1f}× battery")
    if bidmc_rows:
        print(f"  Healthcare numerics:    {bidmc_rows:,}x — {bidmc_rows / battery_rows:.1f}× battery")


if __name__ == "__main__":
    main()
