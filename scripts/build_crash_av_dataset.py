#!/usr/bin/env python3
"""Build a crash/near-crash AV dataset from the existing Waymo ORIUS corpus.

Strategy:
  1. **Mine real near-crash events** from the 1,975 Waymo scenarios:
     - Hard-braking (deceleration > 4 m/s²)
     - Close following (lead gap < 5 m)
     - RSS violations (already labelled)
  2. **Generate synthetic crash perturbations** on real trajectories:
     - Lead-vehicle sudden braking (cut-in + emergency stop)
     - Ego speed overshoot (sensor-spoofed higher speed)
     - Gap collapse (lead decelerates while ego maintains speed)
  3. Output combined ORIUS-contract CSV + crash manifest.

Usage:
    python scripts/build_crash_av_dataset.py
    python scripts/build_crash_av_dataset.py --out-dir data/orius_av/crash/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAYMO_CSV = REPO_ROOT / "data" / "orius_av" / "av" / "processed" / "av_trajectories_orius.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "orius_av" / "crash" / "processed"

# Thresholds
HARD_BRAKE_MS2 = 4.0  # |deceleration| threshold
CLOSE_GAP_M = 5.0  # near-collision gap threshold
CRITICAL_GAP_M = 2.0  # critical gap (certain collision)
SPEED_DROP_RATIO = 0.5  # speed drops by >50% in one step → hard brake

# Synthetic scenario params
N_SYNTHETIC_LEAD_BRAKE = 30  # lead-vehicle emergency brake scenarios
N_SYNTHETIC_GAP_COLLAPSE = 30  # gap collapse scenarios
N_SYNTHETIC_SPEED_SPIKE = 20  # ego speed overshoot (sensor spoof) scenarios
RNG_SEED = 42


def _compute_accel(df: pd.DataFrame) -> pd.Series:
    """Compute longitudinal acceleration from speed column (m/s²)."""
    dt = 0.1  # Waymo 10 Hz
    accel = df["speed_mps"].diff() / dt
    return accel.fillna(0.0)


def mine_real_crashes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Mine real hard-braking and near-collision events from Waymo data."""
    crash_rows = []
    manifest_entries = []

    for vid, vdf in df.groupby("vehicle_id"):
        vdf = vdf.sort_values("step").copy()
        accel = _compute_accel(vdf)

        # Hard braking
        hard_brake_mask = accel < -HARD_BRAKE_MS2
        has_hard_brake = hard_brake_mask.any()

        # Close gap
        if "lead_rel_x_m" in vdf.columns:
            close_gap_mask = vdf["lead_rel_x_m"].fillna(999) < CLOSE_GAP_M
            critical_gap_mask = vdf["lead_rel_x_m"].fillna(999) < CRITICAL_GAP_M
        else:
            close_gap_mask = pd.Series(False, index=vdf.index)
            critical_gap_mask = pd.Series(False, index=vdf.index)

        has_close_gap = close_gap_mask.any()
        has_critical_gap = critical_gap_mask.any()

        # RSS violations
        rss_mask = vdf.get("rss_violation_true", pd.Series(False, index=vdf.index))
        has_rss = rss_mask.any()

        is_crash = has_hard_brake or has_close_gap or has_rss

        if is_crash:
            tagged = vdf.copy()
            tagged["accel_mps2"] = accel.values
            tagged["is_crash_scenario"] = True
            tagged["crash_type"] = "real_mined"
            tagged["has_hard_brake"] = has_hard_brake
            tagged["has_close_gap"] = has_close_gap
            tagged["has_critical_gap"] = has_critical_gap
            tagged["has_rss_violation"] = has_rss
            crash_rows.append(tagged)

            manifest_entries.append(
                {
                    "vehicle_id": vid,
                    "crash_type": "real_mined",
                    "n_steps": len(vdf),
                    "has_hard_brake": bool(has_hard_brake),
                    "has_close_gap": bool(has_close_gap),
                    "has_critical_gap": bool(has_critical_gap),
                    "has_rss_violation": bool(has_rss),
                    "hard_brake_steps": int(hard_brake_mask.sum()),
                    "close_gap_steps": int(close_gap_mask.sum()),
                    "min_gap_m": float(vdf["lead_rel_x_m"].min())
                    if "lead_rel_x_m" in vdf.columns and vdf["lead_rel_x_m"].notna().any()
                    else None,
                    "max_decel_mps2": float(accel.min()),
                    "speed_range_mps": [float(vdf["speed_mps"].min()), float(vdf["speed_mps"].max())],
                }
            )

    crash_df = pd.concat(crash_rows, ignore_index=True) if crash_rows else pd.DataFrame()
    return crash_df, manifest_entries


def _pick_random_trajectories(df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    """Pick n random vehicle_ids that have lead_present data."""
    lead_present = df.get("lead_present")
    mask = (
        pd.Series(True, index=df.index) if lead_present is None else lead_present.fillna(False).astype(bool)
    )
    candidates = df.loc[mask, "vehicle_id"].unique()
    if len(candidates) == 0:
        candidates = df["vehicle_id"].unique()
    n = min(n, len(candidates))
    return list(rng.choice(candidates, size=n, replace=False))


def synth_lead_brake(df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[pd.DataFrame], list[dict]]:
    """Synthesise lead-vehicle emergency brake scenarios.

    Takes a real trajectory and injects a sudden lead-vehicle stop:
    the lead decelerates at -8 to -12 m/s² starting at a random step.
    """
    vids = _pick_random_trajectories(df, N_SYNTHETIC_LEAD_BRAKE, rng)
    results = []
    manifest = []

    for i, vid in enumerate(vids):
        vdf = df[df["vehicle_id"] == vid].sort_values("step").copy()
        if len(vdf) < 20:
            continue

        # Pick injection point in middle 60% of trajectory
        n = len(vdf)
        inject_step = int(rng.integers(int(n * 0.2), int(n * 0.7)))
        brake_decel = rng.uniform(8, 12)  # m/s²
        dt = 0.1

        # Modify lead position: lead decelerates hard
        new_vid = f"synth_lead_brake_{i:03d}"
        vdf = vdf.copy()
        vdf["vehicle_id"] = new_vid

        if "lead_rel_x_m" in vdf.columns and vdf["lead_rel_x_m"].notna().any():
            lead_speed = vdf["lead_speed_mps"].fillna(vdf["speed_mps"]).values.copy()
            lead_rel = vdf["lead_rel_x_m"].fillna(50.0).values.copy()

            for j in range(inject_step, n):
                lead_speed[j] = (
                    max(0, lead_speed[j - 1] - brake_decel * dt) if j > inject_step else lead_speed[j]
                )
                # Gap shrinks because ego maintains speed but lead slows
                ego_speed = vdf.iloc[j]["speed_mps"]
                gap_change = (ego_speed - lead_speed[j]) * dt
                lead_rel[j] = max(0.1, lead_rel[j - 1] - gap_change) if j > inject_step else lead_rel[j]

            vdf["lead_speed_mps"] = lead_speed
            vdf["lead_rel_x_m"] = lead_rel
            vdf["lead_position_m"] = vdf["position_m"] + lead_rel

            min_gap = float(np.min(lead_rel[inject_step:]))
        else:
            min_gap = None

        vdf["is_crash_scenario"] = True
        vdf["crash_type"] = "synth_lead_brake"
        vdf["has_hard_brake"] = False
        vdf["has_close_gap"] = min_gap is not None and min_gap < CLOSE_GAP_M
        vdf["has_critical_gap"] = min_gap is not None and min_gap < CRITICAL_GAP_M
        vdf["has_rss_violation"] = True
        vdf["accel_mps2"] = _compute_accel(vdf).values

        results.append(vdf)
        manifest.append(
            {
                "vehicle_id": new_vid,
                "crash_type": "synth_lead_brake",
                "base_vehicle": vid,
                "inject_step": inject_step,
                "brake_decel_mps2": float(brake_decel),
                "min_gap_m": min_gap,
                "n_steps": len(vdf),
            }
        )

    return results, manifest


def synth_gap_collapse(df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[pd.DataFrame], list[dict]]:
    """Synthesise gap collapse: ego accelerates while lead maintains speed."""
    vids = _pick_random_trajectories(df, N_SYNTHETIC_GAP_COLLAPSE, rng)
    results = []
    manifest = []

    for i, vid in enumerate(vids):
        vdf = df[df["vehicle_id"] == vid].sort_values("step").copy()
        if len(vdf) < 20:
            continue

        new_vid = f"synth_gap_collapse_{i:03d}"
        n = len(vdf)
        inject_step = int(rng.integers(int(n * 0.2), int(n * 0.6)))
        accel_boost = rng.uniform(2, 5)  # m/s² extra acceleration
        dt = 0.1

        vdf = vdf.copy()
        vdf["vehicle_id"] = new_vid

        speeds = vdf["speed_mps"].values.copy()
        positions = vdf["position_m"].values.copy()

        for j in range(inject_step + 1, n):
            speeds[j] = min(30.0, speeds[j - 1] + accel_boost * dt)
            positions[j] = positions[j - 1] + speeds[j] * dt

        vdf["speed_mps"] = speeds
        vdf["position_m"] = positions

        if "lead_rel_x_m" in vdf.columns and vdf["lead_rel_x_m"].notna().any():
            # Lead position stays on original trajectory; gap shrinks
            orig_pos = df[df["vehicle_id"] == vid].sort_values("step")["position_m"].values
            lead_abs = orig_pos + vdf["lead_rel_x_m"].fillna(50.0).values
            new_gap = lead_abs - positions
            new_gap = np.maximum(new_gap, 0.1)
            vdf["lead_rel_x_m"] = new_gap
            vdf["lead_position_m"] = positions + new_gap
            min_gap = float(new_gap[inject_step:].min())
        else:
            min_gap = None

        vdf["is_crash_scenario"] = True
        vdf["crash_type"] = "synth_gap_collapse"
        vdf["has_hard_brake"] = False
        vdf["has_close_gap"] = min_gap is not None and min_gap < CLOSE_GAP_M
        vdf["has_critical_gap"] = min_gap is not None and min_gap < CRITICAL_GAP_M
        vdf["has_rss_violation"] = True
        vdf["accel_mps2"] = _compute_accel(vdf).values

        results.append(vdf)
        manifest.append(
            {
                "vehicle_id": new_vid,
                "crash_type": "synth_gap_collapse",
                "base_vehicle": vid,
                "inject_step": inject_step,
                "accel_boost_mps2": float(accel_boost),
                "min_gap_m": min_gap,
                "n_steps": len(vdf),
            }
        )

    return results, manifest


def synth_speed_spike(df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[pd.DataFrame], list[dict]]:
    """Synthesise speed spikes (sensor-spoofed or actuator-fault scenarios)."""
    vids = _pick_random_trajectories(df, N_SYNTHETIC_SPEED_SPIKE, rng)
    results = []
    manifest = []

    for i, vid in enumerate(vids):
        vdf = df[df["vehicle_id"] == vid].sort_values("step").copy()
        if len(vdf) < 20:
            continue

        new_vid = f"synth_speed_spike_{i:03d}"
        n = len(vdf)
        spike_start = int(rng.integers(int(n * 0.3), int(n * 0.6)))
        spike_duration = int(rng.integers(5, 15))
        spike_magnitude = rng.uniform(5, 15)  # m/s added

        vdf = vdf.copy()
        vdf["vehicle_id"] = new_vid

        speeds = vdf["speed_mps"].values.copy()
        for j in range(spike_start, min(spike_start + spike_duration, n)):
            speeds[j] = speeds[j] + spike_magnitude

        vdf["speed_mps"] = speeds
        vdf["is_crash_scenario"] = True
        vdf["crash_type"] = "synth_speed_spike"
        vdf["has_hard_brake"] = False
        vdf["has_close_gap"] = False
        vdf["has_critical_gap"] = False
        vdf["has_rss_violation"] = False
        vdf["accel_mps2"] = _compute_accel(vdf).values

        results.append(vdf)
        manifest.append(
            {
                "vehicle_id": new_vid,
                "crash_type": "synth_speed_spike",
                "base_vehicle": vid,
                "spike_start": spike_start,
                "spike_duration": spike_duration,
                "spike_magnitude_mps": float(spike_magnitude),
                "max_speed_mps": float(speeds.max()),
                "n_steps": len(vdf),
            }
        )

    return results, manifest


def build_crash_dataset(
    waymo_csv: Path = DEFAULT_WAYMO_CSV,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = RNG_SEED,
) -> tuple[Path, Path]:
    """Build combined crash dataset from real mining + synthetic injection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"Loading Waymo ORIUS corpus: {waymo_csv}")
    df = pd.read_csv(waymo_csv)
    print(f"  {len(df):,} rows, {df['vehicle_id'].nunique()} vehicles")

    # 1. Mine real crash/near-crash events
    print("\n--- Mining real crash events ---")
    real_crash_df, real_manifest = mine_real_crashes(df)
    print(f"  Found {len(real_manifest)} real crash/near-crash scenarios")
    if len(real_manifest) > 0:
        n_hb = sum(1 for m in real_manifest if m["has_hard_brake"])
        n_cg = sum(1 for m in real_manifest if m["has_close_gap"])
        n_cr = sum(1 for m in real_manifest if m["has_critical_gap"])
        n_rss = sum(1 for m in real_manifest if m["has_rss_violation"])
        print(
            f"    Hard brake: {n_hb}, Close gap (<{CLOSE_GAP_M}m): {n_cg}, "
            f"Critical gap (<{CRITICAL_GAP_M}m): {n_cr}, RSS violations: {n_rss}"
        )

    # 2. Synthetic lead-brake scenarios
    print("\n--- Generating synthetic lead-brake scenarios ---")
    lb_dfs, lb_manifest = synth_lead_brake(df, rng)
    print(f"  Generated {len(lb_manifest)} lead-brake scenarios")

    # 3. Synthetic gap-collapse scenarios
    print("\n--- Generating synthetic gap-collapse scenarios ---")
    gc_dfs, gc_manifest = synth_gap_collapse(df, rng)
    print(f"  Generated {len(gc_manifest)} gap-collapse scenarios")

    # 4. Synthetic speed-spike scenarios
    print("\n--- Generating synthetic speed-spike scenarios ---")
    ss_dfs, ss_manifest = synth_speed_spike(df, rng)
    print(f"  Generated {len(ss_manifest)} speed-spike scenarios")

    # Combine all crash data
    all_dfs = [real_crash_df, *lb_dfs, *gc_dfs, *ss_dfs]
    all_dfs = [d for d in all_dfs if len(d) > 0]

    if not all_dfs:
        print("WARNING: No crash scenarios found or generated!")
        return out_dir / "crash_trajectories_orius.csv", out_dir / "crash_manifest.json"

    crash_combined = pd.concat(all_dfs, ignore_index=True)

    # Also include a sample of normal (non-crash) trajectories for contrast
    crash_vids = set()
    for m in real_manifest:
        crash_vids.add(m["vehicle_id"])
    normal_df = df[~df["vehicle_id"].isin(crash_vids)].copy()
    # Sample up to 200 normal trajectories
    normal_vids = normal_df["vehicle_id"].unique()
    n_normal = min(200, len(normal_vids))
    sampled_normal = rng.choice(normal_vids, size=n_normal, replace=False)
    normal_sample = normal_df[normal_df["vehicle_id"].isin(sampled_normal)].copy()
    normal_sample["is_crash_scenario"] = False
    normal_sample["crash_type"] = "normal"
    normal_sample["has_hard_brake"] = False
    normal_sample["has_close_gap"] = False
    normal_sample["has_critical_gap"] = False
    normal_sample["has_rss_violation"] = False
    normal_sample["accel_mps2"] = 0.0  # Placeholder
    for vid in sampled_normal:
        mask = normal_sample["vehicle_id"] == vid
        vdf = normal_sample.loc[mask].sort_values("step")
        normal_sample.loc[mask, "accel_mps2"] = _compute_accel(vdf).values

    full_dataset = pd.concat([crash_combined, normal_sample], ignore_index=True)

    # Ensure ORIUS contract columns
    orius_cols = [
        "vehicle_id",
        "step",
        "position_m",
        "speed_mps",
        "speed_limit_mps",
        "lead_position_m",
        "ts_utc",
        "source_split",
        # Extended columns
        "is_crash_scenario",
        "crash_type",
        "accel_mps2",
        "lead_rel_x_m",
        "lead_speed_mps",
        "rss_safe_gap_m",
        "rss_violation_true",
        "has_hard_brake",
        "has_close_gap",
        "has_critical_gap",
        "has_rss_violation",
    ]
    # Add source_split if missing
    if "source_split" not in full_dataset.columns:
        full_dataset["source_split"] = "waymo_crash"

    # Keep only existing columns (some may not exist)
    existing_cols = [c for c in orius_cols if c in full_dataset.columns]
    full_dataset = full_dataset[existing_cols]

    # Write outputs
    csv_path = out_dir / "crash_trajectories_orius.csv"
    full_dataset.to_csv(csv_path, index=False)

    all_manifest = real_manifest + lb_manifest + gc_manifest + ss_manifest
    manifest_data = {
        "source": str(waymo_csv),
        "seed": seed,
        "total_crash_scenarios": len(all_manifest),
        "real_mined": len(real_manifest),
        "synth_lead_brake": len(lb_manifest),
        "synth_gap_collapse": len(gc_manifest),
        "synth_speed_spike": len(ss_manifest),
        "normal_contrast": int(n_normal),
        "total_rows": len(full_dataset),
        "crash_rows": int(crash_combined.shape[0]),
        "normal_rows": int(normal_sample.shape[0]),
        "thresholds": {
            "hard_brake_mps2": HARD_BRAKE_MS2,
            "close_gap_m": CLOSE_GAP_M,
            "critical_gap_m": CRITICAL_GAP_M,
        },
        "scenarios": all_manifest,
    }
    manifest_path = out_dir / "crash_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2, default=str) + "\n")

    # Summary
    print(f"\n{'=' * 70}")
    print("Crash AV Dataset Complete")
    print(f"{'=' * 70}")
    print(f"  Real mined:     {len(real_manifest)} scenarios")
    print(f"  Synth lead-brake:  {len(lb_manifest)} scenarios")
    print(f"  Synth gap-collapse: {len(gc_manifest)} scenarios")
    print(f"  Synth speed-spike:  {len(ss_manifest)} scenarios")
    print(f"  Normal contrast:    {n_normal} scenarios")
    print(f"  Total rows:         {len(full_dataset):,}")
    print(f"  Output CSV:         {csv_path}")
    print(f"  Crash manifest:     {manifest_path}")
    print(f"{'=' * 70}")

    return csv_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build crash AV dataset from Waymo + synthetic")
    parser.add_argument("--waymo-csv", type=Path, default=DEFAULT_WAYMO_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    if not args.waymo_csv.exists():
        print(f"ERROR: Waymo CSV not found: {args.waymo_csv}")
        return 1

    build_crash_dataset(args.waymo_csv, args.out_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
