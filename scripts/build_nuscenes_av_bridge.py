#!/usr/bin/env python3
"""Bridge nuScenes data to the ORIUS longitudinal AV contract.

This script:
1. Loads a nuScenes dataroot (v1.0-mini or v1.0-trainval).
2. Extracts ego + nearby-vehicle trajectories at 2 Hz key-frame rate.
3. Labels near-crash / hard-braking scenarios using:
   a. Ego deceleration > threshold (hard-brake proxy).
   b. Minimum gap-to-lead below threshold (near-collision proxy).
   c. Scene description keywords ("collision", "crash", "accident", etc.).
4. Outputs an ORIUS-contract CSV + crash-scenario manifest.

Usage:
    python scripts/build_nuscenes_av_bridge.py --dataroot /data/sets/nuscenes --version v1.0-mini
    python scripts/build_nuscenes_av_bridge.py --dataroot /data/sets/nuscenes --version v1.0-trainval

Prerequisites:
    pip install nuscenes-devkit
    Download nuScenes from https://www.nuscenes.org/download (requires account).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# nuScenes crash / near-crash heuristics
# ---------------------------------------------------------------------------
HARD_BRAKE_MS2 = -4.0  # deceleration threshold (m/s²)
NEAR_COLLISION_GAP_M = 3.0  # minimum gap to lead vehicle (m)
CRASH_KEYWORDS = frozenset(
    [
        "collision",
        "crash",
        "accident",
        "hit",
        "rear-end",
        "sideswipe",
        "near-miss",
        "near miss",
        "emergency",
        "hard brake",
        "hard-brake",
        "evasive",
    ]
)
SPEED_LIMIT_DEFAULT_MPS = 13.4  # 30 mph — typical urban nuScenes


def _quaternion_yaw(q: dict) -> float:
    """Extract yaw angle from a nuScenes quaternion {w, x, y, z}."""
    from pyquaternion import Quaternion

    return float(Quaternion(q["w"], q["x"], q["y"], q["z"]).yaw_pitch_roll[0])


def _scene_has_crash_keyword(description: str) -> bool:
    low = description.lower()
    return any(kw in low for kw in CRASH_KEYWORDS)


def _extract_ego_trajectory(nusc: Any, scene: dict) -> pd.DataFrame:
    """Extract ego poses for a scene at key-frame rate (~2 Hz)."""
    rows: list[dict[str, Any]] = []
    sample_token = scene["first_sample_token"]
    step = 0

    while sample_token:
        sample = nusc.get("sample", sample_token)
        # Ego pose from the lidar sample_data (canonical sensor)
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd = nusc.get("sample_data", lidar_token)
        ego = nusc.get("ego_pose", sd["ego_pose_token"])

        x, y, z = ego["translation"]
        ts_us = ego["timestamp"]  # microseconds

        rows.append(
            {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "timestamp_us": int(ts_us),
                "step": step,
            }
        )

        sample_token = sample["next"]
        step += 1

    df = pd.DataFrame(rows)
    if len(df) < 2:
        return df

    # Compute speed from finite differences
    dt = df["timestamp_us"].diff() / 1e6  # seconds
    dx = df["x"].diff()
    dy = df["y"].diff()
    df["speed_mps"] = np.sqrt(dx**2 + dy**2) / dt
    df["speed_mps"] = df["speed_mps"].fillna(0.0)

    # Compute acceleration
    df["accel_mps2"] = df["speed_mps"].diff() / dt
    df["accel_mps2"] = df["accel_mps2"].fillna(0.0)

    # Position along longitudinal axis (cumulative arc-length)
    df["position_m"] = np.sqrt(dx.fillna(0) ** 2 + dy.fillna(0) ** 2).cumsum()

    return df


def _extract_nearby_tracks(nusc: Any, scene: dict, ego_df: pd.DataFrame) -> pd.DataFrame:
    """Extract annotations of nearby vehicles in each key-frame."""
    vehicle_categories = {
        "vehicle.car",
        "vehicle.truck",
        "vehicle.bus.bendy",
        "vehicle.bus.rigid",
        "vehicle.motorcycle",
        "vehicle.bicycle",
        "vehicle.emergency.ambulance",
        "vehicle.emergency.police",
        "vehicle.construction",
        "vehicle.trailer",
    }

    rows: list[dict[str, Any]] = []
    sample_token = scene["first_sample_token"]
    step = 0

    while sample_token:
        sample = nusc.get("sample", sample_token)
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]
            if cat not in vehicle_categories:
                continue

            ax, ay, az = ann["translation"]
            rows.append(
                {
                    "instance_token": ann["instance_token"],
                    "x": float(ax),
                    "y": float(ay),
                    "step": step,
                    "timestamp_us": int(sample["timestamp"]),
                    "category": cat,
                }
            )

        sample_token = sample["next"]
        step += 1

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _compute_lead_gap(ego_df: pd.DataFrame, tracks_df: pd.DataFrame) -> pd.Series:
    """For each ego step, find the closest vehicle ahead (positive x direction)."""
    gaps = pd.Series(np.nan, index=ego_df.index)
    if tracks_df.empty:
        return gaps

    for idx, ego_row in ego_df.iterrows():
        step_tracks = tracks_df[tracks_df["step"] == ego_row["step"]]
        if step_tracks.empty:
            continue

        # Distances in 2D
        dx = step_tracks["x"].values - ego_row["x"]
        dy = step_tracks["y"].values - ego_row["y"]
        dists = np.sqrt(dx**2 + dy**2)

        # Only consider vehicles within 100 m
        mask = dists < 100.0
        if mask.any():
            gaps.iloc[idx] = float(dists[mask].min())

    return gaps


def _label_crash_scenarios(
    ego_df: pd.DataFrame,
    lead_gap: pd.Series,
    scene_description: str,
) -> dict[str, Any]:
    """Label a scene as crash/near-crash based on heuristics."""
    labels: dict[str, Any] = {
        "has_hard_brake": False,
        "has_near_collision": False,
        "has_crash_keyword": False,
        "is_crash_scenario": False,
        "min_lead_gap_m": float("inf"),
        "max_decel_mps2": 0.0,
        "hard_brake_steps": [],
        "near_collision_steps": [],
    }

    if "accel_mps2" in ego_df.columns:
        min_accel = float(ego_df["accel_mps2"].min())
        labels["max_decel_mps2"] = min_accel
        if min_accel < HARD_BRAKE_MS2:
            labels["has_hard_brake"] = True
            labels["hard_brake_steps"] = ego_df.index[ego_df["accel_mps2"] < HARD_BRAKE_MS2].tolist()

    if lead_gap.notna().any():
        min_gap = float(lead_gap.min())
        labels["min_lead_gap_m"] = min_gap
        if min_gap < NEAR_COLLISION_GAP_M:
            labels["has_near_collision"] = True
            labels["near_collision_steps"] = ego_df.index[lead_gap < NEAR_COLLISION_GAP_M].tolist()

    labels["has_crash_keyword"] = _scene_has_crash_keyword(scene_description)

    labels["is_crash_scenario"] = (
        labels["has_hard_brake"] or labels["has_near_collision"] or labels["has_crash_keyword"]
    )
    return labels


def bridge_nuscenes(
    dataroot: str | Path,
    version: str = "v1.0-mini",
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Run the full nuScenes → ORIUS bridge.

    Returns (orius_csv_path, crash_manifest_path).
    """
    from nuscenes.nuscenes import NuScenes

    dataroot = Path(dataroot)
    if out_dir is None:
        out_dir = REPO_ROOT / "data" / "orius_av" / "nuscenes" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nuScenes {version} from {dataroot} ...")
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    print(f"  {len(nusc.scene)} scenes, {len(nusc.sample)} samples")

    all_rows: list[dict[str, Any]] = []
    crash_manifest: list[dict[str, Any]] = []
    scene_summary: list[dict[str, Any]] = []

    for scene_idx, scene in enumerate(nusc.scene):
        scene_name = scene["name"]
        scene_desc = scene.get("description", "")
        print(f"  [{scene_idx + 1}/{len(nusc.scene)}] {scene_name}: {scene_desc[:60]}")

        # Extract ego trajectory
        ego_df = _extract_ego_trajectory(nusc, scene)
        if len(ego_df) < 3:
            print(f"    Skipping (too short: {len(ego_df)} frames)")
            continue

        # Extract nearby vehicle tracks
        tracks_df = _extract_nearby_tracks(nusc, scene, ego_df)

        # Compute lead gap
        lead_gap = _compute_lead_gap(ego_df, tracks_df)

        # Label crash/near-crash
        labels = _label_crash_scenarios(ego_df, lead_gap, scene_desc)

        # Summary
        scene_info = {
            "scene_name": scene_name,
            "description": scene_desc,
            "n_frames": len(ego_df),
            "n_nearby_vehicles": int(tracks_df["instance_token"].nunique()) if not tracks_df.empty else 0,
            "speed_mean_mps": float(ego_df["speed_mps"].mean()),
            "speed_max_mps": float(ego_df["speed_mps"].max()),
            **labels,
        }
        scene_summary.append(scene_info)

        if labels["is_crash_scenario"]:
            crash_manifest.append(scene_info)
            tag = "CRASH"
        else:
            tag = "normal"

        # Build ORIUS rows
        for _, row in ego_df.iterrows():
            orius_row: dict[str, Any] = {
                "vehicle_id": f"ego_{scene_name}",
                "step": int(row["step"]),
                "position_m": float(row.get("position_m", 0.0)),
                "speed_mps": float(row.get("speed_mps", 0.0)),
                "speed_limit_mps": SPEED_LIMIT_DEFAULT_MPS,
                "lead_position_m": (
                    float(row["position_m"] + lead_gap.iloc[int(row["step"])])
                    if pd.notna(lead_gap.iloc[int(row["step"])])
                    else ""
                ),
                "ts_utc": pd.Timestamp(row["timestamp_us"], unit="us", tz="UTC")
                .isoformat()
                .replace("+00:00", "Z"),
                "source_split": "nuscenes_" + version.replace("v1.0-", ""),
                "scene_name": scene_name,
                "is_crash_scenario": labels["is_crash_scenario"],
                "accel_mps2": float(row.get("accel_mps2", 0.0)),
            }
            all_rows.append(orius_row)

        # Also add nearby vehicle tracks in ORIUS format
        if not tracks_df.empty:
            # Compute arc-length position per instance
            for inst_token, inst_df in tracks_df.groupby("instance_token"):
                if len(inst_df) < 2:
                    continue
                inst_df = inst_df.sort_values("step").copy()
                dx = inst_df["x"].diff().fillna(0)
                dy = inst_df["y"].diff().fillna(0)
                inst_df["position_m"] = np.sqrt(dx**2 + dy**2).cumsum()
                dt = inst_df["timestamp_us"].diff() / 1e6
                speed = np.sqrt(dx**2 + dy**2) / dt
                inst_df["speed_mps"] = speed.fillna(0.0)

                for _, r in inst_df.iterrows():
                    all_rows.append(
                        {
                            "vehicle_id": f"npc_{scene_name}_{str(inst_token)[:8]}",
                            "step": int(r["step"]),
                            "position_m": float(r["position_m"]),
                            "speed_mps": float(r["speed_mps"]),
                            "speed_limit_mps": SPEED_LIMIT_DEFAULT_MPS,
                            "lead_position_m": "",
                            "ts_utc": pd.Timestamp(r["timestamp_us"], unit="us", tz="UTC")
                            .isoformat()
                            .replace("+00:00", "Z"),
                            "source_split": "nuscenes_" + version.replace("v1.0-", ""),
                            "scene_name": scene_name,
                            "is_crash_scenario": labels["is_crash_scenario"],
                            "accel_mps2": 0.0,
                        }
                    )

        print(
            f"    {tag}  frames={len(ego_df)}  "
            f"speed=[{ego_df['speed_mps'].min():.1f}, {ego_df['speed_mps'].max():.1f}] m/s  "
            f"min_gap={labels['min_lead_gap_m']:.1f}m  "
            f"max_decel={labels['max_decel_mps2']:.1f} m/s²  "
            f"tracks={scene_info['n_nearby_vehicles']}"
        )

    # ---- Write outputs ----
    orius_df = pd.DataFrame(all_rows)
    orius_path = out_dir / "nuscenes_trajectories_orius.csv"
    orius_df.to_csv(orius_path, index=False)

    crash_path = out_dir / "nuscenes_crash_manifest.json"
    manifest_data = {
        "version": version,
        "dataroot": str(dataroot),
        "total_scenes": len(nusc.scene),
        "crash_scenes": len(crash_manifest),
        "normal_scenes": len(nusc.scene) - len(crash_manifest),
        "heuristics": {
            "hard_brake_threshold_mps2": HARD_BRAKE_MS2,
            "near_collision_gap_m": NEAR_COLLISION_GAP_M,
            "crash_keywords": sorted(CRASH_KEYWORDS),
        },
        "scenes": scene_summary,
        "crash_manifest": crash_manifest,
    }
    crash_path.write_text(json.dumps(manifest_data, indent=2, default=str) + "\n")

    # Summary
    n_crash = len(crash_manifest)
    n_total = len(nusc.scene)
    total_rows = len(orius_df)
    ego_rows = orius_df[orius_df["vehicle_id"].str.startswith("ego_")].shape[0]
    npc_rows = total_rows - ego_rows

    print(f"\n{'=' * 70}")
    print("nuScenes → ORIUS Bridge Complete")
    print(f"{'=' * 70}")
    print(f"  Version:        {version}")
    print(
        f"  Scenes:         {n_total} total, {n_crash} crash/near-crash ({100 * n_crash / max(n_total, 1):.0f}%)"
    )
    print(f"  ORIUS rows:     {total_rows:,} ({ego_rows:,} ego + {npc_rows:,} NPC)")
    print(f"  Output CSV:     {orius_path}")
    print(f"  Crash manifest: {crash_path}")
    print(f"{'=' * 70}")

    return orius_path, crash_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge nuScenes to ORIUS AV contract")
    parser.add_argument(
        "--dataroot",
        type=Path,
        required=True,
        help="Path to nuScenes data root (should contain v1.0-mini/ or v1.0-trainval/)",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
        help="nuScenes version to load (default: v1.0-mini)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/orius_av/nuscenes/processed/)",
    )
    args = parser.parse_args()

    if not args.dataroot.exists():
        print(f"ERROR: dataroot does not exist: {args.dataroot}")
        print("\nTo get nuScenes data:")
        print("  1. Create account at https://www.nuscenes.org/download")
        print("  2. Download v1.0-mini (or v1.0-trainval for full)")
        print("  3. Extract to a directory, e.g. /data/sets/nuscenes/")
        print("  4. Re-run with --dataroot /data/sets/nuscenes")
        return 1

    orius_path, crash_path = bridge_nuscenes(
        dataroot=args.dataroot,
        version=args.version,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
