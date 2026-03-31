#!/usr/bin/env python3
"""Build ORIUS navigation trajectories from external KITTI Odometry raw data."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.data_pipeline.external_raw import EXTERNAL_DATA_ROOT_ENV, get_external_dataset_dir


DEFAULT_OUT = REPO_ROOT / "data" / "navigation" / "processed" / "navigation_orius.csv"


def _find_poses_dir(root: Path) -> Path:
    for candidate in (
        root / "dataset" / "poses",
        root / "poses",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find KITTI poses directory under {root}")


def _find_sequence_dir(root: Path, sequence_id: str) -> Path:
    for candidate in (
        root / "dataset" / "sequences" / sequence_id,
        root / "sequences" / sequence_id,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find KITTI sequence directory for {sequence_id} under {root}")


def _load_times(sequence_dir: Path) -> np.ndarray:
    times_path = sequence_dir / "times.txt"
    if not times_path.exists():
        raise FileNotFoundError(f"Missing KITTI times.txt at {times_path}")
    values = np.loadtxt(times_path, dtype=float)
    if values.ndim == 0:
        values = np.array([float(values)])
    return values


def _load_poses(poses_dir: Path, sequence_id: str) -> np.ndarray:
    poses_path = poses_dir / f"{sequence_id}.txt"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing KITTI poses file at {poses_path}")
    matrix = np.loadtxt(poses_path, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != 12:
        raise ValueError(f"Expected KITTI 3x4 poses flattened to 12 columns, got {matrix.shape[1]}")
    return matrix


def _sequence_frame(sequence_id: str, times: np.ndarray, poses: np.ndarray) -> pd.DataFrame:
    if len(times) != len(poses):
        raise ValueError(f"Sequence {sequence_id} times/poses length mismatch: {len(times)} vs {len(poses)}")

    x = poses[:, 3]
    y = poses[:, 11]
    dt = np.diff(times, prepend=times[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.clip(dt, 1e-6, None)
    vx = np.gradient(x, times, edge_order=1)
    vy = np.gradient(y, times, edge_order=1)
    start = datetime(2011, 9, 26, tzinfo=timezone.utc)

    return pd.DataFrame(
        {
            "robot_id": f"kitti_{sequence_id}",
            "step": np.arange(len(times), dtype=int),
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "ts_utc": [
                (start + timedelta(seconds=float(offset))).isoformat().replace("+00:00", "Z")
                for offset in times
            ],
            "source_sequence": sequence_id,
        }
    )


def build_navigation_dataset(
    out_path: Path,
    *,
    external_root: Path | None = None,
    sequences: list[str] | None = None,
    manifest_out: Path | None = None,
) -> Path:
    """Convert KITTI Odometry sequences into the ORIUS navigation contract."""
    root = get_external_dataset_dir("kitti_odometry", external_root, required=True)
    assert root is not None
    poses_dir = _find_poses_dir(root)

    if sequences:
        wanted = sequences
    else:
        wanted = sorted(path.stem for path in poses_dir.glob("*.txt"))
    if not wanted:
        raise FileNotFoundError(f"No KITTI pose files found under {poses_dir}")

    frames: list[pd.DataFrame] = []
    sequence_rows: dict[str, int] = {}
    for sequence_id in wanted:
        sequence_dir = _find_sequence_dir(root, sequence_id)
        times = _load_times(sequence_dir)
        poses = _load_poses(poses_dir, sequence_id)
        frame = _sequence_frame(sequence_id, times, poses)
        frames.append(frame)
        sequence_rows[sequence_id] = int(len(frame))

    df = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    if manifest_out is not None:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(
            json.dumps(
                {
                    "dataset": "KITTI Odometry",
                    "external_root": str(root),
                    "sequences": wanted,
                    "rows_per_sequence": sequence_rows,
                    "output_csv": str(out_path),
                    "source_env": EXTERNAL_DATA_ROOT_ENV,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build navigation_orius.csv from external KITTI Odometry data")
    parser.add_argument("--external-root", type=Path, default=None, help="Override ORIUS_EXTERNAL_DATA_ROOT")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output navigation CSV")
    parser.add_argument(
        "--sequence",
        dest="sequences",
        action="append",
        default=[],
        help="Specific KITTI sequence to include (repeatable). Defaults to all discovered sequences.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=REPO_ROOT / "data" / "navigation" / "raw" / "external_build_manifest.json",
        help="Write lightweight build metadata here",
    )
    args = parser.parse_args()

    try:
        output = build_navigation_dataset(
            args.out,
            external_root=args.external_root,
            sequences=args.sequences or None,
            manifest_out=args.manifest_out,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    print(f"Navigation dataset -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
