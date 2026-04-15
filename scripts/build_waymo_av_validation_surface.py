#!/usr/bin/env python3
"""Build Stage 2 validation artifacts for the governed Waymo AV surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orius.av_waymo import build_validation_surface


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "orius_av" / "raw" / "waymo_motion" / "validation"
DEFAULT_OUT = REPO_ROOT / "data" / "orius_av" / "av" / "processed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Waymo AV validation artifacts")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW, help="Directory containing validation TFRecord shards")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory for Stage 2 parquet artifacts")
    parser.add_argument("--max-shards", type=int, default=None, help="Optionally limit the number of shards scanned")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Optionally limit the number of scenarios parsed")
    parser.add_argument("--verify-crc", action="store_true", help="Verify TFRecord CRC checksums when possible")
    parser.add_argument("--skip-actor-tracks", action="store_true", help="Do not materialize actor_tracks.parquet")
    args = parser.parse_args()

    report = build_validation_surface(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        max_shards=args.max_shards,
        max_scenarios=args.max_scenarios,
        verify_crc=args.verify_crc,
        write_actor_tracks=not args.skip_actor_tracks,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
