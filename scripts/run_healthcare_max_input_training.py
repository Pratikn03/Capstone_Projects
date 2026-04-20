#!/usr/bin/env python3
"""Build and train the max-input healthcare surface.

This keeps the existing submission-facing healthcare surface intact while
providing a repeatable larger-data lane that merges the richer BIDMC bridge and
the staged MIMIC-III waveform bridge.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.data_pipeline.build_features_healthcare import (  # noqa: E402
    DEFAULT_BIDMC_BRIDGE,
    DEFAULT_MAX_INPUT_OUT,
    DEFAULT_MIMIC3_BRIDGE,
    build_max_input_features,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "train_forecast_healthcare_max_input.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and train the max-input healthcare lane")
    parser.add_argument("--input", dest="inputs", type=Path, action="append", default=None, help="Input healthcare CSV")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MAX_INPUT_OUT, help="Output directory for max-input features")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training config")
    parser.add_argument("--skip-build", action="store_true", help="Skip feature build")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--targets", default=None, help="Optional comma-separated target override")
    args = parser.parse_args()

    inputs = args.inputs or [DEFAULT_BIDMC_BRIDGE, DEFAULT_MIMIC3_BRIDGE]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing healthcare source: {path}")

    if not args.skip_build:
        features_path = build_max_input_features(inputs, args.out_dir)
        print(f"Built max-input healthcare features: {features_path}")

    if not args.skip_train:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        src_path = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
        cmd = [sys.executable, "-m", "orius.forecasting.train", "--config", str(args.config)]
        if args.targets:
            cmd.extend(["--targets", args.targets])
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
