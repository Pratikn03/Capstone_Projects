# /// script
# dependencies = ["pandas", "pyarrow", "torch", "matplotlib", "scikit-learn"]
# ///
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("GRIDPULSE_REPO_ROOT", ".")).resolve()
PYTHON = sys.executable
OUT_DIR = Path(os.environ.get("ORIUS_DEEP_OUT_DIR", "reports/publication"))
MODEL_DIR = Path(os.environ.get("ORIUS_DEEP_MODEL_DIR", "artifacts/deep_oqe"))
PAPER_TABLE_DIR = Path(os.environ.get("ORIUS_PAPER_TABLE_DIR", "paper/assets/tables/generated"))
PAPER_FIG_DIR = Path(os.environ.get("ORIUS_PAPER_FIG_DIR", "paper/assets/figures"))

EXPECTED_GENERATED_FILES = [
    OUT_DIR / "battery_deep_oqe_summary.csv",
    OUT_DIR / "battery_deep_oqe_summary.json",
    OUT_DIR / "battery_deep_oqe_summary.md",
    OUT_DIR / "battery_deep_oqe_buckets.csv",
    OUT_DIR / "fig_battery_deep_oqe_summary.png",
    OUT_DIR / "battery_deep_oqe_safety_metrics.csv",
    OUT_DIR / "battery_deep_oqe_safety_metrics.md",
    OUT_DIR / "fig_battery_deep_oqe_safety_metrics.png",
    OUT_DIR / "battery_deep_oqe_runtime_summary.csv",
    OUT_DIR / "battery_deep_oqe_runtime_traces.csv",
    OUT_DIR / "battery_deep_oqe_fault_family_coverage.csv",
    OUT_DIR / "battery_deep_oqe_runtime.duckdb",
    OUT_DIR / "battery_raw_sequence_track_benchmark.csv",
    OUT_DIR / "battery_raw_sequence_track_benchmark.md",
    OUT_DIR / "battery_raw_sequence_track_slices.csv",
    OUT_DIR / "fig_battery_raw_sequence_track_benchmark.png",
    OUT_DIR / "battery_deep_learning_novelty_register.json",
    OUT_DIR / "battery_deep_learning_novelty_register.md",
    MODEL_DIR / "battery_deepoqe.pt",
    PAPER_TABLE_DIR / "tbl_battery_deep_oqe_summary.tex",
    PAPER_TABLE_DIR / "tbl_battery_deep_oqe_safety_metrics.tex",
    PAPER_TABLE_DIR / "tbl_battery_raw_sequence_track.tex",
    PAPER_FIG_DIR / "fig_battery_deep_oqe_summary.png",
    PAPER_FIG_DIR / "fig_battery_deep_oqe_safety_metrics.png",
    PAPER_FIG_DIR / "fig_battery_raw_sequence_track_benchmark.png",
]


def run(*args: str) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or inspect the bounded battery deep-learning novelty HF job."
    )
    parser.add_argument(
        "--list-outputs",
        action="store_true",
        help="Print the files this job is expected to generate, then exit without running training.",
    )
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    resolved = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> None:
    args = parse_args()
    if args.list_outputs:
        print(
            json.dumps(
                {
                    "downloads_directly": False,
                    "repo_root": str(REPO_ROOT),
                    "generated_files": [repo_relative(path) for path in EXPECTED_GENERATED_FILES],
                    "validators": [
                        "scripts/verify_paper_manifest.py",
                        "scripts/validate_paper_claims.py",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    cmd = [
        PYTHON,
        "scripts/run_battery_deep_novelty.py",
        "--out-dir",
        str(OUT_DIR),
        "--model-dir",
        str(MODEL_DIR),
        "--paper-table-dir",
        str(PAPER_TABLE_DIR),
        "--paper-fig-dir",
        str(PAPER_FIG_DIR),
        "--deep-oqe-epochs",
        os.environ.get("ORIUS_DEEP_OQE_EPOCHS", "12"),
        "--forecast-epochs",
        os.environ.get("ORIUS_DEEP_FORECAST_EPOCHS", "8"),
        "--batch-size",
        os.environ.get("ORIUS_DEEP_BATCH_SIZE", "128"),
    ]
    run(*cmd)
    run(PYTHON, "scripts/verify_paper_manifest.py")
    run(PYTHON, "scripts/validate_paper_claims.py")


if __name__ == "__main__":
    main()
