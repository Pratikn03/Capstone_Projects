#!/usr/bin/env python3
"""Refresh the canonical ORIUS three-domain closure lane."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
REFRESH_DIR = REPO_ROOT / "reports" / "closure_refresh"
PYTHON = sys.executable


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run(label: str, cmd: list[str], logs_dir: Path) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    log_path = logs_dir / f"{label}.log"
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}. See {log_path}.")
    return {
        "label": label,
        "command": cmd,
        "returncode": completed.returncode,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the canonical Battery + AV + Healthcare lane")
    parser.add_argument(
        "--mode",
        choices=["three_domain_lane", "canonical"],
        default="three_domain_lane",
        help="Accepted for backward compatibility; both values refresh the same canonical 3-domain lane.",
    )
    parser.add_argument("--out", type=Path, default=REFRESH_DIR)
    parser.add_argument(
        "--external-root", type=Path, default=None, help="Accepted for compatibility but not required."
    )
    parser.add_argument("--train-missing", action="store_true", help="Accepted for compatibility.")
    parser.add_argument("--repair-invalid-splits", action="store_true", help="Accepted for compatibility.")
    parser.add_argument("--seeds", type=int, default=1, help="Accepted for compatibility.")
    parser.add_argument("--sil-seeds", type=int, default=1, help="Accepted for compatibility.")
    parser.add_argument("--sil-rows", type=int, default=48, help="Accepted for compatibility.")
    parser.add_argument("--horizon", type=int, default=24, help="Accepted for compatibility.")
    args = parser.parse_args()

    logs_dir = args.out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        _run("refresh_manifests", [PYTHON, "scripts/refresh_real_data_manifests.py"], logs_dir),
        _run(
            "validation",
            [
                PYTHON,
                "scripts/run_universal_orius_validation.py",
                "--out",
                "reports/universal_orius_validation",
            ],
            logs_dir,
        ),
        _run(
            "training_audit",
            [
                PYTHON,
                "scripts/run_universal_training_audit.py",
                "--out",
                "reports/orius_framework_proof/training_audit",
            ],
            logs_dir,
        ),
        _run(
            "domain_closure",
            [
                PYTHON,
                "scripts/build_domain_closure_matrix.py",
                "--validation-report",
                "reports/universal_orius_validation/validation_report.json",
                "--training-report",
                "reports/orius_framework_proof/training_audit/training_audit_report.json",
                "--out",
                "reports/universal_orius_validation",
            ],
            logs_dir,
        ),
        _run("monograph_assets", [PYTHON, "scripts/build_orius_monograph_assets.py"], logs_dir),
    ]

    payload = {
        "generated_at_utc": _utc_now_iso(),
        "mode": "three_domain_lane",
        "submission_scope": "battery_av_healthcare",
        "steps": steps,
        "scorecard_path": str((PUBLICATION_DIR / "orius_submission_scorecard.csv").relative_to(REPO_ROOT)),
        "closure_matrix_path": str(
            (PUBLICATION_DIR / "orius_domain_closure_matrix.csv").relative_to(REPO_ROOT)
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "closure_refresh_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
