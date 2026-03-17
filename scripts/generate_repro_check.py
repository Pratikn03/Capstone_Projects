#!/usr/bin/env python3
"""Generate reports/repro_check.json with reproducibility and integrity status."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "reports" / "repro_check.json"

REQUIRED_PATHS = [
    "paper/metrics_manifest.json",
    "paper/claim_matrix.csv",
    "scripts/validate_paper_claims.py",
    "scripts/sync_paper_assets.py",
    "reports/impact_summary.csv",
    "reports/eia930/impact_summary.csv",
    "reports/research_metrics_de.csv",
    "reports/research_metrics_us.csv",
    "configs/dc3s.yaml",
    "configs/optimization.yaml",
]


def main() -> int:
    payload: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {},
        "validator_pass": None,
        "sync_assets_pass": None,
        "negative_test_pass": None,
    }

    for rel in REQUIRED_PATHS:
        p = REPO_ROOT / rel
        payload["paths"][rel] = p.exists()

    # Run validate_paper_claims
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_paper_claims.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    payload["validator_pass"] = r.returncode == 0

    # Run sync_paper_assets --check
    r2 = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "sync_paper_assets.py"), "--check"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    payload["sync_assets_pass"] = r2.returncode == 0

    # Run claim validator negative test
    r3 = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_claim_validator_negative_test.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    payload["negative_test_pass"] = r3.returncode == 0

    payload["all_paths_exist"] = all(payload["paths"].values())
    payload["integrity_ok"] = (
        payload["validator_pass"]
        and payload["sync_assets_pass"]
        and payload["negative_test_pass"]
        and payload["all_paths_exist"]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0 if payload["integrity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
