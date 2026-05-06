#!/usr/bin/env python3
"""Run claim validator negative test: prove it fails when a locked claim is changed.

Temporarily corrupts C001 canonical_value (2.0.0 -> 9.9.9), runs validator,
captures output to reports/claim_validator_negative_test.log, then restores.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIM_MATRIX = REPO_ROOT / "paper" / "claim_matrix.csv"
LOG_PATH = REPO_ROOT / "reports" / "claim_validator_negative_test.log"


def main() -> int:
    content = CLAIM_MATRIX.read_text(encoding="utf-8")
    original = content
    # Replace only the C001 row's canonical_value
    lines = content.splitlines()
    out_lines = []
    for line in lines:
        if line.startswith("C001,"):
            line = line.replace(",2.0.0,", ",9.9.9,", 1)
        out_lines.append(line)
    corrupted = "\n".join(out_lines) + "\n"

    try:
        CLAIM_MATRIX.write_text(corrupted, encoding="utf-8")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "validate_paper_claims.py")],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        log_content = f"""# Claim validator negative test
# Proves validator fails when a locked claim (C001) is intentionally changed.
# Procedure: C001 canonical_value 2.0.0 -> 9.9.9, run validator, capture output.

## Corrupted state
C001 canonical_value was temporarily changed from 2.0.0 to 9.9.9.

## Validator output (exit code {result.returncode})
stdout:
{result.stdout}

stderr:
{result.stderr}

## Expected
Exit code 1 (validator must fail when locked claim is changed).
"""
        LOG_PATH.write_text(log_content, encoding="utf-8")
        if result.returncode != 1:
            print(f"FAIL: Expected exit 1, got {result.returncode}")
            return 1
        print("PASS: Validator correctly failed when C001 was corrupted")
        return 0
    finally:
        CLAIM_MATRIX.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
