#!/usr/bin/env python3
"""Run Paper 2, Paper 3, and CertOS evidence artifacts.

Produces artifact-backed outputs for:
- Paper 2: certificate_half_life_blackout.csv, certificate_half_life_metrics.json
- Paper 3: graceful_degradation_trace.csv (via generate_priority2_artifacts if 48h_trace exists)
- Paper 6: certos_lifecycle.csv, certos_summary.json

Usage:
    python scripts/run_evidence_artifacts.py [--skip-graceful]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _run(cmd: list[str], desc: str) -> bool:
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAIL {desc}: {r.stderr[:200] if r.stderr else r.stdout[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  FAIL {desc}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-graceful", action="store_true", help="Skip graceful (needs 48h_trace)")
    args = parser.parse_args()

    py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path("python3")

    ok = 0
    total = 0

    # Paper 2: half-life + blackout
    total += 1
    if _run([str(py), "scripts/run_certificate_half_life_blackout.py", "--seeds", "42", "--horizons", "0", "4", "12"], "Paper 2 half-life blackout"):
        csv = REPO / "reports/publication/certificate_half_life_blackout.csv"
        if csv.exists() and csv.stat().st_size > 0:
            ok += 1
            print(f"  PASS Paper 2: {csv}")
        else:
            print(f"  FAIL Paper 2: artifact missing or empty")
    else:
        print(f"  BLOCKED Paper 2: script failed")

    # Paper 6: CertOS lifecycle
    total += 1
    if _run([str(py), "scripts/run_certos_lifecycle.py", "--steps", "96", "--out", "reports/certos"], "CertOS lifecycle"):
        csv = REPO / "reports/certos/certos_lifecycle.csv"
        j = REPO / "reports/certos/certos_summary.json"
        if csv.exists() and j.exists() and csv.stat().st_size > 0:
            ok += 1
            print(f"  PASS CertOS: {csv}, {j}")
        else:
            print(f"  FAIL CertOS: artifact missing or empty")
    else:
        print(f"  BLOCKED CertOS: script failed")

    # Paper 3: graceful degradation (depends on 48h_trace)
    if not args.skip_graceful:
        total += 1
        trace_48 = REPO / "reports/publication/48h_trace.csv"
        if trace_48.exists():
            if _run([str(py), "scripts/generate_priority2_artifacts.py"], "Paper 3 graceful"):
                gd = REPO / "reports/publication/graceful_degradation_trace.csv"
                if gd.exists() and gd.stat().st_size > 0:
                    ok += 1
                    print(f"  PASS Paper 3: {gd}")
                else:
                    print(f"  FAIL Paper 3: artifact missing")
            else:
                print(f"  BLOCKED Paper 3: generate_priority2_artifacts failed")
        else:
            print(f"  BLOCKED Paper 3: 48h_trace.csv missing (run generate_48h_trace first)")

    print(f"\n=== Evidence artifacts: {ok}/{total} PASS ===")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
