#!/usr/bin/env python3
"""Evaluate whether the repo may honestly claim Phases 3/4/6 are closed."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from _active_theorem_program import REPORTS_DIR, build_active_theorem_audit_payload


FINDINGS_CSV = REPORTS_DIR / "external_proof_audit_findings.csv"
STATUS_JSON = REPORTS_DIR / "phase_3_4_6_closure_status.json"


def _load_findings() -> dict[str, dict[str, str]]:
    if not FINDINGS_CSV.exists():
        return {}
    with FINDINGS_CSV.open(encoding="utf-8", newline="") as handle:
        return {row["theorem_id"]: row for row in csv.DictReader(handle) if row.get("theorem_id")}


def build_status() -> dict[str, object]:
    payload = build_active_theorem_audit_payload()
    findings = _load_findings()
    defended_rows = [
        row
        for row in payload["theorems"]
        if row["defense_tier"] in {"flagship_defended", "supporting_defended"}
    ]
    review_ids = set(payload["summary"]["flagship_defended_ids"]) | {"T6", "T8"}

    phase3_blockers = [
        {
            "theorem_id": row["theorem_id"],
            "reason": (
                "unresolved theorem-local assumptions"
                if row["unresolved_assumptions"]
                else "legacy aliases still present"
            ),
        }
        for row in defended_rows
        if row["unresolved_assumptions"] or row["legacy_aliases"]
    ]
    phase4_blockers = [
        {
            "theorem_id": row["theorem_id"],
            "reason": f"code correspondence is '{row['code_correspondence']}'",
        }
        for row in defended_rows
        if row["theorem_id"] in {"T3a", "T6", "T11"} and row["code_correspondence"] != "matches"
    ]
    phase6_blockers = []
    for theorem_id in sorted(review_ids):
        finding = findings.get(theorem_id)
        if not finding:
            phase6_blockers.append({"theorem_id": theorem_id, "reason": "missing external-audit finding row"})
            continue
        disposition = str(finding.get("disposition", "")).strip().lower()
        status = str(finding.get("status", "")).strip().lower()
        if disposition in {"", "pending_external_review"} or status in {"", "pending_external_review", "open"}:
            phase6_blockers.append(
                {
                    "theorem_id": theorem_id,
                    "reason": "external-audit disposition not closed",
                }
            )

    phase3_ready = len(phase3_blockers) == 0
    phase4_ready = len(phase4_blockers) == 0
    phase6_ready = len(phase6_blockers) == 0
    overall_ready = phase3_ready and phase4_ready and phase6_ready
    return {
        "phase_3_ready": phase3_ready,
        "phase_4_ready": phase4_ready,
        "phase_6_ready": phase6_ready,
        "overall_ready": overall_ready,
        "completion_claim_allowed": overall_ready,
        "phase_3_blockers": phase3_blockers,
        "phase_4_blockers": phase4_blockers,
        "phase_6_blockers": phase6_blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit nonzero unless Phases 3/4/6 are all closed without blockers",
    )
    args = parser.parse_args(argv)

    status = build_status()
    STATUS_JSON.write_text(json.dumps(status, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    if args.require_complete and not bool(status["overall_ready"]):
        print("[verify_phase_346_closure] FAIL")
        for key in ("phase_3_blockers", "phase_4_blockers", "phase_6_blockers"):
            for blocker in status[key]:
                print(f"- {blocker['theorem_id']}: {blocker['reason']}")
        return 1

    print(
        "[verify_phase_346_closure] "
        + ("PASS" if bool(status["overall_ready"]) else "INCOMPLETE")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
