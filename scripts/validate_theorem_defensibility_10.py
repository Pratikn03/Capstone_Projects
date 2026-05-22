#!/usr/bin/env python3
"""Validate the strict ORIUS theorem-defensibility 10/10 gate.

This gate is intentionally stricter than promotion validation.  Promotion asks
whether a theorem surface has a statement, proof, code, tests, artifacts, and a
claim boundary.  The 10/10 gate asks whether the whole active theorem package is
defensible as a closed release surface:

* no active theorem has partial code correspondence;
* no defended theorem is draft/non-defended;
* no active theorem carries unresolved assumptions;
* active theorem rows have statement/proof/code/test anchors;
* flagship theorem rows have complete promotion cards;
* the Lean formal core has no executable `sorry`, `admit`, or `axiom`;
* optionally, the Lean formal core builds with `lake build`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = Path("reports/publication")
ACTIVE_AUDIT = PUBLICATION / "active_theorem_audit.json"
PROMOTION_MATRIX = PUBLICATION / "theorem_promotion_matrix.json"
FORMAL_DIR = Path("formal")
FORBIDDEN_LEAN_TOKENS = ("sorry", "admit", "axiom")
ALLOWED_RIGOR_RATINGS = {
    "paper_rigorous",
    "proof_runtime_linked",
    "machine_checked_ready",
    "artifact_runtime_linked",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in audit.get("theorems", []) if row.get("status", "active") == "active"]


def _append_blocker(blockers: list[str], checks: dict[str, bool], check: str, blocker: str) -> None:
    checks[check] = False
    blockers.append(blocker)


def _strip_lean_comments(source: str) -> str:
    """Remove Lean line/block comments while preserving executable text."""
    out: list[str] = []
    i = 0
    depth = 0
    while i < len(source):
        if depth == 0 and source.startswith("--", i):
            newline = source.find("\n", i)
            if newline == -1:
                break
            out.append("\n")
            i = newline + 1
            continue
        if source.startswith("/-", i):
            depth += 1
            i += 2
            continue
        if depth > 0:
            if source.startswith("-/", i):
                depth -= 1
                i += 2
            else:
                if source[i] == "\n":
                    out.append("\n")
                i += 1
            continue
        out.append(source[i])
        i += 1
    return "".join(out)


def _lean_token_hits(path: Path) -> list[str]:
    code = _strip_lean_comments(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for token in FORBIDDEN_LEAN_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", code):
            hits.append(token)
    return hits


def validate_formal_core(repo_root: Path = REPO_ROOT, *, run_lean: bool = False) -> dict[str, Any]:
    formal_root = repo_root / FORMAL_DIR
    lean_files = sorted(
        path for path in formal_root.rglob("*.lean") if path.is_file() and not path.name.startswith("._")
    )
    blockers: list[str] = []
    checks: dict[str, bool] = {
        "formal_core_present": bool(lean_files),
        "formal_core_no_sorry_or_admit": True,
        "formal_core_lake_build": True,
    }

    if not lean_files:
        blockers.append("formal core has no Lean files")

    token_findings: list[dict[str, str]] = []
    for path in lean_files:
        hits = _lean_token_hits(path)
        for token in hits:
            token_findings.append({"path": str(path.relative_to(repo_root)), "token": token})
    if token_findings:
        checks["formal_core_no_sorry_or_admit"] = False
        rendered = ", ".join(f"{finding['path']}:{finding['token']}" for finding in token_findings)
        blockers.append(f"formal core contains executable sorry/admit/axiom token(s): {rendered}")

    lake_output = ""
    if run_lean:
        try:
            completed = subprocess.run(
                ["lake", "build"],
                cwd=formal_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            checks["formal_core_lake_build"] = False
            blockers.append(f"lake build could not run: {exc}")
            completed = None
        if completed is not None:
            lake_output = (completed.stdout + completed.stderr).strip()
            if completed.returncode != 0:
                checks["formal_core_lake_build"] = False
                blockers.append(f"lake build failed with exit code {completed.returncode}")

    return {
        "pass": not blockers,
        "checks": checks,
        "blockers": blockers,
        "lean_files": [str(path.relative_to(repo_root)) for path in lean_files],
        "token_findings": token_findings,
        "lake_output": lake_output,
    }


def _validate_active_audit(
    repo_root: Path,
    audit: dict[str, Any],
    matrix: dict[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    rows = _active_rows(audit)
    checks: dict[str, bool] = {
        "active_audit_present": bool(rows),
        "active_code_correspondence_all_match": True,
        "no_draft_defended_rows": True,
        "no_unresolved_assumptions": True,
        "no_broken_rigor_labels": True,
        "statement_and_proof_locations_present": True,
        "code_and_test_anchors_present": True,
        "flagship_promotion_cards_complete": True,
    }
    blockers: list[str] = []

    if not rows:
        blockers.append("active theorem audit has no active theorem rows")

    for row in rows:
        theorem_id = row.get("theorem_id", "<missing>")
        if row.get("code_correspondence") != "matches":
            _append_blocker(
                blockers,
                checks,
                "active_code_correspondence_all_match",
                f"{theorem_id}: code_correspondence must be matches",
            )
        if row.get("defense_tier") == "draft_non_defended":
            _append_blocker(
                blockers,
                checks,
                "no_draft_defended_rows",
                f"{theorem_id}: active defended register contains draft_non_defended row",
            )
        if row.get("unresolved_assumptions"):
            _append_blocker(
                blockers,
                checks,
                "no_unresolved_assumptions",
                f"{theorem_id}: unresolved assumptions remain",
            )
        if row.get("rigor_rating") not in ALLOWED_RIGOR_RATINGS:
            _append_blocker(
                blockers,
                checks,
                "no_broken_rigor_labels",
                f"{theorem_id}: unsupported rigor_rating {row.get('rigor_rating')!r}",
            )
        if not row.get("statement_location") or not row.get("proof_location"):
            _append_blocker(
                blockers,
                checks,
                "statement_and_proof_locations_present",
                f"{theorem_id}: statement/proof location missing",
            )
        if not row.get("code_anchors") or not row.get("test_anchors"):
            _append_blocker(
                blockers,
                checks,
                "code_and_test_anchors_present",
                f"{theorem_id}: code/test anchors missing",
            )

    matrix_rows = {row.get("theorem_id"): row for row in matrix.get("theorems", [])}
    for row in rows:
        if row.get("defense_tier") != "flagship_defended":
            continue
        theorem_id = row["theorem_id"]
        matrix_row = matrix_rows.get(theorem_id)
        if matrix_row is None:
            _append_blocker(
                blockers,
                checks,
                "flagship_promotion_cards_complete",
                f"{theorem_id}: missing promotion matrix row",
            )
            continue
        card_path = repo_root / matrix_row.get("result_card", "")
        if not card_path.exists():
            _append_blocker(
                blockers,
                checks,
                "flagship_promotion_cards_complete",
                f"{theorem_id}: missing result card {matrix_row.get('result_card')}",
            )
            continue
        card = _load_json(card_path)
        required_card_fields = ("proof_file", "code_anchor", "tests", "artifacts", "claim_boundary")
        missing = [field for field in required_card_fields if not card.get(field)]
        if missing:
            _append_blocker(
                blockers,
                checks,
                "flagship_promotion_cards_complete",
                f"{theorem_id}: result card missing {', '.join(missing)}",
            )

    return checks, blockers


def validate(repo_root: Path = REPO_ROOT, *, run_lean: bool = False) -> dict[str, Any]:
    audit_path = repo_root / ACTIVE_AUDIT
    matrix_path = repo_root / PROMOTION_MATRIX
    blockers: list[str] = []
    checks: dict[str, bool] = {}

    if not audit_path.exists():
        blockers.append(f"missing {ACTIVE_AUDIT}")
        return {"pass": False, "score": 0.0, "checks": {"active_audit_present": False}, "blockers": blockers}
    if not matrix_path.exists():
        blockers.append(f"missing {PROMOTION_MATRIX}")
        return {"pass": False, "score": 0.0, "checks": {"promotion_matrix_present": False}, "blockers": blockers}

    audit = _load_json(audit_path)
    matrix = _load_json(matrix_path)
    audit_checks, audit_blockers = _validate_active_audit(repo_root, audit, matrix)
    checks.update(audit_checks)
    blockers.extend(audit_blockers)

    summary = audit.get("summary", {})
    if summary.get("code_correspondence_counts") != {"matches": summary.get("active_theorem_count")}:
        checks["summary_code_correspondence_all_match"] = False
        blockers.append("active audit summary code_correspondence_counts is not all matches")
    else:
        checks["summary_code_correspondence_all_match"] = True

    if summary.get("draft_non_defended_ids"):
        checks["summary_no_draft_ids"] = False
        blockers.append("active audit summary still lists draft_non_defended_ids")
    else:
        checks["summary_no_draft_ids"] = True

    formal = validate_formal_core(repo_root, run_lean=run_lean)
    checks.update(formal["checks"])
    blockers.extend(formal["blockers"])

    failed_checks = sum(1 for passed in checks.values() if not passed)
    score = 10.0 if not blockers else max(0.0, round(10.0 - failed_checks, 2))
    return {
        "pass": not blockers,
        "score": score,
        "checks": checks,
        "blockers": blockers,
        "formal": formal,
        "active_theorem_count": summary.get("active_theorem_count", 0),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the strict ORIUS theorem-defensibility 10/10 gate")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-lean", action="store_true", help="Run `lake build` inside formal/")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "reports/publication/theorem_defensibility_10.json",
        help="Write JSON validation result",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = validate(args.repo_root.resolve(), run_lean=args.run_lean)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        out_path = args.out if args.out.is_absolute() else args.repo_root / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
