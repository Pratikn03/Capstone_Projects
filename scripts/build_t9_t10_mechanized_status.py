#!/usr/bin/env python3
"""Build the active-theorem-program mechanized-kernel status artifact.

The historical T9/T10 artifact name is retained because the T9/T10 promotion
gates read it. The payload also writes a clearer
``mechanized_theorem_program_status.json`` copy. Both record Lean-checked
theorem kernels while keeping domain-assumption discharge separate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
FORMAL_DIR = REPO_ROOT / "formal"
LEAN_ROOT = FORMAL_DIR / "Orius.lean"
FORBIDDEN_TOKENS = ("sorry", "admit", "axiom ", "unsafe_axiom", "by_contra_placeholder")
FALLBACK_THEOREM_IDS = (
    "T1",
    "T2",
    "T3a",
    "T3b",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "T10_T11_ObservationAmbiguitySandwich",
    "T11_AV_BrakeHold",
    "T11_HC_FailSafeRelease",
    "T6_AV_FallbackValidity",
    "T6_HC_FallbackValidity",
    "T_EQ_Battery_RuntimeArtifactPackage",
    "T_EQ_AV_RuntimeArtifactPackage",
    "T_EQ_HC_RuntimeArtifactPackage",
    "L1",
    "L2",
    "L3",
    "L4",
    "T11_Byzantine",
    "T_stale_decay",
    "T_minimax",
    "T_sensor_converse",
    "T_trajectory_PAC",
)


def _active_theorem_ids() -> tuple[str, ...]:
    audit_path = PUBLICATION_DIR / "active_theorem_audit.json"
    if not audit_path.exists():
        return FALLBACK_THEOREM_IDS
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    theorem_ids = tuple(str(row["theorem_id"]) for row in payload.get("theorems", []))
    return theorem_ids or FALLBACK_THEOREM_IDS


def _scan_forbidden_tokens(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing {path.relative_to(REPO_ROOT)}"]
    findings: list[str] = []
    for lean_file in sorted(path.rglob("*.lean") if path.is_dir() else [path]):
        parts = set(lean_file.parts)
        if ".lake" in parts or lean_file.name.startswith("._"):
            continue
        text = lean_file.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                findings.append(f"{lean_file.relative_to(REPO_ROOT)}:{token.strip()}")
    return findings


def _run_lake_build(run_lean: bool) -> tuple[str, str]:
    lake = shutil.which("lake")
    if not lake:
        return "toolchain_missing", "lake executable not found"
    if not run_lean:
        return "not_run", "Lean build was not run; pass --run-lean after installing Lean 4"
    completed = subprocess.run(
        [lake, "build", "Orius"],
        cwd=FORMAL_DIR,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode == 0:
        return "passed", "lake build passed"
    return "failed", (completed.stderr or completed.stdout)[-2000:]


def build_mechanized_status(
    *,
    out_dir: Path = PUBLICATION_DIR,
    run_lean: bool = False,
) -> dict[str, Any]:
    forbidden = _scan_forbidden_tokens(FORMAL_DIR)
    lean_status, detail = _run_lake_build(run_lean)
    complete = not forbidden and lean_status == "passed"
    theorem_status = {
        theorem_id: {
            "status": "kernel_verified" if complete else "blocked",
            "formal_file": "formal/Orius.lean",
            "lean_status": lean_status,
            "mechanization_scope": "theorem_kernel_not_domain_discharge",
            "blocker": "" if complete else detail,
        }
        for theorem_id in _active_theorem_ids()
    }
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pass": complete,
        "lean_status": lean_status,
        "formal_dir": "formal",
        "forbidden_tokens": forbidden,
        "policy": (
            "The active theorem program's mechanized kernel gate passes only after `lake build Orius` succeeds and no "
            "placeholders/unchecked axioms are present. Domain assumptions, empirical witnesses, "
            "and runtime artifacts remain separate promotion gates."
        ),
        "mechanization_scope": "theorem_kernel_not_domain_discharge",
        "root_module": "formal/Orius.lean",
        "theorems": theorem_status,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    (out_dir / "t9_t10_mechanized_proof_status.json").write_text(rendered, encoding="utf-8")
    (out_dir / "mechanized_theorem_program_status.json").write_text(rendered, encoding="utf-8")
    print(f"[build_t9_t10_mechanized_status] pass={payload['pass']} lean_status={lean_status}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PUBLICATION_DIR)
    parser.add_argument("--run-lean", action="store_true")
    args = parser.parse_args()
    build_mechanized_status(out_dir=args.out_dir, run_lean=args.run_lean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
