#!/usr/bin/env python3
"""R1 Release Family Orchestrator.

Coordinates the full R1 evidence pipeline across all datasets under a
single release-family ID.  Does NOT update canonical manifests until all
gates pass.

Stages
------
diagnostic  – GBM-first fast runs for DE + US_MISO + US_PJM + US_ERCOT
full        – Full six-model baseline comparison
cpsbench    – Run CPSBench severity sweeps with the R1 controller set
verify      – Check all datasets pass acceptance + UQ contract
promote     – Copy verified artifacts into canonical paths and update
              release_manifest.json

Usage
-----
    python scripts/run_r1_release.py --stage diagnostic
    python scripts/run_r1_release.py --stage full     --release-id R1_20260312
    python scripts/run_r1_release.py --stage cpsbench  --release-id R1_20260312
    python scripts/run_r1_release.py --stage verify    --release-id R1_20260312
    python scripts/run_r1_release.py --stage deployment --release-id R1_20260312
    python scripts/run_r1_release.py --stage promote   --release-id R1_20260312
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = sys.executable or "python3"

R1_DATASETS = ["DE", "US_MISO", "US_PJM", "US_ERCOT"]


def _generate_release_id() -> str:
    return "R1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run(cmd: list[str], description: str) -> bool:
    print(f"\n{'─'*60}")
    print(f"  {description}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'─'*60}")
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        return True
    except subprocess.CalledProcessError:
        print(f"  ❌ FAILED: {description}")
        return False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _manifest_path_for(dataset: str, run_id: str) -> Path:
    return REPO_ROOT / "artifacts" / "runs" / dataset.lower() / run_id / "registry" / "run_manifest.json"


def _load_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    preflight_path = _resolve_repo_path(manifest.get("preflight_path"))
    if preflight_path is None:
        return {}
    return _load_json(preflight_path)


def _write_manifest_acceptance(manifest_path: Path, *, accepted: bool) -> None:
    payload = _load_json(manifest_path)
    if not payload:
        return
    payload["accepted"] = bool(accepted)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── Stages ────────────────────────────────────────────────────────────────────


def stage_diagnostic(release_id: str) -> dict[str, bool]:
    """Stage 1 – GBM-first candidate runs for all R1 datasets."""
    print(f"\n{'═'*60}")
    print(f"  STAGE 1: DIAGNOSTIC (GBM-first)  release={release_id}")
    print(f"{'═'*60}\n")
    results: dict[str, bool] = {}
    for ds in R1_DATASETS:
        ok = _run(
            [
                PYTHON_BIN, "scripts/train_dataset.py",
                "--dataset", ds,
                "--candidate-run",
                "--run-id", f"{release_id}_diag",
                "--models", "gbm",
                "--no-tune",
                "--no-cv",
            ],
            f"Diagnostic run for {ds}",
        )
        results[ds] = ok
    return results


def stage_full(release_id: str, profile: str = "standard") -> dict[str, bool]:
    """Stage 2 – Full six-model baseline comparison."""
    print(f"\n{'═'*60}")
    print(f"  STAGE 2: FULL BASELINE  release={release_id}  profile={profile}")
    print(f"{'═'*60}\n")
    results: dict[str, bool] = {}
    extra: list[str] = []
    if profile == "aggressive":
        extra = ["--profile", "aggressive"]
    for ds in R1_DATASETS:
        ok = _run(
            [
                PYTHON_BIN, "scripts/train_dataset.py",
                "--dataset", ds,
                "--candidate-run",
                "--run-id", release_id,
                "--models", "gbm,lstm,tcn,nbeats,tft,patchtst",
                "--tune",
                *extra,
            ],
            f"Full training for {ds}",
        )
        results[ds] = ok
    return results


def stage_cpsbench(release_id: str) -> bool:
    """Stage 3 – Run CPSBench severity sweeps."""
    print(f"\n{'═'*60}")
    print(f"  STAGE 3: CPSBench severity sweep  release={release_id}")
    print(f"{'═'*60}\n")
    out_dir = REPO_ROOT / "reports" / "runs" / "cpsbench" / release_id
    return _run(
        [
            PYTHON_BIN, "scripts/run_cpsbench.py",
            "--config", "configs/cpsbench_r1_severity.yaml",
            "--out-dir", str(out_dir),
        ],
        f"CPSBench severity sweep → {out_dir}",
    )


def stage_verify(release_id: str) -> dict[str, dict]:
    """Stage 4 – Verify all datasets pass acceptance gates."""
    print(f"\n{'═'*60}")
    print(f"  STAGE 4: VERIFY  release={release_id}")
    print(f"{'═'*60}\n")
    verification: dict[str, dict] = {}
    all_pass = True

    for ds in R1_DATASETS:
        manifest_path = _manifest_path_for(ds, release_id)
        detail: dict[str, Any] = {
            "dataset": ds,
            "passed": False,
            "manifest_path": str(manifest_path),
        }
        manifest = _load_json(manifest_path)
        if not manifest:
            detail["error"] = f"Missing or invalid run manifest: {manifest_path}"
            verification[ds] = detail
            all_pass = False
            print(f"  ❌ {ds}: {detail['error']}")
            continue

        if str(manifest.get("release_id", "")) != release_id:
            detail["error"] = (
                f"Manifest release_id={manifest.get('release_id')!r} does not match {release_id!r}"
            )
            verification[ds] = detail
            all_pass = False
            print(f"  ❌ {ds}: {detail['error']}")
            continue

        artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}
        reports_dir = _resolve_repo_path(artifacts.get("reports_dir"))
        models_dir = _resolve_repo_path(artifacts.get("models_dir"))
        uncertainty_dir = _resolve_repo_path(artifacts.get("uncertainty_dir"))
        backtests_dir = _resolve_repo_path(artifacts.get("backtests_dir"))
        artifacts_dir = manifest_path.parents[1]
        summary_path = _resolve_repo_path(manifest.get("selection_summary_path"))
        preflight = _load_preflight(manifest)
        targets = preflight.get("expected_targets", manifest.get("targets", []))
        model_types = preflight.get("expected_model_types", [])
        uncertainty_targets = preflight.get("expected_targets", manifest.get("targets", []))

        detail.update(
            {
                "run_id": manifest.get("run_id"),
                "summary_path": str(summary_path) if summary_path is not None else None,
                "reports_dir": str(reports_dir) if reports_dir is not None else None,
                "models_dir": str(models_dir) if models_dir is not None else None,
                "uncertainty_dir": str(uncertainty_dir) if uncertainty_dir is not None else None,
                "backtests_dir": str(backtests_dir) if backtests_dir is not None else None,
                "targets": [],
            }
        )

        missing_paths = [
            name
            for name, path in (
                ("reports_dir", reports_dir),
                ("models_dir", models_dir),
                ("uncertainty_dir", uncertainty_dir),
                ("backtests_dir", backtests_dir),
                ("summary_path", summary_path),
            )
            if path is None or not path.exists()
        ]
        if missing_paths:
            detail["error"] = f"Missing required verification paths: {missing_paths}"
            _write_manifest_acceptance(manifest_path, accepted=False)
            verification[ds] = detail
            all_pass = False
            print(f"  ❌ {ds}: {detail['error']}")
            continue

        if not isinstance(targets, list) or not targets or not isinstance(model_types, list) or not model_types:
            detail["error"] = "Preflight analysis is missing expected_targets or expected_model_types"
            _write_manifest_acceptance(manifest_path, accepted=False)
            verification[ds] = detail
            all_pass = False
            print(f"  ❌ {ds}: {detail['error']}")
            continue

        verify_cmd = [
            PYTHON_BIN,
            "scripts/verify_training_outputs.py",
            "--models-dir",
            str(models_dir),
            "--reports-dir",
            str(reports_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--uncertainty-dir",
            str(uncertainty_dir),
            "--backtests-dir",
            str(backtests_dir),
            "--targets",
            *[str(target) for target in targets],
            "--uncertainty-targets",
            *[str(target) for target in uncertainty_targets],
            "--model-types",
            *[str(model_type) for model_type in model_types],
        ]
        detail["verify_command"] = " ".join(verify_cmd)
        verify_ok = _run(verify_cmd, f"Artifact verification for {ds}")

        summary_payload = _load_json(summary_path)
        if summary_payload:
            detail["accepted"] = bool(summary_payload.get("accepted", False))
            detail["targets"] = [
                {"target": row.get("target"), "accepted": row.get("accepted")}
                for row in summary_payload.get("targets", [])
                if isinstance(row, dict)
            ]
        else:
            detail["accepted"] = False
            detail["error"] = f"Invalid selection summary JSON: {summary_path}"

        detail["passed"] = bool(verify_ok and detail.get("accepted", False))
        if not detail["passed"] and "error" not in detail:
            detail["error"] = "Training outputs failed verification or acceptance gates"

        _write_manifest_acceptance(manifest_path, accepted=detail["passed"])
        verification[ds] = detail
        if not detail["passed"]:
            all_pass = False
            print(f"  ❌ {ds}: {detail.get('error', 'Not accepted')}")
        else:
            print(f"  ✅ {ds}: accepted")

    report_path = REPO_ROOT / "reports" / "runs" / f"{release_id}_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "release_id": release_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_pass": all_pass,
        "datasets": verification,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n📋 Verification: {report_path}")

    if all_pass:
        print(f"\n✅ Release {release_id} passed all gates.")
    else:
        print(f"\n❌ Release {release_id} has failures.  Fix before promoting.")
    return verification


def stage_deployment(release_id: str) -> bool:
    """Stage 5 – Generate deployment evidence artifacts."""
    print(f"\n{'═'*60}")
    print(f"  STAGE 5: DEPLOYMENT EVIDENCE  release={release_id}")
    print(f"{'═'*60}\n")
    return _run(
        [
            PYTHON_BIN, "scripts/generate_deployment_evidence.py",
            "--release-id", release_id,
        ],
        f"Deployment evidence → reports/runs/deployment/{release_id}/",
    )


def stage_promote(release_id: str) -> bool:
    """Stage 6 – Promote verified release into canonical paths."""
    print(f"\n{'═'*60}")
    print(f"  STAGE 5: PROMOTE  release={release_id}")
    print(f"{'═'*60}\n")
    verification_path = REPO_ROOT / "reports" / "runs" / f"{release_id}_verification.json"
    if not verification_path.exists():
        print(f"  ❌ Missing verification report: {verification_path}")
        print("  Run stage_verify successfully before promotion.")
        return False
    try:
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  ❌ Invalid verification report: {exc}")
        return False
    if not bool(verification.get("all_pass", False)):
        print("  ❌ Verification report indicates failed acceptance gates. Promotion blocked.")
        return False
    all_ok = True
    for ds in R1_DATASETS:
        ok = _run(
            [
                PYTHON_BIN, "scripts/train_dataset.py",
                "--dataset", ds,
                "--candidate-run",
                "--run-id", release_id,
                "--reports-only",
                "--promote-on-accept",
            ],
            f"Promote {ds}",
        )
        if not ok:
            all_ok = False

    if all_ok:
        manifest_path = REPO_ROOT / "reports" / "publication" / "release_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "release_id": release_id,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "datasets": R1_DATASETS,
            "status": "promoted",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n✅ Release manifest updated: {manifest_path}")
    return all_ok


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="R1 Release Family Orchestrator")
    parser.add_argument(
        "--stage",
        choices=["diagnostic", "full", "cpsbench", "verify", "deployment", "promote", "all"],
        required=True,
    )
    parser.add_argument("--release-id", default=None)
    parser.add_argument("--profile", choices=["standard", "aggressive"], default="standard")
    args = parser.parse_args()

    rid = args.release_id or _generate_release_id()
    print(f"🆔 Release family: {rid}")

    if args.stage == "diagnostic":
        r = stage_diagnostic(rid)
        return 0 if all(r.values()) else 1
    elif args.stage == "full":
        r = stage_full(rid, profile=args.profile)
        return 0 if all(r.values()) else 1
    elif args.stage == "cpsbench":
        return 0 if stage_cpsbench(rid) else 1
    elif args.stage == "verify":
        result = stage_verify(rid)
        return 0 if all(detail.get("passed", False) for detail in result.values()) else 1
    elif args.stage == "deployment":
        return 0 if stage_deployment(rid) else 1
    elif args.stage == "promote":
        return 0 if stage_promote(rid) else 1
    elif args.stage == "all":
        # Full pipeline: diagnostic → full → cpsbench → verify
        r1 = stage_diagnostic(rid)
        if not all(r1.values()):
            print("  ⚠  Diagnostic failures — continuing with full run anyway")
        r2 = stage_full(rid, profile=args.profile)
        if not all(r2.values()):
            print("  ❌ Full baseline failures")
            return 1
        if not stage_cpsbench(rid):
            return 1
        verified = stage_verify(rid)
        return 0 if all(detail.get("passed", False) for detail in verified.values()) else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
