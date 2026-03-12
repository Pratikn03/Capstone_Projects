#!/usr/bin/env python3
"""R1 Release Family Orchestrator.

Coordinates the full R1 evidence pipeline across all datasets under a
single release-family ID.  Does NOT update canonical manifests until all
gates pass.

Stages
------
diagnostic  – GBM-first fast runs for DE + US_MISO + US_PJM + US_ERCOT
full        – Full baseline comparison (GBM + N-BEATS + TFT + PatchTST)
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
    python scripts/run_r1_release.py --stage promote   --release-id R1_20260312
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    """Stage 2 – Full baseline comparison with all models."""
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
        ds_lower = ds.lower()
        reports_dir = REPO_ROOT / "reports" / "runs" / ds_lower / release_id
        summary_path = reports_dir / "publish" if reports_dir.exists() else reports_dir

        # Check tuning summary
        summary_file = None
        for candidate in [
            summary_path / f"tuning_summary_{ds_lower}.json",
            reports_dir / f"tuning_summary_{ds_lower}.json",
        ]:
            if candidate.exists():
                summary_file = candidate
                break

        detail: dict = {"dataset": ds, "passed": False}
        if summary_file is not None:
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                detail["accepted"] = data.get("accepted", False)
                detail["targets"] = [
                    {"target": t.get("target"), "accepted": t.get("accepted")}
                    for t in data.get("targets", [])
                ]
                detail["passed"] = bool(data.get("accepted", False))
            except (json.JSONDecodeError, OSError) as exc:
                detail["error"] = str(exc)
        else:
            detail["error"] = f"No tuning summary found under {reports_dir}"

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


def stage_promote(release_id: str) -> bool:
    """Stage 5 – Promote verified release into canonical paths."""
    print(f"\n{'═'*60}")
    print(f"  STAGE 5: PROMOTE  release={release_id}")
    print(f"{'═'*60}\n")
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
        choices=["diagnostic", "full", "cpsbench", "verify", "promote", "all"],
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
        stage_verify(rid)
        return 0
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
        stage_cpsbench(rid)
        stage_verify(rid)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
