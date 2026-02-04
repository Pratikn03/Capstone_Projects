"""Build a release bundle with reports + run manifest."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _latest_run_id(runs_dir: Path) -> str | None:
    if not runs_dir.exists():
        return None
    candidates = [p.name for p in runs_dir.iterdir() if p.is_dir()]
    return sorted(candidates)[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="Run ID under artifacts/runs")
    parser.add_argument("--out", default=None, help="Output bundle directory")
    args = parser.parse_args()

    runs_dir = Path("artifacts/runs")
    run_id = args.run_id or _latest_run_id(runs_dir)
    if not run_id:
        raise SystemExit("No run id found. Run the pipeline first.")

    bundle_dir = Path(args.out) if args.out else Path(f"artifacts/submission_bundle_{run_id}")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    reports_src = Path("reports")
    if reports_src.exists():
        shutil.copytree(reports_src, bundle_dir / "reports", dirs_exist_ok=True)

    run_dir = runs_dir / run_id
    if run_dir.exists():
        shutil.copytree(run_dir, bundle_dir / "run_snapshot", dirs_exist_ok=True)

    manifest = {
        "bundle_id": bundle_dir.name,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id,
        "reports_dir": str(reports_src),
        "run_snapshot_dir": str(run_dir),
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[bundle] created {bundle_dir}")


if __name__ == "__main__":
    main()

