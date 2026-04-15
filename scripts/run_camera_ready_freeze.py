#!/usr/bin/env python3
"""Run the strict ORIUS camera-ready freeze lane and record its status."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.data_pipeline.external_raw import get_strict_external_root


PYTHON = sys.executable
FREEZE_DIR = REPO_ROOT / "reports" / "camera_ready"
FREEZE_LOG_DIR = FREEZE_DIR / "logs"
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
SCORECARD_PATH = PUBLICATION_DIR / "orius_submission_scorecard.csv"
PARITY_PATH = PUBLICATION_DIR / "orius_equal_domain_parity_matrix.csv"
GAP_PATH = PUBLICATION_DIR / "orius_93plus_gap_matrix.csv"
REAL_DATA_PREFLIGHT_PATH = REPO_ROOT / "reports" / "real_data_preflight.json"
PACKAGE_MANIFEST_JSON = PUBLICATION_DIR / "orius_camera_ready_package_manifest.json"
PACKAGE_MANIFEST_MD = PUBLICATION_DIR / "orius_camera_ready_package_manifest.md"
FREEZE_LEDGER_JSON = PUBLICATION_DIR / "orius_camera_ready_freeze_ledger.json"
WAIVER_PATH = REPO_ROOT / "paper" / "camera_ready_warning_waivers.yaml"

PDF_OUTPUTS = {
    "dissertation": REPO_ROOT / "paper" / "paper.pdf",
    "review_dossier": PUBLICATION_DIR / "orius_review_dossier.pdf",
    "ieee_main": REPO_ROOT / "paper" / "ieee" / "orius_ieee_main.pdf",
    "ieee_appendix": REPO_ROOT / "paper" / "ieee" / "orius_ieee_appendix.pdf",
    "ieee_professor_main": REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_main.pdf",
    "ieee_professor_appendix_a": REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_appendix_a.pdf",
    "ieee_professor_appendix_b": REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_appendix_b.pdf",
}
REPO_ROOT_TOKEN = "$ORIUS_REPO_ROOT"
EXTERNAL_ROOT_TOKEN = "$ORIUS_EXTERNAL_DATA_ROOT"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sanitize_path_token(value: str, *, external_root: Path) -> str:
    if not value:
        return value
    if not (
        value.startswith("/")
        or value.startswith("./")
        or value.startswith("../")
        or (value.startswith(".") and "/" in value)
    ):
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            return f"{REPO_ROOT_TOKEN}/{candidate.relative_to(REPO_ROOT).as_posix()}"
        except ValueError:
            pass
        try:
            return f"{EXTERNAL_ROOT_TOKEN}/{candidate.relative_to(external_root).as_posix()}"
        except ValueError:
            pass
    return value


def _sanitize_payload(value: Any, *, external_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_payload(child, external_root=external_root) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(child, external_root=external_root) for child in value]
    if isinstance(value, str):
        return _sanitize_path_token(value, external_root=external_root)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_step(label: str, cmd: list[str], *, env: dict[str, str], logs_dir: Path) -> dict[str, Any]:
    display_cmd = [_sanitize_path_token(part, external_root=Path(env["ORIUS_EXTERNAL_DATA_ROOT"])) for part in cmd]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    log_path = logs_dir / f"{label}.log"
    log_path.write_text(
        f"$ {' '.join(display_cmd)}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n",
        encoding="utf-8",
    )
    return {
        "label": label,
        "command": display_cmd,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
    }


def _build_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    external_root = str(args.external_root)
    return [
        (
            "real_data_preflight",
            [
                PYTHON,
                "scripts/verify_real_data_preflight.py",
                "--external-root",
                external_root,
                "--out",
                str(REAL_DATA_PREFLIGHT_PATH),
            ],
        ),
        (
            "refresh_real_data_manifests",
            [PYTHON, "scripts/refresh_real_data_manifests.py"],
        ),
        (
            "universal_training_audit",
            [
                PYTHON,
                "scripts/run_universal_training_audit.py",
                "--out",
                "reports/orius_framework_proof/training_audit",
                *(["--train-missing"] if args.train_missing else []),
                *(["--repair-invalid-splits"] if args.repair_invalid_splits else []),
            ],
        ),
        (
            "universal_sil_validation",
            [
                PYTHON,
                "scripts/run_universal_sil_validation.py",
                "--seeds",
                str(args.sil_seeds),
                "--rows",
                str(args.sil_rows),
                "--out",
                "reports/orius_framework_proof/sil_validation",
            ],
        ),
        (
            "universal_orius_validation",
            [
                PYTHON,
                "scripts/run_universal_orius_validation.py",
                "--seeds",
                str(args.seeds),
                "--horizon",
                str(args.horizon),
                "--out",
                "reports/universal_orius_validation",
            ],
        ),
        (
            "equal_domain_gate",
            [
                PYTHON,
                "scripts/run_orius_canonical_closure_refresh.py",
                "--mode",
                "equal_domain_gate",
                "--external-root",
                external_root,
                *(["--train-missing"] if args.train_missing else []),
                *(["--repair-invalid-splits"] if args.repair_invalid_splits else []),
                "--seeds",
                str(args.seeds),
                "--sil-seeds",
                str(args.sil_seeds),
                "--sil-rows",
                str(args.sil_rows),
                "--horizon",
                str(args.horizon),
            ],
        ),
        ("battery_deep_novelty", [PYTHON, "scripts/run_battery_deep_novelty.py"]),
        ("build_orius_monograph_assets", [PYTHON, "scripts/build_orius_monograph_assets.py"]),
        ("build_orius_ieee_assets", [PYTHON, "scripts/build_orius_ieee_assets.py"]),
        ("build_camera_ready_tables", [PYTHON, "scripts/build_camera_ready_tables.py"]),
        ("build_camera_ready_figures", [PYTHON, "scripts/build_camera_ready_figures.py"]),
        ("verify_paper_manifest_camera_ready", [PYTHON, "scripts/verify_paper_manifest.py", "--camera-ready"]),
        ("validate_paper_claims", [PYTHON, "scripts/validate_paper_claims.py"]),
        ("test_thesis_package_assets", [PYTHON, "-m", "pytest", "-q", "tests/test_thesis_package_assets.py", "--no-cov"]),
        ("thesis_manuscript", ["make", "thesis-manuscript"]),
        ("review_compile", ["make", "review-compile"]),
        ("ieee_pack", ["make", "ieee-pack"]),
        ("ieee_prof_pack", ["make", "ieee-prof-pack"]),
        ("verify_camera_ready_logs", [PYTHON, "scripts/verify_camera_ready_logs.py", "--waivers", str(args.warning_waivers)]),
    ]


def _pdf_manifest() -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for name, path in PDF_OUTPUTS.items():
        payload[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": path.exists(),
            "sha256": _sha256(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }
    return payload


def _scorecard_summary() -> dict[str, dict[str, str]]:
    return {row["target_tier"]: row for row in _read_csv_rows(SCORECARD_PATH)}


def _parity_summary() -> dict[str, dict[str, str]]:
    return {row["domain"]: row for row in _read_csv_rows(PARITY_PATH)}


def _remaining_equal_domain_gaps() -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv_rows(GAP_PATH)
        if row.get("target_tier") == "equal_domain_93" and row.get("severity") in {"critical", "high"}
    ]


def _write_outputs(
    *,
    freeze_run_id: str,
    args: argparse.Namespace,
    steps: list[dict[str, Any]],
    status: str,
    failure_step: str | None,
) -> None:
    scorecard = _scorecard_summary()
    parity = _parity_summary()
    gap_rows = _remaining_equal_domain_gaps()
    payload = {
        "generated_at_utc": _utc_now_iso(),
        "freeze_run_id": freeze_run_id,
        "status": status,
        "failure_step": failure_step,
        "compute_lane": args.compute_lane,
        "external_root": EXTERNAL_ROOT_TOKEN,
        "warning_waivers": str(args.warning_waivers.relative_to(REPO_ROOT)),
        "steps": _sanitize_payload(steps, external_root=args.external_root),
        "scorecard": scorecard,
        "remaining_equal_domain_gaps": gap_rows,
        "parity_rows": parity,
        "pdf_outputs": _pdf_manifest(),
        "hf_job_templates": [
            "scripts/hf_jobs/canonical_closure_refresh_job.py",
            "scripts/hf_jobs/navigation_realdata_closure_job.py",
            "scripts/hf_jobs/aerospace_flight_closure_job.py",
            "scripts/hf_jobs/deep_learning_novelty_job.py",
            "scripts/hf_jobs/calibration_diagnostics_job.py",
            "scripts/hf_jobs/runtime_governance_trace_job.py",
        ],
    }
    FREEZE_LEDGER_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    PACKAGE_MANIFEST_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    equal_domain = scorecard.get("equal_domain_93", {})
    bounded = scorecard.get("bounded_93_candidate", {})
    PACKAGE_MANIFEST_MD.write_text(
        "\n".join(
            [
                "# ORIUS Camera-Ready Package Manifest",
                "",
                f"- Freeze run id: `{freeze_run_id}`",
                f"- Status: `{status}`",
                f"- Failure step: `{failure_step or 'none'}`",
                f"- Compute lane: `{args.compute_lane}`",
                f"- External data root: `{EXTERNAL_ROOT_TOKEN}`",
                f"- Warning waivers: `{args.warning_waivers.relative_to(REPO_ROOT)}`",
                "",
                "## Gate summary",
                "",
                f"- `bounded_93_candidate`: `{bounded.get('readiness_score_100', 'n/a')}/100`",
                f"- `equal_domain_93`: `{equal_domain.get('readiness_score_100', 'n/a')}/100`",
                f"- Equal-domain critical gaps: `{equal_domain.get('critical_gap_count', 'n/a')}`",
                f"- Equal-domain high gaps: `{equal_domain.get('high_gap_count', 'n/a')}`",
                "",
                "## Step status",
                "",
                *[
                    f"- `{step['label']}` -> returncode `{step['returncode']}` ({'ok' if step['ok'] else 'failed'})"
                    for step in steps
                ],
                "",
                "## PDF hashes",
                "",
                *[
                    f"- `{name}`: `{meta['sha256']}`" if meta["exists"] else f"- `{name}`: `missing`"
                    for name, meta in payload["pdf_outputs"].items()
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strict ORIUS camera-ready freeze lane.")
    parser.add_argument("--external-root", type=Path, default=None)
    parser.add_argument("--compute-lane", choices=("local", "hybrid"), default="hybrid")
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--repair-invalid-splits", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--sil-seeds", type=int, default=3)
    parser.add_argument("--sil-rows", type=int, default=96)
    parser.add_argument("--warning-waivers", type=Path, default=WAIVER_PATH)
    args = parser.parse_args()

    args.external_root = (args.external_root.expanduser().resolve() if args.external_root is not None else get_strict_external_root())
    args.warning_waivers = args.warning_waivers.expanduser().resolve()

    FREEZE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    freeze_run_id = f"camera_ready_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_env = os.environ.copy()
    run_env["ORIUS_EXTERNAL_DATA_ROOT"] = str(args.external_root)

    steps: list[dict[str, Any]] = []
    status = "passed"
    failure_step: str | None = None

    try:
        for label, cmd in _build_steps(args):
            step = _run_step(label, cmd, env=run_env, logs_dir=FREEZE_LOG_DIR)
            steps.append(step)
            if not step["ok"]:
                status = "failed"
                failure_step = label
                break
    finally:
        _write_outputs(
            freeze_run_id=freeze_run_id,
            args=args,
            steps=steps,
            status=status,
            failure_step=failure_step,
        )

    print(json.dumps({"freeze_run_id": freeze_run_id, "status": status, "failure_step": failure_step, "steps": steps}, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
