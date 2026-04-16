#!/usr/bin/env python3
"""Run the ORIUS canonical closure refresh with optional supplemental HF support.

This entrypoint keeps the official parity-closing lane separate from the bounded
Hugging Face support lane. It refreshes the canonical repo-local artifacts and
then emits an execution summary that can be cited in manuscript or project
tracking updates.
"""
from __future__ import annotations

import argparse
import csv
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

from orius.data_pipeline.external_raw import get_external_data_root, get_strict_external_root


PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
REFRESH_DIR = REPO_ROOT / "reports" / "closure_refresh"
PYTHON = sys.executable
REPO_ROOT_TOKEN = "$ORIUS_REPO_ROOT"
EXTERNAL_ROOT_TOKEN = "$ORIUS_EXTERNAL_DATA_ROOT"

REAL_DATA_PREFLIGHT_PATH = REPO_ROOT / "reports" / "real_data_preflight.json"
REAL_DATA_STATUS_PATH = REPO_ROOT / "reports" / "real_data_contract_status.json"
TRAINING_AUDIT_DIR = REPO_ROOT / "reports" / "orius_framework_proof" / "training_audit"
SIL_DIR = REPO_ROOT / "reports" / "orius_framework_proof" / "sil_validation"
VALIDATION_DIR = REPO_ROOT / "reports" / "universal_orius_validation"
PARITY_MATRIX_PATH = PUBLICATION_DIR / "orius_equal_domain_parity_matrix.csv"
SCORECARD_PATH = PUBLICATION_DIR / "orius_submission_scorecard.csv"
GAP_MATRIX_PATH = PUBLICATION_DIR / "orius_93plus_gap_matrix.csv"
GATE_LEDGER_JSON_PATH = PUBLICATION_DIR / "orius_equal_domain_gate_ledger.json"
GATE_LEDGER_CSV_PATH = PUBLICATION_DIR / "orius_equal_domain_gate_ledger.csv"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_external_root(external_root: Path | None) -> Path | None:
    if external_root is None:
        return None
    return external_root.expanduser().resolve()


def _sanitize_path_token(value: str, *, external_root: Path | None) -> str:
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
            rel_to_repo = candidate.relative_to(REPO_ROOT)
            return f"{REPO_ROOT_TOKEN}/{rel_to_repo.as_posix()}"
        except ValueError:
            pass
        if external_root is not None:
            try:
                rel_to_external = candidate.relative_to(external_root)
                return f"{EXTERNAL_ROOT_TOKEN}/{rel_to_external.as_posix()}"
            except ValueError:
                pass
        if value.startswith("/Users/"):
            rel = Path(value).name
            return f"{EXTERNAL_ROOT_TOKEN}/{rel}" if rel else EXTERNAL_ROOT_TOKEN
    return value


def _sanitize_payload(value: Any, *, external_root: Path | None) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_payload(child, external_root=external_root) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(child, external_root=external_root) for child in value]
    if isinstance(value, str):
        return _sanitize_path_token(value, external_root=external_root)
    return value


def _run_step(
    label: str,
    cmd: list[str],
    *,
    logs_dir: Path,
    allow_failure: bool = False,
    env: dict[str, str] | None = None,
    external_root: Path | None = None,
) -> dict[str, Any]:
    command_for_display = [_sanitize_path_token(part, external_root=external_root) for part in cmd]
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
        f"$ {' '.join(command_for_display)}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}. See {log_path}.")
    return {
        "label": label,
        "command": command_for_display,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "allow_failure": allow_failure,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
    }


def _dir_has_files(path: Path) -> bool:
    return path.exists() and any(candidate.is_file() for candidate in path.rglob("*"))


def _navigation_external_root_ready(root: Path | None) -> bool:
    if root is None:
        return False
    kitti_root = root / "kitti_odometry"
    poses_candidates = [kitti_root / "dataset" / "poses", kitti_root / "poses"]
    sequence_candidates = [kitti_root / "dataset" / "sequences", kitti_root / "sequences"]
    poses_ready = any(candidate.exists() and any(candidate.glob("*.txt")) for candidate in poses_candidates)
    times_ready = any(candidate.exists() and any(candidate.glob("*/times.txt")) for candidate in sequence_candidates)
    return poses_ready and times_ready


def _append_external_arg(cmd: list[str], external_root: Path | None) -> list[str]:
    if external_root is None:
        return cmd
    return [*cmd, "--external-root", str(external_root)]


def _write_equal_domain_gate_ledger(steps: list[dict[str, Any]], *, external_root: Path | None) -> None:
    payload = {
        "generated_at_utc": _utc_now_iso(),
        "external_data_root": EXTERNAL_ROOT_TOKEN if external_root is not None else None,
        "steps": _sanitize_payload(steps, external_root=external_root),
    }
    GATE_LEDGER_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with GATE_LEDGER_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "returncode", "ok", "allow_failure", "log_path"])
        for step in steps:
            writer.writerow([
                step["label"],
                step["returncode"],
                step["ok"],
                step["allow_failure"],
                step["log_path"],
            ])


def _assert_equal_domain_gate_success() -> None:
    scorecard_rows = {row["target_tier"]: row for row in _read_csv_rows(SCORECARD_PATH)}
    equal_row = scorecard_rows.get("equal_domain_93")
    if not equal_row:
        raise RuntimeError("equal_domain_93 scorecard row is missing after equal-domain gate run.")
    if str(equal_row.get("meets_93_gate", "")).strip().lower() != "true":
        raise RuntimeError("equal_domain_93 did not meet the 93 gate.")
    if int(equal_row.get("critical_gap_count", "999")) != 0:
        raise RuntimeError("equal_domain_93 still has critical gaps.")
    if int(equal_row.get("high_gap_count", "999")) != 0:
        raise RuntimeError("equal_domain_93 still has high gaps.")

    parity_rows = {row["domain"]: row for row in _read_csv_rows(PARITY_MATRIX_PATH)}
    for domain in (
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Industrial Process Control",
        "Medical and Healthcare Monitoring",
        "Navigation and Guidance",
        "Aerospace Control",
    ):
        row = parity_rows.get(domain)
        if row is None:
            raise RuntimeError(f"Missing parity row for {domain}.")
        if row.get("resulting_tier") not in {"reference", "proof_validated"}:
            raise RuntimeError(f"{domain} is not parity-ready in the final matrix.")

    gap_rows = _read_csv_rows(GAP_MATRIX_PATH)
    remaining_equal_domain_gaps = [
        row for row in gap_rows
        if row.get("target_tier") == "equal_domain_93" and row.get("severity") in {"critical", "high"}
    ]
    if remaining_equal_domain_gaps:
        raise RuntimeError("Equal-domain high/critical gaps remain after strict gate run.")

    paper5_rows = {row["domain"]: row for row in _read_csv_rows(VALIDATION_DIR / "paper5_cross_domain_matrix.csv")}
    paper6_rows = {row["domain"]: row for row in _read_csv_rows(VALIDATION_DIR / "paper6_cross_domain_matrix.csv")}
    for domain in ("battery", "vehicle", "industrial", "healthcare", "navigation", "aerospace"):
        if paper5_rows.get(domain, {}).get("status") != "evaluated":
            raise RuntimeError(f"Paper 5 shared-constraint surface is not evaluated for {domain}.")
        if paper6_rows.get(domain, {}).get("status") != "evaluated":
            raise RuntimeError(f"Paper 6 governance surface is not evaluated for {domain}.")


def _write_execution_summary(
    mode: str,
    steps: list[dict[str, Any]],
    out_dir: Path,
    *,
    external_root: Path | None,
) -> None:
    preflight = _read_json(REAL_DATA_PREFLIGHT_PATH)
    real_data_status = _read_json(REAL_DATA_STATUS_PATH)
    training_report = _read_json(TRAINING_AUDIT_DIR / "training_audit_report.json")
    validation_report = _read_json(VALIDATION_DIR / "validation_report.json")
    parity_rows = _read_csv_rows(PARITY_MATRIX_PATH)
    scorecard_rows = {row["target_tier"]: row for row in _read_csv_rows(SCORECARD_PATH)}

    parity_blockers = {
        row["domain"]: row["exact_blocker"]
        for row in parity_rows
        if row.get("domain") in {"Navigation and Guidance", "Aerospace Control"}
    }
    aerospace_public_summary = _read_json(PUBLICATION_DIR / "aerospace_public_flight_runtime_summary.json")
    external_root_raw = os.environ.get("ORIUS_EXTERNAL_DATA_ROOT")

    payload = {
        "generated_at_utc": _utc_now_iso(),
        "mode": mode,
        "repo_root": REPO_ROOT_TOKEN,
        "or_external_data_root_set": bool(external_root_raw),
        "external_data_root": (
            EXTERNAL_ROOT_TOKEN if external_root is not None else None
        ),
        "steps": steps,
        "preflight": _sanitize_payload(preflight, external_root=external_root),
        "real_data_status": _sanitize_payload(real_data_status, external_root=external_root),
        "training_audit": training_report,
        "validation_report": validation_report,
        "scorecard": scorecard_rows,
        "official_parity_blockers": parity_blockers,
        "aerospace_public_flight_support": aerospace_public_summary if isinstance(aerospace_public_summary, dict) else {},
        "supplemental_hf_policy": {
            "promotion_allowed": False,
            "note": "Supplemental Hugging Face evidence remains bounded diagnostics only and cannot override the official parity matrix.",
        },
    }
    payload = _sanitize_payload(payload, external_root=external_root)
    json_path = out_dir / "orius_refresh_execution.json"
    md_path = PUBLICATION_DIR / "orius_refresh_execution.md"
    publication_json_path = PUBLICATION_DIR / "orius_refresh_execution.json"
    json_text = json.dumps(payload, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    publication_json_path.write_text(json_text, encoding="utf-8")

    bounded_score = scorecard_rows.get("bounded_93_candidate", {})
    equal_score = scorecard_rows.get("equal_domain_93", {})
    refreshed_domains = real_data_status.get("refreshed_domains", []) if isinstance(real_data_status, dict) else []
    blocked_domains = real_data_status.get("blocked_domains", []) if isinstance(real_data_status, dict) else []
    training_closed = training_report.get("training_surface_closed_domains", []) if isinstance(training_report, dict) else []
    md_path.write_text(
        "\n".join(
            [
                "# ORIUS Canonical Closure Refresh Execution",
                "",
                f"- Generated: `{payload['generated_at_utc']}`",
                f"- Mode: `{mode}`",
                f"- Canonical raw-data root mounted: `{payload['or_external_data_root_set']}`",
                "",
                "## Official canonical lane",
                "",
                f"- Refreshed raw-contract domains: `{', '.join(refreshed_domains) or 'none'}`",
                f"- Blocked raw-contract domains: `{', '.join(blocked_domains) or 'none'}`",
                f"- Training surfaces closed: `{', '.join(training_closed) or 'none'}`",
                f"- `bounded_93_candidate`: `{bounded_score.get('readiness_score_100', 'n/a')}/100`",
                f"- `equal_domain_93`: `{equal_score.get('readiness_score_100', 'n/a')}/100`",
                "",
                "## Official blockers",
                "",
                f"- Navigation: `{parity_blockers.get('Navigation and Guidance', 'n/a')}`",
                f"- Aerospace: `{parity_blockers.get('Aerospace Control', 'n/a')}`",
                "",
                "## Supplemental Hugging Face lane",
                "",
                "- Supplemental Hugging Face evidence is tracked separately and is not allowed to promote the official parity matrix.",
                (
                    f"- Aerospace public-flight support: {aerospace_public_summary.get('rows_total', 'n/a')} rows across {aerospace_public_summary.get('flight_count', 'n/a')} flights from {aerospace_public_summary.get('repo_id', 'n/a')}."
                    if isinstance(aerospace_public_summary, dict) and aerospace_public_summary
                    else "- Aerospace public-flight support is not yet built in this refresh."
                ),
                "- The bounded support lane cannot serve as a parity-closing source replacement.",
                "",
                "## Step status",
                "",
                *[
                    f"- `{step['label']}` → returncode `{step['returncode']}` ({'ok' if step['ok'] else 'expected blocker' if step['allow_failure'] else 'failed'})"
                    for step in steps
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh ORIUS canonical closure artifacts and bounded HF support surfaces.")
    parser.add_argument(
        "--mode",
        choices=("canonical_only", "canonical_plus_hf_support", "equal_domain_gate"),
        default="canonical_only",
    )
    parser.add_argument("--seeds", type=int, default=1, help="Universal validation seed count.")
    parser.add_argument("--horizon", type=int, default=24, help="Universal validation horizon.")
    parser.add_argument("--sil-seeds", type=int, default=1, help="SIL validation seed count.")
    parser.add_argument("--sil-rows", type=int, default=48, help="SIL validation row count.")
    parser.add_argument("--train-missing", action="store_true", help="Allow the training audit to train missing domains.")
    parser.add_argument("--repair-invalid-splits", action="store_true", help="Allow the training audit to rebuild invalid splits.")
    parser.add_argument("--skip-manuscript", action="store_true", help="Skip the final thesis-manuscript build.")
    parser.add_argument("--external-root", type=Path, default=None, help="Override ORIUS_EXTERNAL_DATA_ROOT for the refresh run.")
    args = parser.parse_args()

    REFRESH_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = REFRESH_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configured_external_root = get_external_data_root(required=False)
    external_root = _normalize_external_root(args.external_root or configured_external_root)
    strict_external_root = get_strict_external_root()
    if args.mode == "equal_domain_gate":
        if args.skip_manuscript:
            raise RuntimeError("--skip-manuscript is not allowed in equal_domain_gate mode.")
        if external_root is None:
            raise RuntimeError("equal_domain_gate requires ORIUS_EXTERNAL_DATA_ROOT to be set.")
        if external_root != strict_external_root:
            raise RuntimeError(
                f"equal_domain_gate requires external root {strict_external_root}, got {external_root}."
            )
    run_env = os.environ.copy()
    if external_root is not None:
        run_env["ORIUS_EXTERNAL_DATA_ROOT"] = str(external_root)

    steps: list[dict[str, Any]] = []
    steps.append(
        _run_step(
            "real_data_preflight",
            _append_external_arg([PYTHON, "scripts/verify_real_data_preflight.py", "--out", str(REAL_DATA_PREFLIGHT_PATH)], external_root),
            logs_dir=logs_dir,
            allow_failure=args.mode != "equal_domain_gate",
            env=run_env,
            external_root=external_root,
        )
    )
    if args.mode == "canonical_plus_hf_support":
        steps.append(
            _run_step(
                "aerospace_public_adsb_runtime",
                _append_external_arg(
                    [PYTHON, "scripts/build_aerospace_public_adsb_runtime.py", "--download"],
                    external_root,
                ),
                logs_dir=logs_dir,
                allow_failure=True,
                env=run_env,
                external_root=external_root,
            )
        )
    if args.mode == "equal_domain_gate" or (external_root is not None and _dir_has_files(external_root / "waymo_open_motion")):
        steps.append(
            _run_step(
                "av_real_dataset_build",
                _append_external_arg(
                    [
                        PYTHON,
                        "scripts/download_av_datasets.py",
                        "--source",
                        "waymo_motion",
                        "--out",
                        "data/orius_av/av/processed/av_trajectories_orius.csv",
                    ],
                    external_root,
                ),
                logs_dir=logs_dir,
                allow_failure=args.mode != "equal_domain_gate",
                env=run_env,
                external_root=external_root,
            )
        )
    if args.mode == "equal_domain_gate" or _navigation_external_root_ready(external_root):
        steps.append(
            _run_step(
                "navigation_real_dataset_build",
                _append_external_arg(
                    [
                        PYTHON,
                        "scripts/build_navigation_real_dataset.py",
                        "--out",
                        "data/navigation/processed/navigation_orius.csv",
                    ],
                    external_root,
                ),
                logs_dir=logs_dir,
                allow_failure=args.mode != "equal_domain_gate",
                env=run_env,
                external_root=external_root,
            )
        )
    if args.mode == "equal_domain_gate" or (external_root is not None and _dir_has_files(external_root / "aerospace_flight_telemetry")):
        steps.append(
            _run_step(
                "aerospace_real_flight_dataset_build",
                _append_external_arg(
                    [
                        PYTHON,
                        "scripts/build_aerospace_real_flight_dataset.py",
                        "--out",
                        "data/aerospace/processed/aerospace_realflight_runtime.csv",
                    ],
                    external_root,
                ),
                logs_dir=logs_dir,
                allow_failure=args.mode != "equal_domain_gate",
                env=run_env,
                external_root=external_root,
            )
        )
    steps.append(
        _run_step(
            "refresh_real_data_manifests",
            [PYTHON, "scripts/refresh_real_data_manifests.py", "--out", str(REAL_DATA_STATUS_PATH)],
            logs_dir=logs_dir,
            env=run_env,
            external_root=external_root,
        )
    )

    audit_cmd = [
        PYTHON,
        "scripts/run_universal_training_audit.py",
        "--out",
        str(TRAINING_AUDIT_DIR),
    ]
    if args.train_missing:
        audit_cmd.append("--train-missing")
    if args.repair_invalid_splits:
        audit_cmd.append("--repair-invalid-splits")
    steps.append(
        _run_step(
            "universal_training_audit",
            audit_cmd,
            logs_dir=logs_dir,
            allow_failure=args.mode != "equal_domain_gate",
            env=run_env,
            external_root=external_root,
        )
    )
    steps.append(
        _run_step(
            "universal_sil_validation",
            [
                PYTHON,
                "scripts/run_universal_sil_validation.py",
                "--seeds",
                str(args.sil_seeds),
                "--rows",
                str(args.sil_rows),
                "--out",
                str(SIL_DIR),
            ],
            logs_dir=logs_dir,
            env=run_env,
            external_root=external_root,
        )
    )
    steps.append(
        _run_step(
            "universal_orius_validation",
            [
                PYTHON,
                "scripts/run_universal_orius_validation.py",
                "--seeds",
                str(args.seeds),
                "--horizon",
                str(args.horizon),
                "--out",
                str(VALIDATION_DIR),
                *(["--equal-domain-gate"] if args.mode == "equal_domain_gate" else []),
            ],
            logs_dir=logs_dir,
            allow_failure=args.mode != "equal_domain_gate",
            env=run_env,
            external_root=external_root,
        )
    )
    steps.append(
        _run_step(
            "domain_closure_matrix",
            [
                PYTHON,
                "scripts/build_domain_closure_matrix.py",
                "--validation-report",
                str(VALIDATION_DIR / "validation_report.json"),
                "--training-report",
                str(TRAINING_AUDIT_DIR / "training_audit_report.json"),
                "--out",
                str(VALIDATION_DIR),
            ],
            logs_dir=logs_dir,
            env=run_env,
            external_root=external_root,
        )
    )
    steps.append(
        _run_step(
            "build_monograph_assets",
            [PYTHON, "scripts/build_orius_monograph_assets.py"],
            logs_dir=logs_dir,
            env=run_env,
            external_root=external_root,
        )
    )
    steps.append(
        _run_step(
            "generate_book_visuals",
            [PYTHON, "scripts/generate_orius_book_visuals.py"],
            logs_dir=logs_dir,
            env=run_env,
            external_root=external_root,
        )
    )
    steps.append(
        _run_step(
            "verify_paper_manifest",
            [PYTHON, "scripts/verify_paper_manifest.py"],
            logs_dir=logs_dir,
            env=run_env,
            external_root=external_root,
        )
    )
    steps.append(
        _run_step(
            "validate_paper_claims",
            [PYTHON, "scripts/validate_paper_claims.py"],
            logs_dir=logs_dir,
            env=run_env,
            external_root=external_root,
        )
    )
    if not args.skip_manuscript:
        steps.append(
            _run_step(
                "thesis_manuscript",
                ["make", "thesis-manuscript"],
                logs_dir=logs_dir,
                env=run_env,
                external_root=external_root,
            )
        )

    _write_execution_summary(args.mode, steps, REFRESH_DIR, external_root=external_root)
    if args.mode == "equal_domain_gate":
        _write_equal_domain_gate_ledger(steps, external_root=external_root)
        _assert_equal_domain_gate_success()
    print(json.dumps({"mode": args.mode, "steps": steps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
