"""End-to-end final publish audit orchestrator.

Produces:
- reports/publish/final_audit_report.md
- reports/publish/final_audit_report.json
- reports/publish/go_no_go_decision.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StepResult:
    name: str
    ok: bool
    return_code: int
    duration_s: float
    command: list[str]
    log_path: str


def _load_publish_cfg(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    return payload.get("publish_audit", {}) if isinstance(payload.get("publish_audit"), dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_checked_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _collect_python_package_versions() -> dict[str, str | None]:
    packages = ["pandas", "duckdb", "numpy", "fastapi", "torch", "lightgbm", "optuna", "pytest"]
    out: dict[str, str | None] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[pkg] = None
    return out


def _snapshot_baseline(out_dir: Path) -> dict[str, Any]:
    dirty_lines = _run_checked_output(["git", "status", "--short"]).splitlines()
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git": {
            "branch": _run_checked_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "head": _run_checked_output(["git", "rev-parse", "HEAD"]),
            "dirty_file_count": len([x for x in dirty_lines if x.strip()]),
            "dirty_files": [x.rstrip() for x in dirty_lines if x.strip()],
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "packages": _collect_python_package_versions(),
        },
    }
    _write_json(out_dir / "baseline_snapshot.json", payload)
    return payload


def _write_scope_manifest(out_dir: Path, publish_cfg: dict[str, Any]) -> dict[str, Any]:
    scope_cfg = publish_cfg.get("scope", {}) if isinstance(publish_cfg.get("scope"), dict) else {}
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "in_scope": scope_cfg.get(
            "include", ["src/orius/**", "services/api/**", "scripts/**", "configs/**", "iot/**"]
        ),
        "non_blocking_out_of_scope": scope_cfg.get(
            "non_blocking_out_of_scope",
            ["frontend visual/style-only changes", "paper formatting/layout artifacts", "generated figures"],
        ),
    }
    _write_json(out_dir / "audit_scope.json", payload)
    return payload


def _write_repro_lock(out_dir: Path, publish_cfg: dict[str, Any]) -> dict[str, Any]:
    repro_cfg = (
        publish_cfg.get("reproducibility", {}) if isinstance(publish_cfg.get("reproducibility"), dict) else {}
    )
    run_id_prefix = str(repro_cfg.get("run_id_prefix", "publish"))
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": f"{run_id_prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        "seeds": repro_cfg.get("seeds", {}),
        "out_dir": str(out_dir),
    }
    _write_json(out_dir / "reproducibility_lock.json", payload)
    return payload


def _capture_baseline_metrics(out_dir: Path) -> dict[str, str]:
    mapping = {
        "DE": REPO_ROOT / "reports/week2_metrics.json",
        "US": REPO_ROOT / "reports/eia930/week2_metrics.json",
    }
    out: dict[str, str] = {}
    for name, src in mapping.items():
        if src.exists():
            dst = out_dir / f"baseline_metrics_{name.lower()}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            out[name] = str(dst)
    return out


def _run_step(
    *,
    name: str,
    cmd: list[str],
    logs_dir: Path,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> tuple[StepResult, str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cmd_env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = cmd_env.get("PYTHONPATH", "")
    cmd_env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path
    )
    if env:
        cmd_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=cmd_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    duration = time.time() - started
    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    log_path = logs_dir / f"{safe_name}.log"
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    step = StepResult(
        name=name,
        ok=proc.returncode == 0,
        return_code=int(proc.returncode),
        duration_s=float(duration),
        command=cmd,
        log_path=str(log_path),
    )
    return step, output


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        obj = None

    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last > first:
        block = raw[first : last + 1]
        try:
            obj = json.loads(block)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _compute_iot_ack_hold_metrics(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "db_exists": False,
            "ack_success_rate": None,
            "hold_rate": None,
            "acks_total": 0,
            "holds_total": 0,
            "commands_total": 0,
        }
    con = duckdb.connect(str(db_path))
    try:
        tables = {str(r[0]) for r in con.execute("SHOW TABLES").fetchall()}
        acked = 0
        acks_total = 0
        if "iot_ack" in tables:
            row = con.execute(
                "SELECT SUM(CASE WHEN status='acked' THEN 1 ELSE 0 END), COUNT(*) FROM iot_ack"
            ).fetchone()
            acked = int(row[0] or 0)
            acks_total = int(row[1] or 0)
        holds = 0
        commands_total = 0
        if "iot_command_queue" in tables:
            row = con.execute(
                "SELECT SUM(CASE WHEN status='hold' THEN 1 ELSE 0 END), COUNT(*) FROM iot_command_queue"
            ).fetchone()
            holds = int(row[0] or 0)
            commands_total = int(row[1] or 0)
    finally:
        con.close()

    return {
        "db_exists": True,
        "ack_success_rate": (acked / acks_total) if acks_total else None,
        "hold_rate": (holds / commands_total) if commands_total else None,
        "acks_total": acks_total,
        "holds_total": holds,
        "commands_total": commands_total,
    }


def _compute_dc3s_typed_readiness(db_path: Path, table_name: str) -> dict[str, Any]:
    if not db_path.exists():
        return {"exists": False}
    con = duckdb.connect(str(db_path))
    try:
        tables = {str(r[0]) for r in con.execute("SHOW TABLES").fetchall()}
        if table_name not in tables:
            return {"exists": False, "reason": f"missing_table:{table_name}"}
        total = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0] or 0)
        row = con.execute(
            f"""
            SELECT
              SUM(CASE WHEN intervened IS NULL THEN 1 ELSE 0 END),
              SUM(CASE WHEN intervention_reason IS NULL THEN 1 ELSE 0 END),
              SUM(CASE WHEN reliability_w IS NULL THEN 1 ELSE 0 END),
              SUM(CASE WHEN drift_flag IS NULL THEN 1 ELSE 0 END),
              SUM(CASE WHEN inflation IS NULL THEN 1 ELSE 0 END),
              SUM(CASE WHEN guarantee_checks_passed IS NULL THEN 1 ELSE 0 END),
              SUM(CASE WHEN guarantee_fail_reasons IS NULL THEN 1 ELSE 0 END),
              SUM(CASE WHEN assumptions_version IS NULL THEN 1 ELSE 0 END)
            FROM {table_name}
            """
        ).fetchone()
    finally:
        con.close()
    n = max(total, 1)
    return {
        "exists": True,
        "rows_total": total,
        "null_counts": {
            "intervened": int(row[0] or 0),
            "intervention_reason": int(row[1] or 0),
            "reliability_w": int(row[2] or 0),
            "drift_flag": int(row[3] or 0),
            "inflation": int(row[4] or 0),
            "guarantee_checks_passed": int(row[5] or 0),
            "guarantee_fail_reasons": int(row[6] or 0),
            "assumptions_version": int(row[7] or 0),
        },
        "null_ratios": {
            "intervened": float((row[0] or 0) / n),
            "intervention_reason": float((row[1] or 0) / n),
            "reliability_w": float((row[2] or 0) / n),
            "drift_flag": float((row[3] or 0) / n),
            "inflation": float((row[4] or 0) / n),
            "guarantee_checks_passed": float((row[5] or 0) / n),
            "guarantee_fail_reasons": float((row[6] or 0) / n),
            "assumptions_version": float((row[7] or 0) / n),
        },
    }


def _load_monitoring_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_iot_loop_db_path() -> Path:
    env_path = os.environ.get("ORIUS_IOT_DUCKDB_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.is_absolute() else REPO_ROOT / path
    candidates = [
        REPO_ROOT / "data/audit/iot_loop.duckdb",
        REPO_ROOT / "data/interim/iot_loop.duckdb",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _evaluate_go_no_go(
    *,
    publish_cfg: dict[str, Any],
    steps: list[StepResult],
    iot_nominal: dict[str, Any] | None,
    iot_ack_hold: dict[str, Any],
    monitoring_summary: dict[str, Any],
) -> dict[str, Any]:
    gates = publish_cfg.get("go_no_go", {}) if isinstance(publish_cfg.get("go_no_go"), dict) else {}
    safety_req = int(gates.get("safety_violations", 0))
    cert_req = float(gates.get("certificate_completeness_rate_min", 0.99))
    ack_req = float(gates.get("ack_success_rate_min", 0.99))
    hold_max = float(gates.get("hold_rate_max", 0.01))
    require_no_critical = bool(gates.get("require_no_critical_alerts", True))

    blockers: list[dict[str, Any]] = []

    failing_steps = [s.name for s in steps if not s.ok]
    if failing_steps:
        blockers.append(
            {
                "category": "pipeline_steps",
                "message": "One or more required steps failed",
                "details": failing_steps,
            }
        )

    safety_violations = int((iot_nominal or {}).get("safety_violations", 999999))
    cert_rate = float((iot_nominal or {}).get("certificate_completeness_rate", 0.0))
    ack_rate = iot_ack_hold.get("ack_success_rate")
    hold_rate = iot_ack_hold.get("hold_rate")

    if safety_violations > safety_req:
        blockers.append(
            {
                "category": "safety_violations",
                "message": f"safety_violations={safety_violations} > {safety_req}",
            }
        )
    if cert_rate < cert_req:
        blockers.append(
            {
                "category": "certificate_completeness",
                "message": f"certificate_completeness_rate={cert_rate:.4f} < {cert_req:.4f}",
            }
        )
    if ack_rate is None or float(ack_rate) < ack_req:
        blockers.append(
            {"category": "ack_success_rate", "message": f"ack_success_rate={ack_rate} < {ack_req:.4f}"}
        )
    if hold_rate is None or float(hold_rate) > hold_max:
        blockers.append({"category": "hold_rate", "message": f"hold_rate={hold_rate} > {hold_max:.4f}"})

    critical_alerts: list[str] = []
    if require_no_critical:
        data_drift = bool((monitoring_summary.get("data_drift") or {}).get("drift", False))
        model_drift = bool(
            (monitoring_summary.get("model_drift") or {}).get("decision", {}).get("drift", False)
        )
        dc3s_triggered = bool((monitoring_summary.get("dc3s_health") or {}).get("triggered", False))
        if data_drift:
            critical_alerts.append("data_drift")
        if model_drift:
            critical_alerts.append("model_drift")
        if dc3s_triggered:
            critical_alerts.append("dc3s_health_triggered")
    if critical_alerts:
        blockers.append(
            {
                "category": "critical_alerts",
                "message": "critical monitoring alerts present",
                "details": critical_alerts,
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "gate_config": gates,
        "metrics": {
            "safety_violations": safety_violations,
            "certificate_completeness_rate": cert_rate,
            "ack_success_rate": ack_rate,
            "hold_rate": hold_rate,
            "critical_alerts": critical_alerts,
        },
        "go": len(blockers) == 0,
        "blockers": blockers,
    }


def _build_markdown_report(
    *,
    baseline: dict[str, Any],
    scope_manifest: dict[str, Any],
    repro_lock: dict[str, Any],
    git_delta_summary: dict[str, Any],
    figure_inventory_summary: dict[str, Any],
    step_results: list[StepResult],
    data_refresh_summary: dict[str, Any],
    na_summary: dict[str, Any],
    leakage_summary: dict[str, Any],
    code_health_summary: dict[str, Any],
    dc3s_readiness: dict[str, Any],
    monitoring_readiness: dict[str, Any],
    go_no_go: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Final Publish Audit Report")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now(UTC).isoformat()}`")
    lines.append(f"- Run ID: `{repro_lock.get('run_id')}`")
    lines.append(f"- GO/NO-GO: **{'GO' if go_no_go.get('go') else 'NO-GO'}**")
    lines.append("")
    lines.append("## Baseline")
    lines.append(f"- Branch: `{(baseline.get('git') or {}).get('branch')}`")
    lines.append(f"- HEAD: `{(baseline.get('git') or {}).get('head')}`")
    lines.append(f"- Dirty files: `{(baseline.get('git') or {}).get('dirty_file_count')}`")
    lines.append("")
    lines.append("## Git Delta")
    lines.append(f"- Changed files: `{git_delta_summary.get('total_changed_files')}`")
    lines.append(f"- In scope: `{git_delta_summary.get('in_scope_files')}`")
    lines.append(f"- Out of scope: `{git_delta_summary.get('out_of_scope_files')}`")
    lines.append(f"- Added lines: `{git_delta_summary.get('added_lines')}`")
    lines.append(f"- Deleted lines: `{git_delta_summary.get('deleted_lines')}`")
    lines.append("")
    lines.append("## Figure Integrity")
    lines.append(f"- Files inventoried: `{figure_inventory_summary.get('files_total')}`")
    lines.append(f"- Critical missing: `{figure_inventory_summary.get('critical_missing')}`")
    lines.append(f"- Critical zero-size: `{figure_inventory_summary.get('critical_zero_size')}`")
    lines.append(f"- Critical OK: `{figure_inventory_summary.get('critical_ok')}`")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- In scope: `{scope_manifest.get('in_scope')}`")
    lines.append(f"- Non-blocking out-of-scope: `{scope_manifest.get('non_blocking_out_of_scope')}`")
    lines.append("")
    lines.append("## Pipeline Steps")
    lines.append("| Step | Status | Return Code | Duration (s) | Log |")
    lines.append("|---|:---:|---:|---:|---|")
    for step in step_results:
        lines.append(
            f"| {step.name} | {'OK' if step.ok else 'FAIL'} | {step.return_code} | {step.duration_s:.1f} | `{step.log_path}` |"
        )
    lines.append("")
    lines.append("## Data Freshness")
    lines.append(f"- Summary: `{data_refresh_summary}`")
    lines.append("")
    lines.append("## NA Audit")
    lines.append(f"- Violations: `{na_summary.get('violations')}`")
    lines.append(f"- Fail: `{na_summary.get('fail')}`")
    lines.append("")
    lines.append("## Leakage Audit")
    lines.append(f"- Fail: `{leakage_summary.get('fail')}`")
    lines.append(f"- Violations: `{len(leakage_summary.get('violations', []))}`")
    lines.append("")
    lines.append("## Code Health")
    lines.append(f"- Fail: `{code_health_summary.get('fail')}`")
    lines.append(f"- Violations: `{len(code_health_summary.get('violations', []))}`")
    lines.append("")
    lines.append("## DC3S Readiness")
    lines.append(f"- Typed column readiness: `{dc3s_readiness}`")
    lines.append("")
    lines.append("## Monitoring Readiness")
    lines.append(f"- Monitoring summary: `{monitoring_readiness}`")
    lines.append("")
    lines.append("## GO/NO-GO Decision")
    lines.append(f"- Decision: **{'GO' if go_no_go.get('go') else 'NO-GO'}**")
    lines.append(f"- Blockers: `{go_no_go.get('blockers')}`")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final pre-publish audit orchestration")
    parser.add_argument("--config", default="configs/publish_audit.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument("--max-runtime-hours", type=float, default=None)
    parser.add_argument("--iot-steps", type=int, default=72)
    parser.add_argument("--run-hooks", action="store_true", help="Enable refresh_data_delta hook execution")
    parser.add_argument("--skip-retrain", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    publish_cfg = _load_publish_cfg(config_path)
    out_dir = Path(
        args.out_dir
        or (
            (publish_cfg.get("reproducibility") or {}).get("out_dir")
            if isinstance(publish_cfg.get("reproducibility"), dict)
            else None
        )
        or "reports/publish"
    )
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    baseline = _snapshot_baseline(out_dir)
    scope_manifest = _write_scope_manifest(out_dir, publish_cfg)
    repro_lock = _write_repro_lock(out_dir, publish_cfg)
    baseline_metrics = _capture_baseline_metrics(out_dir)

    steps: list[StepResult] = []
    outputs: dict[str, Any] = {}

    planned_steps: list[tuple[str, list[str], dict[str, str] | None, float | None]] = [
        (
            "audit_git_delta",
            [
                sys.executable,
                "scripts/audit_git_delta.py",
                "--baseline-ref",
                str(args.baseline_ref),
                "--out-dir",
                str(out_dir),
                "--scope-config",
                str(config_path),
            ],
            None,
            None,
        ),
        (
            "audit_figure_inventory",
            [
                sys.executable,
                "scripts/audit_figure_inventory.py",
                "--out-dir",
                str(out_dir),
                "--manifest",
                "paper/metrics_manifest.json",
                "--run-start-utc",
                str(baseline.get("generated_at")),
            ],
            None,
            None,
        ),
        (
            "refresh_data_delta",
            [sys.executable, "scripts/refresh_data_delta.py", "--dataset", "ALL", "--apply"]
            + (["--run-hooks"] if args.run_hooks else []),
            None,
            None,
        ),
        (
            "rebuild_de_splits_reports_only",
            [sys.executable, "scripts/train_dataset.py", "--dataset", "DE", "--reports-only", "--rebuild"],
            None,
            None,
        ),
        (
            "rebuild_us_splits_reports_only",
            [sys.executable, "scripts/train_dataset.py", "--dataset", "US", "--reports-only", "--rebuild"],
            None,
            None,
        ),
        (
            "backfill_dc3s_typed_columns",
            [sys.executable, "scripts/backfill_dc3s_typed_columns.py"],
            None,
            None,
        ),
        (
            "audit_na_tables",
            [sys.executable, "scripts/audit_na_tables.py", "--config", str(config_path)],
            None,
            None,
        ),
        (
            "audit_leakage",
            [sys.executable, "scripts/audit_leakage.py", "--config", str(config_path)],
            None,
            None,
        ),
        (
            "audit_code_health",
            [sys.executable, "scripts/audit_code_health.py", "--config", str(config_path)],
            None,
            None,
        ),
        (
            "compile_gate",
            [sys.executable, "-m", "compileall", "src", "services", "scripts"],
            {**os.environ, "PYTHONPYCACHEPREFIX": "/tmp/orius_pycache"},
            None,
        ),
    ]

    if not args.skip_retrain:
        planned_steps.extend(
            [
                (
                    "retrain_de_aggressive",
                    [
                        sys.executable,
                        "scripts/train_dataset.py",
                        "--dataset",
                        "DE",
                        "--profile",
                        "aggressive",
                        "--max-runtime-hours",
                        str(float(args.max_runtime_hours) if args.max_runtime_hours else 6.0),
                        "--target-metrics-file",
                        baseline_metrics.get("DE", "reports/week2_metrics.json"),
                    ],
                    None,
                    float(args.max_runtime_hours) * 3600.0 if args.max_runtime_hours else None,
                ),
                (
                    "retrain_us_aggressive",
                    [
                        sys.executable,
                        "scripts/train_dataset.py",
                        "--dataset",
                        "US",
                        "--profile",
                        "aggressive",
                        "--max-runtime-hours",
                        str(float(args.max_runtime_hours) if args.max_runtime_hours else 6.0),
                        "--target-metrics-file",
                        baseline_metrics.get("US", "reports/eia930/week2_metrics.json"),
                    ],
                    None,
                    float(args.max_runtime_hours) * 3600.0 if args.max_runtime_hours else None,
                ),
            ]
        )

    planned_steps.extend(
        [
            ("run_monitoring", [sys.executable, "scripts/run_monitoring.py", "--disable-alerts"], None, None),
        ]
    )

    if not args.skip_validation:
        planned_steps.extend(
            [
                (
                    "core_pytests",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "--no-cov",
                        "tests/test_dc3s_api_smoke.py",
                        "tests/test_dc3s_certificate_store.py",
                        "tests/test_dc3s_health_monitoring.py",
                        "tests/test_retraining_decision_dc3s.py",
                        "tests/test_run_monitoring_with_dc3s.py",
                        "tests/test_retrain_if_needed_dc3s_trigger.py",
                        "tests/test_iot_loop_smoke.py",
                    ],
                    None,
                    None,
                ),
                ("dc3s_demo", ["make", "dc3s-demo"], None, None),
                (
                    "iot_sim_nominal",
                    [
                        sys.executable,
                        "iot/simulator/run_closed_loop.py",
                        "--steps",
                        str(int(args.iot_steps)),
                        "--seed",
                        "17",
                        "--scenario",
                        "nominal",
                    ],
                    None,
                    None,
                ),
                (
                    "iot_sim_dropout",
                    [
                        sys.executable,
                        "iot/simulator/run_closed_loop.py",
                        "--steps",
                        str(int(args.iot_steps)),
                        "--seed",
                        "42",
                        "--scenario",
                        "dropout",
                    ],
                    None,
                    None,
                ),
                ("cpsbench", ["make", "cpsbench"], None, None),
                (
                    "dc3s_ablations",
                    [
                        sys.executable,
                        "scripts/run_ablations.py",
                        "--dc3s",
                        "--output",
                        "reports/publication",
                        "--scenario",
                        "drift_combo",
                        "--horizon",
                        "96",
                        "--seeds",
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                    ],
                    None,
                    None,
                ),
                ("release_check", ["bash", "scripts/release_check.sh"], None, None),
            ]
        )

    for name, cmd, env, timeout in planned_steps:
        step, out = _run_step(name=name, cmd=cmd, logs_dir=logs_dir, env=env, timeout_s=timeout)
        steps.append(step)
        outputs[name] = out

    data_refresh_summary = _extract_json_object(outputs.get("refresh_data_delta", "")) or {}
    na_summary = _extract_json_object(outputs.get("audit_na_tables", "")) or {}
    leakage_summary = _extract_json_object(outputs.get("audit_leakage", "")) or {}
    code_health_summary = _extract_json_object(outputs.get("audit_code_health", "")) or {}
    iot_nominal = _extract_json_object(outputs.get("iot_sim_nominal", ""))
    git_delta = _load_json_file(out_dir / "github_delta_report.json")
    figure_inventory = _load_json_file(out_dir / "figure_inventory.json")
    git_delta_summary = git_delta.get("summary", {}) if isinstance(git_delta.get("summary"), dict) else {}
    figure_inventory_summary = (
        figure_inventory.get("summary", {}) if isinstance(figure_inventory.get("summary"), dict) else {}
    )

    dc3s_cfg_path = REPO_ROOT / "configs/dc3s.yaml"
    dc3s_cfg = yaml.safe_load(dc3s_cfg_path.read_text(encoding="utf-8")) if dc3s_cfg_path.exists() else {}
    dc3s_audit_cfg = (
        ((dc3s_cfg or {}).get("dc3s") or {}).get("audit", {})
        if isinstance((dc3s_cfg or {}).get("dc3s"), dict)
        else {}
    )
    dc3s_db_path = Path(str(dc3s_audit_cfg.get("duckdb_path", "data/audit/dc3s_audit.duckdb")))
    if not dc3s_db_path.is_absolute():
        dc3s_db_path = REPO_ROOT / dc3s_db_path
    dc3s_table = str(dc3s_audit_cfg.get("table_name", "dispatch_certificates"))

    dc3s_readiness = _compute_dc3s_typed_readiness(dc3s_db_path, dc3s_table)
    monitoring_summary = _load_monitoring_summary(REPO_ROOT / "reports/monitoring_summary.json")
    monitoring_readiness = {
        "generated_at": datetime.now(UTC).isoformat(),
        "retraining": monitoring_summary.get("retraining"),
        "dc3s_health": monitoring_summary.get("dc3s_health"),
    }
    _write_json(out_dir / "dc3s_health_readiness.json", dc3s_readiness)
    _write_json(out_dir / "monitoring_readiness.json", monitoring_readiness)

    iot_ack_hold = _compute_iot_ack_hold_metrics(_resolve_iot_loop_db_path())
    go_no_go = _evaluate_go_no_go(
        publish_cfg=publish_cfg,
        steps=steps,
        iot_nominal=iot_nominal,
        iot_ack_hold=iot_ack_hold,
        monitoring_summary=monitoring_summary,
    )

    final_json = {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_ref": str(args.baseline_ref),
        "baseline": baseline,
        "scope_manifest": scope_manifest,
        "reproducibility_lock": repro_lock,
        "steps": [
            {
                "name": s.name,
                "ok": s.ok,
                "return_code": s.return_code,
                "duration_s": s.duration_s,
                "command": s.command,
                "log_path": s.log_path,
            }
            for s in steps
        ],
        "artifacts": {
            "git_delta": git_delta,
            "figure_inventory": figure_inventory,
            "data_refresh_summary": data_refresh_summary,
            "na_summary": na_summary,
            "leakage_summary": leakage_summary,
            "code_health_summary": code_health_summary,
            "dc3s_readiness": dc3s_readiness,
            "monitoring_readiness": monitoring_readiness,
        },
        "validation_metrics": {
            "iot_nominal": iot_nominal,
            "iot_ack_hold": iot_ack_hold,
        },
        "go_no_go": go_no_go,
    }
    _write_json(out_dir / "final_audit_report.json", final_json)
    _write_json(out_dir / "go_no_go_decision.json", go_no_go)

    md = _build_markdown_report(
        baseline=baseline,
        scope_manifest=scope_manifest,
        repro_lock=repro_lock,
        git_delta_summary=git_delta_summary,
        figure_inventory_summary=figure_inventory_summary,
        step_results=steps,
        data_refresh_summary=data_refresh_summary,
        na_summary=na_summary,
        leakage_summary=leakage_summary,
        code_health_summary=code_health_summary,
        dc3s_readiness=dc3s_readiness,
        monitoring_readiness=monitoring_readiness,
        go_no_go=go_no_go,
    )
    (out_dir / "final_audit_report.md").write_text(md, encoding="utf-8")
    print(json.dumps(go_no_go, indent=2))
    if not go_no_go.get("go", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
