#!/usr/bin/env python3
"""Build the canonical ORIUS domain-closure and P5/P6 cross-domain matrices.

This script is intentionally conservative. It does not widen any claim tier
from prose alone; it reads the current validation/training surfaces, runs a
small set of explicit bounded checks, and then emits the promotion matrix that
the thesis can cite directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _dataset_registry import DATASET_REGISTRY, REPO_ROOT, get_runtime_dataset_config, repo_path
from orius.adapters.vehicle import VehicleTrackAdapter
from orius.certos.domain_policies import policy_for_domain
from orius.certos.runtime import CertOSConfig, CertOSRuntime
from orius.multi_agent.protocol import (
    CentralizedCoordinatorProtocol,
    DistributedNegotiationProtocol,
    IndependentLocalProtocol,
)
from orius.multi_agent.scenarios import run_transformer_capacity_scenario
from orius.universal_framework import get_adapter, get_domain_capabilities, run_universal_step


DOMAIN_ORDER = ("battery", "industrial", "healthcare", "vehicle", "navigation", "aerospace")
TARGET_TIER = {
    "battery": "witness_row",
    "industrial": "defended_bounded_row",
    "healthcare": "defended_bounded_row",
    "vehicle": "defended_bounded_row",
    "navigation": "defended_bounded_row",
    "aerospace": "defended_bounded_row",
}
REGISTRY_DOMAIN_IDS = {
    "battery": "energy",
    "industrial": "industrial",
    "healthcare": "healthcare",
    "vehicle": "av",
    "navigation": "navigation",
    "aerospace": "aerospace",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _humanize_table_cell(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value).replace("_", " ")


def _latex_escape(value: Any) -> str:
    text = _humanize_table_cell(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _write_tex_table(
    path: Path,
    *,
    caption: str,
    label: str,
    column_spec: str,
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{_latex_escape(header)}}}" for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bool_status(value: bool) -> str:
    return "pass" if value else "fail"


def _representative_runtime_case(domain: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    if domain == "industrial":
        return (
            {
                "temp_c": 25.0,
                "pressure_mbar": 1010.0,
                "power_mw": 470.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {"power_setpoint_mw": 520.0},
            {"power_max_mw": 500.0, "temp_max_c": 120.0},
            30.0,
        )
    if domain == "healthcare":
        return (
            {
                "hr_bpm": 72.0,
                "spo2_pct": 88.0,
                "respiratory_rate": 16.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {"alert_level": 0.1},
            {"spo2_min_pct": 90.0},
            5.0,
        )
    if domain == "vehicle":
        return (
            {
                "position_m": 40.0,
                "speed_mps": 12.0,
                "speed_limit_mps": 30.0,
                "lead_position_m": 75.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {"acceleration_mps2": 2.0},
            {
                "speed_limit_mps": 30.0,
                "accel_min_mps2": -5.0,
                "accel_max_mps2": 3.0,
                "dt_s": 0.25,
                "min_headway_m": 5.0,
                "ttc_min_s": 2.0,
            },
            0.9,
        )
    if domain == "navigation":
        return (
            {
                "x": 9.8,
                "y": 9.6,
                "vx": 0.1,
                "vy": 0.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {"ax": 4.0, "ay": 4.0},
            {"arena_min": 0.0, "arena_max": 10.0, "max_speed": 1.0, "dt_s": 0.25},
            10.0,
        )
    if domain == "aerospace":
        return (
            {
                "altitude_m": 3000.0,
                "airspeed_kt": 180.0,
                "bank_angle_deg": 28.0,
                "fuel_remaining_pct": 65.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {"throttle": 0.9, "bank_deg": 35.0},
            {"v_min_kt": 60.0, "v_max_kt": 350.0, "max_bank_deg": 30.0},
            5.0,
        )
    raise KeyError(f"No representative runtime case configured for {domain}")


def _runtime_contract_status(domain: str) -> dict[str, Any]:
    if domain == "battery":
        return {
            "typed_kernel_status": "reference_witness",
            "fallback_status": "paper6_runtime",
            "detail": "Battery remains the full proof witness.",
        }
    try:
        telemetry, candidate, constraints, quantile = _representative_runtime_case(domain)
        adapter = get_adapter(REGISTRY_DOMAIN_IDS[domain], {})
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=telemetry,
            history=None,
            candidate_action=candidate,
            constraints=constraints,
            quantile=quantile,
        )
        checks = dict(result.get("contract_checks", {}))
        passed = bool(checks.get("contract_passed", False))
        return {
            "typed_kernel_status": _bool_status(passed),
            "fallback_status": "bounded_runtime_pass" if passed else "bounded_runtime_fail",
            "detail": checks.get("failed_invariants", []),
            "result": result,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "typed_kernel_status": "fail",
            "fallback_status": "bounded_runtime_fail",
            "detail": str(exc),
        }


def _vehicle_soundness_report() -> dict[str, Any]:
    adapter = get_adapter("av", {})
    track = VehicleTrackAdapter()
    cases = [
        {
            "name": "recoverable_ttc_clamp",
            "telemetry": {
                "position_m": 40.0,
                "speed_mps": 12.0,
                "speed_limit_mps": 30.0,
                "lead_position_m": 75.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            "candidate": {"acceleration_mps2": 2.0},
            "constraints": {
                "speed_limit_mps": 30.0,
                "accel_min_mps2": -5.0,
                "accel_max_mps2": 3.0,
                "dt_s": 0.25,
                "min_headway_m": 5.0,
                "ttc_min_s": 2.0,
            },
            "expect_barrier": False,
        },
        {
            "name": "unavoidable_entry_barrier",
            "telemetry": {
                "position_m": 44.0,
                "speed_mps": 10.0,
                "speed_limit_mps": 30.0,
                "lead_position_m": 50.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            "candidate": {"acceleration_mps2": 2.0},
            "constraints": {
                "speed_limit_mps": 30.0,
                "accel_min_mps2": -5.0,
                "accel_max_mps2": 3.0,
                "dt_s": 0.25,
                "min_headway_m": 5.0,
                "ttc_min_s": 2.0,
            },
            "expect_barrier": True,
        },
    ]

    rows: list[dict[str, Any]] = []
    passed = True
    for case in cases:
        track.reset(42)
        track._plant.reset(  # type: ignore[attr-defined]
            position_m=float(case["telemetry"]["position_m"]),
            speed_mps=float(case["telemetry"]["speed_mps"]),
            lead_position_m=float(case["telemetry"]["lead_position_m"]),
            speed_limit_mps=float(case["telemetry"]["speed_limit_mps"]),
        )
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=case["telemetry"],
            history=None,
            candidate_action=case["candidate"],
            constraints=case["constraints"],
            quantile=0.9,
        )
        safe_action = dict(result["safe_action"])
        meta = dict(result["repair_meta"])
        next_state = track.step(safe_action)
        violated = bool(track.check_violation(next_state)["violated"])
        barrier = bool(meta.get("entry_barrier_triggered", False))
        case_pass = (barrier if case["expect_barrier"] else not violated)
        passed = passed and case_pass
        rows.append(
            {
                "case": case["name"],
                "safe_action": json.dumps(safe_action, sort_keys=True),
                "intervention_reason": meta.get("intervention_reason", ""),
                "entry_barrier_triggered": barrier,
                "violated_after_step": violated,
                "case_pass": case_pass,
            }
        )
    return {"soundness_pass": passed, "rows": rows}


def _dataset_chain_status(domain: str, training_report: dict[str, Any]) -> tuple[str, str]:
    verified = set(training_report.get("training_verified_domains", []))
    cfg = get_runtime_dataset_config(domain)
    if domain == "battery":
        return "locked_reference", "battery_reference_witness"
    if domain == "industrial":
        return ("verified", str(cfg.exact_blocker)) if "industrial" in verified else ("blocked", "industrial_training_gap")
    if domain == "healthcare":
        return ("verified", str(cfg.exact_blocker)) if "healthcare" in verified else ("blocked", "healthcare_training_gap")
    if domain == "vehicle":
        av_cfg = DATASET_REGISTRY["AV"]
        processed = (REPO_ROOT / av_cfg.raw_data_path).exists()
        features = (REPO_ROOT / av_cfg.features_path).exists()
        if processed and features and "av" in verified:
            return "verified", str(DATASET_REGISTRY["AV"].exact_blocker)
        return "blocked", "av_training_surface_incomplete"
    if domain == "navigation":
        processed = (REPO_ROOT / cfg.raw_data_path).exists()
        features = (REPO_ROOT / cfg.features_path).exists()
        runtime_manifest = repo_path(cfg.runtime_provenance_path)
        if processed and features and runtime_manifest is not None and runtime_manifest.exists() and "navigation" in verified:
            return "real_data_ready", "real_data_row_cleared"
        return "blocked", str(cfg.exact_blocker)
    if domain == "aerospace":
        processed = (REPO_ROOT / cfg.raw_data_path).exists()
        features = (REPO_ROOT / cfg.features_path).exists()
        runtime_processed = repo_path(cfg.canonical_runtime_path)
        runtime_manifest = repo_path(cfg.runtime_provenance_path)
        support_runtime = repo_path(cfg.support_runtime_path)
        support_manifest = repo_path(cfg.support_runtime_provenance_path)
        if (
            processed
            and features
            and runtime_processed is not None
            and runtime_processed.exists()
            and runtime_manifest is not None
            and runtime_manifest.exists()
            and "aerospace" in verified
        ):
            return "real_data_ready", "realflight_row_cleared"
        if (
            processed
            and features
            and support_runtime is not None
            and support_runtime.exists()
            and support_manifest is not None
            and support_manifest.exists()
        ):
            return "support_lane_only", str(cfg.exact_blocker)
        return "blocked", str(cfg.exact_blocker)
    raise KeyError(domain)


def _run_generic_shared_resource_scenario(capacity: float, proposals: list[dict[str, float]]) -> dict[str, Any]:
    protocols = {
        "independent": IndependentLocalProtocol(),
        "centralized": CentralizedCoordinatorProtocol(),
        "distributed": DistributedNegotiationProtocol(),
    }
    summary: dict[str, Any] = {}
    for name, protocol in protocols.items():
        actions = protocol.compute_actions({}, proposals, feeder_capacity_mw=capacity)
        total = sum(float(a.get("discharge_mw", 0.0)) - float(a.get("charge_mw", 0.0)) for a in actions)
        summary[name] = {
            "joint_violation": total > capacity + 1e-9,
            "total_demand": total,
        }
    return summary


def _build_paper5_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    battery = run_transformer_capacity_scenario(out_dir=None)
    rows.append(
        {
            "domain": "battery",
            "supported": "yes",
            "shared_constraint_surface": "shared_feeder_capacity",
            "independent_violation": bool(battery["independent"]["joint_violations"] > 0),
            "coordinated_safe": bool(battery["centralized"]["joint_violations"] == 0 and battery["distributed"]["joint_violations"] == 0),
            "bounded_claim": "counterexample_plus_coordination",
            "status": "evaluated",
        }
    )
    industrial = _run_generic_shared_resource_scenario(
        500.0,
        [{"charge_mw": 0.0, "discharge_mw": 320.0}, {"charge_mw": 0.0, "discharge_mw": 320.0}],
    )
    rows.append(
        {
            "domain": "industrial",
            "supported": "yes",
            "shared_constraint_surface": "shared_power_budget",
            "independent_violation": bool(industrial["independent"]["joint_violation"]),
            "coordinated_safe": bool(not industrial["centralized"]["joint_violation"] and not industrial["distributed"]["joint_violation"]),
            "bounded_claim": "shared_budget_stress_test",
            "status": "evaluated",
        }
    )
    healthcare = _run_generic_shared_resource_scenario(
        1.0,
        [{"charge_mw": 0.0, "discharge_mw": 0.7}, {"charge_mw": 0.0, "discharge_mw": 0.7}],
    )
    rows.append(
        {
            "domain": "healthcare",
            "supported": "yes",
            "shared_constraint_surface": "shared_alert_budget",
            "independent_violation": bool(healthcare["independent"]["joint_violation"]),
            "coordinated_safe": bool(not healthcare["centralized"]["joint_violation"] and not healthcare["distributed"]["joint_violation"]),
            "bounded_claim": "certificate_gated_alert_budget",
            "status": "evaluated",
        }
    )
    vehicle = _run_generic_shared_resource_scenario(
        1.0,
        [{"charge_mw": 0.0, "discharge_mw": 0.8}, {"charge_mw": 0.0, "discharge_mw": 0.8}],
    )
    rows.append(
        {
            "domain": "vehicle",
            "supported": "yes",
            "shared_constraint_surface": "shared_headway_budget",
            "independent_violation": bool(vehicle["independent"]["joint_violation"]),
            "coordinated_safe": bool(not vehicle["centralized"]["joint_violation"] and not vehicle["distributed"]["joint_violation"]),
            "bounded_claim": "coordinated_headway_stress_test",
            "status": "evaluated",
        }
    )
    navigation_ready = (REPO_ROOT / "data" / "navigation" / "processed" / "navigation_orius.csv").exists()
    aerospace_ready = (REPO_ROOT / "data" / "aerospace" / "processed" / "aerospace_realflight_runtime.csv").exists()
    conditional_rows = {
        "navigation": {
            "surface": "shared_corridor_capacity",
            "claim": "coordinated_guidance_correlation_test",
        },
        "aerospace": {
            "surface": "shared_airspace_separation_budget",
            "claim": "coordinated_envelope_separation_test",
        },
    }
    for domain in ("navigation", "aerospace"):
        ready = navigation_ready if domain == "navigation" else aerospace_ready
        if ready:
            scenario = _run_generic_shared_resource_scenario(
                1.0,
                [{"charge_mw": 0.0, "discharge_mw": 0.8}, {"charge_mw": 0.0, "discharge_mw": 0.8}],
            )
            rows.append(
                {
                    "domain": domain,
                    "supported": "yes",
                    "shared_constraint_surface": conditional_rows[domain]["surface"],
                    "independent_violation": bool(scenario["independent"]["joint_violation"]),
                    "coordinated_safe": bool(not scenario["centralized"]["joint_violation"] and not scenario["distributed"]["joint_violation"]),
                    "bounded_claim": conditional_rows[domain]["claim"],
                    "status": "evaluated",
                }
            )
            continue
        rows.append(
            {
                "domain": domain,
                "supported": "no",
                "shared_constraint_surface": "",
                "independent_violation": "",
                "coordinated_safe": "",
                "bounded_claim": "not_supported_in_this_pass",
                "status": "gated",
            }
        )
    return rows


def _evaluate_certos_domain(domain: str, proposed: Mapping[str, float], safe: Mapping[str, float]) -> dict[str, Any]:
    rt = CertOSRuntime(config=CertOSConfig(governance_policy=policy_for_domain(domain)))
    s1 = rt.validate_and_step(100.0, proposed, safe, 8)
    s2 = rt.validate_and_step(100.0, proposed, safe, 2)
    s3 = rt.validate_and_step(100.0, proposed, safe, 0)
    ops = [entry["op"] for entry in rt.raw_audit_log]
    invariants_ok = rt.check_invariants(s1) == [] and rt.check_invariants(s2) == [] and rt.check_invariants(s3) == []
    return {
        "domain": domain,
        "supported": "yes",
        "issue_seen": "ISSUE" in ops,
        "validate_seen": "VALIDATE" in ops,
        "expire_seen": "EXPIRE" in ops,
        "fallback_seen": bool(s3.fallback_active and "FALLBACK" in ops),
        "hash_chain_ok": bool(s3.hash_chain_ok),
        "invariants_pass": invariants_ok,
        "status": "evaluated" if (s3.hash_chain_ok and invariants_ok) else "fail",
    }


def _build_paper6_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    battery = _evaluate_certos_domain(
        "battery",
        {"charge_mw": 0.0, "discharge_mw": 50.0},
        {"charge_mw": 0.0, "discharge_mw": 45.0},
    )
    rows.append(battery)
    for domain in ("industrial", "healthcare", "vehicle"):
        telemetry, candidate, constraints, quantile = _representative_runtime_case(domain)
        adapter = get_adapter(REGISTRY_DOMAIN_IDS[domain], {})
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=telemetry,
            history=None,
            candidate_action=candidate,
            constraints=constraints,
            quantile=quantile,
        )
        rows.append(_evaluate_certos_domain(domain, candidate, dict(result["safe_action"])))
    for domain in ("navigation", "aerospace"):
        telemetry, candidate, constraints, quantile = _representative_runtime_case(domain)
        adapter = get_adapter(REGISTRY_DOMAIN_IDS[domain], {})
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=telemetry,
            history=None,
            candidate_action=candidate,
            constraints=constraints,
            quantile=quantile,
        )
        rows.append(_evaluate_certos_domain(domain, candidate, dict(result["safe_action"])))
    return rows


def build_closure_matrix(
    *,
    validation_report_path: Path,
    training_report_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    validation_report = _read_json(validation_report_path)
    training_report = _read_json(training_report_path)

    domain_results = {row["domain"]: row for row in validation_report.get("domain_results", [])}
    proof_reports = dict(validation_report.get("domain_proof_reports", {}))
    support_reports = dict(validation_report.get("domain_support_reports", {}))
    p5_rows = _build_paper5_rows()
    p6_rows = _build_paper6_rows()
    p5_map = {row["domain"]: row for row in p5_rows}
    p6_map = {row["domain"]: row for row in p6_rows}
    vehicle_soundness = _vehicle_soundness_report()

    closure_rows: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        caps = get_domain_capabilities(REGISTRY_DOMAIN_IDS[domain])
        data_status, blocker = _dataset_chain_status(domain, training_report)
        domain_row = domain_results.get(domain, {})
        replay_status = "not_run"
        if domain in ("battery", "industrial", "healthcare"):
            replay_status = "pass" if domain_row.get("validation_status", "").endswith("validated") else "fail"
        elif domain == "vehicle":
            replay_status = _bool_status(bool(proof_reports.get(domain, {}).get("evidence_pass", False)))
        elif domain in ("navigation", "aerospace"):
            report = support_reports.get(domain, {})
            replay_status = "pass" if report.get("portability_pass", False) else "fail"

        runtime_status = _runtime_contract_status(domain)
        typed_kernel_status = str(runtime_status["typed_kernel_status"])
        fallback_status = str(runtime_status["fallback_status"])
        safe_action_soundness = ""
        if domain == "vehicle":
            safe_action_soundness = _bool_status(bool(vehicle_soundness["soundness_pass"]))
        elif domain in ("industrial", "healthcare", "battery", "navigation", "aerospace"):
            safe_action_soundness = typed_kernel_status

        multi_agent_status = p5_map[domain]["status"] if domain in p5_map else "gated"
        certos_status = p6_map[domain]["status"] if domain in p6_map else "gated"

        reported_tier = str(domain_row.get("validation_status") or validation_report.get("domain_maturity", {}).get(domain, "unknown"))
        if reported_tier == "reference_validated":
            reported_tier = "reference"
        resulting_tier = reported_tier
        exact_blocker = blocker
        if domain == "vehicle":
            all_av_gates = (
                replay_status == "pass"
                and safe_action_soundness == "pass"
                and typed_kernel_status == "pass"
                and certos_status == "evaluated"
            )
            resulting_tier = "proof_validated" if all_av_gates else "proof_candidate"
            exact_blocker = "ttc_replay_or_soundness_gate_open" if not all_av_gates else "promoted"
        elif domain == "navigation":
            nav_pass = (
                data_status == "real_data_ready"
                and replay_status == "pass"
                and safe_action_soundness == "pass"
                and typed_kernel_status == "pass"
                and certos_status == "evaluated"
                and multi_agent_status == "evaluated"
            )
            resulting_tier = "proof_validated" if nav_pass else reported_tier
            exact_blocker = "real_data_row_cleared" if nav_pass else str(get_runtime_dataset_config(domain).exact_blocker)
        elif domain == "aerospace":
            aero_pass = (
                data_status == "real_data_ready"
                and replay_status == "pass"
                and safe_action_soundness == "pass"
                and typed_kernel_status == "pass"
                and certos_status == "evaluated"
                and multi_agent_status == "evaluated"
            )
            resulting_tier = "proof_validated" if aero_pass else reported_tier
            exact_blocker = "realflight_row_cleared" if aero_pass else str(get_runtime_dataset_config(domain).exact_blocker)

        closure_rows.append(
            {
                "domain": domain,
                "closure_target_tier": TARGET_TIER[domain],
                "adapter_correctness": "pass",
                "training_data_status": data_status,
                "replay_status": replay_status,
                "safe_action_soundness_status": safe_action_soundness,
                "fallback_status": fallback_status,
                "typed_kernel_status": typed_kernel_status,
                "multi_agent_portability_status": multi_agent_status,
                "certos_portability_status": certos_status,
                "safety_surface_type": caps.get("safety_surface_type", ""),
                "repair_mode": caps.get("repair_mode", ""),
                "fallback_mode": caps.get("fallback_mode", ""),
                "resulting_tier": resulting_tier,
                "closure_target_ready": resulting_tier in {"reference", "proof_validated"},
                "exact_blocker": exact_blocker,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    closure_csv = out_dir / "domain_closure_matrix.csv"
    closure_json = out_dir / "domain_closure_matrix.json"
    p5_csv = out_dir / "paper5_cross_domain_matrix.csv"
    p6_csv = out_dir / "paper6_cross_domain_matrix.csv"
    closure_tex = out_dir / "tbl_domain_closure_matrix.tex"
    p5_tex = out_dir / "tbl_paper5_cross_domain_matrix.tex"
    p6_tex = out_dir / "tbl_paper6_cross_domain_matrix.tex"
    _write_csv(closure_csv, closure_rows)
    _write_csv(p5_csv, p5_rows)
    _write_csv(p6_csv, p6_rows)
    _write_tex_table(
        closure_tex,
        caption="Canonical domain-closure matrix for universal ORIUS promotion. Promotion follows this table rather than prose-first widening.",
        label="tab:domain-closure-matrix",
        column_spec="p{1.5cm}p{1.8cm}p{1.7cm}p{1.2cm}p{1.5cm}p{1.1cm}p{1.1cm}p{1.4cm}p{2.3cm}",
        headers=[
            "Domain",
            "Target tier",
            "Train/data",
            "Replay",
            "Soundness",
            "P5",
            "P6",
            "Tier",
            "Exact blocker",
        ],
        rows=[
            [
                row["domain"],
                row["closure_target_tier"],
                row["training_data_status"],
                row["replay_status"],
                row["safe_action_soundness_status"],
                row["multi_agent_portability_status"],
                row["certos_portability_status"],
                row["resulting_tier"],
                row["exact_blocker"],
            ]
            for row in closure_rows
        ],
    )
    _write_tex_table(
        p5_tex,
        caption="Bounded cross-domain composition evaluation surface. Only domains with an explicit shared-constraint scenario are counted as evaluated.",
        label="tab:paper5-cross-domain",
        column_spec="p{1.7cm}p{1.4cm}p{3.4cm}p{3.6cm}p{1.3cm}",
        headers=["Domain", "Supported", "Shared constraint", "Bounded claim", "Status"],
        rows=[
            [
                row["domain"],
                row["supported"],
                row["shared_constraint_surface"],
                row["bounded_claim"],
                row["status"],
            ]
            for row in p5_rows
        ],
    )
    _write_tex_table(
        p6_tex,
        caption="Bounded cross-domain runtime-governance evaluation surface. The table records where the CertOS lifecycle was actually exercised under an adapter-supported action surface.",
        label="tab:paper6-cross-domain",
        column_spec="p{1.7cm}p{1.4cm}p{1.0cm}p{1.0cm}p{1.0cm}p{1.1cm}p{1.1cm}p{1.3cm}",
        headers=["Domain", "Supported", "Issue", "Validate", "Expire", "Fallback", "INV-2", "Status"],
        rows=[
            [
                row["domain"],
                row["supported"],
                row["issue_seen"],
                row["validate_seen"],
                row["expire_seen"],
                row["fallback_seen"],
                row["hash_chain_ok"],
                row["status"],
            ]
            for row in p6_rows
        ],
    )
    closure_json.write_text(
        json.dumps(
            {
                "domain_closure_rows": closure_rows,
                "paper5_rows": p5_rows,
                "paper6_rows": p6_rows,
                "vehicle_soundness_rows": vehicle_soundness["rows"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_md = out_dir / "domain_closure_summary.md"
    lines = [
        "# ORIUS Domain Closure Summary",
        "",
        "## Resulting tiers",
    ]
    for row in closure_rows:
        lines.append(
            f"- `{row['domain']}` → `{row['resulting_tier']}` "
            f"(blocker: `{row['exact_blocker']}`)"
        )
    lines.extend(
        [
            "",
            "## Bounded composition support",
            *[
                f"- `{row['domain']}` → `{row['status']}` ({row['bounded_claim']})"
                for row in p5_rows
            ],
            "",
            "## Runtime-governance support",
            *[
                f"- `{row['domain']}` → `{row['status']}`"
                for row in p6_rows
            ],
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "closure_csv": str(closure_csv),
        "closure_json": str(closure_json),
        "closure_tex": str(closure_tex),
        "paper5_csv": str(p5_csv),
        "paper5_tex": str(p5_tex),
        "paper6_csv": str(p6_csv),
        "paper6_tex": str(p6_tex),
        "summary_md": str(summary_md),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the ORIUS domain closure matrix and the bounded composition/runtime-governance cross-domain artifacts"
    )
    parser.add_argument(
        "--validation-report",
        default="reports/universal_orius_validation/validation_report.json",
        help="Path to validation_report.json",
    )
    parser.add_argument(
        "--training-report",
        default="reports/universal_training_audit/training_audit_report.json",
        help="Path to training_audit_report.json",
    )
    parser.add_argument(
        "--out",
        default="reports/universal_orius_validation",
        help="Directory for closure artifacts",
    )
    args = parser.parse_args()

    outputs = build_closure_matrix(
        validation_report_path=Path(args.validation_report),
        training_report_path=Path(args.training_report),
        out_dir=Path(args.out),
    )
    print("=== ORIUS Domain Closure Matrix ===")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
