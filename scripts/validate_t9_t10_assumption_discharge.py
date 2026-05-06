#!/usr/bin/env python3
"""Validate T9/T10 assumption-discharge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
DOMAINS = ("battery", "av", "healthcare")
THEOREMS = ("T9", "T10")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _path_exists(reference: str, *, out_dir: Path) -> bool:
    if not reference:
        return False
    path = Path(reference)
    if path.is_absolute():
        return path.exists()
    if reference.startswith("reports/publication/"):
        publication_relative = reference.removeprefix("reports/publication/")
        if (out_dir / publication_relative).exists():
            return True
    return (REPO_ROOT / path).exists()


def _is_missing(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return not text or text.startswith("missing") or "not_" in text or "not " in text


def _positive(value: Any, *, minimum: float) -> bool:
    try:
        return float(value) > float(minimum)
    except (TypeError, ValueError):
        return False


def _n_usable_rows_supported(payload: dict[str, Any]) -> bool:
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    min_rows = int(thresholds.get("min_rows", 1000) or 1000)
    return int(payload.get("n_usable_rows", 0) or 0) >= min_rows


def _promotion_ready_is_supported(payload: dict[str, Any], *, out_dir: Path) -> bool:
    theorem_id = payload.get("theorem_id")
    if not payload.get("artifact_source") or not _path_exists(
        str(payload.get("artifact_source")), out_dir=out_dir
    ):
        return False
    if not payload.get("source_trace_path") or not _path_exists(
        str(payload.get("source_trace_path")), out_dir=out_dir
    ):
        return False
    auxiliary_paths = payload.get("auxiliary_trace_paths", [])
    if auxiliary_paths:
        if not isinstance(auxiliary_paths, list):
            return False
        if not all(_path_exists(str(reference), out_dir=out_dir) for reference in auxiliary_paths):
            return False
    if not _n_usable_rows_supported(payload):
        return False
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    minimum = float(thresholds.get("min_positive_rate", 1e-6) or 1e-6)
    if theorem_id == "T9":
        required = (
            "witness_constant_status",
            "degradation_rate_status",
            "boundary_reachability_status",
            "mixing_bridge_status",
            "constants_status",
            "assumptions_status",
        )
        mixing = payload.get("mixing_proxy") if isinstance(payload.get("mixing_proxy"), dict) else {}
        numeric_supported = (
            _positive(payload.get("witness_constant"), minimum=minimum)
            and _positive(payload.get("degradation_rate"), minimum=minimum)
            and _positive(payload.get("boundary_reachability_rate"), minimum=minimum)
            and bool(mixing.get("finite_mixing_proxy"))
        )
    elif theorem_id == "T10":
        required = (
            "tv_bridge_status",
            "unsafe_boundary_mass_status",
            "reliability_sequence_status",
            "boundary_testing_subproblem_status",
            "constants_status",
            "assumptions_status",
        )
        tv_bridge = payload.get("tv_bridge") if isinstance(payload.get("tv_bridge"), dict) else {}
        numeric_supported = (
            _positive(payload.get("unsafe_boundary_mass"), minimum=minimum)
            and _positive(payload.get("boundary_mass"), minimum=minimum)
            and _positive(payload.get("safe_side_mass"), minimum=minimum)
            and bool(tv_bridge.get("passed"))
            and payload.get("le_cam_lower_bound") is not None
        )
    else:
        return False
    return numeric_supported and all(not _is_missing(payload.get(key)) for key in required)


def validate_assumption_discharge(*, out_dir: Path = PUBLICATION_DIR) -> dict[str, Any]:
    findings: list[str] = []
    evidence_dir = out_dir / "theorem_promotion_evidence"
    promotion_ready = True

    for theorem_id in THEOREMS:
        for domain in DOMAINS:
            name = f"{theorem_id}_{domain}"
            path = evidence_dir / f"{name}.json"
            try:
                payload = _read_json(path)
            except FileNotFoundError:
                findings.append(f"missing assumption-discharge artifact: {path}")
                promotion_ready = False
                continue
            if payload.get("theorem_id") != theorem_id or payload.get("domain") != domain:
                findings.append(f"{name}: theorem_id/domain does not match file name")
            if not payload.get("artifact_source"):
                findings.append(f"{name}: missing artifact_source")
            if not _path_exists(str(payload.get("artifact_source", "")), out_dir=out_dir):
                findings.append(f"{name}: artifact_source does not exist")
            if not payload.get("source_trace_path"):
                findings.append(f"{name}: missing source_trace_path")
            elif not _path_exists(str(payload.get("source_trace_path", "")), out_dir=out_dir):
                findings.append(f"{name}: source_trace_path does not exist")
            auxiliary_paths = payload.get("auxiliary_trace_paths", [])
            if auxiliary_paths and not isinstance(auxiliary_paths, list):
                findings.append(f"{name}: auxiliary_trace_paths must be a list")
            elif isinstance(auxiliary_paths, list):
                for reference in auxiliary_paths:
                    if not _path_exists(str(reference), out_dir=out_dir):
                        findings.append(f"{name}: auxiliary_trace_path does not exist: {reference}")
            ready = bool(payload.get("promotion_ready"))
            supported = _promotion_ready_is_supported(payload, out_dir=out_dir)
            if ready and not supported:
                findings.append(
                    f"{name}: promotion_ready=true is not supported by discharged assumptions/constants"
                )
            if ready and payload.get("blocker"):
                findings.append(f"{name}: promotion_ready=true must not keep a blocker")
            if not ready and not payload.get("blocker"):
                findings.append(f"{name}: blocked evidence must explain the blocker")
            promotion_ready = promotion_ready and ready

    return {
        "pass": not findings,
        "promotion_ready": promotion_ready and not findings,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PUBLICATION_DIR)
    args = parser.parse_args()
    result = validate_assumption_discharge(out_dir=args.out_dir)
    print(
        "[validate_t9_t10_assumption_discharge] "
        f"{'PASS' if result['pass'] else 'FAIL'} promotion_ready={result['promotion_ready']}"
    )
    for finding in result["findings"]:
        print(f"- {finding}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
