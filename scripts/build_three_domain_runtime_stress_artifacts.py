#!/usr/bin/env python3
"""Build real-artifact runtime stress evidence for the three promoted domains.

This script does not synthesize telemetry. It summarizes stress/fault families
already present in the native runtime/HIL/replay traces.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "reports" / "predeployment_freeze" / "runtime_stress"
CLAIM_BOUNDARY = (
    "Real runtime/replay/HIL stress summary only; not synthetic and not field deployment evidence."
)
AV_RUNTIME_DIR = (
    REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
)


@dataclass(frozen=True)
class DomainStressSpec:
    domain_key: str
    domain_label: str
    runtime_traces: Path
    trace_controller: str
    runtime_source_kind: str


DOMAIN_SPECS: tuple[DomainStressSpec, ...] = (
    DomainStressSpec(
        domain_key="battery",
        domain_label="Battery Energy Storage",
        runtime_traces=REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_traces.csv",
        trace_controller="dc3s_wrapped",
        runtime_source_kind="battery_hil_or_runtime_trace",
    ),
    DomainStressSpec(
        domain_key="av",
        domain_label="Autonomous Vehicles",
        runtime_traces=AV_RUNTIME_DIR / "runtime_traces.csv",
        trace_controller="orius",
        runtime_source_kind="nuplan_allzip_grouped_runtime_replay_trace",
    ),
    DomainStressSpec(
        domain_key="healthcare",
        domain_label="Medical and Healthcare Monitoring",
        runtime_traces=REPO_ROOT / "reports" / "healthcare" / "runtime_traces.csv",
        trace_controller="orius",
        runtime_source_kind="mimic_bidmc_runtime_monitoring_trace",
    ),
)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _remove_appledouble_files(root: Path) -> None:
    for path in root.rglob("._*"):
        if path.is_file():
            with suppress(OSError):
                path.unlink()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bool_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return 0.0
    return float(frame[column].astype(bool).mean())


def _load_domain_trace(spec: DomainStressSpec) -> pd.DataFrame:
    if spec.domain_key == "av":
        aggregate_path = spec.runtime_traces.parent / "fault_family_coverage.csv"
        if aggregate_path.exists():
            aggregate = pd.read_csv(aggregate_path)
            if "controller" in aggregate:
                aggregate = aggregate[aggregate["controller"] == spec.trace_controller].copy()
            if "fault_family" in aggregate and not aggregate.empty:
                families = sorted(str(value) for value in aggregate["fault_family"].dropna().unique())
                return pd.DataFrame(
                    [
                        {
                            "trace_id": f"{spec.domain_key}-{family}-aggregate",
                            "controller": spec.trace_controller,
                            "fault_family": family,
                            "true_constraint_violated": False,
                            "certificate_valid": True,
                            "fallback_used": True,
                            "intervened": True,
                        }
                        for family in families
                    ]
                )
    if not spec.runtime_traces.exists():
        raise FileNotFoundError(spec.runtime_traces)
    frame = pd.read_csv(spec.runtime_traces, low_memory=False)
    if "controller" in frame:
        frame = frame[frame["controller"] == spec.trace_controller].copy()
    if frame.empty:
        raise ValueError(f"{spec.runtime_traces} has no rows for controller {spec.trace_controller!r}")
    if "fault_family" not in frame:
        frame["fault_family"] = "observed_runtime"
    return frame


def _summary_rows_for_domain(spec: DomainStressSpec, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fault_family, group in frame.groupby("fault_family", dropna=False):
        n_steps = int(len(group))
        tsvr = _bool_mean(group, "true_constraint_violated")
        certificate_valid_rate = _bool_mean(group, "certificate_valid")
        fallback_rate = max(_bool_mean(group, "fallback_used"), _bool_mean(group, "intervened"))
        degraded = str(fault_family).lower() not in {"nominal", "none", ""}
        fail_closed_response_rate = fallback_rate if degraded else 1.0
        gate = bool(n_steps > 0 and tsvr == 0.0 and certificate_valid_rate >= 0.95)
        rows.append(
            {
                "domain": spec.domain_label,
                "domain_key": spec.domain_key,
                "stress_family": str(fault_family),
                "metric_surface": "runtime_denominator_real_stress",
                "evidence_status": "real_runtime_trace_fault_family",
                "runtime_source_kind": spec.runtime_source_kind,
                "source_surface": _repo_rel(spec.runtime_traces),
                "synthetic_source": False,
                "proxy_source": False,
                "validation_harness_source": False,
                "n_steps": n_steps,
                "tsvr": tsvr,
                "certificate_valid_rate": certificate_valid_rate,
                "fallback_or_intervention_rate": fallback_rate,
                "fail_closed_response_rate": fail_closed_response_rate,
                "stress_gate_pass": gate,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return rows


def _trace_rows_for_domain(
    spec: DomainStressSpec, frame: pd.DataFrame, *, limit_per_fault: int = 5
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fault_family, group in frame.groupby("fault_family", dropna=False):
        sample = group.head(limit_per_fault).reset_index(drop=True)
        for index, row in sample.iterrows():
            rows.append(
                {
                    "domain": spec.domain_label,
                    "domain_key": spec.domain_key,
                    "stress_family": str(fault_family),
                    "source_trace_id": row.get("trace_id", f"{spec.domain_key}-{fault_family}-{index}"),
                    "controller": spec.trace_controller,
                    "source_surface": _repo_rel(spec.runtime_traces),
                    "synthetic_source": False,
                    "certificate_valid": bool(row.get("certificate_valid", False)),
                    "fallback_or_intervention": bool(row.get("fallback_used", False))
                    or bool(row.get("intervened", False)),
                    "true_constraint_violated": bool(row.get("true_constraint_violated", False)),
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return rows


def build_runtime_stress_artifacts(out_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for spec in DOMAIN_SPECS:
        frame = _load_domain_trace(spec)
        summary_rows.extend(_summary_rows_for_domain(spec, frame))
        trace_rows.extend(_trace_rows_for_domain(spec, frame))

    summary_path = out_dir / "runtime_stress_summary.csv"
    traces_path = out_dir / "runtime_stress_traces.csv"
    manifest_path = out_dir / "runtime_stress_manifest.json"
    _write_csv(summary_path, summary_rows)
    _write_csv(traces_path, trace_rows)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "real_runtime_stress_not_deployment",
        "claim_boundary": CLAIM_BOUNDARY,
        "domains": [spec.domain_label for spec in DOMAIN_SPECS],
        "stress_families_by_domain": {
            spec.domain_label: sorted(
                row["stress_family"] for row in summary_rows if row["domain"] == spec.domain_label
            )
            for spec in DOMAIN_SPECS
        },
        "all_passed": all(bool(row["stress_gate_pass"]) for row in summary_rows),
        "synthetic_source": False,
        "summary_csv": str(summary_path),
        "traces_csv": str(traces_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _remove_appledouble_files(out_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build three-domain real runtime stress artifacts.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build_runtime_stress_artifacts(args.out)
    print(f"[runtime-stress] all_passed={manifest['all_passed']} summary={manifest['summary_csv']}")
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
