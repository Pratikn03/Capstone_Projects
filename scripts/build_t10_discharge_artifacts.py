#!/usr/bin/env python3
"""Build empirical T10 assumption-discharge artifacts for three domains."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orius.universal_theory.theorem_discharge import (
    DischargeThresholds,
    compute_t10_discharge_from_csv,
    compute_t10_discharge_from_rows,
    iter_csv_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
EVIDENCE_DIR_NAME = "theorem_promotion_evidence"

DOMAIN_SOURCES = {
    "battery": "reports/battery_av/battery/runtime_traces.csv",
    "av": "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_traces.csv",
    "healthcare": "data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv",
}

BATTERY_AUXILIARY_UNSAFE_SOURCE = "reports/publication/48h_trace.csv"


def _env_max_rows() -> int | None:
    raw = os.environ.get("ORIUS_T9_T10_MAX_ROWS")
    if not raw:
        return None
    parsed = int(raw)
    return parsed if parsed > 0 else None


def _thresholds(args: argparse.Namespace) -> DischargeThresholds:
    return DischargeThresholds(
        min_rows=args.min_rows,
        boundary_margin=args.boundary_margin,
        min_positive_rate=args.min_positive_rate,
        tv_bridge_epsilon=args.tv_bridge_epsilon,
        tv_histogram_bins=args.tv_histogram_bins,
    )


def _artifact_ref(theorem_id: str, domain: str) -> str:
    return f"reports/publication/{EVIDENCE_DIR_NAME}/{theorem_id}_{domain}.json"


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _battery_margin(value: float, *, capacity_mwh: float) -> float:
    return min(value, capacity_mwh - value)


def _battery_auxiliary_unsafe_rows(path: Path) -> list[dict[str, Any]]:
    """Convert locked 48h Battery unsafe steps into T10 boundary-law rows."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    true_values = [
        parsed
        for row in source_rows
        if (parsed := _safe_float(row.get("soc_true_mwh"))) is not None and parsed >= 0.0
    ]
    capacity_mwh = max(true_values) if true_values else 10000.0
    if capacity_mwh <= 0.0:
        capacity_mwh = 10000.0

    auxiliary_rows: list[dict[str, Any]] = []
    for row in source_rows:
        if not _as_bool(row.get("true_soc_violated")) and not _as_bool(row.get("violated")):
            continue
        observed_soc = _safe_float(row.get("soc_observed_mwh"))
        true_soc = _safe_float(row.get("soc_true_mwh"))
        reliability = _safe_float(row.get("reliability_w"))
        if observed_soc is None or true_soc is None or reliability is None:
            continue
        auxiliary_rows.append(
            {
                "reliability_w": reliability,
                "observed_margin": _battery_margin(observed_soc, capacity_mwh=capacity_mwh),
                "true_margin": _battery_margin(true_soc, capacity_mwh=capacity_mwh),
                "true_constraint_violated": True,
                "fault_family": "battery_48h_boundary_violation",
                "auxiliary_source_step": row.get("step", ""),
            }
        )
    return auxiliary_rows


def _missing_payload(
    *, domain: str, artifact_ref: str, source_trace_path: str, thresholds: DischargeThresholds
) -> dict[str, Any]:
    payload = compute_t10_discharge_from_rows(
        [],
        domain=domain,
        artifact_source=artifact_ref,
        thresholds=thresholds,
        artifact_exists=False,
    )
    payload["source_trace_path"] = source_trace_path
    payload["blocker"] = f"source_trace_path missing: {source_trace_path}; {payload['blocker']}"
    return payload


def build_t10_discharge_artifacts(
    *,
    repo_root: Path = REPO_ROOT,
    out_dir: Path = PUBLICATION_DIR,
    thresholds: DischargeThresholds | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    cfg = thresholds or DischargeThresholds()
    evidence_dir = out_dir / EVIDENCE_DIR_NAME
    evidence_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    domains: dict[str, Any] = {}

    for domain, source_ref in DOMAIN_SOURCES.items():
        source_path = repo_root / source_ref
        artifact_ref = _artifact_ref("T10", domain)
        if source_path.exists():
            if domain == "battery":
                auxiliary_source_path = repo_root / BATTERY_AUXILIARY_UNSAFE_SOURCE
                auxiliary_unsafe_rows = _battery_auxiliary_unsafe_rows(auxiliary_source_path)
                payload = compute_t10_discharge_from_rows(
                    iter_csv_rows(source_path, max_rows=max_rows),
                    domain=domain,
                    artifact_source=artifact_ref,
                    thresholds=cfg,
                    artifact_exists=True,
                    auxiliary_unsafe_rows=auxiliary_unsafe_rows,
                )
                payload["auxiliary_trace_paths"] = [BATTERY_AUXILIARY_UNSAFE_SOURCE]
                payload["paired_observation_source"] = (
                    "safe law from Battery ORIUS runtime trace; unsafe law from locked "
                    "48h Battery boundary-violation trace transformed to margin units"
                )
                payload["counterfactual_pairing_method"] = (
                    "same-domain boundary-law bridge using observed/true SoC margins "
                    "m = min(soc, capacity - soc)"
                )
            else:
                payload = compute_t10_discharge_from_csv(
                    source_path,
                    domain=domain,
                    artifact_source=artifact_ref,
                    thresholds=cfg,
                    max_rows=max_rows,
                )
            payload["source_trace_path"] = source_ref
        else:
            payload = _missing_payload(
                domain=domain,
                artifact_ref=artifact_ref,
                source_trace_path=source_ref,
                thresholds=cfg,
            )
        payload["generated_at_utc"] = generated_at
        (evidence_dir / f"T10_{domain}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        domains[domain] = {
            "promotion_ready": payload["promotion_ready"],
            "artifact_source": artifact_ref,
            "source_trace_path": source_ref,
            "auxiliary_trace_paths": payload.get("auxiliary_trace_paths", []),
            "blocker": payload["blocker"],
        }

    return {
        "theorem_id": "T10",
        "generated_at_utc": generated_at,
        "promotion_ready": all(item["promotion_ready"] for item in domains.values()),
        "domains": domains,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PUBLICATION_DIR)
    parser.add_argument("--max-rows", type=int, default=_env_max_rows())
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--boundary-margin", type=float, default=0.5)
    parser.add_argument("--min-positive-rate", type=float, default=1e-6)
    parser.add_argument("--tv-bridge-epsilon", type=float, default=0.05)
    parser.add_argument("--tv-histogram-bins", type=int, default=10)
    args = parser.parse_args()
    result = build_t10_discharge_artifacts(
        repo_root=args.repo_root.resolve(),
        out_dir=args.out_dir.resolve(),
        thresholds=_thresholds(args),
        max_rows=args.max_rows,
    )
    print(
        "[build_t10_discharge_artifacts] "
        f"promotion_ready={result['promotion_ready']} domains={','.join(result['domains'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
