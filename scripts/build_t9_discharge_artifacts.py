#!/usr/bin/env python3
"""Build empirical T9 assumption-discharge artifacts for three domains."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orius.universal_theory.theorem_discharge import (
    DischargeThresholds,
    compute_t9_discharge_from_csv,
    compute_t9_discharge_from_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
EVIDENCE_DIR_NAME = "theorem_promotion_evidence"

DOMAIN_SOURCES = {
    "battery": "reports/battery_av/battery/runtime_traces.csv",
    "av": "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_traces.csv",
    "healthcare": "data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv",
}


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
        reliability_degradation_threshold=args.reliability_degradation_threshold,
        mixing_autocorrelation_max=args.mixing_autocorrelation_max,
        mixing_max_lag=args.mixing_max_lag,
    )


def _artifact_ref(theorem_id: str, domain: str) -> str:
    return f"reports/publication/{EVIDENCE_DIR_NAME}/{theorem_id}_{domain}.json"


def _missing_payload(
    *, domain: str, artifact_ref: str, source_trace_path: str, thresholds: DischargeThresholds
) -> dict[str, Any]:
    payload = compute_t9_discharge_from_rows(
        [],
        domain=domain,
        artifact_source=artifact_ref,
        thresholds=thresholds,
        artifact_exists=False,
    )
    payload["source_trace_path"] = source_trace_path
    payload["blocker"] = f"source_trace_path missing: {source_trace_path}; {payload['blocker']}"
    return payload


def build_t9_discharge_artifacts(
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
        artifact_ref = _artifact_ref("T9", domain)
        if source_path.exists():
            payload = compute_t9_discharge_from_csv(
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
        (evidence_dir / f"T9_{domain}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        domains[domain] = {
            "promotion_ready": payload["promotion_ready"],
            "artifact_source": artifact_ref,
            "source_trace_path": source_ref,
            "blocker": payload["blocker"],
        }

    return {
        "theorem_id": "T9",
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
    parser.add_argument("--reliability-degradation-threshold", type=float, default=0.95)
    parser.add_argument("--mixing-autocorrelation-max", type=float, default=0.99)
    parser.add_argument("--mixing-max-lag", type=int, default=128)
    args = parser.parse_args()
    result = build_t9_discharge_artifacts(
        repo_root=args.repo_root.resolve(),
        out_dir=args.out_dir.resolve(),
        thresholds=_thresholds(args),
        max_rows=args.max_rows,
    )
    print(
        "[build_t9_discharge_artifacts] "
        f"promotion_ready={result['promotion_ready']} domains={','.join(result['domains'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
