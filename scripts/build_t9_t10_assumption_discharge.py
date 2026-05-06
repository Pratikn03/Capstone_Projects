#!/usr/bin/env python3
"""Build executable T9/T10 assumption-discharge artifacts.

This script orchestrates the theorem-specific builders.  It never promotes a
theorem by hand: each per-domain JSON artifact is computed from trace data and
then summarized for the promotion validator.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.build_t9_discharge_artifacts import build_t9_discharge_artifacts
    from scripts.build_t10_discharge_artifacts import build_t10_discharge_artifacts
except ModuleNotFoundError:  # pragma: no cover - direct script execution from scripts/
    from build_t9_discharge_artifacts import build_t9_discharge_artifacts  # type: ignore[no-redef]
    from build_t10_discharge_artifacts import build_t10_discharge_artifacts  # type: ignore[no-redef]

from orius.universal_theory.theorem_discharge import DischargeThresholds

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"


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
        tv_bridge_epsilon=args.tv_bridge_epsilon,
        tv_histogram_bins=args.tv_histogram_bins,
    )


def build_assumption_discharge(
    *,
    repo_root: Path = REPO_ROOT,
    out_dir: Path = PUBLICATION_DIR,
    thresholds: DischargeThresholds | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    cfg = thresholds or DischargeThresholds()
    generated_at = datetime.now(UTC).isoformat()
    t9 = build_t9_discharge_artifacts(
        repo_root=repo_root,
        out_dir=out_dir,
        thresholds=cfg,
        max_rows=max_rows,
    )
    t10 = build_t10_discharge_artifacts(
        repo_root=repo_root,
        out_dir=out_dir,
        thresholds=cfg,
        max_rows=max_rows,
    )
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "promotion_ready": bool(t9["promotion_ready"] and t10["promotion_ready"]),
        "theorems": {
            "T9": t9["domains"],
            "T10": t10["domains"],
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "t9_t10_assumption_discharge_scorecard.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


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
    parser.add_argument("--tv-bridge-epsilon", type=float, default=0.05)
    parser.add_argument("--tv-histogram-bins", type=int, default=10)
    args = parser.parse_args()
    result = build_assumption_discharge(
        repo_root=args.repo_root.resolve(),
        out_dir=args.out_dir.resolve(),
        thresholds=_thresholds(args),
        max_rows=args.max_rows,
    )
    print(
        "[build_t9_t10_assumption_discharge] "
        f"promotion_ready={result['promotion_ready']} theorems={','.join(result['theorems'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
