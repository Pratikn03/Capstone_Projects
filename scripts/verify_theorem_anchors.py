#!/usr/bin/env python3
"""
Verify that all T1-T8 theorem code anchors exist and are importable.
Aligns PDF (paper) and system per orius-plan/theorem_to_evidence_map.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ANCHORS = {
    "T1": [
        ("cpsbench_iot.scenarios", "fault generation, x_obs/x_true"),
        ("cpsbench_iot.plant", "BatteryPlant unclamped truth"),
        ("cpsbench_iot.runner", "true-state violation computation"),
    ],
    "T2": [
        ("dc3s.guarantee_checks", "one-step feasibility"),
        ("dc3s.shield", "projection, repair"),
    ],
    "T3": [
        ("dc3s.coverage_theorem", "E[V] bound, coverage guarantee"),
    ],
    "T4": [
        ("cpsbench_iot.baselines", "quality-ignorant baselines"),
        ("cpsbench_iot.scenarios", "admissible fault sequences"),
        ("cpsbench_iot.runner", "violation vs intervention"),
    ],
    "T5": [
        ("dc3s.temporal_theorems", "forward_tube, certificate_validity_horizon"),
    ],
    "T6": [
        ("dc3s.temporal_theorems", "certificate_expiration_bound"),
    ],
    "T7": [
        ("dc3s.temporal_theorems", "zero_dispatch_fallback, certify_fallback_existence"),
    ],
    "T8": [
        ("dc3s.temporal_theorems", "evaluate_graceful_degradation_dominance"),
        ("dc3s.graceful", "plan_graceful_degradation"),
    ],
}


def main() -> int:
    failed = []
    for tid, modules in ANCHORS.items():
        for mod_path, _desc in modules:
            try:
                __import__(f"orius.{mod_path}")
            except Exception as e:
                failed.append((tid, mod_path, str(e)))

    if failed:
        print("Theorem anchor verification FAILED:")
        for tid, mod, err in failed:
            print(f"  {tid} {mod}: {err}")
        return 1

    # Explicitly check T5-T8 functions exist
    from orius.dc3s import temporal_theorems

    required = [
        "forward_tube",
        "certificate_validity_horizon",
        "certificate_half_life",
        "certificate_expiration_bound",
        "zero_dispatch_fallback",
        "certify_fallback_existence",
        "evaluate_graceful_degradation_dominance",
        "should_renew_certificate",
        "should_expire_certificate",
    ]
    for name in required:
        if not hasattr(temporal_theorems, name):
            print(f"  T5-T8: missing {name} in temporal_theorems")
            failed.append(("T5-T8", name, "missing"))

    if failed:
        return 1

    print("All T1-T8 theorem anchors verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
