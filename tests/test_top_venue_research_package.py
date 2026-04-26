from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_source_backed_positioning_has_required_lanes() -> None:
    rows = _read_csv(PUBLICATION_DIR / "source_backed_research_positioning.csv")
    lanes = {row["lane"] for row in rows}

    assert "runtime_assurance" in lanes
    assert "uncertainty_coverage" in lanes
    assert "safety_critical_control" in lanes
    assert "clinical_dataset" in lanes
    assert "clinical_reporting" in lanes
    assert "av_closed_loop" in lanes
    assert "av_stress_simulation" in lanes
    assert all(row["url"].startswith("https://") for row in rows)
    assert all(row["orius_gap"] and row["boundary"] for row in rows)


def test_95plus_scorecard_is_gate_based_not_overclaiming() -> None:
    scorecard = _read_csv(PUBLICATION_DIR / "orius_95plus_uplift_scorecard.csv")
    payload = json.loads((PUBLICATION_DIR / "orius_95plus_uplift_scorecard.json").read_text(encoding="utf-8"))

    dimensions = {row["dimension"] for row in scorecard}
    assert {
        "core_idea_novelty",
        "theory",
        "three_domain_runtime_evidence",
        "external_validation_depth",
        "reproducibility_and_freeze",
        "claim_quality",
    } <= dimensions
    assert all(int(float(row["target_score"])) >= 95 for row in scorecard)
    assert payload["achieved"] is all(row["current_status"] == "pass" for row in scorecard)
    assert any(row["current_status"] != "pass" for row in scorecard)


def test_top_venue_json_contains_source_anchors_and_safe_boundaries() -> None:
    payload = json.loads((PUBLICATION_DIR / "top_venue_research_package.json").read_text(encoding="utf-8"))

    assert payload["status"] == "top_venue_defensible_predeployment_package"
    assert payload["healthcare_boundary"] == "retrospective_source_holdout_time_forward_not_live_clinical"
    assert payload["av_boundary"] == "nuplan_replay_surrogate_not_carla_or_road_deployment"
    assert len(payload["source_anchors"]) >= 12
    assert payload["uplift_95plus_achieved"] is all(
        row["current_status"] == "pass" for row in payload["uplift_scorecard"]
    )
