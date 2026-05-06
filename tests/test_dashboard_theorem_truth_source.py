from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = REPO_ROOT / "frontend" / "src"
THEOREM_PAGE_DIR = FRONTEND_ROOT / "app" / "(dashboard)" / "theorems"
DOMAIN_PAGE = FRONTEND_ROOT / "app" / "(dashboard)" / "domains" / "page.tsx"
SERVER_LOADER = FRONTEND_ROOT / "lib" / "server" / "theorem-data.ts"
AUDIT_JSON = REPO_ROOT / "reports" / "publication" / "active_theorem_audit.json"
MULTIPLICATION_SIGN = "\u00d7"


def _read_all_theorem_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in THEOREM_PAGE_DIR.glob("*.tsx")
        if not path.name.startswith("._")
    )


def test_theorem_dashboard_reads_generated_publication_audit_not_static_rows() -> None:
    page_sources = _read_all_theorem_sources()
    loader_source = SERVER_LOADER.read_text(encoding="utf-8")

    assert "const THEOREMS: Theorem[]" not in page_sources
    assert "active_theorem_audit.json" in loader_source
    assert "theorem_surface_register.csv" in loader_source
    assert "theorem_promotion_gates.csv" in loader_source
    assert "theorem_promotion_scorecard.json" in loader_source
    assert "promotion_package_missing" in loader_source
    assert "PROMOTION_PACKAGE_REQUIRED_IDS" in loader_source
    assert "readFile" in loader_source


def test_dashboard_t9_t10_status_comes_from_generated_promotion_package() -> None:
    loader_source = SERVER_LOADER.read_text(encoding="utf-8")

    assert "PromotionScorecardCandidate" in loader_source
    assert "promotionCandidate" in loader_source
    assert "promotion_ready=false" in loader_source
    assert "T9/T10 status source: reports/publication/theorem_promotion_scorecard.json" in loader_source


def test_dashboard_theorem_labels_follow_authoritative_audit_tiers() -> None:
    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    rows = {row["theorem_id"]: row for row in payload["theorems"]}
    page_sources = _read_all_theorem_sources()
    loader_source = SERVER_LOADER.read_text(encoding="utf-8")

    assert rows["T5"]["surface_kind"] == "definition"
    assert rows["T5"]["defense_tier"] == "draft_non_defended"
    assert rows["T9"]["defense_tier"] == "supporting_defended"
    assert rows["T10"]["defense_tier"] == "supporting_defended"
    assert "battery-specific" in rows["T7"]["scope_note"]
    assert rows["T11"]["defense_tier"] == "flagship_defended"
    assert rows["T11"]["typed_obligations"]

    assert "Definition" in page_sources
    assert "Draft / Non-Defended" in page_sources
    assert "T7_Battery" in page_sources
    assert "Promotion Gates" in page_sources
    assert "constants and assumptions" in loader_source


def test_domain_dashboard_uses_universal_contract_language_and_battery_alias() -> None:
    domain_source = DOMAIN_PAGE.read_text(encoding="utf-8")

    assert (
        f"3 promoted domains {MULTIPLICATION_SIGN} universal contract + domain instantiations"
        in domain_source
    )
    assert f"3 promoted domains {MULTIPLICATION_SIGN} 11 theorems" not in domain_source
    assert "T7_Battery" in domain_source
    assert "C7_BatteryFallback" in domain_source
    assert "T9/T10 are supporting rows" in domain_source
    assert "any compliant domain inherits the ORIUS runtime-assurance contract" in domain_source
