from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from orius.universal_theory.domain_runtime_contracts import (
    AV_BRAKE_HOLD_CONTRACT_ID,
    BATTERY_SAFE_DISPATCH_CONTRACT_ID,
    HEALTHCARE_FAIL_SAFE_CONTRACT_ID,
    universal_contract_manifest_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_universal_contract_manifest.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_universal_contract_manifest", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary() -> dict[str, Any]:
    return {
        "source_theorem": "T11",
        "domains": {
            "battery": {"n_steps": 2, "contract_id": BATTERY_SAFE_DISPATCH_CONTRACT_ID},
            "av": {"n_steps": 2, "contract_id": AV_BRAKE_HOLD_CONTRACT_ID},
            "healthcare": {"n_steps": 2, "contract_id": HEALTHCARE_FAIL_SAFE_CONTRACT_ID},
        },
    }


def _write_manifest_pair(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    summary = _summary()
    manifest = universal_contract_manifest_payload(summary)
    for domain, payload in manifest["domains"].items():
        for slot in (
            "domain_data",
            "forecast_model",
            "uncertainty_estimate",
            "runtime_trace",
            "domain_contract_witness",
        ):
            source = tmp_path / domain / f"{slot}.txt"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"{domain}:{slot}\n", encoding="utf-8")
            payload[slot] = str(source)
    manifest_path = tmp_path / "orius_universal_contract_manifest.json"
    summary_path = tmp_path / "domain_runtime_contract_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path, summary_path, manifest


def test_universal_contract_manifest_validator_accepts_complete_manifest(tmp_path: Path) -> None:
    validator = _load_validator()
    manifest_path, summary_path, _manifest = _write_manifest_pair(tmp_path)

    result = validator.validate_manifest(manifest_path=manifest_path, summary_path=summary_path)

    assert result["pass"] is True
    assert result["failures"] == []


def test_universal_contract_manifest_validator_fails_on_missing_domain_slot(tmp_path: Path) -> None:
    validator = _load_validator()
    manifest_path, summary_path, manifest = _write_manifest_pair(tmp_path)
    del manifest["domains"]["av"]["runtime_trace"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validator.validate_manifest(manifest_path=manifest_path, summary_path=summary_path)

    assert result["pass"] is False
    assert any("av: missing required universal contract slot runtime_trace" in item for item in result["failures"])


def test_universal_contract_manifest_validator_fails_on_av_road_overclaim(tmp_path: Path) -> None:
    validator = _load_validator()
    manifest_path, summary_path, manifest = _write_manifest_pair(tmp_path)
    manifest["domains"]["av"]["claim_boundary"] = "AV road deployment completed."
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validator.validate_manifest(manifest_path=manifest_path, summary_path=summary_path)

    assert result["pass"] is False
    assert any("road deployment" in item for item in result["failures"])


def test_universal_contract_manifest_validator_fails_on_healthcare_clinical_overclaim(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    manifest_path, summary_path, manifest = _write_manifest_pair(tmp_path)
    manifest["domains"]["healthcare"]["claim_boundary"] = "Healthcare clinical deployment approved."
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = validator.validate_manifest(manifest_path=manifest_path, summary_path=summary_path)

    assert result["pass"] is False
    assert any("clinical deployment" in item for item in result["failures"])
