#!/usr/bin/env python3
"""Validate local/server deployment-security hardening gates."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

APPROVED_LOADER_FILES = {
    "src/orius/release/artifact_loader.py",
    "src/orius/forecasting/predict.py",
    "src/orius/anomaly/detection.py",
    "src/orius/dc3s/deep_oqe.py",
}
RUNTIME_SCAN_ROOTS = [
    REPO_ROOT / "services",
    REPO_ROOT / "scripts",
    REPO_ROOT / "src" / "orius" / "release",
    REPO_ROOT / "src" / "orius" / "forecasting",
    REPO_ROOT / "src" / "orius" / "anomaly",
    REPO_ROOT / "src" / "orius" / "dc3s",
    REPO_ROOT / "src" / "orius" / "iot",
]
UNSAFE_LOAD_RE = re.compile(r"\b(?:pickle\.load|joblib\.load|torch\.load)\s*\(")


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _missing_marker(relative: str, markers: list[str]) -> list[str]:
    text = _read(relative)
    return [f"{relative} missing marker: {marker}" for marker in markers if marker not in text]


def _validate_auth_coverage() -> list[str]:
    from scripts.validate_api_auth_coverage import validate as validate_auth

    return [f"auth coverage: {finding}" for finding in validate_auth()]


def _validate_release_contract() -> list[str]:
    from scripts.validate_runtime_release_contract import validate as validate_release_contract

    return [f"runtime release contract: {finding}" for finding in validate_release_contract()]


def _validate_unsafe_loaders() -> list[str]:
    findings: list[str] = []
    for root in RUNTIME_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name.startswith("._"):
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not UNSAFE_LOAD_RE.search(text):
                continue
            if relative not in APPROVED_LOADER_FILES:
                findings.append(
                    f"{relative} uses direct unsafe model loading outside approved loader modules"
                )
                continue
            if "_verify_model_artifact_hash" not in text and "verify_artifact_hash" not in text:
                findings.append(f"{relative} loads model artifacts without SHA256 verification")
            if "torch.load" in text and "weights_only" not in text:
                findings.append(f"{relative} uses torch.load without weights_only=True guard")
    return findings


def validate() -> list[str]:
    findings: list[str] = []
    findings.extend(_validate_auth_coverage())
    findings.extend(_validate_release_contract())
    findings.extend(
        _missing_marker(
            "services/api/routers/dc3s.py",
            ["certificate_signature_required", "get_certificate_keys", "sign_certificate"],
        )
    )
    findings.extend(
        _missing_marker(
            "src/orius/dc3s/certificate.py",
            ["_ensure_event_store", "conflicting certificate overwrite rejected", "get_certificate_key"],
        )
    )
    findings.extend(
        _missing_marker(
            "services/api/routers/iot.py",
            ["verify_device_request", "_verify_device_identity_or_raise", "device_signature"],
        )
    )
    findings.extend(
        _missing_marker(
            "src/orius/iot/store.py",
            ["iot_device_nonce", "record_device_nonce"],
        )
    )
    findings.extend(
        _missing_marker(
            "src/orius/forecasting/predict.py",
            ["load_pickle_artifact", "weights_only=True", "Refusing to load unsigned model artifact"],
        )
    )
    findings.extend(_validate_unsafe_loaders())
    return findings


def main() -> int:
    findings = validate()
    if findings:
        print("[validate_deployment_security] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_deployment_security] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
