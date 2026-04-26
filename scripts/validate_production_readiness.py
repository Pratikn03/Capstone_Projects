#!/usr/bin/env python3
"""Validate deployment-grade ORIUS runtime hardening gates.

This is stricter than publication validation. It checks whether the runtime can
fail closed for auth, model provenance, and certificate provenance. In default
mode it validates code-level fail-closed behavior; in ``--strict`` mode it also
requires production secrets/config to be present.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.dc3s.certificate import (  # noqa: E402
    CERTIFICATE_SIGNATURE_ALGORITHM,
    make_certificate,
    sign_certificate,
    verify_certificate,
)
from orius.forecasting.predict import load_model_bundle  # noqa: E402
from services.api.config import get_api_keys, is_auth_disabled_for_tests  # noqa: E402


REQUIRED_RELEASE_SURFACES = [
    "reports/publication/three_domain_ml_benchmark.csv",
    "reports/publication/certificate_schema_witnesses.csv",
    "reports/publication/domain_runtime_contract_summary.json",
    "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_summary.csv",
    "reports/healthcare/runtime_summary.csv",
]


def _sample_certificate() -> dict[str, object]:
    return make_certificate(
        command_id="production-readiness-smoke",
        device_id="dev-1",
        zone_id="battery",
        controller="production-readiness",
        proposed_action={"charge_mw": 0.0},
        safe_action={"charge_mw": 0.0},
        uncertainty={"lower": [0.0], "upper": [1.0]},
        reliability={"w_t": 1.0},
        drift={"drift": False},
        model_hash="model",
        config_hash="config",
        validity_horizon_H_t=1,
        expires_at_step=1,
        theorem_contracts={"T11": "production_readiness_smoke"},
    )


def _check_auth(strict: bool, findings: list[str], warnings: list[str]) -> None:
    get_api_keys.cache_clear()
    if is_auth_disabled_for_tests():
        env = os.getenv("ORIUS_ENV", "")
        if env.lower() not in {"test", "testing", "pytest"}:
            findings.append("auth bypass is active outside a test environment")
    api_keys = get_api_keys()
    if strict and not api_keys:
        findings.append("strict mode requires ORIUS_API_KEYS or configured security.api_keys")
    elif not api_keys:
        warnings.append("no API keys configured; API will fail closed until ORIUS_API_KEYS is set")


def _check_certificate_signing(strict: bool, findings: list[str], warnings: list[str]) -> None:
    secret = os.getenv("ORIUS_CERTIFICATE_SIGNING_KEY")
    if strict and (secret is None or len(secret) < 32):
        findings.append("strict mode requires ORIUS_CERTIFICATE_SIGNING_KEY with at least 32 characters")
        return
    if not secret:
        warnings.append("certificate signing key is not configured; signed release certificates cannot be emitted")
        secret = "local-readiness-smoke-secret-with-32-plus-chars"

    cert = _sample_certificate()
    signed = sign_certificate(cert, secret=secret, key_id="production-readiness-smoke")
    if signed.get("signature_algorithm") != CERTIFICATE_SIGNATURE_ALGORITHM:
        findings.append("signed certificate did not record the canonical signature algorithm")
        return
    verification = verify_certificate(signed, require_signature=True, signature_secret=secret)
    if not verification["valid"]:
        findings.append(f"signed certificate failed verification: {verification}")

    tampered = dict(signed)
    tampered["safe_action"] = {"charge_mw": 99.0}
    tampered_verification = verify_certificate(tampered, require_signature=True, signature_secret=secret)
    if tampered_verification["valid"]:
        findings.append("tampered signed certificate verified as valid")


def _check_model_provenance(findings: list[str]) -> None:
    old_env = os.environ.get("ORIUS_ENV")
    old_require = os.environ.get("ORIUS_REQUIRE_MODEL_HASH")
    try:
        os.environ["ORIUS_ENV"] = "production"
        os.environ.pop("ORIUS_REQUIRE_MODEL_HASH", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            path.write_bytes(pickle.dumps({"model": "stub", "feature_cols": [], "target": "load_mw"}))
            try:
                load_model_bundle(path)
            except RuntimeError as exc:
                if "without sha256 manifest" not in str(exc):
                    findings.append(f"production model provenance failed with unexpected error: {exc}")
            else:
                findings.append("production model loading accepted an unsigned/unhashed pickle artifact")
    finally:
        if old_env is None:
            os.environ.pop("ORIUS_ENV", None)
        else:
            os.environ["ORIUS_ENV"] = old_env
        if old_require is None:
            os.environ.pop("ORIUS_REQUIRE_MODEL_HASH", None)
        else:
            os.environ["ORIUS_REQUIRE_MODEL_HASH"] = old_require


def _check_release_surfaces(findings: list[str]) -> None:
    for rel_path in REQUIRED_RELEASE_SURFACES:
        path = REPO_ROOT / rel_path
        if not path.exists():
            findings.append(f"required production-readiness surface missing: {rel_path}")


def validate(strict: bool = False) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    warnings: list[str] = []
    _check_auth(strict, findings, warnings)
    _check_certificate_signing(strict, findings, warnings)
    _check_model_provenance(findings)
    _check_release_surfaces(findings)
    return findings, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Require production secrets/config to be present")
    args = parser.parse_args()

    findings, warnings = validate(strict=args.strict)
    if warnings:
        print("[validate_production_readiness] WARN")
        for warning in warnings:
            print(f"- {warning}")
    if findings:
        print("[validate_production_readiness] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_production_readiness] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
