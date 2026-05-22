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
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.dc3s.certificate import (
    CERTIFICATE_SIGNATURE_ALGORITHM,
    make_certificate,
    sign_certificate,
    verify_certificate,
)
from orius.forecasting.predict import load_model_bundle
from orius.release.artifact_loader import model_hash_required
from orius.security.policy import (
    artifact_manifest_required,
    certificate_signature_required,
    device_signature_required,
    get_active_certificate_key_id,
    get_certificate_keys,
    get_device_ca_bundle_path,
    get_device_keys,
    get_secret_backend,
    implemented_managed_secret_backend,
    load_external_command_secrets,
    mtls_required,
    secret_backend_is_managed,
)
from scripts.validate_95_validation_manifest import DEFAULT_MANIFEST, validate_manifest
from scripts.validate_deployment_security import validate as validate_deployment_security
from scripts.validate_runtime_release_contract import validate as validate_release_contract
from services.api.config import get_api_keys, is_auth_disabled_for_tests

REQUIRED_RELEASE_SURFACES = [
    "reports/publication/three_domain_ml_benchmark.csv",
    "reports/publication/certificate_schema_witnesses.csv",
    "reports/publication/domain_runtime_contract_summary.json",
    "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_summary.csv",
    "reports/healthcare/runtime_summary.csv",
]
DEPLOYMENT_ENVS = {"staging", "production", "prod", "deploy", "deployment"}
TRUE_VALUES = {"1", "true", "yes", "y", "on", "required", "strict"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUE_VALUES


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


def _check_secret_backend(deployment_grade: bool, findings: list[str], warnings: list[str]) -> None:
    backend = get_secret_backend()
    if deployment_grade:
        if not secret_backend_is_managed() or not implemented_managed_secret_backend():
            findings.append(
                "deployment-grade mode requires an implemented managed secret backend "
                "(ORIUS_SECRET_BACKEND=external_command with ORIUS_SECRETS_COMMAND); "
                "KMS/HSM/vault labels alone are not accepted"
            )
        env_secret_sources = [
            name
            for name in (
                "ORIUS_CERTIFICATE_KEYS",
                "ORIUS_CERTIFICATE_SIGNING_KEY",
                "ORIUS_DEVICE_KEYS",
                "ORIUS_SECRETS_FILE",
            )
            if os.getenv(name, "").strip()
        ]
        if env_secret_sources:
            findings.append(
                "deployment-grade mode must not source certificate/device secrets "
                f"from env/local files: {', '.join(env_secret_sources)}"
            )
        try:
            managed_payload = load_external_command_secrets()
        except Exception as exc:
            findings.append(f"deployment-grade managed secret command failed: {exc}")
            managed_payload = {}
        if not isinstance(managed_payload.get("certificate_keys"), dict):
            findings.append("deployment-grade managed secret payload missing certificate_keys")
        if not isinstance(managed_payload.get("device_keys"), dict):
            findings.append("deployment-grade managed secret payload missing device_keys")
    elif backend in {"env", "local", "local_file", "file"}:
        warnings.append(
            f"secret backend is {backend}; acceptable for local/server research but not KMS/HSM-grade"
        )


def _check_strict_profile_expectations(strict: bool, findings: list[str]) -> None:
    if not strict:
        return
    env = os.getenv("ORIUS_ENV", "").strip().lower()
    if env not in DEPLOYMENT_ENVS:
        findings.append("strict mode requires ORIUS_ENV=production or ORIUS_ENV=staging")
    if not _truthy_env("ORIUS_REQUIRE_CERT_SIGNATURE"):
        findings.append("strict mode requires ORIUS_REQUIRE_CERT_SIGNATURE=1")
    if not _truthy_env("ORIUS_REQUIRE_DEVICE_SIGNATURE"):
        findings.append("strict mode requires ORIUS_REQUIRE_DEVICE_SIGNATURE=1")
    if not get_device_keys():
        findings.append(
            "strict mode requires configured ORIUS_DEVICE_KEYS or device_keys from "
            "ORIUS_SECRETS_FILE/ORIUS_SECRETS_COMMAND"
        )
    if not _truthy_env("ORIUS_REQUIRE_ARTIFACT_MANIFEST"):
        findings.append("strict mode requires ORIUS_REQUIRE_ARTIFACT_MANIFEST=1")
    if not _truthy_env("ORIUS_REQUIRE_MODEL_HASH"):
        findings.append("strict mode requires ORIUS_REQUIRE_MODEL_HASH=1")
    active_key_id = os.getenv("ORIUS_CERTIFICATE_ACTIVE_KEY_ID") or os.getenv("ORIUS_CERTIFICATE_KEY_ID")
    if not active_key_id:
        findings.append(
            "strict mode requires ORIUS_CERTIFICATE_ACTIVE_KEY_ID for auditable certificate key rotation"
        )
    elif active_key_id not in get_certificate_keys():
        findings.append("strict mode active certificate key ID is not present in configured keys")


def _check_certificate_signing(strict: bool, findings: list[str], warnings: list[str]) -> None:
    certificate_keys = get_certificate_keys()
    active_key_id = get_active_certificate_key_id()
    secret = os.getenv("ORIUS_CERTIFICATE_SIGNING_KEY") or certificate_keys.get(active_key_id)
    if strict and (secret is None or len(secret) < 32):
        findings.append(
            "strict mode requires ORIUS_CERTIFICATE_SIGNING_KEY or ORIUS_CERTIFICATE_KEYS "
            "with an active key of at least 32 characters"
        )
        return
    if not secret:
        warnings.append(
            "certificate signing key is not configured; signed release certificates cannot be emitted"
        )
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


def _check_deployment_identity_and_signing(deployment_grade: bool, findings: list[str]) -> None:
    if not deployment_grade:
        return
    if not certificate_signature_required():
        findings.append("deployment-grade mode requires ORIUS_REQUIRE_CERT_SIGNATURE=1")
    if not get_certificate_keys():
        findings.append("deployment-grade mode requires configured ORIUS_CERTIFICATE_KEYS")
    if not device_signature_required():
        findings.append("deployment-grade mode requires ORIUS_REQUIRE_DEVICE_SIGNATURE=1")
    if not get_device_keys():
        findings.append("deployment-grade mode requires configured ORIUS_DEVICE_KEYS")
    if not mtls_required():
        findings.append("deployment-grade mode requires ORIUS_REQUIRE_MTLS=1 at the device ingress layer")
    ca_bundle = get_device_ca_bundle_path()
    if ca_bundle is None:
        findings.append("deployment-grade mode requires ORIUS_DEVICE_CA_BUNDLE")
    elif not ca_bundle.exists():
        findings.append(f"deployment-grade device CA bundle does not exist: {ca_bundle}")


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


def _check_model_policy(deployment_grade: bool, findings: list[str]) -> None:
    if not deployment_grade:
        return
    if not artifact_manifest_required():
        findings.append("deployment-grade mode requires ORIUS_REQUIRE_ARTIFACT_MANIFEST=1")
    if not model_hash_required():
        findings.append("deployment-grade mode requires strict model hash verification")


def _check_release_surfaces(findings: list[str]) -> None:
    for rel_path in REQUIRED_RELEASE_SURFACES:
        path = REPO_ROOT / rel_path
        if not path.exists():
            findings.append(f"required production-readiness surface missing: {rel_path}")


def _check_operations_runbook(deployment_grade: bool, findings: list[str]) -> None:
    if not deployment_grade:
        return
    runbook = REPO_ROOT / "docs" / "incident_response.md"
    if not runbook.exists():
        findings.append("deployment-grade mode requires docs/incident_response.md")
        return
    text = runbook.read_text(encoding="utf-8").lower()
    for marker in (
        "slo",
        "rollback",
        "failure budget",
        "certificate key compromise",
        "device revocation",
        "model artifact rollback",
        "physical actuation stop",
    ):
        if marker not in text:
            findings.append(f"incident response runbook missing marker: {marker}")


def _check_final_release_and_validation(deployment_grade: bool, findings: list[str]) -> None:
    if not deployment_grade:
        return
    manifests = sorted((REPO_ROOT / "reports" / "predeployment_freeze").glob("*/predeployment_release_manifest.json"))
    if not manifests:
        findings.append("deployment-grade mode requires a frozen predeployment_release_manifest.json")
    if not DEFAULT_MANIFEST.exists():
        findings.append(f"deployment-grade mode requires 95 validation manifest: {DEFAULT_MANIFEST}")
    else:
        validation_findings = validate_manifest(DEFAULT_MANIFEST)
        if validation_findings:
            findings.extend(f"95 validation manifest: {finding}" for finding in validation_findings)


def _check_strict_release_contract(deployment_grade: bool, findings: list[str]) -> None:
    if not deployment_grade:
        return
    contract_findings = validate_release_contract(strict=True)
    findings.extend(f"strict runtime release contract: {finding}" for finding in contract_findings)


def _check_deployment_security_umbrella(deployment_grade: bool, findings: list[str]) -> None:
    if not deployment_grade:
        return
    security_findings = validate_deployment_security()
    findings.extend(f"deployment security: {finding}" for finding in security_findings)


def _check_git_clean(deployment_grade: bool, findings: list[str]) -> None:
    if not deployment_grade:
        return
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        findings.append("deployment-grade mode could not inspect git status")
    elif completed.stdout.strip():
        findings.append("deployment-grade mode requires a clean git tree before field/deployment claims")


def _current_git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _check_selected_release_manifest(
    deployment_grade: bool,
    findings: list[str],
    release_manifest: Path | None,
) -> None:
    if not deployment_grade:
        return
    selected = release_manifest
    env_manifest = os.getenv("ORIUS_RELEASE_MANIFEST", "").strip()
    if selected is None and env_manifest:
        selected = Path(env_manifest)
    if selected is None:
        findings.append(
            "deployment-grade mode requires an explicit --release-manifest or ORIUS_RELEASE_MANIFEST; "
            "historical freeze manifests are not accepted by glob"
        )
        return
    path = selected if selected.is_absolute() else REPO_ROOT / selected
    if not path.exists():
        findings.append(f"deployment-grade selected release manifest missing: {path}")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        findings.append(f"deployment-grade selected release manifest is invalid JSON: {exc}")
        return
    if payload.get("dirty_worktree") is not False:
        findings.append("deployment-grade selected release manifest must have dirty_worktree=false")
    manifest_commit = str(payload.get("git_commit") or "").strip()
    current_head = _current_git_head()
    if current_head and manifest_commit and manifest_commit != current_head:
        findings.append(
            "deployment-grade selected release manifest git_commit does not match current HEAD"
        )
    if not manifest_commit:
        findings.append("deployment-grade selected release manifest missing git_commit")
    if not payload.get("release_id"):
        findings.append("deployment-grade selected release manifest missing release_id")


def validate(
    strict: bool = False,
    deployment_grade: bool = False,
    release_manifest: Path | None = None,
) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    warnings: list[str] = []
    strict = strict or deployment_grade
    _check_auth(strict, findings, warnings)
    _check_secret_backend(deployment_grade, findings, warnings)
    _check_strict_profile_expectations(strict, findings)
    _check_certificate_signing(strict, findings, warnings)
    _check_deployment_identity_and_signing(deployment_grade, findings)
    _check_model_provenance(findings)
    _check_model_policy(deployment_grade, findings)
    _check_release_surfaces(findings)
    _check_deployment_security_umbrella(deployment_grade, findings)
    _check_strict_release_contract(deployment_grade, findings)
    _check_operations_runbook(deployment_grade, findings)
    _check_final_release_and_validation(deployment_grade, findings)
    _check_selected_release_manifest(deployment_grade, findings, release_manifest)
    _check_git_clean(deployment_grade, findings)
    return findings, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="Require production secrets/config to be present"
    )
    parser.add_argument(
        "--deployment-grade",
        action="store_true",
        help=(
            "Require field/deployment gates: managed secrets, mTLS/device identity, "
            "strict artifact manifests, validation manifest, and clean git tree"
        ),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=None,
        help="Explicit release manifest required for deployment-grade validation.",
    )
    args = parser.parse_args()

    findings, warnings = validate(
        strict=args.strict,
        deployment_grade=args.deployment_grade,
        release_manifest=args.release_manifest,
    )
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
