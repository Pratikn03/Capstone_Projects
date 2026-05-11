"""Central deployment-security policy and secret resolution."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

DEPLOYMENT_ENVS = {"staging", "production", "prod", "deploy", "deployment"}
TRUE_VALUES = {"1", "true", "yes", "y", "on", "required", "strict"}
DEFAULT_CERTIFICATE_KEY_ID = "orius.local.hmac"
LOCAL_SECRET_BACKENDS = {"env", "local", "local_file", "file"}
MANAGED_SECRET_BACKENDS = {
    "external_command",
    "aws_kms",
    "gcp_kms",
    "azure_key_vault",
    "hsm",
    "kms",
    "vault",
}


def _env_name() -> str:
    return os.getenv("ORIUS_ENV", "dev").strip().lower()


def is_deployment_env() -> bool:
    return _env_name() in DEPLOYMENT_ENVS


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUE_VALUES


def _parse_secret_payload(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(raw)
    return dict(parsed or {}) if isinstance(parsed, dict) else {}


def _parse_mapping(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    raw = value.strip()
    if not raw:
        return {}
    path = Path(raw)
    if path.exists() and path.is_file():
        raw = path.read_text(encoding="utf-8")
    return _parse_secret_payload(raw)


def get_secret_backend() -> str:
    """Return the configured secret backend class.

    ``env`` and ``local_file`` are acceptable for local/server research runs.
    Field/deployment-grade gates require a managed backend such as an external
    command, KMS, HSM, or vault-backed provider.
    """

    configured = os.getenv("ORIUS_SECRET_BACKEND", "").strip().lower().replace("-", "_")
    if configured:
        return configured
    if os.getenv("ORIUS_SECRETS_COMMAND", "").strip():
        return "external_command"
    if os.getenv("ORIUS_SECRETS_FILE", "").strip():
        return "local_file"
    return "env"


def secret_backend_is_managed() -> bool:
    return get_secret_backend() in MANAGED_SECRET_BACKENDS


def implemented_managed_secret_backend() -> bool:
    """Return true when this process can actually fetch managed secrets.

    Cloud KMS/HSM/Vault labels are accepted as documentation labels, but this
    local runtime currently implements managed retrieval through an explicit
    external command. Deployment-grade validation must not pass on labels alone.
    """

    return get_secret_backend() == "external_command" and bool(
        os.getenv("ORIUS_SECRETS_COMMAND", "").strip()
    )


def load_external_command_secrets() -> dict[str, Any]:
    command = os.getenv("ORIUS_SECRETS_COMMAND", "").strip()
    if not command:
        return {}
    try:
        timeout = float(os.getenv("ORIUS_SECRETS_COMMAND_TIMEOUT_SECONDS", "5"))
    except ValueError:
        timeout = 5.0
    args = shlex.split(command)
    if not args:
        return {}
    completed = subprocess.run(  # noqa: S603 - command is explicit deployment configuration.
        args,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return _parse_secret_payload(completed.stdout)


def load_security_secrets() -> dict[str, Any]:
    """Load optional secrets from a local file and/or managed command.

    The file is intentionally optional so test/dev environments can rely on
    explicit env vars. Deployment-grade gates should use ``external_command``,
    KMS, HSM, or a vault-backed provider rather than repo-local files.
    """

    secrets: dict[str, Any] = {}
    secrets_file = os.getenv("ORIUS_SECRETS_FILE", "").strip()
    if secrets_file:
        path = Path(secrets_file)
        if not path.exists():
            raise RuntimeError(f"ORIUS_SECRETS_FILE does not exist: {path}")
        secrets.update(_parse_secret_payload(path.read_text(encoding="utf-8")))
    if os.getenv("ORIUS_SECRETS_COMMAND", "").strip():
        secrets.update(load_external_command_secrets())
    return secrets


def get_certificate_keys() -> dict[str, str]:
    secrets = load_security_secrets()
    keys: dict[str, str] = {}

    for source in (
        secrets.get("certificate_keys"),
        secrets.get("ORIUS_CERTIFICATE_KEYS"),
        _parse_mapping(os.getenv("ORIUS_CERTIFICATE_KEYS")),
    ):
        if isinstance(source, dict):
            keys.update({str(k): str(v) for k, v in source.items() if v not in (None, "")})

    legacy_key = os.getenv("ORIUS_CERTIFICATE_SIGNING_KEY", "")
    if legacy_key:
        legacy_id = os.getenv("ORIUS_CERTIFICATE_KEY_ID") or DEFAULT_CERTIFICATE_KEY_ID
        keys.setdefault(str(legacy_id), str(legacy_key))
    return keys


def get_active_certificate_key_id() -> str:
    configured = os.getenv("ORIUS_CERTIFICATE_ACTIVE_KEY_ID") or os.getenv("ORIUS_CERTIFICATE_KEY_ID")
    if configured:
        return str(configured)
    keys = get_certificate_keys()
    return next(iter(keys), DEFAULT_CERTIFICATE_KEY_ID)


def get_certificate_key(key_id: str | None = None) -> str | None:
    keys = get_certificate_keys()
    selected = key_id or get_active_certificate_key_id()
    return keys.get(str(selected))


def certificate_signature_required() -> bool:
    return is_deployment_env() or _truthy_env("ORIUS_REQUIRE_CERT_SIGNATURE")


def _normalize_device_keys(raw: Any) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    if not isinstance(raw, dict):
        return normalized
    for device_id, value in raw.items():
        if isinstance(value, dict):
            normalized[str(device_id)] = {
                str(key_id): str(secret) for key_id, secret in value.items() if secret not in (None, "")
            }
        elif ":" in str(device_id):
            dev_id, key_id = str(device_id).split(":", 1)
            normalized.setdefault(dev_id, {})[key_id] = str(value)
    return normalized


def get_device_keys() -> dict[str, dict[str, str]]:
    secrets = load_security_secrets()
    keys: dict[str, dict[str, str]] = {}
    for source in (
        secrets.get("device_keys"),
        secrets.get("ORIUS_DEVICE_KEYS"),
        _parse_mapping(os.getenv("ORIUS_DEVICE_KEYS")),
    ):
        for device_id, device_keys in _normalize_device_keys(source).items():
            keys.setdefault(device_id, {}).update(device_keys)
    return keys


def get_device_key(device_id: str, key_id: str) -> str | None:
    return get_device_keys().get(str(device_id), {}).get(str(key_id))


def device_signature_required() -> bool:
    return is_deployment_env() or _truthy_env("ORIUS_REQUIRE_DEVICE_SIGNATURE")


def artifact_manifest_required() -> bool:
    return _truthy_env("ORIUS_REQUIRE_ARTIFACT_MANIFEST")


def mtls_required() -> bool:
    return _truthy_env("ORIUS_REQUIRE_MTLS")


def get_device_ca_bundle_path() -> Path | None:
    raw = os.getenv("ORIUS_DEVICE_CA_BUNDLE", "").strip()
    return Path(raw) if raw else None


def _normalize_revoked_device_credentials(raw: Any) -> set[tuple[str, str | None]]:
    revoked: set[tuple[str, str | None]] = set()
    if isinstance(raw, list | tuple | set):
        for value in raw:
            text = str(value)
            if ":" in text:
                device_id, key_id = text.split(":", 1)
                revoked.add((device_id, key_id))
            elif text:
                revoked.add((text, None))
    elif isinstance(raw, dict):
        for device_id, value in raw.items():
            if value is True:
                revoked.add((str(device_id), None))
            elif isinstance(value, list | tuple | set):
                revoked.update((str(device_id), str(key_id)) for key_id in value)
            elif isinstance(value, dict):
                revoked.update(
                    (str(device_id), str(key_id))
                    for key_id, is_revoked in value.items()
                    if bool(is_revoked)
                )
            elif value not in (None, "", False):
                revoked.add((str(device_id), str(value)))
    return revoked


def get_revoked_device_credentials() -> set[tuple[str, str | None]]:
    secrets = load_security_secrets()
    revoked: set[tuple[str, str | None]] = set()
    for source in (
        secrets.get("revoked_devices"),
        secrets.get("revoked_device_keys"),
        secrets.get("ORIUS_REVOKED_DEVICE_KEYS"),
        _parse_mapping(os.getenv("ORIUS_REVOKED_DEVICE_KEYS")),
    ):
        revoked.update(_normalize_revoked_device_credentials(source))
    return revoked


def is_device_credential_revoked(device_id: str, key_id: str | None = None) -> bool:
    revoked = get_revoked_device_credentials()
    return (str(device_id), None) in revoked or (
        key_id is not None and (str(device_id), str(key_id)) in revoked
    )
