"""Central deployment-security policy and secret resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

DEPLOYMENT_ENVS = {"staging", "production", "prod", "deploy", "deployment"}
TRUE_VALUES = {"1", "true", "yes", "y", "on", "required", "strict"}
DEFAULT_CERTIFICATE_KEY_ID = "orius.local.hmac"


def _env_name() -> str:
    return os.getenv("ORIUS_ENV", "dev").strip().lower()


def is_deployment_env() -> bool:
    return _env_name() in DEPLOYMENT_ENVS


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUE_VALUES


def _parse_mapping(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    raw = value.strip()
    if not raw:
        return {}
    path = Path(raw)
    if path.exists() and path.is_file():
        raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(raw)
    return dict(parsed or {}) if isinstance(parsed, dict) else {}


def load_security_secrets() -> dict[str, Any]:
    """Load optional local secrets from ORIUS_SECRETS_FILE.

    The file is intentionally optional so test/dev environments can rely on
    explicit env vars. Real secrets must stay outside Git.
    """

    secrets_file = os.getenv("ORIUS_SECRETS_FILE", "").strip()
    if not secrets_file:
        return {}
    path = Path(secrets_file)
    if not path.exists():
        raise RuntimeError(f"ORIUS_SECRETS_FILE does not exist: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(parsed or {}) if isinstance(parsed, dict) else {}


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
