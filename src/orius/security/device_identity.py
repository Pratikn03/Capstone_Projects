"""Per-device HMAC identity for IoT request paths."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .policy import device_signature_required, get_device_key

DEVICE_SIGNATURE_ALGORITHM = "HMAC-SHA256"
SIGNATURE_FIELD = "device_signature"
_EXCLUDED_FIELDS = {SIGNATURE_FIELD}


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and "T" in value:
        with contextlib.suppress(ValueError):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_safe(v) for v in value]
    return value


def _canonical_device_payload(payload: Mapping[str, Any]) -> bytes:
    safe = {str(k): _to_json_safe(v) for k, v in dict(payload).items() if k not in _EXCLUDED_FIELDS}
    return json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sign_device_payload(payload: Mapping[str, Any], secret: str | bytes) -> str:
    key = secret if isinstance(secret, bytes) else str(secret).encode("utf-8")
    if not key:
        raise RuntimeError("device signing key is required")
    return hmac.new(key, _canonical_device_payload(payload), hashlib.sha256).hexdigest()


def _parse_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if value in (None, ""):
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _max_skew_seconds() -> int:
    with contextlib.suppress(ValueError):
        return int(os.getenv("ORIUS_DEVICE_MAX_CLOCK_SKEW_SECONDS", "300"))
    return 300


def has_device_signature_fields(payload: Mapping[str, Any]) -> bool:
    return any(
        payload.get(field) not in (None, "")
        for field in ("device_key_id", "device_ts_utc", "device_nonce", SIGNATURE_FIELD)
    )


def verify_device_request(payload: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Verify a device HMAC request payload.

    When strict mode is disabled and no HMAC fields are present, verification is
    skipped. If any HMAC fields are present, the request is fully verified.
    """

    strict = device_signature_required()
    present = has_device_signature_fields(payload)
    if not strict and not present:
        return {"valid": True, "required": False, "verified": False, "reason": None}
    required = ("device_id", "device_key_id", "device_ts_utc", "device_nonce", SIGNATURE_FIELD)
    missing = [field for field in required if payload.get(field) in (None, "")]
    if missing:
        return {
            "valid": False,
            "required": strict,
            "verified": False,
            "reason": "device signature required",
            "missing": missing,
        }

    device_id = str(payload["device_id"])
    key_id = str(payload["device_key_id"])
    secret = get_device_key(device_id, key_id)
    if not secret:
        return {"valid": False, "required": strict, "verified": False, "reason": "device key missing"}

    observed_ts = _parse_utc(payload.get("device_ts_utc"))
    if observed_ts is None:
        return {"valid": False, "required": strict, "verified": False, "reason": "device timestamp invalid"}
    reference = (now or datetime.now(UTC)).astimezone(UTC)
    skew = abs((reference - observed_ts).total_seconds())
    if skew > _max_skew_seconds():
        return {"valid": False, "required": strict, "verified": False, "reason": "device timestamp stale"}

    expected = sign_device_payload(payload, secret)
    observed = str(payload.get(SIGNATURE_FIELD))
    if not hmac.compare_digest(observed, expected):
        return {
            "valid": False,
            "required": strict,
            "verified": False,
            "reason": "device signature invalid",
            "expected_signature": expected,
            "observed_signature": observed,
        }
    return {
        "valid": True,
        "required": strict,
        "verified": True,
        "reason": None,
        "device_id": device_id,
        "device_key_id": key_id,
        "device_nonce": str(payload["device_nonce"]),
    }
