"""Local append-only byte records, not scientific approval or durable custody.

Exclusive creation prevents this API from overwriting an existing path. The
owner can still delete or replace files: this is not WORM storage, a signature,
an external timestamp, or outcome custody. A PROTOCOL type does not establish a
scientific lock. Callers must retain the returned whole-envelope SHA256 outside
the record and provide it when reading.

Readers require the same canonical UTF-8 encoding as writers. Only native JSON
types are accepted; nesting beyond 128 container levels is rejected. Symlinks
in any supplied path component are rejected (including aliases such as /tmp on
macOS). Interrupted writes leave their partial file in place and return no
identity; there is no cleanup or recovery claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_RECORD_TYPES = frozenset({"PROTOCOL", "ACTION", "OUTCOME", "REVIEW"})
_ENVELOPE_FIELDS = frozenset({"schema_version", "record_type", "payload", "payload_sha256"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_DEPTH = 128


def _validate_json(value: object, depth: int = 0) -> None:
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return
    if kind is float:
        if not math.isfinite(value):  # type: ignore[arg-type]
            raise ValueError("JSON numbers must be finite")
        return
    if kind not in (list, dict):
        raise ValueError("Only native JSON types are supported")
    if depth >= _MAX_DEPTH:
        raise ValueError("JSON nesting exceeds 128 container levels")
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be native strings")
            _validate_json(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json(item, depth + 1)


def canonical_json_bytes(value: object) -> bytes:
    """Encode native JSON values; normalize invalid encoding/value errors."""
    try:
        _validate_json(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError("Value cannot be encoded as canonical JSON") from exc


def _validate_record_type(record_type: object) -> None:
    if type(record_type) is not str or record_type not in _RECORD_TYPES:
        raise ValueError("Unrecognized record type")


def _validate_sha256(value: object) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:  # type: ignore[arg-type]
        raise ValueError("Expected a lowercase SHA256 hex digest")


@contextmanager
def _parent_descriptor(path: Path) -> Iterator[tuple[int, str]]:
    """Pin each opened directory; never traverse a symlink with a later open."""
    parts = path.parts
    if not parts or not path.name:
        raise ValueError("A record filename is required")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor or ".", directory_flags)
    try:
        parent_parts = parts[1:-1] if path.is_absolute() else parts[:-1]
        for component in parent_parts:
            next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor, path.name
    finally:
        os.close(descriptor)


def write_record(path: Path, payload: dict[str, Any], *, record_type: str) -> str:
    """Exclusively create and fsync a record; never remove a partial failure.

    Return the SHA256 of the entire canonical envelope. Payload validation here
    covers JSON representation only, not scientific content or protocol status.
    Filesystem errors propagate; invalid JSON or record types raise ValueError.
    """
    _validate_record_type(record_type)
    if type(payload) is not dict:
        raise ValueError("Record payload must be a native JSON object")
    payload_bytes = canonical_json_bytes(payload)
    envelope = {
        "schema_version": 2,
        "record_type": record_type,
        "payload": payload,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }
    encoded = canonical_json_bytes(envelope)
    with _parent_descriptor(path) as (parent, filename):
        descriptor = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("Record write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Nonfinite JSON constant")


def read_record(path: Path, *, expected_sha256: str, record_type: str) -> dict[str, Any]:
    """Verify bytes against a mandatory external identity and return payload.

    Verification covers byte identity and envelope structure only. It does not
    establish when bytes were created or whether their scientific claims hold.
    Malformed contents and identity errors raise ValueError; OS errors propagate.
    """
    _validate_sha256(expected_sha256)
    _validate_record_type(record_type)
    with _parent_descriptor(path) as (parent, filename):
        # NONBLOCK lets us reject a FIFO without waiting for a writer.
        descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Record must be a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                encoded = stream.read()
        finally:
            os.close(descriptor)
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("Record does not match its external SHA256 identity")
    try:
        envelope = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        if type(envelope) is not dict or envelope.keys() != _ENVELOPE_FIELDS:
            raise ValueError("Invalid envelope fields")
        if type(envelope["schema_version"]) is not int or envelope["schema_version"] != 2:
            raise ValueError("Unsupported schema version")
        _validate_record_type(envelope["record_type"])
        if envelope["record_type"] != record_type:
            raise ValueError("Unexpected record type")
        payload = envelope["payload"]
        if type(payload) is not dict:
            raise ValueError("Record payload must be an object")
        _validate_sha256(envelope["payload_sha256"])
        if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != envelope["payload_sha256"]:
            raise ValueError("Payload SHA256 mismatch")
        if canonical_json_bytes(envelope) != encoded:
            raise ValueError("Envelope bytes are not canonical JSON")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError("Invalid audit record") from exc
    return dict(payload)
