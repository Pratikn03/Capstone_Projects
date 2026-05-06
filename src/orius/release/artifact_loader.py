"""Approved artifact loading with fail-closed hash verification.

All code paths that deserialize model artifacts should pass through this
module. In dev/test, missing hashes remain usable for exploratory research.
In staging/production or when artifact manifests are required, missing or
mismatched hashes fail before deserialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

from orius.security.policy import artifact_manifest_required, is_deployment_env

TRUE_VALUES = {"1", "true", "yes", "on", "required", "strict"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUE_VALUES


def model_hash_required() -> bool:
    """Return whether model/artifact hash manifests are mandatory."""

    return is_deployment_env() or artifact_manifest_required() or _truthy_env("ORIUS_REQUIRE_MODEL_HASH")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iter_manifest_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("artifacts", "hashes", "files", "rows"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        rows: list[dict[str, Any]] = []
        for path, digest in payload.items():
            if isinstance(digest, str):
                rows.append({"path": path, "sha256": digest})
            elif isinstance(digest, dict):
                row = dict(digest)
                row.setdefault("path", path)
                rows.append(row)
        return rows
    return []


def _path_matches_manifest_row(artifact_path: Path, row_path: str) -> bool:
    if not row_path:
        return False
    candidate = Path(row_path)
    artifact_abs = str(artifact_path.resolve())
    if candidate.is_absolute():
        try:
            return candidate.resolve() == artifact_path.resolve()
        except FileNotFoundError:
            return str(candidate) == artifact_abs
    return (
        row_path == str(artifact_path)
        or artifact_abs.endswith(f"/{row_path}")
        or row_path == artifact_path.name
    )


def _expected_sha256_from_manifest(artifact_path: Path, manifest_path: Path) -> str | None:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in _iter_manifest_rows(payload):
        row_path = str(row.get("path") or row.get("artifact_path") or row.get("file") or "")
        digest = row.get("sha256") or row.get("hash") or row.get("digest")
        if isinstance(digest, str) and _path_matches_manifest_row(artifact_path, row_path):
            return digest.strip().lower()
    return None


def _manifest_candidates(path: Path) -> list[Path]:
    candidates: list[Path] = []
    env_manifest = os.getenv("ORIUS_MODEL_HASH_MANIFEST") or os.getenv("ORIUS_ARTIFACT_MANIFEST")
    if env_manifest:
        candidates.append(Path(env_manifest))
    candidates.extend(
        [
            path.parent / "model_hashes.json",
            path.parent / "artifact_hashes.json",
            path.parent / "frozen_artifact_hashes.json",
            Path("reports/predeployment_freeze/frozen_artifact_hashes.json"),
        ]
    )
    return candidates


def expected_artifact_sha256(path: str | Path) -> str | None:
    artifact_path = Path(path)
    sidecar = artifact_path.with_name(f"{artifact_path.name}.sha256")
    if sidecar.exists():
        raw = sidecar.read_text(encoding="utf-8").strip()
        if raw:
            return raw.split()[0].strip().lower()
    for manifest_path in _manifest_candidates(artifact_path):
        try:
            expected = _expected_sha256_from_manifest(artifact_path, manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        if expected:
            return expected
    return None


def verify_artifact_hash(path: str | Path, *, required: bool | None = None) -> str | None:
    """Verify SHA256 before deserialization.

    Returns the observed hash when a manifest/sidecar is present, otherwise
    ``None`` in non-strict dev/test mode.
    """

    artifact_path = Path(path)
    expected = expected_artifact_sha256(artifact_path)
    must_verify = model_hash_required() if required is None else bool(required)
    if expected is None:
        if must_verify:
            raise RuntimeError(
                f"Refusing to load unsigned model artifact without sha256 manifest: {artifact_path}"
            )
        return None
    actual = sha256_file(artifact_path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"Model artifact hash mismatch for {artifact_path}: expected {expected}, observed {actual}"
        )
    return actual


def load_pickle_artifact(path: str | Path, *, required: bool | None = None) -> Any:
    artifact_path = Path(path)
    verify_artifact_hash(artifact_path, required=required)
    with artifact_path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - hash-verified approved loader


def load_joblib_artifact(path: str | Path, *, required: bool | None = None) -> Any:
    artifact_path = Path(path)
    verify_artifact_hash(artifact_path, required=required)
    import joblib

    return joblib.load(artifact_path)


def load_torch_artifact(
    path: str | Path,
    *,
    required: bool | None = None,
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Any:
    artifact_path = Path(path)
    verify_artifact_hash(artifact_path, required=required)
    import torch

    try:
        return torch.load(artifact_path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(artifact_path, map_location=map_location)
    except Exception:
        if model_hash_required() if required is None else bool(required):
            raise
        return torch.load(artifact_path, map_location=map_location)
