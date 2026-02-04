"""Utilities: artifact registry."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA‑256 hash for a file."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def register_models(
    models_dir: Path,
    registry_path: Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Record model artifact metadata into a local registry file."""
    models_dir = Path(models_dir)
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    models = []
    if models_dir.exists():
        for path in sorted(list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pkl"))):
            models.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
            )

    snapshot = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id,
        "models_dir": str(models_dir),
        "models": models,
    }

    history: list[dict[str, Any]] = []
    if registry_path.exists():
        existing = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and "history" in existing:
            history = list(existing.get("history", []))
        elif isinstance(existing, dict):
            history = [existing]
        elif isinstance(existing, list):
            history = existing

    history.append(snapshot)
    payload = {"latest": snapshot, "history": history}
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
