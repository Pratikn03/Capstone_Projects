"""Release manifest: single source of truth for run integrity.

One file at the end of every release that lets a reviewer answer:
"did all models see the same data?" by reading one field
(``inputs.splits_sha256``) and "is this run reproducible?" by comparing
``environment.git_commit`` and ``environment.dep_versions`` to their own
checkout.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orius.utils.manifest import get_git_status

CRITICAL_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "lightgbm",
    "xgboost",
    "torch",
    "ngboost",
    "prophet",
    "darts",
    "flaml",
    "statsmodels",
    "pyarrow",
)


@dataclass
class StepStatus:
    name: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ReleaseManifest:
    region: str
    release_id: str
    started_at: str
    finished_at: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    steps: list[StepStatus] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _resolve_package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name.replace("-", "_"))
    except Exception:
        return None
    for attr in ("__version__", "VERSION", "version"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
        if isinstance(value, tuple):
            return ".".join(str(p) for p in value)
    return None


def collect_environment() -> dict[str, Any]:
    git = get_git_status()
    deps = {pkg: _resolve_package_version(pkg) for pkg in CRITICAL_PACKAGES}
    return {
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_uncommitted_changes": bool(git.get("uncommitted_changes")),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dep_versions": deps,
    }


def write_release_manifest(manifest: ReleaseManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "region": manifest.region,
        "release_id": manifest.release_id,
        "started_at": manifest.started_at,
        "finished_at": manifest.finished_at,
        "inputs": manifest.inputs,
        "environment": manifest.environment,
        "artifacts": manifest.artifacts,
        "steps": [asdict(step) for step in manifest.steps],
        "summary": manifest.summary,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
