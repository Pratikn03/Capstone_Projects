"""Training run manifest: reproducibility tracking."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_git_hash() -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def get_git_status() -> dict[str, Any]:
    """Get git repository status."""
    status = {
        "commit": get_git_hash(),
        "branch": None,
        "uncommitted_changes": False,
    }

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            status["branch"] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            status["uncommitted_changes"] = bool(result.stdout.strip())
    except Exception:
        return status

    return status


def get_pip_freeze() -> list[str]:
    """Get list of installed Python packages."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")
    except Exception:
        return []
    return []


def get_system_info() -> dict[str, Any]:
    """Get Python and system information."""
    import platform

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def create_run_manifest(
    config_path: Path | str,
    output_dir: Path | str,
    run_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    data_manifest_path: Path | str | None = "paper/assets/data/data_manifest.json",
    data_manifest_sha256: str | None = None,
    split_boundaries: dict[str, Any] | None = None,
    schema_hash: str | None = None,
    config_snapshot_text: str | None = None,
) -> dict[str, Any]:
    """
    Create a comprehensive training run manifest for reproducibility.

    Args:
        config_path: Path to training config YAML
        output_dir: Directory where manifest will be saved
        run_id: Optional run identifier (defaults to timestamp)
        extra_metadata: Optional additional metadata to include

    Returns:
        Manifest dictionary
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    resolved_data_manifest = None
    if data_manifest_path:
        resolved_data_manifest = Path(data_manifest_path)
        if not resolved_data_manifest.is_absolute():
            resolved_data_manifest = Path.cwd() / resolved_data_manifest

    data_manifest_payload: dict[str, Any] = {}
    if resolved_data_manifest and resolved_data_manifest.exists():
        try:
            data_manifest_payload = json.loads(resolved_data_manifest.read_text(encoding="utf-8"))
        except Exception:
            data_manifest_payload = {}
        if data_manifest_sha256 is None:
            data_manifest_sha256 = _sha256_path(resolved_data_manifest)
        if schema_hash is None and isinstance(data_manifest_payload, dict):
            schema_hash = data_manifest_payload.get("schema_hash") or data_manifest_payload.get(
                "manifest_sha256"
            )
        if split_boundaries is None and isinstance(data_manifest_payload, dict):
            split_boundaries = data_manifest_payload.get("split_boundaries")

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "git": get_git_status(),
        "system": get_system_info(),
        "packages": get_pip_freeze(),
        "config": {
            "path": str(config_path),
            "exists": config_path.exists(),
        },
        "data_manifest_path": str(resolved_data_manifest) if resolved_data_manifest is not None else None,
        "data_manifest_sha256": data_manifest_sha256,
        "split_boundaries": split_boundaries if isinstance(split_boundaries, dict) else {},
        "schema_hash": schema_hash,
    }

    # Copy the effective config to output directory for snapshot. If the
    # caller applies CLI overrides in memory, config_snapshot_text must carry
    # that mutated configuration so the manifest reflects the actual run.
    if config_path.exists() or config_snapshot_text is not None:
        config_snapshot = output_dir / f"config_{run_id}.yaml"
        if config_snapshot_text is not None:
            config_snapshot.write_text(config_snapshot_text, encoding="utf-8")
        else:
            shutil.copy2(config_path, config_snapshot)
        manifest["config"]["snapshot"] = str(config_snapshot)
        manifest["config"]["effective_snapshot"] = config_snapshot_text is not None

        # Also read config content
        with contextlib.suppress(Exception):
            manifest["config"]["content"] = (
                config_snapshot_text
                if config_snapshot_text is not None
                else config_path.read_text(encoding="utf-8")
            )

    # Add extra metadata
    if extra_metadata:
        manifest["metadata"] = extra_metadata

    # Save manifest
    manifest_path = output_dir / f"manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)

    return manifest


def load_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """Load a training run manifest."""
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def compare_manifests(path1: Path | str, path2: Path | str) -> dict[str, Any]:
    """
    Compare two training run manifests to identify differences.

    Useful for debugging why two runs produced different results.
    """
    m1 = load_manifest(path1)
    m2 = load_manifest(path2)

    differences = {
        "run_ids": (m1.get("run_id"), m2.get("run_id")),
        "git_commits": (m1.get("git", {}).get("commit"), m2.get("git", {}).get("commit")),
        "git_match": m1.get("git", {}).get("commit") == m2.get("git", {}).get("commit"),
        "python_versions": (
            m1.get("system", {}).get("python_version"),
            m2.get("system", {}).get("python_version"),
        ),
        "config_match": m1.get("config", {}).get("content") == m2.get("config", {}).get("content"),
    }

    # Compare package versions
    packages1 = set(m1.get("packages", []))
    packages2 = set(m2.get("packages", []))
    differences["package_diff"] = {
        "only_in_run1": sorted(packages1 - packages2)[:10],  # Limit output
        "only_in_run2": sorted(packages2 - packages1)[:10],
        "total_diff_count": len(packages1.symmetric_difference(packages2)),
    }

    return differences


def print_manifest_summary(manifest: dict[str, Any]) -> None:
    """Print a human-readable summary of a training manifest."""

    manifest.get("git", {})

    manifest.get("system", {})

    manifest.get("config", {})

    len(manifest.get("packages", []))
