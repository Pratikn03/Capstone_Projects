"""Training run manifest: reproducibility tracking."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import shutil


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
        pass
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
        pass
    
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
        pass
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


def create_run_manifest(
    config_path: Path | str,
    output_dir: Path | str,
    run_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
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
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": get_git_status(),
        "system": get_system_info(),
        "packages": get_pip_freeze(),
        "config": {
            "path": str(config_path),
            "exists": config_path.exists(),
        },
    }
    
    # Copy config file to output directory for snapshot
    if config_path.exists():
        config_snapshot = output_dir / f"config_{run_id}.yaml"
        shutil.copy2(config_path, config_snapshot)
        manifest["config"]["snapshot"] = str(config_snapshot)
        
        # Also read config content
        try:
            manifest["config"]["content"] = config_path.read_text(encoding="utf-8")
        except Exception:
            pass
    
    # Add extra metadata
    if extra_metadata:
        manifest["metadata"] = extra_metadata
    
    # Save manifest
    manifest_path = output_dir / f"manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
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
        "only_in_run1": sorted(list(packages1 - packages2))[:10],  # Limit output
        "only_in_run2": sorted(list(packages2 - packages1))[:10],
        "total_diff_count": len(packages1.symmetric_difference(packages2)),
    }
    
    return differences


def print_manifest_summary(manifest: dict[str, Any]) -> None:
    """Print a human-readable summary of a training manifest."""
    print(f"Run ID: {manifest.get('run_id')}")
    print(f"Timestamp: {manifest.get('timestamp')}")
    
    git_info = manifest.get("git", {})
    print(f"\nGit:")
    print(f"  Commit: {git_info.get('commit', 'N/A')}")
    print(f"  Branch: {git_info.get('branch', 'N/A')}")
    print(f"  Uncommitted: {git_info.get('uncommitted_changes', False)}")
    
    sys_info = manifest.get("system", {})
    print(f"\nSystem:")
    print(f"  Python: {sys_info.get('python_version', 'N/A')[:50]}")
    print(f"  Platform: {sys_info.get('platform', 'N/A')}")
    
    config_info = manifest.get("config", {})
    print(f"\nConfig:")
    print(f"  Path: {config_info.get('path', 'N/A')}")
    print(f"  Snapshot: {config_info.get('snapshot', 'N/A')}")
    
    pkg_count = len(manifest.get("packages", []))
    print(f"\nPackages: {pkg_count} installed")
