"""Validate configuration YAML files using pydantic models."""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.utils.config import validate_config


def main() -> None:
    cfg_dir = Path("configs")
    if not cfg_dir.exists():
        raise SystemExit("Missing configs directory.")

    errors: list[str] = []
    for path in sorted(cfg_dir.glob("*.yaml")):
        try:
            validate_config(path)
            print(f"[config] OK: {path}")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            print(f"[config] FAIL: {path} -> {exc}")

    if errors:
        raise SystemExit("Config validation failed.")
    print("[config] All configs validated.")


if __name__ == "__main__":
    main()

