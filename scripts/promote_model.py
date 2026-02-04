"""Promote (pin) model bundles in configs/forecast.yaml for rollback control."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target name (e.g., load_mw)")
    parser.add_argument("--path", required=True, help="Model bundle path to pin")
    parser.add_argument("--config", default="configs/forecast.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    else:
        cfg = {}

    cfg.setdefault("models", {})
    cfg["models"][args.target] = args.path

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[promote] pinned {args.target} -> {args.path} in {cfg_path}")


if __name__ == "__main__":
    main()

