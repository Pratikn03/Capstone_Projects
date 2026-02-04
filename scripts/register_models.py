"""Register model artifacts into a local registry file."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.utils.registry import register_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--out", default="artifacts/registry/models.json")
    parser.add_argument("--run-id", default=None, help="Optional run id for registry snapshot")
    args = parser.parse_args()

    payload = register_models(Path(args.models_dir), Path(args.out), run_id=args.run_id)
    print(f"[registry] wrote {args.out} (models={len(payload['latest']['models'])})")


if __name__ == "__main__":
    main()

