"""Run the default CPSBench-IoT suite and verify publication artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.cpsbench_iot.runner import REQUIRED_OUTPUTS, run_suite
from gridpulse.cpsbench_iot.scenarios import DEFAULT_SCENARIOS


DEFAULT_SEEDS = [11, 22, 33, 44, 55]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPSBench-IoT default benchmark suite")
    parser.add_argument("--out-dir", default="reports/publication")
    parser.add_argument("--horizon", type=int, default=168)
    return parser.parse_args()


def _verify_artifacts(out_dir: Path) -> None:
    missing = []
    for name in REQUIRED_OUTPUTS:
        path = out_dir / name
        if (not path.exists()) or path.stat().st_size == 0:
            missing.append(str(path))
    if missing:
        raise SystemExit(f"Missing required CPSBench artifacts: {missing}")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    summary = run_suite(
        scenarios=list(DEFAULT_SCENARIOS),
        seeds=list(DEFAULT_SEEDS),
        out_dir=out_dir,
        horizon=int(args.horizon),
    )
    _verify_artifacts(out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
