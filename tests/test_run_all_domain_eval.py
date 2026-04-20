from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_all_domain_eval.py"


def _load_eval_script():
    spec = importlib.util.spec_from_file_location("run_all_domain_eval", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_script = _load_eval_script()


def test_load_domain_frame_requires_defended_surface_by_default(tmp_path: Path) -> None:
    cfg = {
        "csv": tmp_path / "missing_navigation.csv",
        "synthetic_fn": lambda n_rows, rng: pd.DataFrame({"value": rng.normal(size=n_rows)}),
    }

    with pytest.raises(FileNotFoundError, match="legacy lower-tier path"):
        eval_script._load_domain_frame(
            "support_only",
            cfg,
            np.random.default_rng(42),
            16,
            allow_support_tier=False,
        )


def test_load_domain_frame_uses_legacy_synthetic_only_when_explicitly_enabled(tmp_path: Path) -> None:
    cfg = {
        "csv": tmp_path / "missing_navigation.csv",
        "synthetic_fn": lambda n_rows, rng: pd.DataFrame({"value": rng.normal(size=n_rows)}),
    }

    frame, source = eval_script._load_domain_frame(
        "support_only",
        cfg,
        np.random.default_rng(42),
        16,
        allow_support_tier=True,
    )

    assert source == "synthetic"
    assert not frame.empty


def test_all_domain_eval_runs_under_explicit_support_tier_and_emits_results(tmp_path: Path) -> None:
    out_dir = tmp_path / "multi_domain"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--rows",
            "32",
            "--allow-support-tier",
            "--out",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Running Autonomous Vehicles" in proc.stdout
    assert "Running Medical Monitoring (ICU Vitals)" in proc.stdout
    report = json.loads((out_dir / "all_domain_results.json").read_text(encoding="utf-8"))
    rows = {row["domain"]: row for row in report["results"]}

    assert set(rows) == {"av", "healthcare"}
    assert all(row["status"] == "ok" for row in rows.values())
    assert rows["healthcare"]["data_source"] == "locked_csv"
    assert any(float(row["mean_reliability"]) < 0.99 for row in rows.values())

    comparison_tex = (out_dir / "tbl_all_domain_comparison.tex").read_text(encoding="utf-8")
    assert "Medical Monitoring (ICU Vitals)" in comparison_tex
    assert "locked\\_csv" in comparison_tex

    assert (out_dir / "fig_all_domain_comparison.png").exists()
