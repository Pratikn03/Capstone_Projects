from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_all_domain_eval_includes_navigation_and_nontrivial_reliability(tmp_path: Path) -> None:
    out_dir = tmp_path / "multi_domain"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_all_domain_eval.py",
            "--rows",
            "32",
            "--out",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Running Navigation" in proc.stdout
    report = json.loads((out_dir / "all_domain_results.json").read_text(encoding="utf-8"))
    rows = {row["domain"]: row for row in report["results"]}

    assert set(rows) == {"aerospace", "av", "navigation", "healthcare", "industrial"}
    assert rows["navigation"]["data_source"] == "synthetic"
    assert all(row["status"] == "ok" for row in rows.values())
    assert any(float(row["mean_reliability"]) < 0.99 for row in rows.values())

    comparison_tex = (out_dir / "tbl_all_domain_comparison.tex").read_text(encoding="utf-8")
    assert "Navigation" in comparison_tex
    assert "locked\\_csv" in comparison_tex
    assert "synthetic" in comparison_tex

    assert (out_dir / "fig_all_domain_comparison.png").exists()
