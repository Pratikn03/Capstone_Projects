from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def test_verify_paper_manifest_camera_ready_promotes_warnings_to_errors(tmp_path: Path) -> None:
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "assets" / "data").mkdir(parents=True)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "placeholder.txt").write_text("ok\n", encoding="utf-8")
    (tmp_path / "paper" / "assets" / "data" / "metrics_snapshot.json").write_text(
        json.dumps({"ok": True}), encoding="utf-8"
    )

    (tmp_path / "paper" / "manifest.yaml").write_text(
        "\n".join(
            [
                "figures:",
                "  FIG_USED:",
                "    path: assets/placeholder.txt",
                "  FIG_UNUSED:",
                "    path: assets/placeholder.txt",
                "tables: {}",
                "data: {}",
                "configs: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "paper" / "paper.tex").write_text(
        "\n".join(
            [
                r"\documentclass[12pt,oneside]{book}",
                r"\begin{document}",
                r"\part{Problem, Principle, and Gap}",
                r"\part{ORIUS Framework and Theorem Ladder}",
                r"\part{Witness Domain Evidence}",
                r"\part{Universalization and Extensions}",
                r"\part{Submission Boundary}",
                r"\appendix",
                r"\newcommand{\PaperFigure}[2]{}",
                r"\PaperFigure{FIG_USED}{Used token activates the legacy manifest surface.}",
                r"\bibliography{paper/bibliography/orius_monograph}",
                r"\end{document}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    baseline = _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_paper_manifest.py")],
        cwd=tmp_path,
    )
    assert baseline.returncode == 0
    assert "Unused manifest tokens" in baseline.stdout

    strict = _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_paper_manifest.py"), "--camera-ready"],
        cwd=tmp_path,
    )
    assert strict.returncode == 1
    assert "camera-ready warning promoted to error" in strict.stderr


def test_verify_paper_manifest_skips_unused_legacy_tokens_when_macro_surface_is_inactive(tmp_path: Path) -> None:
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "assets" / "data").mkdir(parents=True)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "placeholder.txt").write_text("ok\n", encoding="utf-8")
    (tmp_path / "paper" / "assets" / "data" / "metrics_snapshot.json").write_text(
        json.dumps({"ok": True}), encoding="utf-8"
    )

    (tmp_path / "paper" / "manifest.yaml").write_text(
        "\n".join(
            [
                "figures:",
                "  FIG_UNUSED:",
                "    path: assets/placeholder.txt",
                "tables: {}",
                "data: {}",
                "configs: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "paper" / "paper.tex").write_text(
        "\n".join(
            [
                r"\documentclass[12pt,oneside]{book}",
                r"\begin{document}",
                r"\part{Problem, Principle, and Gap}",
                r"\part{ORIUS Framework and Theorem Ladder}",
                r"\part{Witness Domain Evidence}",
                r"\part{Universalization and Extensions}",
                r"\part{Submission Boundary}",
                r"\appendix",
                r"\chapter{No Legacy Manifest Macros}",
                r"Camera-ready prose now uses direct includes rather than \PaperFigure tokens.",
                r"\bibliography{paper/bibliography/orius_monograph}",
                r"\end{document}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    strict = _run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_paper_manifest.py"), "--camera-ready"],
        cwd=tmp_path,
    )
    assert strict.returncode == 0
    assert "unused-token audit skipped" in strict.stdout


def test_verify_camera_ready_logs_blocks_duplicate_destinations_even_with_waivers(tmp_path: Path) -> None:
    log_path = tmp_path / "paper.log"
    waiver_path = tmp_path / "waivers.yaml"

    waiver_path.write_text(
        "\n".join(
            [
                "logs:",
                "  paper.log:",
                r"    - '^Underfull \\hbox .*'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    log_path.write_text(r"Underfull \hbox (badness 10000) in paragraph at lines 1--1" + "\n", encoding="utf-8")
    ok = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_camera_ready_logs.py"),
            "--waivers",
            str(waiver_path),
            "--log",
            str(log_path),
        ],
        cwd=tmp_path,
    )
    assert ok.returncode == 0

    log_path.write_text(
        "\n".join(
            [
                r"Underfull \hbox (badness 10000) in paragraph at lines 1--1",
                "pdfTeX warning (ext4): destination with the same identifier (name{page.1}) has been already used, duplicate ignored",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    blocked = _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_camera_ready_logs.py"),
            "--waivers",
            str(waiver_path),
            "--log",
            str(log_path),
        ],
        cwd=tmp_path,
    )
    assert blocked.returncode == 1
    assert "duplicate_destination" in blocked.stderr
