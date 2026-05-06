from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APPENDIX_TEX = REPO_ROOT / "appendices" / "app_c_flagship_proofs.tex"
APPENDIX_PDF = REPO_ROOT / "appendices" / "app_c_flagship_proofs.pdf"
APPENDIX_SCRIPT = REPO_ROOT / "scripts" / "build_app_c_flagship_proofs.py"


def test_app_c_flagship_proofs_sources_are_present_and_labeled() -> None:
    assert APPENDIX_TEX.exists()
    assert APPENDIX_SCRIPT.exists()

    text = APPENDIX_TEX.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())

    assert r"\section*{C.1 Strict Topological Notation and Dynamics}" in text
    assert "Flagship Proofs for ORIUS" in text
    assert r"\label{thm:t1}" in text
    assert r"\label{thm:t2}" in text
    assert r"\label{thm:t3a}" in text
    assert r"\label{thm:t4}" in text
    assert r"\label{thm:t11}" in text
    assert r"\label{thm:t_trajectory_pac}" in text
    assert r"\label{thm:t8}" not in text
    assert r"\usepackage[utf8]{inputenc}" in text
    assert "Assumption Register" in normalized_text
    assert "Appendix B" in normalized_text
    assert "registry-canonical flagship-core execution draft" in normalized_text
    assert "T3b" in text
    assert "T6" in text
    assert "T8" in text
    assert r"\texttt{appendices/app\_c\_full\_proofs.tex}" in text

    positions = [
        text.index(r"\label{thm:t1}"),
        text.index(r"\label{thm:t2}"),
        text.index(r"\label{thm:t3a}"),
        text.index(r"\label{thm:t4}"),
        text.index(r"\label{thm:t11}"),
        text.index(r"\label{thm:t_trajectory_pac}"),
    ]
    assert positions == sorted(positions)


def test_app_c_flagship_proofs_build_script_generates_pdf_when_toolchain_exists(tmp_path: Path) -> None:
    if shutil.which("latexmk") is None and shutil.which("pdflatex") is None:
        pytest.skip("latexmk/pdflatex not available")

    output_pdf = tmp_path / "app_c_flagship_proofs.pdf"
    completed = subprocess.run(
        [sys.executable, str(APPENDIX_SCRIPT), "--output-pdf", str(output_pdf)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(output_pdf) in completed.stdout
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0
