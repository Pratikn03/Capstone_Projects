from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
APPENDIX_TEX = REPO_ROOT / "appendices" / "app_c_all_theorems.tex"
APPENDIX_PDF = REPO_ROOT / "appendices" / "app_c_all_theorems.pdf"
APPENDIX_SCRIPT = REPO_ROOT / "scripts" / "build_app_c_all_theorems.py"


def test_app_c_all_theorems_sources_are_present_and_labeled() -> None:
    assert APPENDIX_TEX.exists()
    assert APPENDIX_SCRIPT.exists()

    text = APPENDIX_TEX.read_text(encoding="utf-8")

    assert r"\documentclass[12pt,oneside]{book}" in text
    assert r"\input{preamble.tex}" in text
    assert r"\setcounter{chapter}{2}" in text
    assert r"\input{appendices/app_c_full_proofs.tex}" in text
    assert "All ORIUS Theorem Surfaces" in text


def test_app_c_all_theorems_build_script_generates_pdf_when_toolchain_exists() -> None:
    if shutil.which("latexmk") is None and shutil.which("pdflatex") is None:
        pytest.skip("latexmk/pdflatex not available")

    completed = subprocess.run(
        [sys.executable, str(APPENDIX_SCRIPT)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "appendices/app_c_all_theorems.pdf" in completed.stdout
    assert APPENDIX_PDF.exists()
    assert APPENDIX_PDF.stat().st_size > 0
