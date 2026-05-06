from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOK_MD = REPO_ROOT / "reports" / "publication" / "phase3_flagship_v1_proof_book.md"
BOOK_PDF = REPO_ROOT / "reports" / "publication" / "phase3_flagship_v1_proof_book.pdf"
BOOK_SCRIPT = REPO_ROOT / "scripts" / "build_phase3_flagship_v1_proof_book.py"
PUBLICATION_README = REPO_ROOT / "reports" / "publication" / "README.md"


def test_phase3_proof_book_sources_are_present_and_normalized() -> None:
    assert BOOK_MD.exists()
    assert BOOK_SCRIPT.exists()

    book_text = BOOK_MD.read_text(encoding="utf-8")
    readme_text = PUBLICATION_README.read_text(encoding="utf-8")

    for theorem_id in ("T1", "T2", "T3a", "T4", "T11", "T_trajectory_PAC"):
        assert theorem_id in book_text

    assert "`T7` is the fallback-existence row. `T8` is graceful-degradation dominance." in book_text
    assert "phase3_flagship_v1_proof_book.md" in readme_text
    assert "phase3_flagship_v1_proof_book.pdf" in readme_text
    assert "repo-normalized implementation book" in readme_text


def test_phase3_proof_book_build_script_generates_pdf_when_toolchain_exists() -> None:
    if shutil.which("pandoc") is None or shutil.which("xelatex") is None:
        pytest.skip("pandoc/xelatex not available")

    completed = subprocess.run(
        [sys.executable, str(BOOK_SCRIPT)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "reports/publication/phase3_flagship_v1_proof_book.pdf" in completed.stdout
    assert BOOK_PDF.exists()
    assert BOOK_PDF.stat().st_size > 0
