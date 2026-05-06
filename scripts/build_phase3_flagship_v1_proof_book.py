#!/usr/bin/env python3
"""Build the standalone Phase 3 proof-book PDF from the tracked Markdown source."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = REPO_ROOT / "reports" / "publication" / "phase3_flagship_v1_proof_book.md"
OUTPUT_PDF = REPO_ROOT / "reports" / "publication" / "phase3_flagship_v1_proof_book.pdf"


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"missing required tool: {name}")


def main() -> int:
    if not SOURCE_MD.exists():
        raise SystemExit(f"missing source markdown: {SOURCE_MD}")

    _require_tool("pandoc")
    _require_tool("xelatex")

    cmd = [
        "pandoc",
        str(SOURCE_MD.relative_to(REPO_ROOT)),
        "--from",
        "gfm",
        "--pdf-engine=xelatex",
        "--toc",
        "--number-sections",
        "-V",
        "geometry:margin=1in",
        "-V",
        "colorlinks=true",
        "-o",
        str(OUTPUT_PDF.relative_to(REPO_ROOT)),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    print(OUTPUT_PDF.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
