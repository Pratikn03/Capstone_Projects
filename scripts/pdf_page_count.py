#!/usr/bin/env python3
"""Print the page count for a PDF file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _read_with_python_library(pdf_path: Path) -> int | None:
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name, fromlist=["PdfReader"])
            reader = module.PdfReader(str(pdf_path))
            return len(reader.pages)
        except Exception:
            continue
    return None


def _read_with_mdls(pdf_path: Path) -> int | None:
    try:
        completed = subprocess.run(
            ["mdls", "-name", "kMDItemNumberOfPages", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    for token in completed.stdout.split():
        if token.isdigit():
            return int(token)
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: pdf_page_count.py <pdf-path>", file=sys.stderr)
        return 2
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"missing PDF: {pdf_path}", file=sys.stderr)
        return 1
    page_count = _read_with_python_library(pdf_path)
    if page_count is None:
        page_count = _read_with_mdls(pdf_path)
    if page_count is None:
        print(f"could not determine page count for {pdf_path}", file=sys.stderr)
        return 1
    print(page_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
