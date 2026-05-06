#!/usr/bin/env python3
"""Build the standalone Appendix C all-theorems bundle."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEX = REPO_ROOT / "appendices" / "app_c_all_theorems.tex"
OUTPUT_PDF = REPO_ROOT / "appendices" / "app_c_all_theorems.pdf"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the standalone Appendix C all-theorems bundle")
    parser.add_argument("--output-pdf", default=str(OUTPUT_PDF))
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _build_with_latexmk(outdir: Path) -> None:
    _run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={outdir}",
            str(SOURCE_TEX.relative_to(REPO_ROOT)),
        ]
    )


def _build_with_pdflatex(outdir: Path) -> None:
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={outdir}",
        str(SOURCE_TEX.relative_to(REPO_ROOT)),
    ]
    _run(cmd)
    _run(cmd)


def main() -> int:
    args = _parse_args()
    output_pdf = Path(args.output_pdf)
    if not output_pdf.is_absolute():
        output_pdf = REPO_ROOT / output_pdf
    if not SOURCE_TEX.exists():
        raise SystemExit(f"missing source TeX: {SOURCE_TEX}")

    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk is None and pdflatex is None:
        raise SystemExit("missing required LaTeX toolchain: need latexmk or pdflatex")

    with tempfile.TemporaryDirectory(prefix="app_c_all_theorems_") as tmp:
        outdir = Path(tmp)
        if latexmk is not None:
            _build_with_latexmk(outdir)
        else:
            _build_with_pdflatex(outdir)

        built_pdf = outdir / f"{SOURCE_TEX.stem}.pdf"
        if not built_pdf.exists():
            raise SystemExit(f"expected PDF not found: {built_pdf}")
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_pdf, output_pdf)

    try:
        print(output_pdf.relative_to(REPO_ROOT))
    except ValueError:
        print(output_pdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
