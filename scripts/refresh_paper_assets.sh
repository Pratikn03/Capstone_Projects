#!/usr/bin/env bash
# Full paper asset refresh: validation, figures, tables, LaTeX.
# Run from repo root: bash scripts/refresh_paper_assets.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

echo "=== Refreshing paper assets (from $ROOT) ==="
"$PY" scripts/run_universal_orius_validation.py
"$PY" scripts/generate_architecture_diagram.py --paper paper/assets/figures/fig01_architecture
"$PY" scripts/generate_paper_figures_fig12_fig13_fig58.py
"$PY" scripts/sync_paper_figures.py
"$PY" scripts/generate_multi_domain_figure.py
"$PY" scripts/build_paper_table_tex.py

echo "=== Building paper ==="
(cd paper && pdflatex -interaction=nonstopmode paper.tex)

echo "Done. PDF: paper/paper.pdf"
