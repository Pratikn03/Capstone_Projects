from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "notebooks"
PUBLICATION = REPO_ROOT / "reports" / "publication"

REQUIRED_NOTEBOOKS = [
    "00_orius_research_notebook_index.ipynb",
    "20_final_release_results_analysis.ipynb",
    "21_utility_preserving_safety_analysis.ipynb",
    "22_theorem_audit_traceability.ipynb",
    "23_freeze_release_reproducibility_audit.ipynb",
    "24_publication_package_quality_audit.ipynb",
]


def test_required_orius_research_notebooks_exist_and_validate() -> None:
    for name in REQUIRED_NOTEBOOKS:
        path = NOTEBOOKS / name
        assert path.exists(), path
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert len(nb.cells) >= 4
        assert nb.cells[0].cell_type == "markdown"
        assert "ORIUS" in "".join(nb.cells[0].source)


def test_notebook_inventory_tracks_active_research_suite() -> None:
    inventory_path = PUBLICATION / "orius_research_notebook_inventory.csv"
    assert inventory_path.exists()
    rows = list(csv.DictReader(inventory_path.open("r", encoding="utf-8", newline="")))
    by_name = {Path(row["notebook"]).name: row for row in rows}
    for name in REQUIRED_NOTEBOOKS:
        assert name in by_name
        assert by_name[name]["status"] == "active"
        assert by_name[name]["primary_artifacts"]


def test_notebooks_are_analysis_only_surfaces() -> None:
    forbidden = {
        "train_dataset.py",
        "run_three_domain_offline_freeze.py",
        "git push",
        "git commit",
    }
    for name in REQUIRED_NOTEBOOKS:
        path = NOTEBOOKS / name
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text


def test_generated_research_notebooks_execute_as_plain_python() -> None:
    matplotlib.use("Agg")
    for name in REQUIRED_NOTEBOOKS:
        path = NOTEBOOKS / name
        nb = nbformat.read(path, as_version=4)
        namespace = {"display": lambda *_args, **_kwargs: None}
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            code = "".join(cell.source)
            exec(compile(code, f"{path}:{index}", "exec"), namespace)
