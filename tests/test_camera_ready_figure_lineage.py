from __future__ import annotations

import json
from pathlib import Path

from scripts import build_camera_ready_figure_lineage as lineage


def _configure_tmp_repo(tmp_path: Path, monkeypatch) -> None:
    publication = tmp_path / "reports" / "publication"
    monkeypatch.setattr(lineage, "REPO", tmp_path)
    monkeypatch.setattr(lineage, "PUBLICATION_DIR", publication)
    monkeypatch.setattr(lineage, "LINEAGE_JSON", publication / "camera_ready_figure_lineage.json")
    monkeypatch.setattr(lineage, "LINEAGE_CSV", publication / "camera_ready_figure_lineage.csv")
    monkeypatch.setattr(lineage, "LINEAGE_MD", publication / "camera_ready_figure_lineage.md")
    monkeypatch.setattr(lineage, "DESIGN_JSON", publication / "camera_ready_design_sources.json")
    monkeypatch.setattr(lineage, "TEX_SURFACES", (tmp_path / "paper" / "ieee",))
    monkeypatch.setattr(
        lineage,
        "INVENTORY_ROOTS",
        (
            tmp_path / "paper" / "assets" / "figures",
            tmp_path / "paper" / "ieee" / "generated",
            tmp_path / "reports",
        ),
    )
    monkeypatch.setattr(
        lineage,
        "RESOLUTION_ROOTS",
        (
            tmp_path,
            tmp_path / "paper" / "assets" / "figures",
            tmp_path / "reports" / "publication",
            tmp_path / "reports" / "industrial" / "figures",
        ),
    )


def _write(path: Path, data: bytes = b"ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_lineage_resolves_latex_figures_and_writes_manifests(tmp_path: Path, monkeypatch) -> None:
    _configure_tmp_repo(tmp_path, monkeypatch)
    (tmp_path / "paper" / "ieee").mkdir(parents=True)
    (tmp_path / "paper" / "ieee" / "main.tex").write_text(
        "\n".join(
            [
                r"\includegraphics[width=\linewidth]{fig_data}",
                r"\includegraphics[width=\linewidth]{fig01_architecture.pdf}",
                r"\ORIUSIncludeArchivedGraphic[width=\linewidth]{reports/industrial/figures/old.png}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write(tmp_path / "reports" / "publication" / "fig_data.png")
    _write(tmp_path / "paper" / "assets" / "figures" / "fig01_architecture.pdf", b"%PDF-1.4\nok\n")
    _write(tmp_path / "scripts" / "generate_publication_figures.py")
    _write(tmp_path / "scripts" / "generate_architecture_diagram.py")
    _write(tmp_path / "reports" / "publication" / "artifact_traceability_table.csv")

    payload = lineage.build_entries()
    assert payload["totals"]["unresolved_latex_references"] == 0
    assert payload["totals"]["verification_errors"] == 0
    assert {entry["classification"] for entry in payload["entries"] if entry["latex_used"]} == {
        "data_plot",
        "legacy_archive",
        "static_diagram",
    }

    lineage.write_outputs(payload)
    assert (tmp_path / "reports" / "publication" / "camera_ready_figure_lineage.json").exists()
    assert (tmp_path / "reports" / "publication" / "camera_ready_figure_lineage.csv").exists()
    assert (tmp_path / "reports" / "publication" / "camera_ready_figure_lineage.md").exists()
    design = json.loads(
        (tmp_path / "reports" / "publication" / "camera_ready_design_sources.json").read_text()
    )
    assert design["design_sources"][0]["tool"] == "figma"
    assert lineage.verify_against_saved(payload) == []


def test_lineage_blocks_latex_data_plot_without_tracked_sources(tmp_path: Path, monkeypatch) -> None:
    _configure_tmp_repo(tmp_path, monkeypatch)
    (tmp_path / "paper" / "ieee").mkdir(parents=True)
    (tmp_path / "paper" / "ieee" / "main.tex").write_text(
        r"\includegraphics[width=\linewidth]{fig_violation_rate}" + "\n",
        encoding="utf-8",
    )
    _write(tmp_path / "reports" / "publication" / "fig_violation_rate.png")

    payload = lineage.build_entries()
    errors = [
        error for entry in payload["entries"] if entry["latex_used"] for error in entry["verification_errors"]
    ]

    assert "missing_source_script" in errors
    assert "missing_source_data" in errors
