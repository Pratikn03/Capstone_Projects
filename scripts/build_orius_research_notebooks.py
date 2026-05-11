#!/usr/bin/env python3
"""Build the canonical ORIUS research-analysis notebooks.

The notebooks are lightweight analysis surfaces over locked artifacts. They do
not retrain models or mutate release artifacts; they make the final evidence
auditable from a reader-friendly Jupyter interface.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11",
    },
}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def _setup_cell() -> nbf.NotebookNode:
    return _code(
        """
        from __future__ import annotations

        import csv
        import json
        from pathlib import Path

        import matplotlib.pyplot as plt
        import pandas as pd

        ROOT = Path.cwd()
        if not (ROOT / "reports").exists():
            ROOT = Path.cwd().parent
        PUBLICATION = ROOT / "reports" / "publication"
        SPLIT_ROOT = ROOT / "reports" / "split_training"
        RELEASE_ID = (SPLIT_ROOT / "latest_release_id.txt").read_text().strip()
        FREEZE = ROOT / "reports" / "predeployment_freeze" / RELEASE_ID

        def read_csv(relpath: str) -> pd.DataFrame:
            path = ROOT / relpath
            if not path.exists():
                raise FileNotFoundError(path)
            return pd.read_csv(path)

        def read_json(relpath: str) -> dict:
            path = ROOT / relpath
            if not path.exists():
                raise FileNotFoundError(path)
            return json.loads(path.read_text())

        def display_path(relpath: str) -> None:
            path = ROOT / relpath
            print(f"{relpath}: {'exists' if path.exists() else 'missing'}")
        """
    )


def _build_notebook(title: str, objective: str, cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.update(KERNEL_METADATA)
    nb.cells = [
        _md(
            f"""
            # {title}

            **Objective.** {objective}

            **Run mode.** Analysis only. This notebook reads locked ORIUS artifacts
            and does not retrain models, rewrite release manifests, or mutate runtime traces.
            """
        ),
        *cells,
    ]
    return nb


def _write_notebook(path: Path, nb: nbf.NotebookNode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def _notebook_specs() -> list[dict[str, Any]]:
    return [
        {
            "path": NOTEBOOK_DIR / "00_orius_research_notebook_index.ipynb",
            "title": "ORIUS Research Notebook Index",
            "objective": "Inventory the active research notebooks and map each notebook to a defended ORIUS claim surface.",
            "cells": [
                _setup_cell(),
                _md(
                    """
                    ## Active notebook coverage

                    The active ORIUS research program is Battery, AV, and Healthcare.
                    Older industrial, navigation, and aerospace notebooks are historical
                    only unless their artifact rows are explicitly promoted again.
                    """
                ),
                _code(
                    """
                    inventory = pd.read_csv(PUBLICATION / "orius_research_notebook_inventory.csv")
                    inventory
                    """
                ),
                _code(
                    """
                    required = inventory[inventory["status"] == "active"]
                    missing = [
                        row.notebook
                        for row in required.itertuples()
                        if not (ROOT / row.notebook).exists()
                    ]
                    assert not missing, missing
                    required[["notebook", "claim_surface", "primary_artifacts"]]
                    """
                ),
            ],
        },
        {
            "path": NOTEBOOK_DIR / "20_final_release_results_analysis.ipynb",
            "title": "Final Release Results Analysis",
            "objective": "Reproduce the final paper-facing training, runtime, utility, and freeze tables from locked CPU release artifacts.",
            "cells": [
                _setup_cell(),
                _md("## Final release tables"),
                _code(
                    """
                    training = read_csv("reports/publication/final_training_quality_for_paper.csv")
                    runtime = read_csv("reports/publication/final_runtime_safety_for_paper.csv")
                    freeze = read_csv("reports/publication/final_freeze_validation_for_paper.csv")
                    utility = read_csv("reports/publication/utility_preserving_safety_scorecard.csv")

                    print("release:", RELEASE_ID)
                    display(training)
                    display(runtime)
                    display(utility[["domain", "safety_reference_controller", "excess_tsvr_over_safety_reference", "utility_gain_over_safety_reference", "utility_preserving_safety_gate"]])
                    display(freeze)
                    """
                ),
                _md("## Runtime TSVR reduction"),
                _code(
                    """
                    ax = runtime.plot(
                        x="domain",
                        y=["baseline_tsvr", "orius_tsvr"],
                        kind="bar",
                        figsize=(9, 4),
                        title="Final claim-governing TSVR",
                    )
                    ax.set_ylabel("TSVR")
                    ax.grid(axis="y", alpha=0.25)
                    plt.tight_layout()
                    """
                ),
                _code(
                    """
                    assert (runtime["orius_tsvr"] <= runtime["baseline_tsvr"]).all()
                    assert (freeze["pass"].astype(str).str.lower() == "yes").all()
                    assert (utility["utility_preserving_safety_gate"].astype(str).str.lower() == "true").all()
                    """
                ),
            ],
        },
        {
            "path": NOTEBOOK_DIR / "21_utility_preserving_safety_analysis.ipynb",
            "title": "Utility-Preserving Safety Analysis",
            "objective": "Audit the claim that ORIUS is not merely a shutdown, always-brake, or always-alert policy.",
            "cells": [
                _setup_cell(),
                _md("## Scorecard semantics"),
                _code(
                    """
                    scorecard = read_csv("reports/publication/utility_preserving_safety_scorecard.csv")
                    cols = [
                        "domain",
                        "safety_reference_controller",
                        "orius_tsvr",
                        "safety_reference_tsvr",
                        "excess_tsvr_over_safety_reference",
                        "orius_utility",
                        "safety_reference_utility",
                        "utility_gain_over_safety_reference",
                        "utility_preserving_safety_gate",
                        "claim_boundary",
                    ]
                    scorecard[cols]
                    """
                ),
                _code(
                    """
                    plot_df = scorecard.copy()
                    plot_df["utility_delta_over_safety_reference"] = pd.to_numeric(
                        plot_df["utility_delta_over_safety_reference"], errors="coerce"
                    )
                    ax = plot_df.plot(
                        x="domain",
                        y="utility_delta_over_safety_reference",
                        kind="bar",
                        legend=False,
                        figsize=(9, 4),
                        title="Useful work preserved over fail-safe reference",
                    )
                    ax.set_ylabel("Utility delta")
                    ax.grid(axis="y", alpha=0.25)
                    plt.tight_layout()
                    """
                ),
                _code(
                    """
                    assert (scorecard["excess_tsvr_over_safety_reference"].astype(float) <= 1e-3).all()
                    assert (scorecard["utility_preserving_safety_gate"].astype(str).str.lower() == "true").all()
                    """
                ),
            ],
        },
        {
            "path": NOTEBOOK_DIR / "22_theorem_audit_traceability.ipynb",
            "title": "Theorem Audit and Traceability",
            "objective": "Inspect theorem status, proof tier, code anchors, tests, and remaining non-defended rows.",
            "cells": [
                _setup_cell(),
                _md("## Active theorem audit"),
                _code(
                    """
                    audit = read_csv("reports/publication/active_theorem_audit.csv")
                    audit[[
                        "theorem_id",
                        "title",
                        "surface_kind",
                        "defense_tier",
                        "proof_tier",
                        "code_correspondence",
                        "weakest_step",
                    ]]
                    """
                ),
                _code(
                    """
                    tier_counts = audit["defense_tier"].value_counts().rename_axis("defense_tier").reset_index(name="count")
                    display(tier_counts)
                    ax = tier_counts.plot(x="defense_tier", y="count", kind="bar", legend=False, figsize=(8, 4))
                    ax.set_title("Theorem defense-tier distribution")
                    ax.set_ylabel("Rows")
                    plt.tight_layout()
                    """
                ),
                _md(
                    """
                    ## Reviewer-facing theorem debt

                    These rows are not failures by themselves, but they must not be
                    described as active flagship theorems unless their proof, code, test,
                    artifact, and manuscript gates are promoted.
                    """
                ),
                _code(
                    """
                    debt = audit[~audit["defense_tier"].isin(["flagship_defended", "supporting_defended"])]
                    debt[["theorem_id", "title", "defense_tier", "scope_note", "remediation_class"]]
                    """
                ),
            ],
        },
        {
            "path": NOTEBOOK_DIR / "23_freeze_release_reproducibility_audit.ipynb",
            "title": "Freeze and Release Reproducibility Audit",
            "objective": "Verify the split-training release, downstream freeze validators, stress gates, and artifact hashes.",
            "cells": [
                _setup_cell(),
                _md("## Release manifest and hashes"),
                _code(
                    """
                    model_gate = read_json(f"reports/split_training/{RELEASE_ID}/candidate_model_quality_gate.json")
                    downstream = read_json(f"reports/predeployment_freeze/{RELEASE_ID}/downstream_post_split_results.json")
                    runtime_stress = read_json(f"reports/predeployment_freeze/{RELEASE_ID}/final_runtime_stress_gates_post_split.json")
                    manifest = read_json(f"reports/predeployment_freeze/{RELEASE_ID}/predeployment_release_manifest.json")
                    hashes = read_json(f"reports/predeployment_freeze/{RELEASE_ID}/frozen_artifact_hashes.json")

                    summary = {
                        "release_id": RELEASE_ID,
                        "model_gate_pass": model_gate["pass"],
                        "release_models": sum(1 for row in model_gate["models"] if row.get("release_model")),
                        "model_blockers": len(model_gate["blockers"]),
                        "downstream_all_passed": downstream["all_passed"],
                        "downstream_validators": len(downstream["results"]),
                        "runtime_stress_all_passed": runtime_stress["all_passed"],
                        "manifest_all_passed": manifest["all_passed"],
                        "hashed_artifacts": len(hashes["artifacts"]),
                    }
                    pd.DataFrame([summary])
                    """
                ),
                _code(
                    """
                    assert model_gate["pass"]
                    assert len(model_gate["blockers"]) == 0
                    assert downstream["all_passed"]
                    assert runtime_stress["all_passed"]
                    assert manifest["all_passed"]
                    assert len(hashes["artifacts"]) > 0
                    """
                ),
            ],
        },
        {
            "path": NOTEBOOK_DIR / "24_publication_package_quality_audit.ipynb",
            "title": "Publication Package Quality Audit",
            "objective": "Check that final manuscript tables, figures, PDF, and claim-boundary surfaces exist.",
            "cells": [
                _setup_cell(),
                _md("## Final manuscript artifact checks"),
                _code(
                    """
                    required = [
                        "reports/publication/tbl_final_training_quality.tex",
                        "reports/publication/tbl_final_runtime_safety.tex",
                        "reports/publication/tbl_final_utility_preserving_safety.tex",
                        "reports/publication/tbl_final_freeze_validation.tex",
                        "reports/publication/fig_final_training_picp90.png",
                        "reports/publication/fig_final_runtime_tsvr.png",
                        "reports/publication/fig_final_utility_delta.png",
                        "reports/publication/final_paper_results_summary.json",
                        "reports/publication/orius_book_final_results.pdf",
                    ]
                    rows = []
                    for relpath in required:
                        path = ROOT / relpath
                        rows.append({"artifact": relpath, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0})
                    qa = pd.DataFrame(rows)
                    qa
                    """
                ),
                _code(
                    """
                    assert qa["exists"].all()
                    assert (qa["bytes"] > 0).all()
                    summary = read_json("reports/publication/final_paper_results_summary.json")
                    summary
                    """
                ),
            ],
        },
    ]


def _inventory_rows(specs: list[dict[str, Any]]) -> list[dict[str, str]]:
    existing = [
        ("01_eda.ipynb", "legacy_battery_eda", "battery data exploration", "active_legacy"),
        ("02_baselines.ipynb", "legacy_battery_forecasting", "battery baselines", "active_legacy"),
        ("03_feature_pipeline.ipynb", "legacy_feature_pipeline", "feature construction", "active_legacy"),
        ("04_train_models.ipynb", "legacy_training", "battery model training", "active_legacy"),
        ("05_inference_intervals.ipynb", "legacy_uncertainty", "interval analysis", "active_legacy"),
        ("06_error_analysis.ipynb", "legacy_error_analysis", "forecast residual analysis", "active_legacy"),
        ("07_production_run.ipynb", "legacy_runbook", "production runbook", "active_legacy"),
        ("08_weather_features.ipynb", "legacy_weather_features", "optional weather features", "active_legacy"),
        ("09_walk_forward_report.ipynb", "legacy_backtest", "walk-forward report", "active_legacy"),
        ("10_optimization_engine.ipynb", "legacy_optimization", "dispatch optimization", "active_legacy"),
        ("11_monitoring_drift.ipynb", "legacy_monitoring", "drift monitoring", "active_legacy"),
        ("12_api_dashboard_smoke_test.ipynb", "legacy_ui_api", "API/dashboard smoke test", "active_legacy"),
        ("13_runbook_end_to_end.ipynb", "legacy_reproducibility", "end-to-end runbook", "active_legacy"),
        ("14_de_us_gap_analysis.ipynb", "battery_data_gap", "DE/US battery data gap", "active"),
        ("15_av_domain_validation.ipynb", "av_domain_validation", "AV runtime validation", "active"),
        ("17_healthcare_domain_validation.ipynb", "healthcare_domain_validation", "Healthcare runtime validation", "active"),
        ("19_universal_theorem_visualization.ipynb", "theorem_visualization", "universal theorem visualization", "active"),
    ]
    rows = [
        {
            "notebook": f"notebooks/{name}",
            "claim_surface": claim,
            "primary_artifacts": artifacts,
            "status": status,
        }
        for name, claim, artifacts, status in existing
    ]
    generated_artifacts = {
        "00_orius_research_notebook_index.ipynb": "reports/publication/orius_research_notebook_inventory.csv",
        "20_final_release_results_analysis.ipynb": "reports/publication/final_*_for_paper.csv | reports/publication/tbl_final_*.tex",
        "21_utility_preserving_safety_analysis.ipynb": "reports/publication/utility_preserving_safety_scorecard.csv",
        "22_theorem_audit_traceability.ipynb": "reports/publication/active_theorem_audit.csv | reports/publication/theorem_result_cards/*.json",
        "23_freeze_release_reproducibility_audit.ipynb": "reports/split_training/latest_release_id.txt | reports/predeployment_freeze/<release>/*.json",
        "24_publication_package_quality_audit.ipynb": "reports/publication/tbl_final_*.tex | reports/publication/fig_final_*.png | reports/publication/orius_book_final_results.pdf",
    }
    for spec in specs:
        name = Path(spec["path"]).name
        rows.append(
            {
                "notebook": f"notebooks/{name}",
                "claim_surface": str(spec["title"]),
                "primary_artifacts": generated_artifacts.get(name, ""),
                "status": "active",
            }
        )
    return rows


def _write_inventory(rows: list[dict[str, str]]) -> None:
    csv_path = PUBLICATION_DIR / "orius_research_notebook_inventory.csv"
    md_path = NOTEBOOK_DIR / "README.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["notebook", "claim_surface", "primary_artifacts", "status"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# ORIUS Research Notebooks",
        "",
        "These notebooks are analysis surfaces over locked artifacts. They are not the source of truth for release claims; the CSV/JSON/TeX/PDF artifacts under `reports/publication/` and the freeze artifacts remain authoritative.",
        "",
        "| Notebook | Claim surface | Primary artifacts | Status |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['notebook']}` | {row['claim_surface']} | `{row['primary_artifacts']}` | {row['status']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_notebooks() -> list[dict[str, str]]:
    specs = _notebook_specs()
    for spec in specs:
        nb = _build_notebook(spec["title"], spec["objective"], spec["cells"])
        _write_notebook(spec["path"], nb)
    rows = _inventory_rows(specs)
    _write_inventory(rows)
    return rows


def main() -> int:
    rows = build_notebooks()
    print(f"[build_orius_research_notebooks] wrote {len(rows)} inventory rows at notebooks/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
