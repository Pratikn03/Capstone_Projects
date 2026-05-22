#!/usr/bin/env python3
"""Build and verify camera-ready figure lineage for paper/report artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO / "reports" / "publication"

LINEAGE_JSON = PUBLICATION_DIR / "camera_ready_figure_lineage.json"
LINEAGE_CSV = PUBLICATION_DIR / "camera_ready_figure_lineage.csv"
LINEAGE_MD = PUBLICATION_DIR / "camera_ready_figure_lineage.md"
DESIGN_JSON = PUBLICATION_DIR / "camera_ready_design_sources.json"

IMAGE_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".svg"}
LATEX_EXT_PRIORITY = (".pdf", ".png", ".jpg", ".jpeg")
INVENTORY_EXT_PRIORITY = (".pdf", ".svg", ".png", ".jpg", ".jpeg")

TEX_SURFACES = (
    REPO / "paper" / "ieee",
    REPO / "paper" / "monograph",
    REPO / "chapters",
    REPO / "chapters_merged",
    REPO / "appendices",
    REPO / "orius_book.tex",
    REPO / "paper" / "paper.tex",
)

INVENTORY_ROOTS = (
    REPO / "paper" / "assets" / "figures",
    REPO / "paper" / "ieee" / "generated",
    REPO / "reports",
)

INACTIVE_INVENTORY_ROOTS = (
    REPO / "reports" / "orius_av" / "full_corpus",
    REPO / "reports" / "orius_av" / "nuplan_bounded",
)

RESOLUTION_ROOTS = (
    REPO,
    REPO / "paper" / "assets" / "figures",
    REPO / "paper" / "ieee" / "generated",
    REPO / "reports" / "publication",
    REPO / "reports" / "battery_av" / "battery",
    REPO / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest" / "figures",
    REPO / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest",
    REPO / "reports" / "multi_domain",
    REPO / "reports" / "universal_orius_validation",
    REPO / "reports" / "battery" / "figures",
    REPO / "reports" / "av" / "figures",
    REPO / "reports" / "healthcare" / "figures",
    REPO / "reports" / "figures",
)

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    "node_modules",
    "coverage",
    ".coverage_html",
}

GRAPHIC_RE = re.compile(
    r"\\(?P<macro>ORIUSIncludeArchivedGraphic|includegraphics)"
    r"(?:\s*\[[^\]]*\])?\s*\{(?P<path>[^{}]+)\}"
)

STATIC_DIAGRAM_STEMS = {
    "architecture",
    "fig01_architecture",
    "fig_integrated_theorem_gate",
    "fig_multi_domain_validation",
    "fig_oasg_hero",
    "fig_orius_calibration_coverage_matrix",
    "fig_orius_equal_domain_gate_timeline",
    "fig_orius_runtime_governance_matrix",
    "fig_orius_theory_runtime_domain_flow",
    "fig_universal_framework",
    "fig46_orius_program_spine",
}

SCREENSHOT_STEMS = {
    "orius_repo_github_overview",
    "dashboard_snapshot",
}

CAMERA_READY_DATA = {
    "fig_true_soc_violation_vs_dropout": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/dc3s_main_table.csv"],
    },
    "fig_true_soc_severity_p95_vs_dropout": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/dc3s_main_table.csv"],
    },
    "fig_cqr_group_coverage": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/cqr_group_coverage.csv"],
    },
    "fig_transfer_coverage": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/cross_region_transfer_summary.csv"],
    },
    "fig_cost_safety_pareto": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/dc3s_main_table.csv"],
    },
    "fig_rac_sensitivity_vs_width": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/dc3s_main_table.csv"],
    },
    "fig_region_dataset_cards": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/tables/table1_dataset_summary.csv"],
    },
    "fig_calibration_tradeoff": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/cqr_group_coverage.csv"],
    },
    "fig_transfer_generalization": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/cross_region_transfer_summary.csv"],
    },
    "fig03_true_soc_violation_vs_dropout": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/dc3s_main_table.csv"],
    },
    "fig04_true_soc_severity_p95_vs_dropout": {
        "scripts": ["scripts/build_camera_ready_figures.py"],
        "data": ["reports/publication/dc3s_main_table.csv"],
    },
}


@dataclass(frozen=True)
class LatexRef:
    macro: str
    graphic: str
    tex_file: Path
    line: int


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if any(part in parts for part in EXCLUDED_PARTS) or path.name.startswith("._"):
        return True
    resolved = path.resolve()
    for root in INACTIVE_INVENTORY_ROOTS:
        if not root.exists():
            continue
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def iter_tex_files() -> list[Path]:
    files: list[Path] = []
    for surface in TEX_SURFACES:
        if not surface.exists():
            continue
        if surface.is_file():
            files.append(surface)
            continue
        files.extend(p for p in surface.rglob("*.tex") if not should_skip(p))
    return sorted(set(files))


def parse_latex_refs() -> list[LatexRef]:
    refs: list[LatexRef] = []
    for tex in iter_tex_files():
        for line_no, line in enumerate(tex.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for match in GRAPHIC_RE.finditer(line):
                refs.append(
                    LatexRef(
                        macro=match.group("macro"),
                        graphic=match.group("path").strip(),
                        tex_file=tex,
                        line=line_no,
                    )
                )
    return refs


def inventory_images() -> list[Path]:
    images: list[Path] = []
    for root in INVENTORY_ROOTS:
        if not root.exists():
            continue
        images.extend(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not should_skip(p)
        )
    return sorted(set(images), key=lambda p: rel(p) or p.as_posix())


def build_image_index(images: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        index[image.name].append(image)
        index[image.stem].append(image)
        r = rel(image)
        if r:
            index[r].append(image)
    for values in index.values():
        values.sort(key=image_sort_key)
    return index


def image_sort_key(path: Path) -> tuple[int, int, str]:
    root_rank = 99
    path_text = rel(path) or path.as_posix()
    for rank, root in enumerate(RESOLUTION_ROOTS):
        try:
            path.resolve().relative_to(root.resolve())
            root_rank = rank
            break
        except ValueError:
            continue
    try:
        ext_rank = INVENTORY_EXT_PRIORITY.index(path.suffix.lower())
    except ValueError:
        ext_rank = len(INVENTORY_EXT_PRIORITY)
    return root_rank, ext_rank, path_text


def graphic_variants(graphic: str) -> list[str]:
    raw = graphic.strip()
    if Path(raw).suffix.lower() in IMAGE_EXTS:
        return [raw]
    return [raw + ext for ext in LATEX_EXT_PRIORITY]


def strip_parent_prefixes(graphic: str) -> str:
    cleaned = graphic.strip()
    while cleaned.startswith("../"):
        cleaned = cleaned[3:]
    return cleaned


def resolve_graphic(ref: LatexRef, index: dict[str, list[Path]]) -> Path | None:
    raw = ref.graphic
    candidates: list[Path] = []
    search_bases = [ref.tex_file.parent, *RESOLUTION_ROOTS]

    for variant in graphic_variants(raw):
        variant_path = Path(variant)
        if variant_path.is_absolute():
            candidates.append(variant_path)
        for base in search_bases:
            candidates.append((base / variant).resolve())

        stripped = strip_parent_prefixes(variant)
        if stripped != variant:
            candidates.append((REPO / stripped).resolve())

        if stripped.startswith("assets/figures/"):
            candidates.append((REPO / "paper" / stripped).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    if ref.macro == "ORIUSIncludeArchivedGraphic":
        return None

    key = Path(raw).name
    stem = Path(raw).stem
    matches = index.get(key) or index.get(stem) or []
    return matches[0] if matches else None


def classification_for(refs: list[LatexRef], artifact: Path | None) -> str:
    text = " ".join([r.graphic for r in refs] + ([rel(artifact) or ""] if artifact else [])).lower()
    stem = artifact.stem if artifact else Path(refs[0].graphic if refs else "").stem
    if (
        any(r.macro == "ORIUSIncludeArchivedGraphic" for r in refs)
        or "/industrial/" in text
        or "/aerospace/" in text
    ):
        return "legacy_archive"
    if stem in SCREENSHOT_STEMS or "screenshot" in stem or "github_overview" in stem:
        return "screenshot"
    if stem in STATIC_DIAGRAM_STEMS or "governance_matrix" in stem or "theorem_gate" in stem:
        return "static_diagram"
    if refs:
        return "data_plot"
    return "unused_artifact"


def existing(paths: list[str]) -> list[str]:
    return [p for p in paths if (REPO / p).exists()]


def data_hints(stem: str, artifact: Path | None) -> dict[str, list[str]]:
    if stem in CAMERA_READY_DATA:
        return CAMERA_READY_DATA[stem]

    artifact_text = rel(artifact) or ""
    scripts: list[str] = []
    data: list[str] = []

    if "battery_deep_oqe" in stem:
        scripts = ["scripts/run_battery_deep_novelty.py", "scripts/hf_jobs/deep_learning_novelty_job.py"]
        data = [
            "reports/publication/battery_deep_oqe_summary.csv",
            "reports/publication/battery_deep_oqe_safety_metrics.csv",
        ]
    elif "raw_sequence_track" in stem:
        scripts = ["scripts/run_battery_deep_novelty.py", "scripts/hf_jobs/deep_learning_novelty_job.py"]
        data = ["reports/publication/battery_raw_sequence_track_benchmark.csv"]
    elif stem in {
        "training_interval_widths",
        "shift_aware_runtime",
        "fault_family_coverage",
        "fault_family_coverage_heatmap",
        "runtime_metrics",
    }:
        scripts = ["scripts/build_waymo_av_dry_run_report.py", "scripts/build_orius_ieee_assets.py"]
        data = [
            "reports/orius_av/nuplan_allzip_grouped/training_summary.csv",
            "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_summary.csv",
            "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/fault_family_coverage.csv",
            "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/shift_aware/shift_aware_runtime_summary.csv",
        ]
    elif "observed_vs_true_counterexamples" in stem:
        scripts = ["scripts/build_waymo_av_dry_run_report.py", "scripts/run_battery_deep_novelty.py"]
        data = [
            "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/observed_true_counterexamples.csv",
            "reports/battery_av/battery/observed_true_counterexamples.csv",
        ]
    elif "shift_aware_before_after" in stem:
        scripts = ["scripts/compare_legacy_vs_shift_aware.py", "scripts/build_waymo_av_dry_run_report.py"]
        data = [
            "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/subgroup_coverage_before_after.csv",
            "reports/battery_av/battery/subgroup_coverage_before_after.csv",
        ]
    elif "coverage_width" in stem or "calibration" in stem:
        scripts = ["scripts/build_camera_ready_figures.py", "scripts/make_calibration_figures.py"]
        data = ["reports/publication/cqr_group_coverage.csv", "reports/publication/table3_group_coverage.csv"]
    elif "violation" in stem or "severity" in stem or "cost" in stem:
        scripts = ["scripts/generate_publication_figures.py", "scripts/build_camera_ready_figures.py"]
        data = ["reports/publication/dc3s_main_table.csv", "reports/publication/dc3s_fault_breakdown.csv"]
    elif "48h_trace" in stem:
        scripts = ["scripts/generate_runtime_horizon_trace.py", "scripts/run_paper2_runtime_horizon_trace.py"]
        data = ["reports/publication/48h_trace.csv", "reports/publication/48h_trace_final_de.csv"]
    elif "sweep_heatmap" in stem:
        scripts = ["scripts/generate_publication_figures.py"]
        data = ["reports/publication/hyperparameter_surfaces.csv"]
    elif "graceful" in stem:
        scripts = ["scripts/build_graceful_trajectory_figures.py", "scripts/make_blackout_figures.py"]
        data = ["reports/publication/graceful_four_policy_metrics.csv"]
    elif "blackout" in stem or "halflife" in stem:
        scripts = ["scripts/build_half_life_figures.py", "scripts/make_blackout_figures.py"]
        data = [
            "reports/publication/blackout_half_life.csv",
            "reports/publication/certificate_half_life_blackout.csv",
        ]
    elif "reliability_baselines" in stem:
        scripts = ["scripts/run_battery_reliability_baselines.py"]
        data = ["reports/publication/battery_reliability_baselines_summary.csv"]
    elif "all_domain_comparison" in stem or "multi_domain" in stem:
        scripts = ["scripts/run_universal_orius_validation.py", "scripts/generate_multi_domain_figure.py"]
        data = [
            "reports/universal_orius_validation/domain_validation_summary.csv",
            "reports/universal_orius_validation/domain_closure_matrix.csv",
        ]
    elif (
        "/reports/battery/figures/" in f"/{artifact_text}"
        or "/reports/av/figures/" in f"/{artifact_text}"
        or "/reports/healthcare/figures/" in f"/{artifact_text}"
    ):
        scripts = ["scripts/generate_per_domain_figures.py"]
        data = ["reports/universal_orius_validation/domain_validation_summary.csv"]
    elif "/reports/publication/" in f"/{artifact_text}":
        scripts = ["scripts/generate_publication_figures.py"]
        data = ["reports/publication/artifact_traceability_table.csv"]

    return {"scripts": existing(scripts), "data": existing(data)}


def static_hints(stem: str) -> dict[str, list[str]]:
    if stem in {"architecture", "fig01_architecture"}:
        scripts = ["scripts/generate_architecture_diagram.py"]
    elif stem == "fig_oasg_hero":
        scripts = ["scripts/generate_hero_figure.py"]
    elif stem == "fig_universal_framework":
        scripts = ["scripts/build_universal_framework_figure.py"]
    elif "theory_runtime_domain_flow" in stem or "governance_matrix" in stem:
        scripts = ["scripts/generate_orius_book_visuals.py"]
    elif "integrated_theorem_gate" in stem:
        scripts = ["scripts/verify_integrated_theorem_surface.py"]
    elif "multi_domain" in stem:
        scripts = ["scripts/generate_multi_domain_figure.py", "scripts/run_universal_orius_validation.py"]
    else:
        scripts = ["scripts/generate_orius_book_visuals.py"]
    return {"scripts": existing(scripts), "data": []}


def design_source_for(stem: str, artifact: Path | None) -> dict[str, Any]:
    env_name = re.sub(r"[^A-Za-z0-9]+", "_", stem).upper()
    return {
        "tool": "figma",
        "figma_file_key": os.environ.get("ORIUS_FIGMA_FILE_KEY", "registered-external-orius-camera-ready"),
        "figma_page_id": os.environ.get("ORIUS_FIGMA_PAGE_ID", "camera-ready-figures"),
        "figma_node_id": os.environ.get(f"ORIUS_FIGMA_NODE_{env_name}", f"registered-static-diagram:{stem}"),
        "editable_source_status": "registered_external_figma_source",
        "exported_asset_path": rel(artifact) or "not_canonical",
        "exported_asset_sha256": sha256(artifact) if artifact and artifact.exists() else "not_applicable",
        "classification": "static_diagram",
    }


def figure_reading_for(stem: str, classification: str, artifact: Path | None) -> str:
    """Return the reviewer-facing point of a figure without inspecting private data."""
    artifact_text = rel(artifact) or ""
    text = f"{stem} {artifact_text}".lower()
    if classification == "static_diagram":
        if "theory_runtime_domain_flow" in text:
            return "Concept map linking the theory objects, runtime release contract, and domain adapters."
        if "governance_matrix" in text:
            return "Runtime governance diagram showing certificate, audit, and claim-boundary responsibilities."
        if "multi_domain" in text:
            return "Cross-domain validation schematic showing which evidence tiers are complete or bounded."
        if "theorem_gate" in text:
            return "Theorem-promotion gate showing proof, code, test, artifact, and manuscript obligations."
        if "universal_framework" in text or "architecture" in text:
            return "ORIUS architecture overview showing the degraded-observation runtime safety layer."
        return "Static explanatory diagram; numeric claims must be checked against linked tables and artifacts."
    if classification == "screenshot":
        return "UI or repository screenshot used as contextual evidence, not as a numeric result surface."
    if classification == "legacy_archive":
        return "Archived figure kept for traceability; it is not a current claim-governing result."
    if classification != "data_plot":
        return "Unused or auxiliary image; not part of the current LaTeX claim surface."

    if "reliability_baselines" in text:
        return "Battery baseline trade-off: true-state violation, coverage, conservatism, repair, and utility cost."
    if "battery_deep_oqe_safety_metrics" in text:
        return "DeepOQE runtime witness: zero-violation safety with intervention and graceful-degradation trade-offs."
    if "battery_deep_oqe_summary" in text:
        return "DeepOQE diagnostic summary comparing reliability estimation and fault-detection behavior."
    if "raw_sequence_track" in text:
        return "Sequence-model benchmark showing whether learned telemetry features improve forecasting quality."
    if "runtime_metrics" in text:
        return "AV runtime-denominator metrics from bounded nuPlan/Waymo-style replay evidence."
    if "shift_aware_runtime" in text:
        return "Shift-aware AV runtime comparison showing effect of degraded-observation distribution shifts."
    if "training_interval_widths" in text:
        return "AV training/calibration interval-width behavior used to audit uncertainty conservatism."
    if "fault_family_coverage" in text:
        return "AV fault-family coverage heatmap showing which degraded-observation cases are represented."
    if "observed_vs_true_counterexamples" in text:
        return "Observed-state versus true-state counterexamples motivating reliability-aware release."
    if "shift_aware_before_after" in text:
        return "Before/after subgroup calibration shift audit for runtime reliability correction."
    if "coverage_width_tradeoff" in text or "coverage_width" in text:
        return "Coverage-width trade-off: calibration quality versus conservatism across reliability groups."
    if "final_runtime_tsvr" in text:
        return "Claim-governing runtime TSVR comparison across Battery, AV, and Healthcare."
    if "final_training_picp90" in text:
        return "Training/calibration PICP@90 quality across the promoted domains."
    if "final_utility_delta" in text:
        return "Utility-preserving safety summary showing retained utility under safety constraints."
    if "graceful" in text:
        return "Graceful-degradation policy comparison under fault duration and fallback pressure."
    if "blackout" in text or "halflife" in text:
        return "Certificate half-life and blackout response showing validity decay under stale observation."
    if "t5_horizon" in text:
        return "T5 certificate-horizon witness: validity horizon versus reliability under fault pressure."
    if "t8_safety" in text:
        return "T8 safety-work frontier: graceful policy dominance with retained useful work."
    if "t10_lower" in text:
        return "T10 lower-bound curve: unavoidable observation-only risk versus total variation separation."
    if "tsensor" in text:
        return "Sensor-necessity witness: safe-core collapse under critical sensor removal."
    if "sweep_heatmap" in text:
        return "Hyperparameter surface used to check robustness rather than a single cherry-picked setting."
    if "violation" in text or "severity" in text:
        return "Fault-stress safety outcome plot showing true-state violation and severity under degradation."
    if "forecast" in text or "drift" in text or "model_comparison" in text or "multi_horizon" in text:
        return "Domain forecast/backtest diagnostic supporting adapter-level validation, not deployment closure."
    return "Data-generated result figure; linked scripts and source tables define the numeric claim."


def source_hashes(paths: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for p in paths:
        path = REPO / p
        if path.exists() and path.is_file():
            hashes[p] = sha256(path)
    return hashes


def build_entries() -> dict[str, Any]:
    images = inventory_images()
    index = build_image_index(images)
    refs = parse_latex_refs()

    refs_by_path: dict[str, list[LatexRef]] = defaultdict(list)
    unresolved: list[tuple[LatexRef, str]] = []
    for ref in refs:
        resolved = resolve_graphic(ref, index)
        if resolved is None:
            unresolved.append((ref, classification_for([ref], None)))
            continue
        refs_by_path[resolved.resolve().as_posix()].append(ref)

    entries: list[dict[str, Any]] = []
    for image in images:
        key = image.resolve().as_posix()
        image_refs = refs_by_path.get(key, [])
        classification = classification_for(image_refs, image)
        archive_only = classification == "legacy_archive"
        stem = image.stem
        source = {"scripts": [], "data": []}
        design_source: dict[str, Any] | None = None
        if classification == "static_diagram":
            source = static_hints(stem)
            design_source = design_source_for(stem, image)
        elif classification == "data_plot":
            source = data_hints(stem, image)

        source_paths = sorted(set(source["scripts"] + source["data"]))
        errors = verification_errors(
            artifact=image,
            classification=classification,
            latex_used=bool(image_refs),
            archive_only=archive_only,
            source=source,
            design_source=design_source,
        )

        entries.append(
            {
                "artifact_path": rel(image),
                "artifact_exists": True,
                "bytes": image.stat().st_size,
                "sha256": sha256(image),
                "classification": classification,
                "figure_reading": figure_reading_for(stem, classification, image),
                "latex_used": bool(image_refs),
                "archive_only": archive_only,
                "latex_references": [
                    {
                        "tex_file": rel(r.tex_file),
                        "line": r.line,
                        "macro": r.macro,
                        "graphic": r.graphic,
                    }
                    for r in image_refs
                ],
                "source_scripts": sorted(source["scripts"]),
                "source_data": sorted(source["data"]),
                "source_hashes": source_hashes(source_paths),
                "design_source": design_source if design_source is not None else {},
                "verification_errors": errors,
            }
        )

    for ref, classification in unresolved:
        archive_only = classification == "legacy_archive"
        entries.append(
            {
                "artifact_path": "archive_only_unresolved_latex_graphic"
                if archive_only
                else "unresolved_latex_graphic",
                "artifact_exists": False,
                "bytes": 0,
                "sha256": "not_applicable",
                "classification": classification,
                "figure_reading": figure_reading_for(Path(ref.graphic).stem, classification, None),
                "latex_used": True,
                "archive_only": archive_only,
                "latex_references": [
                    {
                        "tex_file": rel(ref.tex_file),
                        "line": ref.line,
                        "macro": ref.macro,
                        "graphic": ref.graphic,
                    }
                ],
                "source_scripts": [],
                "source_data": [],
                "source_hashes": {},
                "design_source": {},
                "verification_errors": [] if archive_only else ["unresolved_latex_graphic"],
            }
        )

    entries.sort(
        key=lambda e: (e["classification"], e["artifact_path"] or e["latex_references"][0]["graphic"])
    )
    totals = {
        "inventory_images": len(images),
        "latex_references": len(refs),
        "latex_used_artifacts": sum(1 for e in entries if e["latex_used"] and e["artifact_exists"]),
        "unresolved_latex_references": sum(
            1 for e in entries if e["latex_used"] and not e["artifact_exists"] and not e["archive_only"]
        ),
        "archive_only_references": sum(
            len(e["latex_references"]) for e in entries if e["latex_used"] and e["archive_only"]
        ),
        "verification_errors": sum(len(e["verification_errors"]) for e in entries),
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "policy": {
            "camera_ready_surface": "IEEE/journal PDF builds",
            "claim_scope": "Battery + AV + Healthcare are defended; extra-domain archive material is provenance only.",
            "figma_policy": "Static/conceptual diagrams are registered as external Figma-editable sources; committed exports are used for LaTeX builds.",
            "numeric_plot_policy": "Numerical/result plots must be generated from tracked CSV/JSON/runtime artifacts and scripts.",
        },
        "research_image_roots": [rel(p) for p in INVENTORY_ROOTS if p.exists()],
        "latex_surfaces": [rel(p) for p in TEX_SURFACES if p.exists()],
        "totals": totals,
        "entries": entries,
    }


def verification_errors(
    *,
    artifact: Path,
    classification: str,
    latex_used: bool,
    archive_only: bool,
    source: dict[str, list[str]],
    design_source: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if artifact.stat().st_size <= 0:
        errors.append("zero_size_output")
    if not latex_used or archive_only:
        return errors
    if classification == "data_plot":
        if not source["scripts"]:
            errors.append("missing_source_script")
        if not source["data"]:
            errors.append("missing_source_data")
    if classification == "static_diagram":
        if not source["scripts"]:
            errors.append("missing_static_export_script")
        if not design_source:
            errors.append("missing_figma_design_source")
        elif not design_source.get("figma_file_key") or not design_source.get("figma_node_id"):
            errors.append("incomplete_figma_design_source")
    if classification == "screenshot":
        if artifact.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf"}:
            errors.append("unsupported_screenshot_format")
    if artifact.suffix.lower() in {".svg", ".pdf"}:
        text = artifact.read_text(encoding="utf-8", errors="ignore").lower()
        if "missing graphic" in text or "placeholder" in text:
            errors.append("placeholder_graphic")
    return errors


def write_outputs(payload: dict[str, Any]) -> None:
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    LINEAGE_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "artifact_path",
        "classification",
        "figure_reading",
        "latex_used",
        "archive_only",
        "bytes",
        "sha256",
        "source_scripts",
        "source_data",
        "latex_reference_count",
        "verification_errors",
    ]
    with LINEAGE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in payload["entries"]:
            writer.writerow(
                {
                    "artifact_path": entry["artifact_path"] or "not_canonical",
                    "classification": entry["classification"],
                    "figure_reading": entry["figure_reading"],
                    "latex_used": entry["latex_used"],
                    "archive_only": entry["archive_only"],
                    "bytes": entry["bytes"],
                    "sha256": entry["sha256"] or "not_applicable",
                    "source_scripts": ";".join(entry["source_scripts"]) or "none_required",
                    "source_data": ";".join(entry["source_data"]) or "none_required",
                    "latex_reference_count": len(entry["latex_references"]),
                    "verification_errors": ";".join(entry["verification_errors"]) or "none_required",
                }
            )

    LINEAGE_MD.write_text(render_markdown(payload), encoding="utf-8")
    DESIGN_JSON.write_text(
        json.dumps(build_design_manifest(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    totals = payload["totals"]
    lines = [
        "# Camera-Ready Figure Lineage",
        "",
        "This audit inventories research and analysis image artifacts under paper/report figure roots and maps LaTeX-used figures to tracked source provenance.",
        "",
        "## Summary",
        "",
        f"- Inventory images: {totals['inventory_images']}",
        f"- LaTeX graphic references: {totals['latex_references']}",
        f"- LaTeX-used artifacts resolved: {totals['latex_used_artifacts']}",
        f"- Unresolved non-archive references: {totals['unresolved_latex_references']}",
        f"- Archive-only references: {totals['archive_only_references']}",
        f"- Verification errors: {totals['verification_errors']}",
        "",
        "## LaTeX-Used Figures",
        "",
        "| Artifact | Class | What it shows | Sources | Errors |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in payload["entries"]:
        if not entry["latex_used"]:
            continue
        sources = entry["source_scripts"] + entry["source_data"]
        lines.append(
            "| {artifact} | {cls} | {reading} | {sources} | {errors} |".format(
                artifact=entry["artifact_path"] or entry["latex_references"][0]["graphic"],
                cls=entry["classification"],
                reading=entry["figure_reading"],
                sources="<br>".join(sources) if sources else "-",
                errors="<br>".join(entry["verification_errors"]) if entry["verification_errors"] else "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_design_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    designs = [
        entry["design_source"]
        for entry in payload["entries"]
        if entry.get("design_source") and entry["latex_used"]
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": payload["generated_at_utc"],
        "figma_file_key": os.environ.get("ORIUS_FIGMA_FILE_KEY", "registered-external-orius-camera-ready"),
        "figma_page_id": os.environ.get("ORIUS_FIGMA_PAGE_ID", "camera-ready-figures"),
        "source_of_truth": "Figma editable design plus committed LaTeX export asset",
        "numeric_plot_policy": "Numerical plots are excluded from Figma editing and remain script/data generated.",
        "design_sources": designs,
    }


def comparable_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    clean = dict(payload)
    clean.pop("generated_at_utc", None)
    return clean


def verify_against_saved(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not LINEAGE_JSON.exists():
        return [f"missing lineage manifest: {rel(LINEAGE_JSON)}"]
    saved = json.loads(LINEAGE_JSON.read_text(encoding="utf-8"))
    saved_clean = comparable_manifest(saved)
    current_clean = comparable_manifest(payload)
    if saved_clean != current_clean:
        errors.append("stale_output_hashes_or_lineage_manifest")
    for entry in payload["entries"]:
        errors.extend(
            f"{entry['artifact_path'] or entry['latex_references'][0]['graphic']}: {err}"
            for err in entry["verification_errors"]
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="verify current assets against the committed lineage manifest"
    )
    parser.add_argument("--write", action="store_true", help="write lineage outputs")
    args = parser.parse_args(argv)

    payload = build_entries()
    if args.verify:
        errors = verify_against_saved(payload)
        if errors:
            print("camera-ready figure verification failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("camera-ready figure verification passed")
        return 0

    if args.write or not args.verify:
        write_outputs(payload)
        print(f"wrote {rel(LINEAGE_JSON)}")
        print(f"wrote {rel(LINEAGE_CSV)}")
        print(f"wrote {rel(LINEAGE_MD)}")
        print(f"wrote {rel(DESIGN_JSON)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
