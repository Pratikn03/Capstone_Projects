#!/usr/bin/env python3
"""Generate parity and flow visuals for the ORIUS universal-first monograph."""
from __future__ import annotations

import csv
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "assets" / "figures"
PARITY_CSV = PUBLICATION_DIR / "orius_equal_domain_parity_matrix.csv"
CALIBRATION_CSV = PUBLICATION_DIR / "orius_calibration_diagnostics_matrix.csv"
RUNTIME_CSV = PUBLICATION_DIR / "orius_runtime_budget_matrix.csv"
GOVERNANCE_CSV = PUBLICATION_DIR / "orius_governance_lifecycle_matrix.csv"
GATE_LEDGER_CSV = PUBLICATION_DIR / "orius_equal_domain_gate_ledger.csv"
REFRESH_EXECUTION_JSON = PUBLICATION_DIR / "orius_refresh_execution.json"


STATUS_COLOR = {
    "pass": "#8fd19e",
    "verified": "#8fd19e",
    "locked_reference": "#77b7ff",
    "reference_witness": "#77b7ff",
    "safe_hold_validated": "#8fd19e",
    "evaluated": "#8fd19e",
    "proof_validated": "#8fd19e",
    "reference": "#77b7ff",
    "bounded_runtime_pass": "#cfe8a9",
    "bounded_brake_fallback_validated": "#cfe8a9",
    "pass_under_ttc_entry_barrier_contract": "#cfe8a9",
    "gated": "#ffe082",
    "gated_pending_shared_constraint_surface": "#ffe082",
    "portability_only": "#ffe082",
    "shadow_synthetic": "#ffd54f",
    "blocked_real_data_gap": "#ffb3b3",
    "placeholder_surface": "#ffb3b3",
    "experimental_replay_only": "#ffb3b3",
    "experimental_placeholder": "#ffb3b3",
    "experimental": "#ffb3b3",
}


def _sanitize(label: str) -> str:
    return label.replace("_", " ")


def _cell_color(value: str) -> str:
    return STATUS_COLOR.get(value, "#f2f2f2")


def _write_png(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")


def build_parity_figure() -> None:
    with PARITY_CSV.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    col_specs = [
        ("domain", "Domain"),
        ("dataset_raw_source_status", "Raw"),
        ("processed_train_validation_status", "Train"),
        ("replay_status", "Replay"),
        ("safe_action_soundness", "Soundness"),
        ("fallback_semantics", "Fallback"),
        ("certos_lifecycle_support", "CertOS"),
        ("multi_agent_support", "P5"),
        ("resulting_tier", "Tier"),
        ("exact_blocker", "Current Blocker"),
    ]
    headers = [label for _, label in col_specs]
    data = []
    colors = []
    for row in rows:
        row_text = []
        row_colors = []
        for key, _ in col_specs:
            raw = row[key]
            if key == "domain":
                row_text.append(fill(raw.replace(" and ", " /\n"), 20))
                row_colors.append("#e8f1fb")
            elif key == "exact_blocker":
                row_text.append(fill(_sanitize(raw), 24))
                row_colors.append("#fafafa")
            else:
                row_text.append(fill(_sanitize(raw), 16))
                row_colors.append(_cell_color(raw))
        data.append(row_text)
        colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(17, 5.8))
    ax.axis("off")
    ax.set_title(
        "ORIUS Equal-Domain Parity Gate\nUniversal-first structure with explicit evidence asymmetry",
        fontsize=16,
        weight="bold",
        pad=18,
    )
    col_widths = [0.16, 0.08, 0.08, 0.09, 0.13, 0.12, 0.09, 0.09, 0.08, 0.18]
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc="center",
        colColours=["#2c4f7c"] * len(headers),
        colWidths=col_widths,
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    table.scale(1, 2.2)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("white")
            continue
        cell.set_facecolor(colors[r - 1][c])
        cell.set_edgecolor("white")
        if c in (0, 9):
            cell.set_text_props(ha="left")

    ax.text(
        0.0,
        -0.11,
        "Green = defended or evaluated, Blue = reference witness, Amber = bounded/gated, Red = open blocker.",
        transform=ax.transAxes,
        fontsize=9,
        color="#334155",
    )

    _write_png(fig, PUBLICATION_DIR / "fig_orius_equal_domain_parity_matrix.png")
    _write_png(fig, PAPER_FIG_DIR / "fig_orius_equal_domain_parity_matrix.png")
    plt.close(fig)


def _box(ax, x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, weight="bold")
    return (x, y, w, h)


def _center_right(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w, y + h / 2)


def _center_left(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, _, h = box
    return (x, y + h / 2)


def build_flow_figure() -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title(
        "ORIUS Theory-to-Runtime-to-Domain Flow",
        fontsize=17,
        weight="bold",
        pad=16,
    )
    ax.text(
        0.5,
        0.935,
        "One universal safety argument, six domain instantiations, one parity gate",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )

    hazard = _box(ax, 0.03, 0.66, 0.15, 0.12, "Physical-AI Hazard\nunder degraded observation", "#eef2ff", "#4f46e5")
    theory = _box(ax, 0.24, 0.66, 0.16, 0.12, "Theory Bridge\nOASG, repair,\ntemporal validity", "#ecfeff", "#0891b2")
    runtime = _box(ax, 0.46, 0.66, 0.17, 0.12, "Runtime Kernel\nDetect · Calibrate ·\nConstrain · Shield · Certify", "#fff7ed", "#c2410c")
    domains = _box(ax, 0.69, 0.66, 0.14, 0.12, "Domain Adapters\nbattery · AV\nhealthcare", "#f0fdf4", "#15803d")
    parity = _box(ax, 0.86, 0.66, 0.11, 0.12, "Parity Gate\npromotion only\nby evidence", "#fef2f2", "#b91c1c")

    for src, dst in ((hazard, theory), (theory, runtime), (runtime, domains), (domains, parity)):
        ax.add_patch(
            FancyArrowPatch(
                _center_right(src),
                _center_left(dst),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.8,
                color="#1f2937",
            )
        )

    pills = [
        ("Battery", 0.16, "#77b7ff"),
        ("AV", 0.30, "#8fd19e"),
        ("Industrial", 0.44, "#8fd19e"),
        ("Healthcare", 0.58, "#8fd19e"),
        ("Navigation", 0.72, "#ffd54f"),
        ("Aerospace", 0.86, "#ffb3b3"),
    ]
    ax.text(0.06, 0.47, "Current domain status", fontsize=11, weight="bold", color="#0f172a")
    for label, x, fc in pills:
        pill = FancyBboxPatch(
            (x - 0.055, 0.39),
            0.11,
            0.055,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=fc,
            edgecolor="#334155",
            linewidth=1.2,
        )
        ax.add_patch(pill)
        ax.text(x, 0.417, label, ha="center", va="center", fontsize=9, weight="bold")

    note = _box(
        ax,
        0.06,
        0.16,
        0.88,
        0.14,
        "Universal-first book claim: ORIUS is one safety architecture with one runtime contract.\nEqual-domain rhetoric is governed by the parity gate: battery is the reference witness; industrial,\nhealthcare, and bounded AV are defended; navigation and aerospace remain explicitly open.",
        "#f8fafc",
        "#94a3b8",
    )
    _ = note

    _write_png(fig, PUBLICATION_DIR / "fig_orius_theory_runtime_domain_flow.png")
    _write_png(fig, PAPER_FIG_DIR / "fig_orius_theory_runtime_domain_flow.png")
    plt.close(fig)


def _binary_from_text(value: str) -> float:
    lowered = str(value).strip().lower()
    if not lowered:
        return 0.0
    blockers = ("blocked", "placeholder", "gated", "experimental", "portability_only", "shadow_synthetic")
    return 0.0 if any(token in lowered for token in blockers) else 1.0


def build_closure_timeline_figure() -> None:
    steps: list[dict[str, object]] = []
    if GATE_LEDGER_CSV.exists():
        with GATE_LEDGER_CSV.open("r", encoding="utf-8", newline="") as handle:
            steps = list(csv.DictReader(handle))
    elif REFRESH_EXECUTION_JSON.exists():
        import json

        payload = json.loads(REFRESH_EXECUTION_JSON.read_text(encoding="utf-8"))
        steps = payload.get("steps", [])
    if not steps:
        return

    labels = [str(step.get("step", step.get("label", ""))) for step in steps]
    values = [1 if str(step.get("ok", "")).lower() == "true" or step.get("ok") is True else 0 for step in steps]
    colors = ["#2e7d32" if value == 1 else "#c62828" for value in values]

    fig_height = max(3.6, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, [1] * len(labels), color=colors, edgecolor="white")
    ax.set_yticks(y_pos, labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["fail", "pass"])
    ax.set_title("ORIUS Equal-Domain Gate Timeline", fontsize=15, weight="bold", pad=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    for idx, value in enumerate(values):
        ax.text(0.5, idx, "pass" if value == 1 else "fail", ha="center", va="center", color="white", weight="bold", fontsize=9)

    _write_png(fig, PUBLICATION_DIR / "fig_orius_equal_domain_gate_timeline.png")
    _write_png(fig, PAPER_FIG_DIR / "fig_orius_equal_domain_gate_timeline.png")
    plt.close(fig)


def build_calibration_matrix_figure() -> None:
    if not CALIBRATION_CSV.exists():
        return
    rows = list(csv.DictReader(CALIBRATION_CSV.open("r", encoding="utf-8", newline="")))
    if not rows:
        return
    domains = [row["domain"] for row in rows]
    matrix = [
        [
            _binary_from_text(row.get("coverage_by_fault_mode", "")),
            _binary_from_text(row.get("coverage_by_oqe_bucket", "")),
            min(max(float(row.get("calibration_completeness_pct", "0") or 0.0), 0.0), 100.0) / 100.0,
        ]
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.7 * len(domains))))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(domains)), domains)
    ax.set_xticks(range(3), ["Fault slices", "OQE buckets", "Completeness"])
    ax.set_title("ORIUS Calibration Coverage Matrix", fontsize=15, weight="bold", pad=12)
    for row_idx, row in enumerate(rows):
        ax.text(2, row_idx, f"{float(row.get('calibration_completeness_pct', '0') or 0.0):.1f}%", ha="center", va="center", color="white", fontsize=9, weight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    _write_png(fig, PUBLICATION_DIR / "fig_orius_calibration_coverage_matrix.png")
    _write_png(fig, PAPER_FIG_DIR / "fig_orius_calibration_coverage_matrix.png")
    plt.close(fig)


def build_runtime_governance_matrix_figure() -> None:
    if not (RUNTIME_CSV.exists() and GOVERNANCE_CSV.exists()):
        return
    runtime_rows = list(csv.DictReader(RUNTIME_CSV.open("r", encoding="utf-8", newline="")))
    governance_rows = {row["domain"]: row for row in csv.DictReader(GOVERNANCE_CSV.open("r", encoding="utf-8", newline=""))}
    if not runtime_rows:
        return
    domains = [row["domain"] for row in runtime_rows]
    matrix = []
    for row in runtime_rows:
        governance = governance_rows.get(row["domain"], {})
        matrix.append(
            [
                _binary_from_text(row.get("certos_status", "")),
                _binary_from_text(governance.get("shared_constraint_status", "")),
                min(max(float(governance.get("governance_completeness_pct", "0") or 0.0), 0.0), 100.0) / 100.0,
            ]
        )
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.7 * len(domains))))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrBr", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(domains)), domains)
    ax.set_xticks(range(3), ["CertOS", "Shared constraint", "Completeness"])
    ax.set_title("ORIUS Runtime and Governance Matrix", fontsize=15, weight="bold", pad=12)
    for row_idx, domain in enumerate(domains):
        completeness = float(governance_rows.get(domain, {}).get("governance_completeness_pct", "0") or 0.0)
        ax.text(2, row_idx, f"{completeness:.1f}%", ha="center", va="center", color="white", fontsize=9, weight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    _write_png(fig, PUBLICATION_DIR / "fig_orius_runtime_governance_matrix.png")
    _write_png(fig, PAPER_FIG_DIR / "fig_orius_runtime_governance_matrix.png")
    plt.close(fig)


def main() -> None:
    build_parity_figure()
    build_flow_figure()
    build_closure_timeline_figure()
    build_calibration_matrix_figure()
    build_runtime_governance_matrix_figure()
    print("Generated ORIUS book visuals.")


if __name__ == "__main__":
    main()
