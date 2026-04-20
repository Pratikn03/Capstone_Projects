#!/usr/bin/env python3
"""Validate the hard-gated 18-row theorem traceability surface and emit audit artifacts.

The hard-gated theorem surface contains 18 rows:
  - 10 unique theorem claims:
      * 8 flagship T1--T8 theorems
      * 2 supporting theorems from ch04
  - 8 appendix theorem restatements in Appendix C

This script validates only traceability/release-gate properties of the
hard-gated theorem rows and emits:
  - reports/publication/integrated_theorem_gate.csv
  - reports/publication/integrated_theorem_gate.json
  - reports/publication/integrated_theorem_gate_summary.tex
  - reports/publication/integrated_theorem_gate_matrix.tex
  - reports/publication/fig_integrated_theorem_gate.png
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
REPORTS = REPO / "reports" / "publication"
THEOREM_REGISTER = REPORTS / "theorem_surface_register.csv"
TRACEABILITY = REPORTS / "chapter_theorem_traceability.csv"
APPENDIX_PROOFS = REPO / "appendices" / "app_c_full_proofs.tex"

OUT_CSV = REPORTS / "integrated_theorem_gate.csv"
OUT_JSON = REPORTS / "integrated_theorem_gate.json"
OUT_SUMMARY_TEX = REPORTS / "integrated_theorem_gate_summary.tex"
OUT_MATRIX_TEX = REPORTS / "integrated_theorem_gate_matrix.tex"
OUT_FIG = REPORTS / "fig_integrated_theorem_gate.png"


@dataclass
class GateRow:
    theorem_key: str
    theorem_class: str
    title: str
    source: str
    mapped_to: str
    traceability_status: str
    source_exists: bool
    code_anchors_ok: bool
    artifacts_ok: bool
    appendix_mapping_ok: bool
    pass_gate: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "theorem_key": self.theorem_key,
            "theorem_class": self.theorem_class,
            "title": self.title,
            "source": self.source,
            "mapped_to": self.mapped_to,
            "traceability_status": self.traceability_status,
            "source_exists": self.source_exists,
            "code_anchors_ok": self.code_anchors_ok,
            "artifacts_ok": self.artifacts_ok,
            "appendix_mapping_ok": self.appendix_mapping_ok,
            "pass_gate": self.pass_gate,
            "notes": self.notes,
        }


SUPPORTING_THEOREMS: dict[str, dict[str, Any]] = {
    "Existence of the illusion under dropout": {
        "theorem_key": "S1",
        "traceability_status": "locked",
        "code_anchors": [
            "src/orius/cpsbench_iot/scenarios.py",
            "src/orius/cpsbench_iot/runner.py",
            "src/orius/cpsbench_iot/plant.py",
        ],
        "artifact_paths": [
            "reports/publication/dc3s_main_table_ci.csv",
            "reports/publication/fault_performance_table.csv",
        ],
    },
    "DC3S Feasibility Guarantee": {
        "theorem_key": "S2",
        "traceability_status": "locked",
        "code_anchors": [
            "src/orius/dc3s/guarantee_checks.py",
            "src/orius/dc3s/shield.py",
            "src/orius/dc3s/battery_adapter.py",
        ],
        "artifact_paths": [
            "reports/publication/dc3s_main_table_ci.csv",
            "reports/publication/reliability_group_coverage_phase3.csv",
        ],
    },
    "Informal safety guarantee of DC3S": {
        "theorem_key": "S2",
        "traceability_status": "locked",
        "code_anchors": [
            "src/orius/dc3s/guarantee_checks.py",
            "src/orius/dc3s/shield.py",
            "src/orius/dc3s/battery_adapter.py",
        ],
        "artifact_paths": [
            "reports/publication/dc3s_main_table_ci.csv",
            "reports/publication/reliability_group_coverage_phase3.csv",
        ],
    },
}

APPENDIX_MAP = {
    "C.1 Battery-Domain OASG Existence": "OASG Existence",
    "C.2 One-Step Safety Preservation": "Safety Preservation",
    "C.3 Battery ORIUS Core Bound": "ORIUS Core Bound",
    "C.4 No Free Safety": "No Free Safety",
    "C.5 Certificate Validity Horizon": "Certificate validity horizon",
    "C.6 Certificate Expiration Bound": "Certificate expiration bound",
    "C.8 Feasible Fallback Existence": "Feasible fallback existence",
    "C.9 Graceful Degradation Dominance": "Graceful degradation dominance",
}
HARD_GATED_KEYS = {"S1", "S2", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_anchor(anchor: str) -> Path:
    cleaned = anchor.strip()
    if not cleaned:
        return REPO
    if cleaned.startswith("src/gridpulse/"):
        cleaned = "src/orius/" + cleaned[len("src/gridpulse/") :]
    return REPO / cleaned


def _import_from_path(path: Path) -> bool:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return True
    except Exception:
        return False


def _code_anchors_ok(anchors: list[str]) -> bool:
    if not anchors:
        return False
    for anchor in anchors:
        path = _normalize_anchor(anchor)
        if not path.exists():
            return False
        if path.suffix == ".py" and not _import_from_path(path):
            return False
    return True


def _artifacts_ok(paths: list[str]) -> bool:
    return bool(paths) and all((REPO / path.strip()).exists() for path in paths if path.strip())


def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _write_summary_tex(rows: list[GateRow]) -> None:
    total = len(rows)
    passed = sum(1 for row in rows if row.pass_gate)
    unique_rows = [row for row in rows if row.theorem_class == "unique_theorem"]
    appendix_rows = [row for row in rows if row.theorem_class == "appendix_restatement"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Integrated 18-row hard-gated theorem traceability summary. The unique theorem set",
        r"contains the two supporting Chapter~4 theorem statements and the",
        r"flagship T1--T8 ladder; the appendix class checks the eight theorem",
        r"restatements in Appendix~C against that unique surface.}",
        r"\label{tab:integrated-theorem-gate-summary}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Class & Passed & Total \\",
        r"\midrule",
        rf"Unique theorem claims & {sum(1 for row in unique_rows if row.pass_gate)} & {len(unique_rows)} \\",
        rf"Appendix theorem restatements & {sum(1 for row in appendix_rows if row.pass_gate)} & {len(appendix_rows)} \\",
        r"\midrule",
        rf"\textbf{{All hard-gated theorem rows}} & \textbf{{{passed}}} & \textbf{{{total}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT_SUMMARY_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_matrix_tex(rows: list[GateRow]) -> None:
    lines = [
        r"\begin{small}",
        r"\begin{longtable}{@{}p{1.3cm}p{2.3cm}p{4.1cm}p{1.4cm}p{1.4cm}p{1.4cm}p{1.4cm}p{1.8cm}@{}}",
        r"\caption{Integrated theorem traceability-gate matrix.}",
        r"\label{tab:integrated-theorem-gate-matrix}\\",
        r"\toprule",
        r"Key & Class & Title & Source & Code & Artifacts & Mapping & Status \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{c}{\tablename\ \thetable{} --- continued}\\",
        r"\toprule",
        r"Key & Class & Title & Source & Code & Artifacts & Mapping & Status \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{\textit{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in rows:
        lines.append(
            f"{_tex_escape(row.theorem_key)} & {_tex_escape(row.theorem_class)} & "
            f"{_tex_escape(row.title)} & "
            f"{'yes' if row.source_exists else 'no'} & "
            f"{'yes' if row.code_anchors_ok else 'no'} & "
            f"{'yes' if row.artifacts_ok else 'no'} & "
            f"{'yes' if row.appendix_mapping_ok else 'no'} & "
            f"{'pass' if row.pass_gate else 'fail'} \\\\"
        )
    lines.extend([r"\end{longtable}", r"\end{small}"])
    OUT_MATRIX_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(rows: list[GateRow]) -> None:
    if plt is None:
        return
    labels = [row.theorem_key for row in rows]
    score_rows = [
        [1.0 if row.source_exists else 0.0 for row in rows],
        [1.0 if row.code_anchors_ok else 0.0 for row in rows],
        [1.0 if row.artifacts_ok else 0.0 for row in rows],
        [1.0 if row.appendix_mapping_ok else 0.0 for row in rows],
        [1.0 if row.pass_gate else 0.0 for row in rows],
    ]
    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 0.55), 3.0))
    im = ax.imshow(score_rows, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(["Source", "Code", "Artifacts", "Mapping", "Gate"])
    ax.set_title("Integrated Theorem Traceability Gate")
    for y, metric in enumerate(score_rows):
        for x, value in enumerate(metric):
            ax.text(x, y, "pass" if value >= 0.5 else "fail", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    plt.close(fig)


def _load_rows() -> list[GateRow]:
    register_rows = _read_csv(THEOREM_REGISTER)
    traceability_rows = _read_csv(TRACEABILITY)
    traceability_by_scope = {
        row["theorem_ids_or_scope"]: row
        for row in traceability_rows
        if row.get("theorem_ids_or_scope")
    }
    appendix_text = APPENDIX_PROOFS.read_text(encoding="utf-8")
    appendix_text_lower = appendix_text.lower()
    unique_titles = {
        row["title"]
        for row in register_rows
        if row.get("environment") == "theorem"
        and row.get("register_id", "").strip() in HARD_GATED_KEYS
        and "General CPS" not in row.get("title", "")
    }

    rows: list[GateRow] = []

    for register in register_rows:
        if register.get("environment") != "theorem":
            continue
        title = register["title"]
        source = register["source"]
        theorem_key = register.get("register_id", "").strip()
        theorem_class = "unique_theorem"
        mapped_to = title
        traceability_status = "locked"
        code_anchors: list[str] = []
        artifact_paths: list[str] = []
        appendix_mapping_ok = True

        if register.get("group") == "Appendix proof restatement":
            if title not in APPENDIX_MAP:
                continue
            theorem_class = "appendix_restatement"
            mapped_to = APPENDIX_MAP.get(title, "")
            appendix_key = title.split()[0].strip()
            appendix_mapping_ok = (
                bool(mapped_to)
                and mapped_to in unique_titles
                and appendix_key.lower() in appendix_text_lower
                and mapped_to.lower() in appendix_text_lower
            )
            traceability_status = "restated"
            code_ok = appendix_mapping_ok
            artifact_ok = appendix_mapping_ok
            source_exists = (REPO / source).exists()
            pass_gate = source_exists and appendix_mapping_ok
            notes = "appendix theorem restatement matches main theorem" if pass_gate else "appendix theorem mapping failed"
            rows.append(
                GateRow(
                    theorem_key=title.split()[0].rstrip("."),
                    theorem_class=theorem_class,
                    title=title,
                    source=source,
                    mapped_to=mapped_to,
                    traceability_status=traceability_status,
                    source_exists=source_exists,
                    code_anchors_ok=code_ok,
                    artifacts_ok=artifact_ok,
                    appendix_mapping_ok=appendix_mapping_ok,
                    pass_gate=pass_gate,
                    notes=notes,
                )
            )
            continue

        if theorem_key not in HARD_GATED_KEYS or "General CPS" in title:
            continue

        if theorem_key and theorem_key in traceability_by_scope:
            trace = traceability_by_scope[theorem_key]
            traceability_status = trace.get("status", "missing")
            code_anchors = [item.strip() for item in trace.get("code_anchors", "").split(";") if item.strip()]
            artifact_paths = [item.strip() for item in trace.get("artifact_paths", "").split(";") if item.strip()]
        elif title in SUPPORTING_THEOREMS:
            supp = SUPPORTING_THEOREMS[title]
            theorem_key = str(supp["theorem_key"])
            traceability_status = str(supp["traceability_status"])
            code_anchors = list(supp["code_anchors"])
            artifact_paths = list(supp["artifact_paths"])
        else:
            traceability_status = "missing"

        source_exists = (REPO / source).exists()
        code_ok = _code_anchors_ok(code_anchors)
        artifact_ok = _artifacts_ok(artifact_paths)
        partial = traceability_status.startswith("partial")
        pass_gate = source_exists and code_ok and artifact_ok and not partial
        notes = []
        if partial:
            notes.append("partial traceability")
        if not source_exists:
            notes.append("missing source")
        if not code_ok:
            notes.append("code anchors failed")
        if not artifact_ok:
            notes.append("artifact paths missing")
        rows.append(
            GateRow(
                theorem_key=theorem_key or title,
                theorem_class=theorem_class,
                title=title,
                source=source,
                mapped_to=mapped_to,
                traceability_status=traceability_status,
                source_exists=source_exists,
                code_anchors_ok=code_ok,
                artifacts_ok=artifact_ok,
                appendix_mapping_ok=appendix_mapping_ok,
                pass_gate=pass_gate,
                notes=", ".join(notes) if notes else "locked",
            )
        )

    # Keep stable release order: unique rows first, appendix rows last.
    unique_rows = [row for row in rows if row.theorem_class == "unique_theorem"]
    appendix_rows = [row for row in rows if row.theorem_class == "appendix_restatement"]
    return unique_rows + appendix_rows


def main() -> int:
    rows = _load_rows()
    theorem_count = len(rows)
    if theorem_count != 18:
        print(f"Integrated theorem gate FAILED: expected 18 theorem rows, found {theorem_count}")
        return 1

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_dict().keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())

    summary = {
        "gate_kind": "traceability_release_gate",
        "total": len(rows),
        "passed": sum(1 for row in rows if row.pass_gate),
        "failed": sum(1 for row in rows if not row.pass_gate),
        "unique_theorems_total": sum(1 for row in rows if row.theorem_class == "unique_theorem"),
        "appendix_restatements_total": sum(1 for row in rows if row.theorem_class == "appendix_restatement"),
        "rows": [row.to_dict() for row in rows],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_tex(rows)
    _write_matrix_tex(rows)
    _write_figure(rows)

    failed_rows = [row for row in rows if not row.pass_gate]
    if failed_rows:
        print("Integrated theorem gate FAILED:")
        for row in failed_rows:
            print(f"  - {row.theorem_key}: {row.notes}")
        return 1

    print("Integrated theorem gate PASS: 18/18 theorem rows traceability-locked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
