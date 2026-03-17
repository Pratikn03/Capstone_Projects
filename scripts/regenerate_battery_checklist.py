#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports/publication"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    claim_rows = _read_csv(PUBLICATION_DIR / "battery_claim_evidence_register.csv")
    chapter_rows = _read_csv(PUBLICATION_DIR / "chapter_coverage_register.csv")
    gap_rows = _read_csv(PUBLICATION_DIR / "battery_plan_gap_audit.csv")

    theorem_rows = [
        row
        for row in claim_rows
        if row["theorem_or_scope"].startswith("T") and row["theorem_or_scope"][1:].isdigit()
    ]
    assumption_rows = [row for row in claim_rows if row["theorem_or_scope"] == "A1-A8"]
    locked_chapters = [row for row in chapter_rows if row["status"] == "locked"]
    partial_chapters = [row for row in chapter_rows if row["status"].startswith("partial")]
    complete_gap_rows = [row for row in gap_rows if row["status"] == "complete"]
    partial_gap_rows = [row for row in gap_rows if row["status"].startswith("partial")]

    md_lines = [
        "# Verified Battery Checklist",
        "",
        "This file is generated from the publication traceability registers.",
        "",
        "## Battery 8-Theorem Register",
        "",
        "| Scope | Chapter | Status | Code Or Script | Artifact |",
        "|---|---|---|---|---|",
    ]
    for row in theorem_rows:
        md_lines.append(
            f"| {row['theorem_or_scope']} | {row['chapter']} | {row['status']} | "
            f"`{row['code_or_script']}` | `{row['artifact_path']}` |"
        )
    if assumption_rows:
        md_lines.extend(
            [
                "",
                "## Assumption Register",
                "",
                "| Scope | Chapter | Status | Code Or Script | Artifact |",
                "|---|---|---|---|---|",
            ]
        )
        for row in assumption_rows:
            md_lines.append(
                f"| {row['theorem_or_scope']} | {row['chapter']} | {row['status']} | "
                f"`{row['code_or_script']}` | `{row['artifact_path']}` |"
            )

    md_lines.extend(
        [
            "",
            "## Chapter Coverage",
            "",
            f"- Locked chapters: {len(locked_chapters)}",
            f"- Partial chapters: {len(partial_chapters)}",
            "",
            "| Chapter | Status | Primary Artifact |",
            "|---|---|---|",
        ]
    )
    for row in chapter_rows:
        md_lines.append(f"| {row['chapter']} | {row['status']} | `{row['primary_artifact']}` |")

    md_lines.extend(
        [
            "",
            "## Hardening Gap Audit",
            "",
            f"- Complete items: {len(complete_gap_rows)}",
            f"- Partial items: {len(partial_gap_rows)}",
            "",
            "| Item | Status | Evidence | Note |",
            "|---|---|---|---|",
        ]
    )
    for row in gap_rows:
        md_lines.append(
            f"| {row['item']} | {row['status']} | `{row['evidence_or_path']}` | {row['note']} |"
        )

    md_lines.extend(
        [
            "",
            "## Remaining Partial Surfaces",
            "",
        ]
    )
    if partial_chapters or partial_gap_rows:
        for row in partial_chapters:
            md_lines.append(f"- Chapter `{row['chapter']}` remains `{row['status']}` with primary artifact `{row['primary_artifact']}`.")
        for row in partial_gap_rows:
            md_lines.append(f"- Gap item `{row['item']}` remains `{row['status']}`: {row['note']}.")
    else:
        md_lines.append("- None.")

    out_md = PUBLICATION_DIR / "battery_checklist_verified.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    theorem_md_lines = [
        "# Battery Theorem Checklist",
        "",
        "This file is generated from `battery_claim_evidence_register.csv`.",
        "",
        "## Battery 8-Theorem Register",
        "",
        "| Scope | Chapter | Status | Code Or Script | Artifact |",
        "|---|---|---|---|---|",
    ]
    for row in theorem_rows:
        theorem_md_lines.append(
            f"| {row['theorem_or_scope']} | {row['chapter']} | {row['status']} | "
            f"`{row['code_or_script']}` | `{row['artifact_path']}` |"
        )
    theorem_md_lines.extend(
        [
            "",
            "## Assumption Register",
            "",
            "| Scope | Chapter | Status | Code Or Script | Artifact |",
            "|---|---|---|---|---|",
        ]
    )
    for row in assumption_rows:
        theorem_md_lines.append(
            f"| {row['theorem_or_scope']} | {row['chapter']} | {row['status']} | "
            f"`{row['code_or_script']}` | `{row['artifact_path']}` |"
        )
    (PUBLICATION_DIR / "battery_theorem_checklist.md").write_text("\n".join(theorem_md_lines) + "\n", encoding="utf-8")

    out_json = PUBLICATION_DIR / "battery_checklist_summary.json"
    out_json.write_text(
        json.dumps(
            {
                "theorem_items": len(theorem_rows),
                "assumption_register_items": len(assumption_rows),
                "locked_chapters": len(locked_chapters),
                "partial_chapters": len(partial_chapters),
                "complete_gap_items": len(complete_gap_rows),
                "partial_gap_items": len(partial_gap_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (PUBLICATION_DIR / "battery_theorem_checklist.json").write_text(
        json.dumps(
            {
                "theorem_items": len(theorem_rows),
                "assumption_register_items": len(assumption_rows),
                "theorems": theorem_rows,
                "assumptions": assumption_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
