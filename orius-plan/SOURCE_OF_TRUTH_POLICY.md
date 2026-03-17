# ORIUS Source-of-Truth Policy

**Status**: Canonical. All edits and artifact alignment follow this hierarchy.

---

## 1. Canonical Sources (Edit Here)

| Tier | Source | Location | Role |
|------|--------|----------|------|
| **A** | Repo thesis LaTeX | `orius_battery_409page_figures_upgraded_main.tex` + `chapters/` + `appendices/` | **Primary source of edits.** All theorem statements, proofs, and narrative flow. |
| **A** | Paper LaTeX | `paper/paper.tex` + `paper/longform/` | Conference paper; imports thesis chapters via longform blocks. |
| **A** | Locked evidence | `reports/publication/*.csv`, `reports/impact_summary.csv` | Empirical results. Never overwrite without re-lock. |
| **B** | Theorem register | `appendices/app_m_verified_theorems_and_gap_audit.tex` | Canonical battery-8 (T1–T8). |
| **B** | Evidence map | `orius-plan/theorem_to_evidence_map.md` | Code anchors and locked evidence per theorem. |
| **B** | Cross-source mapping | `orius-plan/THEOREM_REGISTER_MAPPING.md` | Maps artifact zip and paper labels to canonical T1–T8. |

---

## 2. Reference-Only (Do Not Edit as Source)

| Source | Location | Role |
|--------|----------|------|
| Agent artifact zip | `agent-artifacts-zip_*.zip` | Delivery/export package. Contains `ORIUS_COMPLETE_MATHEMATICAL_PROOFS.md`, thesis variants, peer reviews. **Not canonical.** |
| Repo thesis PDF | `orius_battery_409page_figures_upgraded_main.pdf` | Compiled output. Regenerate from LaTeX. |
| Paper PDF | `paper/paper.pdf` | Compiled output. Regenerate from LaTeX. |

---

## 3. Edit Flow

1. **Thesis content**: Edit `chapters/chXX_*.tex` and `appendices/app_*.tex`. Rebuild PDF with thesis build.
2. **Paper content**: Edit `paper/paper.tex` or longform imports. Rebuild with `make paper-compile`.
3. **Theorem alignment**: When artifact zip or paper use different theorem labels, update `THEOREM_REGISTER_MAPPING.md` and `theorem_to_evidence_map.md` to document the mapping. **Do not** change repo thesis to match artifact zip.
4. **Artifact zip**: When producing a new artifact package, include `orius-plan/THEOREM_REGISTER_MAPPING.md` and align zip content to repo thesis (battery-8) where applicable.

---

## 4. Agent Artifact Analysis

Run `make analyze-artifact` (or `python scripts/analyze_agent_artifact.py <path-to-zip>`) to:

- Extract theorem names from `ORIUS_COMPLETE_MATHEMATICAL_PROOFS.md` in the zip
- Compare against canonical battery-8
- Report mismatches, missing sections, and zip-only extensions
- Verify `THEOREM_REGISTER_MAPPING.md` is present in the zip

Use this before accepting agent-generated content as aligned with the repo thesis.
