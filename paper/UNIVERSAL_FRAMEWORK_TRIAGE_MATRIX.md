# ORIUS Universal-Framework Merge Triage Matrix

This matrix records the comparison used to reframe `paper/paper.tex` into the flagship ORIUS manuscript while keeping all claims grounded in repo-verified evidence.

## Source policy

- Canonical sources of edit truth: repo thesis LaTeX, `paper/paper.tex`, locked reports, theorem/evidence maps.
- Reference-only surfaces: `agent-artifacts-zip_*.zip`, `ORIUS_PhD_Thesis_300pages.pdf`, compiled PDFs.
- Dispositions:
  - `import`: repo-backed and safe to foreground in the paper
  - `rephrase`: useful structure or rhetoric, but wording must be softened to match repo evidence
  - `reject`: artifact-only or incompatible with canonical repo truth

## Comparison matrix

| Candidate surface | Artifact zip / thesis300 tendency | Repo-backed reality | Disposition | Paper action |
|---|---|---|---|---|
| ORIUS as the organizing frame | Strong universal-framework rhetoric | Universal framework exists in code, paper, manifest, and cross-domain assets | import | Lead title, abstract, introduction, and conclusion with ORIUS rather than introducing it only late |
| Battery as the deepest validated surface | Sometimes overshadowed by broader claims | Locked theorem and strongest empirical evidence are battery-scoped | import | Make battery-primary evidence boundary explicit throughout |
| Multi-domain adapter architecture | Presented as broad framework contribution | Implemented in `src/orius/universal_framework/` and surfaced in paper assets | import | Keep and foreground the universal framework section and adapter descriptions |
| Cross-domain harness evidence | Often phrased as broad validation success | Locked harness exists, but TSVR table is mixed and not a dominance result | rephrase | Present as framework-portability and unified-harness evidence, not universal superiority |
| Hardware validation / deployment maturity | Artifact package claims stronger hardware validation | Repo paper only supports runtime rehearsal, not full production-hardware proof | reject | Keep HIL/field deployment as future work unless explicitly supported by locked repo artifacts |
| “127 hours hardware validation” | Claimed in artifact package | Not supported by canonical repo evidence | reject | Do not import anywhere into paper claims |
| “6 domains fully validated” | Asserted aggressively in artifact package | Harness executes six tracks, but results are mixed and evidence depth is uneven | reject | Replace with bounded language about execution coverage and portability |
| Universal guarantees across all physical AI systems | Strong artifact/thesis300 rhetoric | Repo theorem map and paper scope remain battery-scoped | reject | Keep universal framework framing, but avoid universal safety-guarantee language |
| Artifact theorem numbering | Zip uses alternate numbering and omits canonical battery theorems | Canonical mapping is battery T1-T8 in repo theorem files | reject | Preserve repo theorem mapping and do not adopt artifact numbering in paper |
| Longform theorem and hardening narrative | Thesis300/artifact emphasize fuller structure | Already imported through `paper/longform/` from repo thesis | import | Retain longform battery theorem and hardening blocks in the primary paper build |

## Merge result

- The paper is now framed as an ORIUS flagship manuscript.
- The battery domain remains the primary locked empirical and theorem-validation surface.
- Cross-domain sections are retained as framework and harness evidence, not as universal dominance claims.
- Artifact-only claims and theorem numbering remain excluded from the paper source.
