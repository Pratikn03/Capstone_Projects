# scripts/_archive — one-shot tooling

Scripts in this directory are **archived one-shot artifacts**, not part of any
active pipeline.  They are preserved here (instead of deleted) so their git
history stays intact and they remain referenceable, but they are not run by
`make` targets, not exercised by tests, and not part of the defended ORIUS
evidence package.

## Contents

- `audit.py` — ad-hoc code-quality audit script run once during cleanup.
  Superseded by `scripts/audit_code_health.py`.
- `restructure_thesis.py` — one-shot script that produced the
  `chapters_merged/` reorganization from the original `chapters/` tree.  Has
  served its purpose; the active manuscript lives under `paper/ieee/`.
- `CODEX_IMPLEMENTATION_UPDATE.md` — historical implementation note from
  earlier ORIUS development; preserved for provenance.

Anything else added here should similarly be (a) inert, (b) reproducible from
elsewhere, and (c) annotated in this README.
