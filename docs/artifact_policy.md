# ORIUS Artifact Policy

This repository separates source code and small canonical evidence from local
research data, generated runtime traces, model artifacts, and tool caches.

## Commit Allowed

- Source code, tests, configs, and documentation.
- Small publication summaries that are claim-governing and reviewed.
- Release manifests, hash registries, theorem audits, and compact witness files.
- Final paper assets only when they are intentionally curated.

## Commit Blocked

- Raw datasets, NuPlan/CARLA/Waymo archives, MIMIC/eICU extracts, and local data
  derivatives.
- Model binaries and checkpoints: `.pkl`, `.pt`, `.onnx`, `.ckpt`, `.safetensors`.
- Runtime databases and heavyweight traces: `.duckdb`, `.parquet`,
  `runtime_traces.csv`, certificate-expiry traces, and full folder inventories.
- Build and cache outputs: `.next`, `node_modules`, `.pytest_cache`,
  `.ruff_cache`, `.hypothesis`, `.playwright-mcp`, `__pycache__`.
- Local screenshots, smoke images, AppleDouble files, and Finder sidecars.

## Enforcement

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_generated_artifact_policy.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_no_appledouble.py --exclude-active
```

If a generated artifact must become canonical evidence, add a compact summary or
manifest instead of the raw artifact. Any exception must be explicitly allowlisted
in the validator with a comment explaining why it is reproducibility-critical.
