# AI Artifact Sanitization Checklist

Use this checklist before sending any ORIUS artifact to a hosted AI system.

## 1. Decide the artifact class

| Artifact class | Default policy | Safe hosted form |
|---|---|---|
| `paper/` excerpts | Upload allowed | Only the exact excerpt needed |
| `paper/monograph/` excerpts | Upload allowed | Only the exact excerpt needed |
| `reports/publication/` tables and figures | Upload allowed | Public-facing tables, figures, parity matrices, and aggregate summaries |
| `configs/` | Usually allowed | Exclude secrets and private paths |
| `src/` code | Conditional | Only the relevant file excerpt or interface |
| `reports/runs/` | Local-only by default | Aggregate metrics only |
| `data/*/processed/` | Local-only by default | Schema, row counts, and aggregate stats only |
| `data/*/raw/` | Local-only | Do not upload |
| audit ledgers / runtime traces | Local-only by default | Small redacted snippets only if necessary |

## 2. Redaction rules

Before upload:

- remove raw rows
- remove IDs and trace keys
- remove file paths that expose private storage layout when not needed
- reduce logs to the failing lines
- reduce metrics to aggregate summaries
- prefer a schema sample over a data sample

## 3. Safe question shapes

Good hosted-model questions:

- “Tighten this theorem statement.”
- “Critique this claim boundary.”
- “Review this API surface.”
- “Explain this failing stack trace.”
- “Compare these aggregate metrics.”

Unsafe hosted-model questions:

- “Analyze this full raw dataset dump.”
- “Find a signal in this patient telemetry table.”
- “Review this entire private repo export.”
- “Infer the distribution from these processed rows.”

## 4. ORIUS-specific local-only surfaces

Keep these local unless heavily redacted:

- `data/battery/`
- `data/healthcare/`
- `data/industrial/`
- `data/av/`
- `data/aerospace/`
- `reports/runs/`
- `reports/certos/` runtime trace details
- `reports/multi_agent/` raw scenario traces

## 5. Upload preflight

Answer all of these before upload:

- Is this artifact public-facing already?
- Can the same question be answered from an excerpt instead of the full file?
- Have raw rows been removed?
- Have identifiers been removed?
- Have I reduced the artifact to schema, summary, or the exact failing lines?
- Would I be comfortable with this exact payload appearing in a reviewer appendix?

If any answer is `no`, do not upload the artifact yet.
