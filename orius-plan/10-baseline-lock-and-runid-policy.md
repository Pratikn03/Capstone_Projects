# Baseline Lock and Run-ID Policy

- Baseline lock run id: `LOCK_20260317T025648Z`
- Claim-evidence matrix: `reports/publication/claim_evidence_matrix.csv`
- Run-ID policy: `reports/publication/run_id_policy.json`
- Baseline manifest: `reports/publication/baseline_lock_manifest.json`

## Policy
- Use UTC timestamps only.
- Use stage-prefixed run ids: `BASE_`, `P1_`, `P2_`, `P3_`, `FINAL_`.
- Every generated artifact must be listed in a manifest with SHA256.
- Claims in manuscript chapters must map to theorem IDs or matrix rows.
