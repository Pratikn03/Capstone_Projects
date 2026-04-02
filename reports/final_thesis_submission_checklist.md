# Final Thesis Submission Checklist

Use this checklist when turning the current ORIUS repo into the final student
submission package.

## Manuscript authority and evidence

- [ ] `paper/paper.tex` is the only manuscript authority referenced in docs and governance
- [ ] every strong thesis claim maps to `paper/claim_matrix.csv`
- [ ] all locked values resolve to tracked artifacts, not ignored local caches
- [ ] evidence tiers are stated consistently: reference witness / proof-validated / proof-candidate / shadow-synthetic / experimental

## Literature and framing

- [ ] related-work chapter covers Papers 1–6 as internal program units
- [ ] external review covers conformal, adaptive conformal, runtime assurance, shielding, and drift/anomaly detection
- [ ] applied-review prose includes energy, robotics/navigation, industrial, healthcare, and IoT/CPS contexts
- [ ] universal terminology is used outside energy-domain chapters

## Figures, tables, and appendices

- [ ] every figure and table cited in the thesis appears in the artifact appendix
- [ ] release manifest, benchmark tables, and latency summaries are present and tracked
- [ ] literature matrix, gap matrix, and maturity matrix are present in `reports/publication/`
- [ ] appendix-facing audit/support files are referenced from documentation

## Formatting and institutional cleanup

- [ ] title page no longer carries `Draft` wording if this is the final submission build
- [ ] acknowledgments no longer say `This draft`
- [ ] solo-submission title page intentionally omits committee/advisor metadata
- [ ] table of contents, page numbering, and front/back matter are correct in the compiled PDF
- [ ] bibliography is consistently numerical and IEEE-style in presentation

## Submission bundle

- [ ] final thesis PDF
- [ ] manuscript sources
- [ ] tracked artifact appendix
- [ ] reproducibility note
- [ ] code/data availability statement
- [ ] institutional declarations and signatures, if required
