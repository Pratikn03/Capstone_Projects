# Shift-Aware Uncertainty Audit Map

## Existing implementation
- Conformal intervals and adaptive alpha primitives were already present in `conformal.py`.
- DC3S pipeline already carried reliability/drift/inflation into certificates.
- Publication reporting already had reliability group coverage scripts.

## Missing before this change
- No runtime-validity score under shift.
- No subgroup-aware coverage object reusable at runtime.
- No adaptive quantile trace emitted as first-class artifact.
- No certificate fields for uncertainty-validity governance.

## Extension points added
- New `shift_aware` package for state, ACI update, subgroup tracking, validity scoring, policy, and artifacts.
- New `build_runtime_interval(...)` integration point in conformal path.
- DC3S certificate schema extension with backward-compatible nullable fields.

## Backward compatibility
- Legacy mode remains default (`enabled: false`).
- Existing certificate readers continue to work because new fields are optional.
