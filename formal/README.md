# ORIUS Formal Proof Track

This directory is the gated Lean 4 proof track for the ORIUS theorem program. It is intentionally
not counted as promoted proof evidence until `lake build` passes and the
publication status artifact records `lean_status=passed`.

The current files encode theorem kernels only. They do not discharge the
domain-specific witness constants, mixing bridge, TV bridge, boundary-mass,
runtime-certificate, or empirical artifact obligations required for flagship
promotion.
