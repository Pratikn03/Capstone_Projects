"""Internal numeric coercion that preserves unavailable NumPy observations."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_float_array(value: Any) -> np.ndarray:
    """Coerce numeric input without exposing values hidden by an active mask.

    ``np.asarray`` discards a ``MaskedArray``'s mask. Convert through NumPy's
    mask-aware array constructor instead, including masks inside nested input,
    and represent missing entries as NaN. No row is dropped or imputed: scalar
    certificate/evidence validators reject these entries, while batch/routing
    callers apply their existing per-cell unavailable policy. Unmasked values
    and the input shape are unchanged, and the caller's array is not mutated.
    """
    array: np.ma.MaskedArray[Any, np.dtype[np.float64]] = np.ma.asarray(value, dtype=float)
    return np.asarray(array.filled(np.nan), dtype=float)
