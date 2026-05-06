"""Shared JSON-safe serialization helpers for API responses."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi.encoders import jsonable_encoder

_CUSTOM_ENCODERS = {
    np.ndarray: lambda value: value.tolist(),
    np.generic: lambda value: value.item(),
}


def api_jsonable(value: Any) -> Any:
    """Normalize runtime payloads into FastAPI/Pydantic-safe data."""
    return jsonable_encoder(value, custom_encoder=_CUSTOM_ENCODERS)
