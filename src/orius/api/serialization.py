"""Shared JSON-safe serialization helpers for API responses."""
from __future__ import annotations

from typing import Any

from fastapi.encoders import jsonable_encoder
import numpy as np


_CUSTOM_ENCODERS = {
    np.ndarray: lambda value: value.tolist(),
    np.generic: lambda value: value.item(),
}


def api_jsonable(value: Any) -> Any:
    """Normalize runtime payloads into FastAPI/Pydantic-safe data."""
    return jsonable_encoder(value, custom_encoder=_CUSTOM_ENCODERS)
