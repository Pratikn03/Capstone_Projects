"""Model and uncertainty bundle loading for the source-neutral AV runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from orius.av_waymo.runtime import load_runtime_bundles as _load_runtime_bundles


def load_runtime_bundles(models_dir: str | Path, *, artifact_prefix: str = "waymo_av") -> dict[str, Any]:
    """Load AV runtime bundles while preserving legacy artifact prefixes."""

    return cast(dict[str, Any], _load_runtime_bundles(models_dir, artifact_prefix=artifact_prefix))
