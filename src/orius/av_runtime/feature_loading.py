"""Feature loading helpers for AV runtime replay."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from orius.av_waymo.runtime import _load_runtime_test_step_features


def load_runtime_test_step_features(
    step_features_path: str | Path,
    *,
    max_scenarios: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load test split step features for a bounded runtime replay."""

    return cast(
        tuple[pd.DataFrame, list[str]],
        _load_runtime_test_step_features(step_features_path, max_scenarios=max_scenarios),
    )
