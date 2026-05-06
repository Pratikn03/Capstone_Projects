"""Runtime replay entry points for source-neutral AV evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from orius.av_waymo.runtime import WaymoAVDomainAdapter
from orius.av_waymo.runtime import run_runtime_dry_run as _run_runtime_dry_run

AVRuntimeDomainAdapter = WaymoAVDomainAdapter


def run_runtime_dry_run(
    *,
    replay_windows_path: str | Path,
    step_features_path: str | Path,
    models_dir: str | Path,
    out_dir: str | Path,
    max_scenarios: int | None = None,
    artifact_prefix: str = "waymo_av",
    runtime_policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run bounded AV runtime replay through the compatibility implementation."""

    return cast(
        dict[str, Any],
        _run_runtime_dry_run(
            replay_windows_path=replay_windows_path,
            step_features_path=step_features_path,
            models_dir=models_dir,
            out_dir=out_dir,
            max_scenarios=max_scenarios,
            artifact_prefix=artifact_prefix,
            runtime_policy_config=runtime_policy_config,
        ),
    )
