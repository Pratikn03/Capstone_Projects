"""Shared report-generation context objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReportContext:
    repo_root: Path
    features_path: Path
    splits_dir: Path
    models_dir: Path
    reports_dir: Path
    publication_dir: Path
    uncertainty_artifacts_dir: Path | None = None
    backtests_dir: Path | None = None
    current_dataset: str | None = None
    targets: list[str] | None = None


def ensure_dir(path: Path) -> None:
    """Create a report output directory if needed."""

    path.mkdir(parents=True, exist_ok=True)
