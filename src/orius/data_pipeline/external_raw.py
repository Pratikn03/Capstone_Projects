"""Helpers for resolving external raw datasets outside the git repository.

Large real-data sources should live on an external volume and be referenced
through ``ORIUS_EXTERNAL_DATA_ROOT`` instead of being copied into the repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DATA_ROOT_ENV = "ORIUS_EXTERNAL_DATA_ROOT"


@dataclass(frozen=True)
class ExternalDatasetSpec:
    """Description of an external raw dataset location."""

    key: str
    directory_name: str
    description: str


EXTERNAL_DATASETS: dict[str, ExternalDatasetSpec] = {
    "waymo_open_motion": ExternalDatasetSpec(
        key="waymo_open_motion",
        directory_name="waymo_open_motion",
        description="Waymo Open Motion Dataset raw root",
    ),
    "argoverse2_motion": ExternalDatasetSpec(
        key="argoverse2_motion",
        directory_name="argoverse2_motion",
        description="Argoverse 2 Motion raw root",
    ),
    "argoverse2_sensor": ExternalDatasetSpec(
        key="argoverse2_sensor",
        directory_name="argoverse2_sensor",
        description="Argoverse 2 Sensor raw root",
    ),
    "kitti_odometry": ExternalDatasetSpec(
        key="kitti_odometry",
        directory_name="kitti_odometry",
        description="KITTI Odometry raw root",
    ),
}


def get_external_data_root(explicit_root: Path | None = None, *, required: bool = False) -> Path | None:
    """Return the configured external data root, if any."""
    if explicit_root is not None:
        root = Path(explicit_root).expanduser()
    else:
        env_value = os.environ.get(EXTERNAL_DATA_ROOT_ENV, "").strip()
        root = Path(env_value).expanduser() if env_value else None

    if root is None:
        if required:
            raise FileNotFoundError(
                f"External data root is not configured. Set {EXTERNAL_DATA_ROOT_ENV} or pass --external-root."
            )
        return None

    root = root.resolve()
    if required and not root.exists():
        raise FileNotFoundError(
            f"External data root does not exist: {root}. Set {EXTERNAL_DATA_ROOT_ENV} to a valid directory."
        )
    return root


def get_external_dataset_dir(
    dataset_key: str,
    explicit_root: Path | None = None,
    *,
    required: bool = False,
) -> Path | None:
    """Return the dataset-specific directory under the external data root."""
    if dataset_key not in EXTERNAL_DATASETS:
        raise KeyError(f"Unknown external dataset key: {dataset_key}")
    root = get_external_data_root(explicit_root, required=required)
    if root is None:
        return None
    dataset_dir = root / EXTERNAL_DATASETS[dataset_key].directory_name
    if required and not dataset_dir.exists():
        raise FileNotFoundError(
            f"Missing {EXTERNAL_DATASETS[dataset_key].description} at {dataset_dir}."
        )
    return dataset_dir


def repo_manifest_path(domain: str) -> Path:
    """Return the lightweight repo-tracked manifest for a domain's external raw source."""
    return REPO_ROOT / "data" / domain / "raw" / "external_sources_manifest.json"

