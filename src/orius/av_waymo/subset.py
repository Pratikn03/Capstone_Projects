"""Deterministic subset selection for the Waymo AV dry run."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import build_validation_surface


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def scenario_hash_rank(scenario_id: str) -> int:
    """Stable integer rank for deterministic scenario selection."""
    return int(_sha256_text(str(scenario_id))[:16], 16)


def _load_or_build_validation_index(
    *,
    raw_dir: Path,
    processed_dir: Path,
    scenario_index_path: Path | None = None,
) -> pd.DataFrame:
    index_path = scenario_index_path or (processed_dir / "scenario_index.parquet")
    if not index_path.exists():
        build_validation_surface(raw_dir=raw_dir, out_dir=processed_dir, write_actor_tracks=False)
    return pd.read_parquet(index_path)


def select_dry_run_subset(
    scenario_index: pd.DataFrame,
    *,
    target_count: int = 1_000,
) -> pd.DataFrame:
    """Select an approximately balanced deterministic subset across shards."""
    df = scenario_index.copy()
    if "usable" in df.columns:
        df = df[df["usable"].astype(bool)].copy()
    if df.empty:
        raise ValueError("No usable scenarios are available for subset selection.")

    df["scenario_hash_rank"] = df["scenario_id"].map(lambda value: scenario_hash_rank(str(value)))
    df = df.sort_values(["shard_id", "scenario_hash_rank", "scenario_id"]).reset_index(drop=True)

    shard_groups = {str(shard_id): group.copy() for shard_id, group in df.groupby("shard_id", sort=True)}
    shard_ids = sorted(shard_groups)
    base_quota = max(1, int(target_count) // max(len(shard_ids), 1))
    remainder = max(0, int(target_count) - (base_quota * len(shard_ids)))

    selected_parts: list[pd.DataFrame] = []
    leftovers: list[pd.DataFrame] = []
    for index, shard_id in enumerate(shard_ids):
        group = (
            shard_groups[shard_id].sort_values(["scenario_hash_rank", "scenario_id"]).reset_index(drop=True)
        )
        shard_quota = base_quota + (1 if index < remainder else 0)
        take = min(len(group), shard_quota)
        if take > 0:
            selected_parts.append(group.iloc[:take].copy())
        if take < len(group):
            leftovers.append(group.iloc[take:].copy())

    selected = (
        pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=df.columns)
    )
    if len(selected) < min(target_count, len(df)) and leftovers:
        needed = min(target_count, len(df)) - len(selected)
        refill = (
            pd.concat(leftovers, ignore_index=True)
            .sort_values(["scenario_hash_rank", "scenario_id"])
            .iloc[:needed]
            .copy()
        )
        selected = pd.concat([selected, refill], ignore_index=True)

    selected = (
        selected.drop_duplicates(subset=["scenario_id"])
        .sort_values(["shard_id", "scenario_hash_rank", "scenario_id"])
        .reset_index(drop=True)
    )
    selected["subset_rank"] = selected.index.astype(int)
    return selected


def build_subset_manifest(
    *,
    raw_dir: str | Path,
    processed_dir: str | Path,
    target_count: int = 1_000,
    scenario_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic dry-run subset manifest and supporting files."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    scenario_index = _load_or_build_validation_index(
        raw_dir=raw_path,
        processed_dir=processed_path,
        scenario_index_path=Path(scenario_index_path) if scenario_index_path is not None else None,
    )
    selected = select_dry_run_subset(scenario_index, target_count=target_count)

    used_shards = sorted({str(item) for item in selected["shard_id"].tolist()})
    raw_hashes = {shard_id: _sha256_file(raw_path / shard_id) for shard_id in used_shards}

    manifest_path = processed_path / "dry_run_subset_manifest.parquet"
    selected.to_parquet(manifest_path, index=False)

    manifest_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_dir": str(raw_path),
        "processed_dir": str(processed_path),
        "target_count": int(target_count),
        "selected_count": int(len(selected)),
        "used_shards": used_shards,
        "raw_file_hashes": raw_hashes,
        "artifacts": {
            "subset_manifest_parquet": str(manifest_path),
        },
        "selection_rule": "deterministic_stratified_by_shard_then_sha256_scenario_id",
        "scenario_ids": selected["scenario_id"].tolist(),
    }
    json_path = processed_path / "dry_run_subset_manifest.json"
    json_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    manifest_payload["artifacts"]["subset_manifest_json"] = str(json_path)
    return manifest_payload
