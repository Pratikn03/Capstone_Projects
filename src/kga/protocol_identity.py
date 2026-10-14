"""Strict JSON identity contracts for prospective protocol version 2.

Sample IDs denote conservative underlying units, globally unique before any
checkpoint expansion. Aliases must be canonicalized upstream: these checks do
not establish independent sampling. Later checkpoint expansion may reuse a fixed
episode; duplicating that episode in the split manifest is prohibited.

Distinct run IDs, seeds and hashes record independence prerequisites, not proof
of independently executed training. A later lock command must dereference and
verify the actual training manifests. Neither validator accesses files or data.
"""

import math
import re
import unicodedata
from typing import Any, cast

_SPLITS = ("development", "calibration", "test")
_WINDOWS = ("U", "V", "E")
_SEEDS = ("training_seed", "initialization_seed", "minibatch_seed", "augmentation_seed")
_HASHES = ("sha256", "training_manifest_sha256")
_CHECKPOINT_IDS = ("checkpoint_id", "training_run_id")


def _object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be a JSON object")
    result = cast(dict[str, Any], value)
    if any(type(key) is not str for key in result) or set(result) != keys:
        raise ValueError(f"{context} must contain exactly its schema fields")
    return result


def _list(value: object, context: str) -> list[Any]:
    if type(value) is not list or not value:
        raise ValueError(f"{context} must be a nonempty JSON array")
    return cast(list[Any], value)


def _identifier(value: object, context: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{context} must be a string")
    result = cast(str, value)
    if not result or result.strip() != result or any(unicodedata.category(char) == "Cc" for char in result):
        raise ValueError(f"{context} must be nonempty, stripped and free of control characters")
    return result


def _version(value: object) -> None:
    if type(value) is not int or value != 2:
        raise ValueError("schema_version must be the integer 2")


def _unique(value: Any, seen: set, context: str) -> None:
    if value in seen:
        raise ValueError(f"{context} must be unique")
    seen.add(value)


def validate_split_manifest(value: object) -> dict:
    """Validate partition identities and return a fresh JSON-compatible copy.

    This conservative unit contract does not prove independence or detect aliases.
    Only identity fields are accepted; private-path or outcome fields are rejected.
    """
    root = _object(value, {"schema_version", "dataset_id", "dataset_version", "environments", "episodes"}, "split")
    _version(root["schema_version"])
    dataset_id = _identifier(root["dataset_id"], "dataset_id")
    dataset_version = _identifier(root["dataset_version"], "dataset_version")
    environments = _object(root["environments"], set(_SPLITS), "environments")
    environment_split: dict[str, str] = {}
    copied_environments: dict[str, list[str]] = {}
    for split in _SPLITS:
        copied_environments[split] = []
        for raw_id in _list(environments[split], "environment list"):
            environment_id = _identifier(raw_id, "environment_id")
            if environment_id in environment_split:
                raise ValueError("environment IDs must be unique within and across splits")
            environment_split[environment_id] = split
            copied_environments[split].append(environment_id)

    episodes = []
    seen_episodes: set[str] = set()
    seen_samples: set[str] = set()
    represented_environments: set[str] = set()
    for raw_episode in _list(root["episodes"], "episodes"):
        episode = _object(raw_episode, {"episode_id", "environment_id", "split", "windows"}, "episode")
        episode_id = _identifier(episode["episode_id"], "episode_id")
        _unique(episode_id, seen_episodes, "episode_id")
        environment_id = _identifier(episode["environment_id"], "environment_id")
        split = _identifier(episode["split"], "split")
        if split not in _SPLITS or environment_split.get(environment_id) != split:
            raise ValueError("episode environment must belong to its declared split")
        represented_environments.add(environment_id)
        windows = _object(episode["windows"], set(_WINDOWS), "windows")
        copied_windows: dict[str, list[str]] = {}
        for window in _WINDOWS:
            copied_windows[window] = []
            for raw_id in _list(windows[window], "window"):
                sample_id = _identifier(raw_id, "sample_id")
                _unique(sample_id, seen_samples, "underlying sample_id across the manifest")
                copied_windows[window].append(sample_id)
        episodes.append(
            {"episode_id": episode_id, "environment_id": environment_id, "split": split, "windows": copied_windows}
        )
    if represented_environments != set(environment_split):
        raise ValueError("every listed environment must have at least one episode")
    return {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "environments": copied_environments,
        "episodes": episodes,
    }


def validate_checkpoint_manifest(value: object) -> dict:
    """Validate recorded training prerequisites and return a fresh JSON copy.

    Hash uniqueness does not prove training independence or authenticate the
    referenced training manifests; dereferencing remains a later lock prerequisite.
    """
    root = _object(value, {"schema_version", "checkpoints"}, "checkpoint manifest")
    _version(root["schema_version"])
    rows = _list(root["checkpoints"], "checkpoints")
    if len(rows) < 5:
        raise ValueError("at least five checkpoints are required")
    unique_fields = (*_CHECKPOINT_IDS, *_SEEDS, *_HASHES)
    seen: dict[str, set] = {field: set() for field in unique_fields}
    checkpoints = []
    for raw_row in rows:
        row = _object(raw_row, {*unique_fields, "source_accuracy"}, "checkpoint")
        copied: dict[str, Any] = {}
        for field in _CHECKPOINT_IDS:
            copied[field] = _identifier(row[field], field)
        for field in _SEEDS:
            seed = row[field]
            if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
                raise ValueError(f"{field} must be a nonnegative signed-64-bit integer")
            copied[field] = seed
        for field in _HASHES:
            digest = row[field]
            if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256 hexadecimal digest")
            copied[field] = digest
        accuracy = row["source_accuracy"]
        if type(accuracy) is not float or not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError("source_accuracy must be a finite float in [0, 1]")
        copied["source_accuracy"] = accuracy
        for field in unique_fields:
            _unique(copied[field], seen[field], field)
        checkpoints.append(copied)
    return {"schema_version": 2, "checkpoints": checkpoints}
