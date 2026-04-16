"""Waymo Motion validation utilities for the governed AV surface.

This package now covers the Waymo AV dry-run path:

- Stage 2 validation over raw TFRecord shards
- deterministic subset selection
- scenario replay windows with AV fault injection
- feature-table generation, LightGBM training, and conformal calibration
- Waymo-native runtime adapter and dry-run evaluation
"""
from __future__ import annotations

from .dataset import (
    ANCHOR_CURRENT_INDEX,
    TOTAL_SCENARIO_STEPS,
    build_validation_surface,
    decode_motion_scenario,
    select_anchor_neighbors,
    validate_scenario,
)
from .replay import FAULT_FAMILIES, WaymoReplayTrackAdapter, build_replay_surface, compute_state_safety_metrics
from .runtime import WaymoAVDomainAdapter, load_runtime_bundles, run_runtime_dry_run
from .subset import build_subset_manifest, scenario_hash_rank, select_dry_run_subset
from .tfrecord import (
    iter_tfrecord_records,
    parse_example_bytes,
    serialize_example_features,
    write_tfrecord_records,
)
from .training import (
    HORIZON_LABELS,
    TARGETS,
    build_feature_tables,
    default_shift_aware_config,
    load_model_bundle,
    predict_interval_from_bundle,
    train_dry_run_models,
)

__all__ = [
    "ANCHOR_CURRENT_INDEX",
    "FAULT_FAMILIES",
    "HORIZON_LABELS",
    "TOTAL_SCENARIO_STEPS",
    "TARGETS",
    "WaymoAVDomainAdapter",
    "WaymoReplayTrackAdapter",
    "build_feature_tables",
    "build_replay_surface",
    "build_subset_manifest",
    "build_validation_surface",
    "compute_state_safety_metrics",
    "default_shift_aware_config",
    "decode_motion_scenario",
    "iter_tfrecord_records",
    "load_model_bundle",
    "load_runtime_bundles",
    "parse_example_bytes",
    "predict_interval_from_bundle",
    "run_runtime_dry_run",
    "scenario_hash_rank",
    "select_anchor_neighbors",
    "select_dry_run_subset",
    "serialize_example_features",
    "train_dry_run_models",
    "validate_scenario",
    "write_tfrecord_records",
]
