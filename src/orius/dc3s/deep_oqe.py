"""Learned DeepOQE utilities for battery-scoped degraded-observation experiments.

This module intentionally keeps the runtime contract narrow:
- the model consumes a short history of telemetry-quality features
- the output is still a bounded reliability score ``w_t`` in ``[min_w, 1]``
- runtime callers may fall back to the heuristic OQE path if the learned model
  is unavailable or explicitly disabled
"""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from orius.release.artifact_loader import load_torch_artifact

FEATURE_NAMES: tuple[str, ...] = (
    "load_mw",
    "renewables_mw",
    "prev_load_mw",
    "prev_renewables_mw",
    "delta_load_mw",
    "delta_renewables_mw",
    "missing_fraction",
    "delay_seconds",
    "out_of_order",
    "spike_ratio",
    "ooo_fraction_in_order",
    "stale_count_load_mw",
    "stale_count_renewables_mw",
    "fault_dropout",
    "fault_stale_sensor",
    "fault_delay_jitter",
    "fault_out_of_order",
    "fault_spikes",
)


@dataclass(frozen=True, slots=True)
class DeepOQEConfig:
    input_dim: int = len(FEATURE_NAMES)
    seq_len: int = 8
    hidden_size: int = 48
    num_layers: int = 1
    dropout: float = 0.10
    min_w: float = 0.05


class DeepOQEModel(nn.Module):
    """Small GRU-based reliability encoder.

    The model is intentionally lightweight so it can run inside the ORIUS
    runtime loop without introducing a new deployment surface.
    """

    def __init__(self, cfg: DeepOQEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DeepOQEConfig()
        dropout = self.cfg.dropout if self.cfg.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=self.cfg.input_dim,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(self.cfg.hidden_size),
            nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_size, 1),
        )
        self.fault_head = nn.Sequential(
            nn.LayerNorm(self.cfg.hidden_size),
            nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size // 2),
            nn.GELU(),
            nn.Linear(max(1, self.cfg.hidden_size // 2), 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _hidden = self.encoder(sequence)
        pooled = encoded[:, -1, :]
        reliability_logit = self.reliability_head(pooled)
        fault_logit = self.fault_head(pooled)
        reliability = torch.sigmoid(reliability_logit)
        reliability = self.cfg.min_w + (1.0 - self.cfg.min_w) * reliability
        return reliability.squeeze(-1), fault_logit.squeeze(-1)


def extract_feature_vector(
    *,
    event: Mapping[str, Any],
    last_event: Mapping[str, Any] | None,
    missing_fraction: float,
    delay_seconds: float,
    out_of_order: bool,
    spike_ratio: float,
    ooo_fraction_in_order: float,
    stale_tracker: Mapping[str, Any] | None,
    fault_flags: Mapping[str, Any] | None,
) -> np.ndarray:
    """Create the DeepOQE feature vector for one telemetry step."""

    previous = dict(last_event or {})
    stale_counts = {}
    if isinstance(stale_tracker, Mapping):
        stale_counts = dict(stale_tracker.get("unchanged_counts", {}) or {})
    faults = dict(fault_flags or {})

    def _num(payload: Mapping[str, Any], key: str) -> float:
        value = payload.get(key)
        if isinstance(value, bool):
            return float(value)
        try:
            out = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(out):
            return 0.0
        return out

    load_now = _num(event, "load_mw")
    renew_now = _num(event, "renewables_mw")
    load_prev = _num(previous, "load_mw")
    renew_prev = _num(previous, "renewables_mw")

    row = np.asarray(
        [
            load_now,
            renew_now,
            load_prev,
            renew_prev,
            load_now - load_prev,
            renew_now - renew_prev,
            float(missing_fraction),
            float(delay_seconds),
            float(bool(out_of_order)),
            float(spike_ratio),
            float(ooo_fraction_in_order),
            float(stale_counts.get("load_mw", 0.0)),
            float(stale_counts.get("renewables_mw", 0.0)),
            float(bool(faults.get("dropout", False))),
            float(bool(faults.get("stale_sensor", False))),
            float(bool(faults.get("delay_jitter", False))),
            float(bool(faults.get("out_of_order", False))),
            float(bool(faults.get("spikes", False))),
        ],
        dtype=np.float32,
    )
    return row


def _history_buffer(adaptive_state: Mapping[str, Any] | None) -> list[list[float]]:
    if not isinstance(adaptive_state, Mapping):
        return []
    history = adaptive_state.get("deep_oqe_history", [])
    if not isinstance(history, Sequence):
        return []
    rows: list[list[float]] = []
    for item in history:
        if not isinstance(item, Sequence):
            continue
        try:
            row = [float(v) for v in item]
        except (TypeError, ValueError):
            continue
        if len(row) == len(FEATURE_NAMES):
            rows.append(row)
    return rows


def prepare_sequence(
    feature: np.ndarray,
    *,
    adaptive_state: Mapping[str, Any] | None,
    seq_len: int,
) -> np.ndarray:
    """Create a fixed-length history tensor ending at ``feature``."""

    history = _history_buffer(adaptive_state)
    history.append(feature.astype(np.float32).tolist())
    history = history[-max(int(seq_len), 1) :]
    if len(history) < seq_len:
        pad = [[0.0] * len(FEATURE_NAMES) for _ in range(seq_len - len(history))]
        history = pad + history
    return np.asarray(history, dtype=np.float32)


def update_history(
    adaptive_state: MutableMapping[str, Any] | None,
    feature: np.ndarray,
    *,
    seq_len: int,
) -> None:
    """Persist the recent DeepOQE feature history inside the runtime state."""

    if not isinstance(adaptive_state, MutableMapping):
        return
    history = _history_buffer(adaptive_state)
    history.append(feature.astype(np.float32).tolist())
    adaptive_state["deep_oqe_history"] = history[-max(int(seq_len), 1) :]


def save_model(
    path: str | Path,
    *,
    model: DeepOQEModel,
    cfg: DeepOQEConfig,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, target)
    return target


@lru_cache(maxsize=8)
def load_model(path: str) -> tuple[DeepOQEModel, DeepOQEConfig, dict[str, Any]]:
    model_path = Path(path)
    payload = load_torch_artifact(model_path, map_location="cpu", weights_only=True)
    cfg = DeepOQEConfig(**dict(payload.get("config", {})))
    model = DeepOQEModel(cfg)
    state_dict = payload.get("state_dict", {})
    model.load_state_dict(state_dict)
    model.eval()
    metadata = dict(payload.get("metadata", {}))
    return model, cfg, metadata


def describe_checkpoint(path: str | Path) -> dict[str, Any]:
    model, cfg, metadata = load_model(str(Path(path)))
    return {
        "path": str(Path(path)),
        "config": asdict(cfg),
        "metadata": metadata,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }


def write_checkpoint_card(path: str | Path, out_path: str | Path) -> Path:
    summary = describe_checkpoint(path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DeepOQE Checkpoint Card",
        "",
        f"- Checkpoint: `{summary['path']}`",
        f"- Parameters: `{summary['n_parameters']}`",
        f"- Config: `{json.dumps(summary['config'], sort_keys=True)}`",
        f"- Metadata: `{json.dumps(summary['metadata'], sort_keys=True)}`",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
