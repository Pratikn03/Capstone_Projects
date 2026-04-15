"""Shared domain-native reliability helpers for non-battery adapters."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping, Sequence

from orius.dc3s.quality import compute_reliability


FeatureSource = str | Callable[[Mapping[str, Any]], Any]


def _source_descriptor(source: FeatureSource) -> str:
    if isinstance(source, str):
        return source
    name = getattr(source, "__name__", source.__class__.__name__)
    return f"derived:{name}"


def _feature_value(state: Mapping[str, Any], source: FeatureSource) -> Any:
    if isinstance(source, str):
        return state.get(source)
    return source(state)


def _build_event(
    state: Mapping[str, Any],
    feature_sources: Mapping[str, FeatureSource],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    event: dict[str, Any] = {"ts_utc": state.get("ts_utc", state.get("timestamp", ""))}
    basis_signals: list[dict[str, str]] = []
    for signal_name, source in feature_sources.items():
        event[signal_name] = _feature_value(state, source)
        basis_signals.append(
            {
                "signal_name": str(signal_name),
                "source": _source_descriptor(source),
            }
        )
    return event, basis_signals


def assess_domain_reliability(
    *,
    domain_id: str,
    state: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None,
    feature_sources: Mapping[str, FeatureSource],
    expected_cadence_s: float,
    reliability_cfg: Mapping[str, Any] | None,
    ftit_cfg: Mapping[str, Any] | None,
    runtime_surface: str,
    closure_tier: str,
) -> tuple[float, Mapping[str, Any]]:
    """Score reliability on domain-native signals and record the feature basis used."""
    event, basis_signals = _build_event(state, feature_sources)
    last_event = None
    if history:
        last_event, _ = _build_event(history[-1], feature_sources)
    w_t, flags = compute_reliability(
        event,
        last_event,
        expected_cadence_s=float(expected_cadence_s),
        reliability_cfg=reliability_cfg or {},
        ftit_cfg=ftit_cfg or {},
    )
    basis = {
        "domain": str(domain_id),
        "timestamp_field": "ts_utc",
        "domain_native": True,
        "signal_names": [row["signal_name"] for row in basis_signals],
        "source_fields": [row["source"] for row in basis_signals],
        "signals": basis_signals,
    }
    return float(w_t), {
        "flags": dict(flags),
        "runtime_surface": str(runtime_surface),
        "closure_tier": str(closure_tier),
        "reliability_feature_basis": basis,
    }
