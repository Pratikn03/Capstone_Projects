"""Deterministic CPSBench-IoT episode generation with configurable faults and drift."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SCENARIOS = ("nominal", "dropout", "delay_jitter", "out_of_order", "spikes", "drift_combo")
FAULT_COLUMNS = (
    "dropout",
    "delay_jitter",
    "out_of_order",
    "spikes",
    "stale_sensor",
    "covariate_drift",
    "label_drift",
)


@dataclass(frozen=True)
class EpisodeArtifacts:
    """Container for generated scenario artifacts."""

    x_obs: pd.DataFrame
    x_true: pd.DataFrame
    event_log: pd.DataFrame


def _base_episode(rng: np.random.Generator, horizon: int) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=horizon, freq="h", tz="UTC")
    idx = np.arange(horizon, dtype=float)
    hour = ts.hour.to_numpy(dtype=float)
    dow = ts.dayofweek.to_numpy(dtype=float)

    load = (
        50000.0
        + 6500.0 * np.sin((2.0 * np.pi * hour / 24.0) - 0.8)
        + 2200.0 * np.sin((2.0 * np.pi * dow / 7.0) - 0.4)
        + rng.normal(0.0, 450.0, size=horizon)
    )
    wind = 8500.0 + 2200.0 * np.sin((2.0 * np.pi * (idx + 5.0) / 24.0)) + rng.normal(0.0, 550.0, size=horizon)
    solar_shape = np.sin(np.pi * (hour - 6.0) / 12.0)
    solar = np.maximum(0.0, 7000.0 * solar_shape) + rng.normal(0.0, 180.0, size=horizon)
    renewables = np.maximum(0.0, wind + np.maximum(0.0, solar))

    peak_mask = ((hour >= 17.0) & (hour <= 21.0)).astype(float)
    offpeak_mask = ((hour >= 0.0) & (hour <= 5.0)).astype(float)
    price = 58.0 + 14.0 * peak_mask - 6.0 * offpeak_mask + rng.normal(0.0, 1.2, size=horizon)
    carbon = np.clip(430.0 - 0.0018 * renewables + rng.normal(0.0, 4.0, size=horizon), 180.0, 700.0)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "load_mw": np.maximum(load, 1000.0),
            "renewables_mw": renewables,
            "price_per_mwh": np.maximum(price, 1.0),
            "carbon_kg_per_mwh": carbon,
        }
    )


def _init_event_log(timestamps: pd.Series) -> pd.DataFrame:
    event_log = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "arrived_timestamp": pd.to_datetime(timestamps, utc=True),
        }
    )
    for col in FAULT_COLUMNS:
        event_log[col] = 0
    event_log["delay_steps"] = 0
    event_log["out_of_order_steps"] = 0
    event_log["spike_sigma"] = 0.0
    return event_log


def _apply_dropout(obs: pd.DataFrame, event_log: pd.DataFrame, rng: np.random.Generator, dropout_rate: float = 0.08) -> None:
    rate = max(0.0, min(float(dropout_rate), 1.0))
    if rate <= 0.0:
        return
    count = max(1, int(rate * len(obs)))
    idx = rng.choice(len(obs), size=count, replace=False)
    obs.loc[idx, ["load_mw", "renewables_mw"]] = np.nan
    event_log.loc[idx, "dropout"] = 1


def _apply_delay_jitter(
    event_log: pd.DataFrame,
    rng: np.random.Generator,
    delay_seconds: float = 1.0,
    delay_rate: float = 0.30,
) -> None:
    rate = max(0.0, min(float(delay_rate), 1.0))
    if rate <= 0.0:
        return
    count = max(1, int(rate * len(event_log)))
    idx = rng.choice(len(event_log), size=count, replace=False)
    delay = max(float(delay_seconds), 0.0)
    jitter = rng.normal(loc=delay, scale=max(0.25 * delay, 0.1), size=count)
    event_log.loc[idx, "arrived_timestamp"] = event_log.loc[idx, "arrived_timestamp"] + pd.to_timedelta(jitter, unit="s")
    event_log.loc[idx, "delay_jitter"] = 1
    event_log.loc[idx, "delay_steps"] = np.maximum(1, np.rint(np.abs(jitter) / 3600.0)).astype(int)


def _apply_out_of_order(event_log: pd.DataFrame, rng: np.random.Generator, out_of_order_rate: float = 0.10) -> None:
    if len(event_log) < 8:
        return
    rate = max(0.0, min(float(out_of_order_rate), 1.0))
    if rate <= 0.0:
        return
    count = max(1, int(rate * len(event_log)))
    idx = rng.choice(np.arange(4, len(event_log)), size=count, replace=False)
    rewind = rng.integers(1, 4, size=count)
    event_log.loc[idx, "arrived_timestamp"] = event_log.loc[idx, "arrived_timestamp"] - pd.to_timedelta(rewind, unit="h")
    event_log.loc[idx, "out_of_order"] = 1
    event_log.loc[idx, "out_of_order_steps"] = rewind.astype(int)


def _apply_spikes(
    obs: pd.DataFrame,
    event_log: pd.DataFrame,
    rng: np.random.Generator,
    spike_sigma: float = 1.0,
    spike_rate: float = 0.05,
) -> None:
    rate = max(0.0, min(float(spike_rate), 1.0))
    if rate <= 0.0:
        return
    count = max(1, int(rate * len(obs)))
    idx = rng.choice(len(obs), size=count, replace=False)
    sigma = max(float(spike_sigma), 0.0)
    load_scale = 1.0 + np.abs(rng.normal(loc=0.35 * sigma, scale=0.12 + 0.08 * sigma, size=count))
    ren_scale = np.clip(1.0 + rng.normal(loc=0.0, scale=0.18 + 0.10 * sigma, size=count), 0.2, 2.2)
    obs.loc[idx, "load_mw"] = obs.loc[idx, "load_mw"] * load_scale
    obs.loc[idx, "renewables_mw"] = np.maximum(0.0, obs.loc[idx, "renewables_mw"] * ren_scale)
    event_log.loc[idx, "spikes"] = 1
    event_log.loc[idx, "spike_sigma"] = float(sigma)


def _apply_stale_sensor(obs: pd.DataFrame, event_log: pd.DataFrame, rng: np.random.Generator) -> None:
    if len(obs) < 12:
        return
    starts = rng.choice(np.arange(1, len(obs) - 4), size=2, replace=False)
    for start in starts:
        end = min(start + 4, len(obs))
        for col in ("load_mw", "renewables_mw"):
            obs.loc[start:end - 1, col] = float(obs.loc[start - 1, col])
        event_log.loc[start:end - 1, "stale_sensor"] = 1


def _apply_drift(
    obs: pd.DataFrame,
    true: pd.DataFrame,
    event_log: pd.DataFrame,
    drift_timestep: int,
) -> None:
    start = int(max(0, min(len(obs) - 1, drift_timestep)))
    # Covariate drift: sensor-side renewables become biased low.
    obs.loc[start:, "renewables_mw"] = np.maximum(0.0, obs.loc[start:, "renewables_mw"] * 0.72)
    event_log.loc[start:, "covariate_drift"] = 1

    # Label drift: underlying load regime shifts upward after change-point.
    ramp = np.linspace(1.0, 1.15, num=len(true) - start)
    true.loc[start:, "load_mw"] = true.loc[start:, "load_mw"].to_numpy(dtype=float) * ramp + 1300.0
    event_log.loc[start:, "label_drift"] = 1


def generate_episode(
    scenario: str,
    seed: int,
    horizon: int = 168,
    drift_timestep: int | None = None,
    fault_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate deterministic CPSBench artifacts.

    Returns:
        x_obs: observed (faulted) telemetry frame
        x_true: latent/true frame
        event_log: per-timestep fault + drift metadata
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    scenario_norm = str(scenario).strip().lower()
    rng = np.random.default_rng(int(seed))
    overrides = dict(fault_overrides or {})
    dropout_rate = float(overrides.get("dropout_rate", 0.08))
    delay_seconds = float(overrides.get("delay_seconds", 1.0))
    delay_rate = float(overrides.get("delay_rate", 0.30))
    out_of_order_rate = float(overrides.get("out_of_order_rate", 0.10))
    spike_sigma = float(overrides.get("spike_sigma", 1.0))
    spike_rate = float(overrides.get("spike_rate", 0.05))
    x_true = _base_episode(rng, horizon=horizon)
    x_obs = x_true.copy(deep=True)
    event_log = _init_event_log(x_true["timestamp"])

    if scenario_norm == "nominal":
        pass
    elif scenario_norm == "dropout":
        _apply_dropout(x_obs, event_log, rng, dropout_rate=dropout_rate)
    elif scenario_norm == "delay_jitter":
        _apply_delay_jitter(event_log, rng, delay_seconds=delay_seconds, delay_rate=delay_rate)
    elif scenario_norm == "out_of_order":
        _apply_out_of_order(event_log, rng, out_of_order_rate=out_of_order_rate)
    elif scenario_norm == "spikes":
        _apply_spikes(x_obs, event_log, rng, spike_sigma=spike_sigma, spike_rate=spike_rate)
    elif scenario_norm == "stale_sensor":
        _apply_stale_sensor(x_obs, event_log, rng)
    elif scenario_norm == "drift_combo":
        _apply_delay_jitter(event_log, rng, delay_seconds=delay_seconds, delay_rate=delay_rate)
        _apply_dropout(x_obs, event_log, rng, dropout_rate=dropout_rate)
        _apply_stale_sensor(x_obs, event_log, rng)
        _apply_spikes(x_obs, event_log, rng, spike_sigma=spike_sigma, spike_rate=spike_rate)
        _apply_out_of_order(event_log, rng, out_of_order_rate=out_of_order_rate)
        _apply_drift(x_obs, x_true, event_log, drift_timestep if drift_timestep is not None else horizon // 2)
    else:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Supported scenarios: {', '.join(DEFAULT_SCENARIOS + ('stale_sensor',))}"
        )

    # Optional domain-shift controls used by transfer stress evaluation.
    load_scale = float(overrides.get("load_scale", 1.0))
    renewables_scale = float(overrides.get("renewables_scale", 1.0))
    price_scale = float(overrides.get("price_scale", 1.0))
    carbon_scale = float(overrides.get("carbon_scale", 1.0))
    load_bias_mw = float(overrides.get("load_bias_mw", 0.0))
    renewables_bias_mw = float(overrides.get("renewables_bias_mw", 0.0))
    seasonal_shift_hours = int(overrides.get("seasonal_shift_hours", 0))

    for frame in (x_obs, x_true):
        frame["load_mw"] = np.maximum(0.0, frame["load_mw"] * load_scale + load_bias_mw)
        frame["renewables_mw"] = np.maximum(0.0, frame["renewables_mw"] * renewables_scale + renewables_bias_mw)
        frame["price_per_mwh"] = np.maximum(0.0, frame["price_per_mwh"] * price_scale)
        frame["carbon_kg_per_mwh"] = np.maximum(0.0, frame["carbon_kg_per_mwh"] * carbon_scale)

    if seasonal_shift_hours != 0 and len(x_obs) > 0:
        for frame in (x_obs, x_true):
            for col in ("load_mw", "renewables_mw", "price_per_mwh", "carbon_kg_per_mwh"):
                frame[col] = np.roll(frame[col].to_numpy(dtype=float), seasonal_shift_hours)

    # Forward-fill measurement dropouts for downstream baseline controllers while preserving event flags.
    x_obs[["load_mw", "renewables_mw"]] = x_obs[["load_mw", "renewables_mw"]].ffill().bfill()
    x_obs["timestamp"] = pd.to_datetime(x_obs["timestamp"], utc=True)
    x_true["timestamp"] = pd.to_datetime(x_true["timestamp"], utc=True)
    event_log = event_log.sort_values(["timestamp"]).reset_index(drop=True)
    return x_obs.reset_index(drop=True), x_true.reset_index(drop=True), event_log.reset_index(drop=True)
