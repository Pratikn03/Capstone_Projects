from __future__ import annotations

import numpy as np
import pandas as pd

from gridpulse.cpsbench_iot.baselines import aci_conformal_dispatch
from gridpulse.cpsbench_iot.scenarios import FAULT_COLUMNS, generate_episode


def _to_telemetry_events(x_obs: pd.DataFrame, event_log: pd.DataFrame) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for idx in range(len(x_obs)):
        payload: dict[str, object] = {
            "ts_utc": pd.to_datetime(event_log.loc[idx, "arrived_timestamp"], utc=True).isoformat(),
            "device_id": "bench-device",
            "zone_id": "DE",
            "load_mw": float(x_obs.loc[idx, "load_mw"]),
            "renewables_mw": float(x_obs.loc[idx, "renewables_mw"]),
        }
        for fault_col in FAULT_COLUMNS:
            payload[fault_col] = bool(event_log.loc[idx, fault_col])
        events.append(payload)
    return events


def test_aci_conformal_dispatch_smoke() -> None:
    horizon = 24
    x_obs, x_true, event_log = generate_episode("drift_combo", seed=11, horizon=horizon)
    telemetry_events = _to_telemetry_events(x_obs=x_obs, event_log=event_log)

    res = aci_conformal_dispatch(
        load_forecast=x_obs["load_mw"].to_numpy(dtype=float),
        renewables_forecast=x_obs["renewables_mw"].to_numpy(dtype=float),
        load_true=x_true["load_mw"].to_numpy(dtype=float),
        telemetry_events=telemetry_events,
        price=x_obs["price_per_mwh"].to_numpy(dtype=float),
    )

    assert res["policy"] == "aci_conformal"
    assert len(res["interval_lower"]) == horizon
    assert len(res["interval_upper"]) == horizon
    assert len(res["proposed_charge_mw"]) == horizon
    assert len(res["proposed_discharge_mw"]) == horizon
    assert len(res["safe_charge_mw"]) == horizon
    assert len(res["safe_discharge_mw"]) == horizon
    assert len(res["soc_mwh"]) == horizon
    assert len(res["certificates"]) == horizon

    lower = np.asarray(res["interval_lower"], dtype=float)
    upper = np.asarray(res["interval_upper"], dtype=float)
    assert np.all(lower <= upper)
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert np.all(np.isfinite(np.asarray(res["proposed_charge_mw"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(res["proposed_discharge_mw"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(res["safe_charge_mw"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(res["safe_discharge_mw"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(res["soc_mwh"], dtype=float)))
    assert np.isfinite(float(res["expected_cost_usd"]))
    if res["carbon_kg"] is not None:
        assert np.isfinite(float(res["carbon_kg"]))
