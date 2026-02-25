from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SOCTelemetryFaultConfig:
    dropout_prob: float = 0.0
    stale_prob: float = 0.0
    noise_std_mwh: float = 0.0
    seed: int = 0


class SOCTelemetryChannel:
    """Produce observed SOC from true SOC with deterministic fault injection."""

    def __init__(self, cfg: SOCTelemetryFaultConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.seed))
        self.last_obs: float | None = None

    def observe(self, soc_true: float) -> tuple[float, dict[str, float | bool]]:
        meta: dict[str, float | bool] = {"dropout": False, "stale": False, "noise": 0.0}
        if self.rng.random() < float(self.cfg.dropout_prob):
            meta["dropout"] = True
            if self.last_obs is None:
                self.last_obs = float(soc_true)
            return float(self.last_obs), meta

        if self.last_obs is not None and self.rng.random() < float(self.cfg.stale_prob):
            meta["stale"] = True
            return float(self.last_obs), meta

        noise = float(self.rng.normal(0.0, float(self.cfg.noise_std_mwh)))
        meta["noise"] = noise
        obs = float(soc_true + noise)
        self.last_obs = obs
        return obs, meta
