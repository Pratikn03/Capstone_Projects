from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SOCTelemetryFaultConfig:
    dropout_prob: float = 0.0          # probability SOC packet missing at a step
    stale_prob: float = 0.0            # probability SOC repeats previous value
    noise_std_mwh: float = 0.0         # measurement noise
    seed: int = 0


class SOCTelemetryChannel:
    """
    Produces soc_obs from soc_true with fault injection.
    This is intentionally simple and deterministic per seed.
    """
    def __init__(self, cfg: SOCTelemetryFaultConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.seed))
        self.last_obs: float | None = None

    def observe(self, soc_true: float) -> tuple[float, dict]:
        meta = {"dropout": False, "stale": False, "noise": 0.0}
        # dropout -> reuse last_obs (or truth if first)
        if self.rng.random() < float(self.cfg.dropout_prob):
            meta["dropout"] = True
            if self.last_obs is None:
                self.last_obs = float(soc_true)
            return float(self.last_obs), meta

        # stale -> repeat last
        if self.last_obs is not None and self.rng.random() < float(self.cfg.stale_prob):
            meta["stale"] = True
            return float(self.last_obs), meta

        # normal measurement + noise
        noise = float(self.rng.normal(0.0, float(self.cfg.noise_std_mwh)))
        meta["noise"] = noise
        obs = float(soc_true + noise)
        self.last_obs = obs
        return obs, meta
