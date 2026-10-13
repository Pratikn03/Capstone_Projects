"""kga.guards.streaming_monitor -- Sequential e-value supermartingale drift monitor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class MonitorStatus:
    """Snapshot of sequential drift monitoring state."""
    step: int
    wealth: float
    threshold: float
    tripped: bool
    last_evalue: float
    consecutive_alarms: int
    trip_reason: Optional[str] = None


class StreamingDriftMonitor:
    """Sequential anytime drift and harm monitor using Ville's inequality.
    
    Tracks cumulative betting wealth W_t = prod_{i=1}^t E_i.
    Under the safe null hypothesis, E_t is a supermartingale (E[E_t | F_{t-1}] <= 1),
    guaranteeing that P(exists t: W_t >= 1 / alpha) <= alpha for all horizons.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        wealth_cap: float = 1e6,
        discount_factor: float = 1.0,
    ) -> None:
        self.alpha = float(alpha)
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        
        self.threshold = 1.0 / self.alpha
        self.wealth_cap = float(wealth_cap)
        self.discount_factor = float(discount_factor)

        self._step: int = 0
        self._wealth: float = 1.0
        self._last_evalue: float = 1.0
        self._tripped: bool = False
        self._trip_reason: Optional[str] = None
        self._history: List[float] = []

    @property
    def current_wealth(self) -> float:
        return self._wealth

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def update(self, evalue: float) -> MonitorStatus:
        """Observe new e-value E_t from current evaluation batch or stream step."""
        e = float(evalue)
        if not np.isfinite(e) or e < 0.0:
            e = 0.0  # Safe lower clamp

        self._step += 1
        self._last_evalue = e

        # Discounted wealth update: W_t = (W_{t-1}^gamma) * E_t
        if self.discount_factor < 1.0:
            discounted_prev = self._wealth ** self.discount_factor
            self._wealth = min(self.wealth_cap, discounted_prev * e)
        else:
            self._wealth = min(self.wealth_cap, self._wealth * e)

        self._history.append(self._wealth)

        if self._wealth >= self.threshold and not self._tripped:
            self._tripped = True
            self._trip_reason = (
                f"Martingale wealth {self._wealth:.2f} crossed anytime threshold "
                f"{self.threshold:.2f} (alpha={self.alpha:.2f}) at step {self._step}"
            )

        return MonitorStatus(
            step=self._step,
            wealth=self._wealth,
            threshold=self.threshold,
            tripped=self._tripped,
            last_evalue=e,
            consecutive_alarms=1 if self._tripped else 0,
            trip_reason=self._trip_reason,
        )

    def update_from_margin(self, delta_sample: float, margin_null: float = 0.0, c: float = 0.5) -> MonitorStatus:
        """Construct an anytime betting factor e-value from a observed performance difference.
        
        E_t = 1 + lambda * (margin_null - delta_sample)
        where positive observed benefit decreases wealth, and negative benefit (harm) increases wealth.
        """
        # Bounded bet lambda in [0, 1]
        lam = float(c)
        harm = margin_null - delta_sample
        # Truncated betting term to ensure non-negativity
        e = max(0.0, 1.0 + lam * np.clip(harm, -1.0, 1.0))
        return self.update(e)

    def reset(self) -> None:
        """Reset martingale betting wealth to baseline 1.0."""
        self._step = 0
        self._wealth = 1.0
        self._last_evalue = 1.0
        self._tripped = False
        self._trip_reason = None
        self._history.clear()
