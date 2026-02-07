"""Forecast plotting helpers with confidence bands."""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt


def plot_forecast_with_bands(
    yhat: Sequence[float],
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    title: str = "Forecast (with intervals)",
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yhat, label="forecast", linewidth=2)

    if lower is not None and upper is not None:
        ax.fill_between(range(len(yhat)), lower, upper, alpha=0.25, label="PI band")

    ax.set_title(title)
    ax.set_xlabel("horizon step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig
