"""kga.observability.metrics -- Production Prometheus metrics collector and exporter."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple


class InMemoryMetricsCollector:
    """Thread-safe, zero-dependency Prometheus-compatible metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Counters: (name, label_tuple) -> count
        self._counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        # Gauges: (name, label_tuple) -> value
        self._gauges: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}

    def inc_counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        label_tuple = tuple(sorted(labels.items()))
        with self._lock:
            key = (name, label_tuple)
            self._counters[key] = self._counters.get(key, 0.0) + value

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        label_tuple = tuple(sorted(labels.items()))
        with self._lock:
            key = (name, label_tuple)
            self._gauges[key] = float(value)

    def generate_prometheus_text(self) -> str:
        """Render metrics in standard Prometheus text exposition format (0.0.4)."""
        lines: List[str] = []
        with self._lock:
            # Group counters
            counter_names = sorted(set(k[0] for k in self._counters.keys()))
            for name in counter_names:
                lines.append(f"# TYPE {name} counter")
                for (metric_name, label_tuple), val in sorted(self._counters.items()):
                    if metric_name == name:
                        if label_tuple:
                            label_str = ",".join(f'{k}="{v}"' for k, v in label_tuple)
                            lines.append(f"{name}{{{label_str}}} {val}")
                        else:
                            lines.append(f"{name} {val}")

            # Group gauges
            gauge_names = sorted(set(k[0] for k in self._gauges.keys()))
            for name in gauge_names:
                lines.append(f"# TYPE {name} gauge")
                for (metric_name, label_tuple), val in sorted(self._gauges.items()):
                    if metric_name == name:
                        if label_tuple:
                            label_str = ",".join(f'{k}="{v}"' for k, v in label_tuple)
                            lines.append(f"{name}{{{label_str}}} {val}")
                        else:
                            lines.append(f"{name} {val}")

        return "\n".join(lines) + "\n"


# Global singleton instance for easy import
METRICS = InMemoryMetricsCollector()
