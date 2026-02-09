"""
Optimization: Dispatch plan persistence helpers.

This module handles saving and loading dispatch plans to/from disk.
Dispatch plans contain the battery charge/discharge schedule optimized
for cost and carbon objectives.

Typical usage:
    >>> from gridpulse.optimizer.dispatch import save_dispatch
    >>> save_dispatch(plan_dict, "artifacts/dispatch_plans/latest.json")
"""
from __future__ import annotations
import json
from pathlib import Path


def save_dispatch(plan: dict, out_path: str):
    """Save a dispatch plan to disk as JSON.
    
    Args:
        plan: Dictionary containing dispatch schedule and metadata
              Expected keys: soc, charge_mw, discharge_mw, grid_mw, etc.
        out_path: Output file path (parent directories created if needed)
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(plan, indent=2), encoding="utf-8")
