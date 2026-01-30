from __future__ import annotations
import json
from pathlib import Path

def save_dispatch(plan: dict, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(plan, indent=2), encoding="utf-8")
