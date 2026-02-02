"""Model registry: model store."""
from __future__ import annotations
from pathlib import Path
import shutil

def promote(candidate_path: str, prod_path: str):
    # Key: manage model artifacts and promotion logic
    prod = Path(prod_path)
    prod.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, prod_path)
    return str(prod)
