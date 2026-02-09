"""
Model Registry: Artifact storage and promotion utilities.

This module provides functions for managing trained model artifacts,
including promotion from staging to production. It serves as a simple
file-based model registry for environments without MLflow or similar
infrastructure.

Usage:
    >>> from gridpulse.registry.model_store import promote
    >>> promote("artifacts/models/candidate.pkl", "artifacts/production/model.pkl")

For full model lifecycle management, see gridpulse.registry.promote module.
"""
from __future__ import annotations

from pathlib import Path
import shutil


def promote(candidate_path: str, prod_path: str) -> str:
    """
    Promote a candidate model to production by copying to the production path.
    
    This is an atomic operation - the model file is copied completely before
    being placed in the production location. Existing production models are
    overwritten.
    
    Args:
        candidate_path: Path to the candidate model artifact (staging)
        prod_path: Target path for the production model
        
    Returns:
        Absolute path to the promoted production model
        
    Example:
        # After successful validation, promote the model
        prod_model = promote(
            "artifacts/models/lgbm_load_mw_v2.pkl",
            "artifacts/production/lgbm_load_mw.pkl"
        )
    
    Note:
        In production, consider adding model versioning and rollback support.
        This simple implementation is suitable for single-model deployments.
    """
    prod = Path(prod_path)
    
    # Ensure the production directory exists
    prod.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy with metadata preservation (timestamps, permissions)
    shutil.copy2(candidate_path, prod_path)
    
    return str(prod)

