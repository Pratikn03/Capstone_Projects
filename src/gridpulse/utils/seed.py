"""
Utilities: Random seed management for reproducibility.

This module provides centralized seed setting to ensure reproducible results
across Python's random module, NumPy, and PyTorch. Reproducibility is critical
for ML experiments to enable result validation and debugging.

Why different seeds are needed:
- Python's `random` module: Used by data loading shufflers
- NumPy's random: Used by sklearn, feature engineering, and data splitting
- PyTorch: Used by neural network weight initialization and dropout

Usage:
    >>> from gridpulse.utils.seed import set_seed
    >>> set_seed(42)  # Call at the start of any training script
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> int:
    """
    Set deterministic random seeds across all libraries for reproducibility.
    
    This function should be called at the very beginning of training scripts,
    before any random operations occur. It configures:
    
    1. PYTHONHASHSEED environment variable (affects dict ordering in Python 3.3+)
    2. Python's built-in random module
    3. NumPy's random number generator
    4. PyTorch's random generators (gracefully skipped if PyTorch unavailable)
    
    Args:
        seed: Integer seed value. Default is 42 (the answer to everything).
        
    Returns:
        The seed that was set (useful for logging/verification).
        
    Example:
        # At the start of your training script:
        from gridpulse.utils.seed import set_seed
        seed = set_seed(2024)
        print(f"Reproducibility seed set to: {seed}")
        
    Note:
        For full PyTorch reproducibility, you may also need to set:
        - CUBLAS_WORKSPACE_CONFIG=:4096:8 (for CUDA operations)
        - Use torch.use_deterministic_algorithms(True)
        
        However, these may impact performance significantly.
    """
    # Set Python's hash seed for deterministic string hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Set Python's built-in random module
    random.seed(seed)
    
    # Set NumPy's random generator
    np.random.seed(seed)
    
    # Try to set PyTorch seeds (may not be installed in all environments)
    try:
        import torch
        
        # CPU random seed
        torch.manual_seed(seed)
        
        # GPU random seeds (if CUDA is available)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic operations (trades speed for reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    except ImportError:
        # PyTorch not installed - this is fine for non-DL workflows
        pass
    except Exception:
        # Other PyTorch configuration errors - log but continue
        pass
    
    return seed

