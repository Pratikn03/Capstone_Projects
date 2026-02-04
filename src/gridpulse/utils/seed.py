"""Utilities: seed."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> int:
    """Set deterministic seeds for python, numpy, and torch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed
