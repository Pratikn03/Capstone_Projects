"""kga.integrations.torch_adapter -- PyTorch and Torchvision model integration adapter."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np


class TorchModelAdapter:
    """Production adapter bridging PyTorch nn.Module models with the KGA Autonomous Safety Gateway.
    
    Provides standardized forward inference, softmax probability conversion,
    and label-free evidence feature extraction.
    """

    def __init__(
        self,
        model: Any,
        device: Optional[str] = None,
        is_adapted: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.is_adapted = is_adapted

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Execute forward pass and return softmax probabilities as numpy array."""
        try:
            import torch

            # Determine device
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(dev)
            self.model.eval()

            tensor_x = torch.as_tensor(x, dtype=torch.float32, device=dev)
            with torch.no_grad():
                logits = self.model(tensor_x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=-1)
            return probs.detach().cpu().numpy()
        except ImportError:
            # Zero-dependency duck-typing fallback for environments without torch
            if callable(self.model):
                out = self.model(x)
                arr = np.asarray(out, dtype=float)
                # Apply softmax if logits
                if np.any(arr < 0.0) or np.any(arr > 1.0) or not np.allclose(np.sum(arr, axis=-1), 1.0, atol=1e-2):
                    exp_arr = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
                    arr = exp_arr / np.sum(exp_arr, axis=-1, keepdims=True)
                return arr
            raise RuntimeError("PyTorch is required for TorchModelAdapter with nn.Module instances")

    @staticmethod
    def extract_standard_evidence_features(
        x: np.ndarray,
        base_probs: np.ndarray,
        adapted_probs: np.ndarray,
    ) -> np.ndarray:
        """Extract standardized 4-dimensional label-free evidence vector Z.
        
        Features:
        1. Mean prediction confidence difference (entropy reduction).
        2. Prediction disagreement rate between f0 and fa.
        3. Mean maximum predicted probability (sharpness).
        4. Average top-2 probability margin.
        """
        p0 = np.asarray(base_probs, dtype=float)
        pa = np.asarray(adapted_probs, dtype=float)

        # 1. Disagreement rate
        preds_0 = np.argmax(p0, axis=-1)
        preds_a = np.argmax(pa, axis=-1)
        disagree_rate = float(np.mean(preds_0 != preds_a))

        # 2. Mean confidence (max prob)
        conf_0 = np.max(p0, axis=-1)
        conf_a = np.max(pa, axis=-1)
        conf_diff = float(np.mean(conf_a - conf_0))
        mean_conf_a = float(np.mean(conf_a))

        # 3. Top-2 margin
        sorted_pa = np.sort(pa, axis=-1)
        if sorted_pa.shape[-1] >= 2:
            top2_margins = sorted_pa[..., -1] - sorted_pa[..., -2]
            mean_margin = float(np.mean(top2_margins))
        else:
            mean_margin = 1.0

        return np.array([disagree_rate, conf_diff, mean_conf_a, mean_margin], dtype=float)
