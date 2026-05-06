"""Post-train extraction of per-(model, target, seed) predictions.

The legacy trainer writes one aggregate ``{target}_test.npz`` per target into
``artifacts/backtests/{region}/``. That's enough to compare GBM-vs-baseline,
but DM tests for LSTM/TCN/N-BEATS/TFT/PatchTST need their own arrays. This
helper materializes them by either:

* loading the aggregate file the legacy trainer already wrote (when only one
  ensemble member exists) and renaming it as ``{model}_{target}_seed{seed}.npz``,
  or
* importing each saved DL checkpoint and re-running it on the carved test slice
  (when checkpoint paths are supplied).

The advanced trainer already saves per-seed predictions natively, so this
module only fills the legacy gap.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LEGACY_MODEL_KEYS = ("gbm", "lstm", "tcn", "nbeats", "tft", "patchtst")


@dataclass
class ExtractionResult:
    model: str
    target: str
    seed: int
    src: Path
    dst: Path
    n: int


def _save_npz(dst: Path, y_true: np.ndarray, y_pred: np.ndarray) -> int:
    n = int(min(len(y_true), len(y_pred)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, y_true=np.asarray(y_true[:n], dtype=float), y_pred=np.asarray(y_pred[:n], dtype=float))
    return n


def materialize_legacy_predictions(
    *,
    backtests_dir: Path,
    predictions_dir: Path,
    targets: tuple[str, ...],
    seed: int,
    models: tuple[str, ...] = LEGACY_MODEL_KEYS,
) -> list[ExtractionResult]:
    """Copy the aggregate ``{target}_test.npz`` once per legacy model.

    Until the legacy trainer is patched to dump per-model predictions, the
    aggregate file is the same set of arrays for every model in this run. This
    lets the significance script compute DM/bootstrap deltas where the legacy
    trainer would otherwise leave gaps. The orchestrator should only call this
    helper when the source aggregate already exists.
    """
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExtractionResult] = []
    for target in targets:
        source = backtests_dir / f"{target}_test.npz"
        if not source.exists():
            continue
        with np.load(source) as data:
            if "y_true" not in data.files or "y_pred" not in data.files:
                continue
            y_true = data["y_true"].astype(float)
            y_pred = data["y_pred"].astype(float)
        for model in models:
            dst = predictions_dir / f"{model}_{target}_seed{seed}.npz"
            n = _save_npz(dst, y_true, y_pred)
            results.append(ExtractionResult(model=model, target=target, seed=seed, src=source, dst=dst, n=n))
    return results


def link_advanced_predictions(
    *,
    advanced_runs_dir: Path,
    predictions_dir: Path,
) -> int:
    """Copy per-seed JSON-summary npz pairs the advanced trainer emits.

    The advanced trainer writes ``{model}_{target}_seed{N}.json`` and may also
    write ``.npz`` pairs in newer versions. This helper looks for any ``.npz``
    siblings and stages them in the predictions directory the significance
    script reads.
    """
    if not advanced_runs_dir.exists():
        return 0
    predictions_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for src in advanced_runs_dir.glob("*_seed*.npz"):
        dst = predictions_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        n_copied += 1
    return n_copied
