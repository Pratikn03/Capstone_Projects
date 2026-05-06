#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from _battery_wrappers_common import REPO_ROOT, ensure_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Make probing figures wrapper")
    p.add_argument("--out-dir", default="reports/probing")
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    df = pd.read_csv(REPO_ROOT / "reports/publication/active_probing_spoofing_detection.csv")
    row = df.iloc[0]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["precision", "recall"], [float(row["precision"]), float(row["recall"])])
    ax.set_ylim(0, 1)
    ax.set_title("Active Probing Detection Quality")
    fig.tight_layout()
    fig.savefig(out / "probing_detection_quality.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
