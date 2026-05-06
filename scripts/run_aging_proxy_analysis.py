#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd
from _battery_wrappers_common import REPO_ROOT, ensure_dir, write_manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Run aging proxy analysis wrapper")
    p.add_argument("--out-dir", default="reports/aging")
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    ft = pd.read_csv(REPO_ROOT / "reports/publication/fault_performance_table.csv").rename(
        columns={"TSVR (%)": "tsvr_pct", "Sev P95 (MWh)": "sev_p95", "Controller": "controller"}
    )
    base = ft[ft["controller"] == "deterministic_lp"][
        ["Fault Dimension", "Severity", "tsvr_pct", "sev_p95"]
    ].rename(columns={"tsvr_pct": "base_tsvr", "sev_p95": "base_sev"})
    dc = ft[ft["controller"] == "dc3s_ftit"][["Fault Dimension", "Severity", "tsvr_pct", "sev_p95"]].rename(
        columns={"tsvr_pct": "dc_tsvr", "sev_p95": "dc_sev"}
    )
    m = base.merge(dc, on=["Fault Dimension", "Severity"], how="inner")
    m["avoided_violation_pct"] = m["base_tsvr"] - m["dc_tsvr"]
    m["avoided_severity_mwh"] = m["base_sev"] - m["dc_sev"]
    m["annual_fade_proxy_pct"] = m["avoided_severity_mwh"].clip(lower=0) * 0.04
    m["ten_year_depreciation_avoided_usd"] = m["annual_fade_proxy_pct"].clip(lower=0) * 10 * 2500
    m.to_csv(out / "asset_preservation_proxy_table.csv", index=False)
    write_manifest(
        out,
        "run_aging_proxy_analysis_manifest.json",
        {"rows": int(len(m)), "source": "reports/publication/fault_performance_table.csv"},
    )


if __name__ == "__main__":
    main()
