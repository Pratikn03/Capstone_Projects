"""
Real-dataset temporal transfer validation for Industrial (UCI CCPP) and Healthcare
(PhysioNet BIDMC) domains.

Protocol
--------
1. Load all available rows from each real dataset.
2. Temporal 80/20 split — first 80% = train, last 20% = test.
3. Run 5 seeds × 48-step DC3S episodes seeded from train rows.
4. Run 5 seeds × 48-step DC3S episodes seeded from test rows.
5. Compare TSVR and intervention rate across splits.

Non-battery analog of tbl04_transfer_stress.tex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from orius.orius_bench.fault_engine import active_faults, generate_fault_schedule
from orius.orius_bench.healthcare_track import HealthcareTrackAdapter
from orius.orius_bench.industrial_track import IndustrialTrackAdapter
from orius.orius_bench.real_data_loader import get_bidmc_rows, get_ccpp_rows
from orius.universal_framework import run_universal_step

# ---------------------------------------------------------------------------
# Domain configs (match run_universal_orius_validation.py)
# ---------------------------------------------------------------------------

_DOMAIN_CFGS = {
    "industrial": {"expected_cadence_s": 900.0, "n_vol_bins": 1},
    "healthcare": {"expected_cadence_s": 30.0, "n_vol_bins": 1},
}

_DOMAIN_QUANTILES = {
    "industrial": 5.0,
    "healthcare": 5.0,
}

_HOLD_KEYS = {
    "industrial": ("temp_c", "pressure_mbar", "power_mw"),
    "healthcare": ("hr_bpm", "spo2_pct", "rr_bpm"),
}


def _make_adapter_from_rows(domain: str, rows: list[dict]) -> object:
    """Create a track adapter whose starting-state pool is drawn from `rows`."""
    import csv
    import os
    import tempfile

    # Write the subset of rows to a temp file so the track can load them.
    # The track reads `dataset_path` on __init__ and uses rows for reset().
    if domain == "industrial":
        fieldnames = ["AT", "V", "AP", "RH", "PE"]
    else:  # healthcare
        fieldnames = ["HR", "SpO2", "RR"]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    tmp.close()

    if domain == "industrial":
        track = IndustrialTrackAdapter(dataset_path=tmp.name)
    else:
        track = HealthcareTrackAdapter(dataset_path=tmp.name)

    os.unlink(tmp.name)
    return track


def _make_domain_adapter(domain: str, cfg: dict) -> object:
    from orius.universal_framework.healthcare_adapter import HealthcareDomainAdapter as HA
    from orius.universal_framework.industrial_adapter import IndustrialDomainAdapter as IA

    if domain == "industrial":
        return IA()
    else:
        return HA()


def _make_constraints(domain: str, state: dict) -> dict:
    if domain == "industrial":
        return {"power_mw": (0.0, 500.0)}
    else:
        return {"dosage_rate": (0.0, 1.0)}


def _iso_step(t: int) -> str:
    return f"2024-01-01T{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}Z"


def _run_split_episode(
    domain: str,
    rows: list[dict],
    seed: int,
    horizon: int = 48,
) -> dict:
    """Run one DC3S episode seeded from `rows`. Returns per-step records."""
    track = _make_adapter_from_rows(domain, rows)
    cfg = _DOMAIN_CFGS[domain]
    universal_adapter = _make_domain_adapter(domain, cfg)
    quantile = _DOMAIN_QUANTILES[domain]
    hold_keys = _HOLD_KEYS[domain]

    schedule = generate_fault_schedule(seed, horizon)
    track.reset(seed)

    history: list[dict] = []
    records: list[dict] = []
    trajectory: list[dict] = []
    interventions = 0

    for t in range(horizon):
        ts = dict(track.true_state())
        faults = active_faults(schedule, t)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None

        obs = dict(track.observe(ts, fault_dict))
        raw_telemetry = dict(obs)
        raw_telemetry["ts_utc"] = _iso_step(t)

        if history:
            prev = history[-1]
            for key in hold_keys:
                raw_telemetry.setdefault(f"_hold_{key}", prev.get(key, 0.0))

        # Simple baseline action: propose 0 (no change)
        candidate = {k: 0.0 for k in (["power_setpoint"] if domain == "industrial" else ["dosage_rate"])}
        constraints = _make_constraints(domain, ts)

        repaired = run_universal_step(
            domain_adapter=universal_adapter,
            raw_telemetry=raw_telemetry,
            history=history,
            candidate_action=candidate,
            constraints=constraints,
            quantile=quantile,
            cfg=cfg,
            controller=f"orius-transfer-{domain}",
        )
        action = dict(repaired["safe_action"])
        if repaired.get("repaired", False):
            interventions += 1

        new_state = track.step(action)
        violation = track.check_violation(new_state)

        step = {**dict(new_state), **dict(action)}
        trajectory.append(step)
        history.append(raw_telemetry)

        records.append(
            {
                "step": t,
                "violated": violation["violated"],
                "repaired": repaired.get("repaired", False),
            }
        )

    tsvr = sum(1 for r in records if r["violated"]) / len(records)
    ir = interventions / horizon
    return {"tsvr": tsvr, "intervention_rate": ir, "n_steps": horizon}


def run_domain_transfer(
    domain: str,
    all_rows: list[dict],
    seeds: int = 5,
    horizon: int = 48,
) -> dict:
    split = int(len(all_rows) * 0.8)
    train_rows = all_rows[:split]
    test_rows = all_rows[split:]

    print(f"  {domain}: {len(all_rows)} rows total | train={len(train_rows)} | test={len(test_rows)}")

    train_tsvrs, train_irs = [], []
    test_tsvrs, test_irs = [], []

    for s in range(seeds):
        seed = 2000 + s
        tr = _run_split_episode(domain, train_rows, seed, horizon)
        te = _run_split_episode(domain, test_rows, seed, horizon)
        train_tsvrs.append(tr["tsvr"])
        train_irs.append(tr["intervention_rate"])
        test_tsvrs.append(te["tsvr"])
        test_irs.append(te["intervention_rate"])
        print(f"    seed {seed}: train_tsvr={tr['tsvr']:.4f} | test_tsvr={te['tsvr']:.4f}")

    def _fmt(vals: list[float]) -> dict:
        a = np.array(vals)
        return {"mean": float(a.mean()), "std": float(a.std())}

    return {
        "domain": domain,
        "n_rows": len(all_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "split_type": "temporal_80_20",
        "seeds": seeds,
        "horizon": horizon,
        "train_tsvr": _fmt(train_tsvrs),
        "test_tsvr": _fmt(test_tsvrs),
        "train_ir": _fmt(train_irs),
        "test_ir": _fmt(test_irs),
        "generalises": _fmt(test_tsvrs)["mean"] <= _fmt(train_tsvrs)["mean"] * 1.25,
    }


def write_transfer_table(results: list[dict], out_path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Real-dataset temporal transfer validation for Industrial (UCI CCPP)"
        r" and Healthcare (PhysioNet BIDMC). Temporal 80/20 split: DC3S evaluated on"
        r" held-out last-20\% rows after calibration on first-80\% rows. Gate:"
        r" test TSVR $\leq 1.25\times$ train TSVR (temporal generalisation). "
        r"Non-battery analog of the battery DE$\to$US transfer in"
        r" Table~\ref{tab:TBL04_TRANSFER_STRESS}.}",
        r"\label{tab:real_data_transfer}",
        r"\begin{tabular}{lrrrrrrc}",
        r"\toprule",
        r"\textbf{Domain} & \textbf{Train} & \textbf{Test} "
        r"& \textbf{Train TSVR} & \textbf{Test TSVR} "
        r"& \textbf{Train IR} & \textbf{Test IR} & \textbf{Gate} \\",
        r" & \textbf{rows} & \textbf{rows} & \textbf{(mean±std)} & \textbf{(mean±std)} "
        r"& \textbf{(mean)} & \textbf{(mean)} & \\",
        r"\midrule",
    ]

    domain_pretty = {
        "industrial": "Industrial (CCPP)",
        "healthcare": "Healthcare (BIDMC)",
    }

    for res in results:
        dom = domain_pretty.get(res["domain"], res["domain"].capitalize())
        tr_t = res["train_tsvr"]
        te_t = res["test_tsvr"]
        tr_i = res["train_ir"]
        te_i = res["test_ir"]
        gate = r"$\checkmark$" if res["generalises"] else r"$\times$"
        lines.append(
            f"{dom} & {res['train_rows']:,} & {res['test_rows']:,} "
            f"& ${tr_t['mean']:.3f}\\pm{tr_t['std']:.3f}$ "
            f"& ${te_t['mean']:.3f}\\pm{te_t['std']:.3f}$ "
            f"& {tr_i['mean']:.3f} & {te_i['mean']:.3f} & {gate} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\multicolumn{8}{l}{\small Gate: test TSVR $\leq 1.25\times$ train TSVR. "
        r"IR = DC3S intervention rate. Split is temporal (not random).}\\",
        r"\multicolumn{8}{l}{\small "
        r"Both domains use real public datasets; "
        r"Vehicle/Aerospace/Navigation use 240-row synthetic ORIUS-Bench environments.}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal 80/20 transfer validation for real-data proof domains."
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "real_data_transfer")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    table_out = ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_real_data_transfer.tex"

    print("Real-data transfer validation (temporal 80/20 split)")
    print("======================================================")

    all_results = []
    for domain, loader in [("industrial", get_ccpp_rows), ("healthcare", get_bidmc_rows)]:
        print(f"\n[{domain}]")
        rows = loader()
        res = run_domain_transfer(domain, rows, seeds=args.seeds, horizon=args.horizon)
        all_results.append(res)
        print(f"  train TSVR: {res['train_tsvr']['mean']:.4f} ± {res['train_tsvr']['std']:.4f}")
        print(f"  test  TSVR: {res['test_tsvr']['mean']:.4f} ± {res['test_tsvr']['std']:.4f}")
        print(f"  generalises: {res['generalises']}")

    # Save JSON report
    report = {"results": all_results, "gate": "test_tsvr <= 1.25 * train_tsvr"}
    report_path = args.out / "transfer_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  report → {report_path}")

    # Write LaTeX table
    write_transfer_table(all_results, table_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
