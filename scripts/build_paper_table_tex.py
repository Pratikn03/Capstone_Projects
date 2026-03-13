#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def repo_to_paper_rel(path: Path, repo_root: Path, paper_dir: Path) -> str:
    abs_path = (repo_root / path).resolve() if not path.is_absolute() else path.resolve()
    return str(abs_path.relative_to(paper_dir.resolve()))


def _tex_escape(value: object) -> str:
    rendered = "" if value is None else str(value)
    return (
        rendered.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _fmt(value: object, precision: int = 2, pct: bool = False) -> str:
    if _is_missing(value):
        return "---"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _tex_escape(value)
    if pct:
        return f"{number:.{precision}f}"
    return f"{number:.{precision}f}"


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _read_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _table_wrapper(*, title: str, label: str, tabular: list[str], size: str = r"\small") -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        size,
        rf"\caption{{{title}}}",
        rf"\label{{tab:{label}}}",
    ]
    lines.extend(tabular)
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _render_generic_table(token: str, csv_path: Path, title: str) -> str:
    df = _read_csv(csv_path).fillna("---")
    tabular = df.to_latex(index=False, escape=True, na_rep="---")
    return _table_wrapper(
        title=title,
        label=token,
        tabular=[
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{0.96}",
            r"\resizebox{\linewidth}{!}{%",
            tabular,
            r"}",
        ],
        size=r"\scriptsize",
    )


def _render_tbl08(token: str, csv_path: Path, title: str) -> str:
    df = pd.read_csv(csv_path, dtype=str).fillna("---")
    target_order = ["Load", "Wind", "Solar"]
    model_order = ["GBM", "N-BEATS", "PatchTST", "TFT"]
    filtered = df[
        (df["Region"] == "DE")
        & (df["Target"].isin(target_order))
        & (df["Model"].isin(model_order))
    ].copy()
    filtered["Target"] = pd.Categorical(filtered["Target"], categories=target_order, ordered=True)
    filtered["Model"] = pd.Categorical(filtered["Model"], categories=model_order, ordered=True)
    filtered = filtered.sort_values(["Target", "Model"])
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"\textbf{Region} & \textbf{Target} & \textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{sMAPE (\%)} & $\mathbf{R^2}$ & \textbf{PICP@90} & \textbf{Width (MW)} \\",
        r"\midrule",
    ]
    current_target: str | None = None
    for _, row in filtered.iterrows():
        target = str(row["Target"])
        if current_target is not None and target != current_target:
            lines.append(r"\midrule")
        current_target = target
        model = _tex_escape(row["Model"])
        if model == "GBM":
            model = r"\textbf{GBM}"

        def maybe_bold(value: str) -> str:
            escaped = _tex_escape(value)
            if str(row["Model"]) == "GBM" and escaped != "---":
                return rf"\textbf{{{escaped}}}"
            return escaped

        lines.append(
            f"{_tex_escape(row['Region'])} & {_tex_escape(row['Target'])} & {model} & "
            f"{maybe_bold(str(row['RMSE']))} & {maybe_bold(str(row['MAE']))} & "
            f"{maybe_bold(str(row['sMAPE (%)']))} & {maybe_bold(str(row['R2']))} & "
            f"{_tex_escape(row['PICP@90 (%)'])} & {_tex_escape(row['Interval Width (MW)'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return _table_wrapper(title=title, label=token, tabular=lines, size=r"\scriptsize")


def _render_tbl03(token: str, csv_path: Path, title: str) -> str:
    df = _read_csv(csv_path).copy()
    if "group" in df.columns:
        df["group"] = df["group"].replace({"mid": "med"})
    lines = [
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Target & Regime & PICP@90 (\%) & Width (MW) & $n$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{_tex_escape(str(row.get('target', '')).replace('_mw', '').title())} & "
            f"{_tex_escape(str(row.get('group', '')).upper())} & "
            f"{_fmt((row.get('picp_90') or 0) * 100, precision=1, pct=True)} & "
            f"{_fmt(row.get('mean_width'), precision=1)} & "
            f"{int(row.get('sample_count', 0))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table_wrapper(title=title, label=token, tabular=lines)


def _render_tbl04(token: str, csv_path: Path, title: str) -> str:
    df = pd.read_csv(csv_path, dtype=str).fillna("---")
    region_order = ["DE", "US"]
    stress_order = ["Nominal", "Dropout", "Drift combo"]
    controller_order = ["DC3S", "Deterministic"]
    df["Region"] = pd.Categorical(df["Region"], categories=region_order, ordered=True)
    df["Stress"] = pd.Categorical(df["Stress"], categories=stress_order, ordered=True)
    df["Controller"] = pd.Categorical(df["Controller"], categories=controller_order, ordered=True)
    df = df.sort_values(["Region", "Stress", "Controller"])
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"\textbf{Region} & \textbf{Stress} & \textbf{Controller} & \textbf{PICP@90 (\%)} & \textbf{Width (MW)} & \textbf{Viol. Rate (\%)} & \textbf{Severity P95 (MWh)} & \textbf{Cost $\Delta$ (\%)} \\",
        r"\midrule",
    ]
    current_region: str | None = None
    for _, row in df.iterrows():
        region = str(row["Region"])
        if current_region is not None and region != current_region:
            lines.append(r"\midrule")
        current_region = region
        controller = _tex_escape(str(row["Controller"]))
        if controller == "DC3S":
            controller = r"\textbf{DC3S}"
        lines.append(
            f"{_tex_escape(region)} & {_tex_escape(str(row['Stress']))} & {controller} & "
            f"{_tex_escape(str(row['PICP@90 (%)']))} & {_tex_escape(str(row['Width (MW)']))} & "
            f"{_tex_escape(str(row['Viol. Rate (%)']))} & {_tex_escape(str(row['Severity P95 (MWh)']))} & "
            f"{_tex_escape(str(row['Cost Delta (%)']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return _table_wrapper(title=title, label=token, tabular=lines, size=r"\scriptsize")


def _render_tbl02(token: str, csv_path: Path, title: str) -> str:
    df = _read_csv(csv_path).copy()
    keep_columns = [
        "analysis_scope",
        "fault_dimension",
        "n_pairs",
        "true_soc_violation_rate_baseline_mean",
        "true_soc_violation_rate_dc3s_mean",
        "true_soc_violation_rate_rel_reduction",
        "true_soc_violation_rate_wilcoxon_p",
        "true_soc_violation_severity_p95_baseline_mean",
        "true_soc_violation_severity_p95_dc3s_mean",
        "true_soc_violation_severity_p95_rel_reduction",
        "true_soc_violation_severity_p95_wilcoxon_p",
        "passes_all_thresholds",
    ]
    condensed = df[keep_columns].copy()
    condensed["analysis_scope"] = condensed["analysis_scope"].replace(
        {
            "primary_aggregate_fault_sweep": "primary",
            "secondary_fault_dimension": "secondary",
        }
    )
    condensed["fault_dimension"] = condensed["fault_dimension"].replace({"delay_jitter": "delay"})
    lines = [
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrrc}",
        r"\toprule",
        r"Scope & Fault & $n$ & Viol. base & Viol. DC3S & Viol. red. & $p_v$ & Sev. base & Sev. DC3S & Sev. red. & $p_s$ & Pass \\",
        r"\midrule",
    ]
    for _, row in condensed.iterrows():
        lines.append(
            f"{_tex_escape(row['analysis_scope'])} & "
            f"{_tex_escape(row['fault_dimension'])} & "
            f"{int(row['n_pairs'])} & "
            f"{_fmt(row['true_soc_violation_rate_baseline_mean'], precision=2)} & "
            f"{_fmt(row['true_soc_violation_rate_dc3s_mean'], precision=2)} & "
            f"{_fmt(float(row['true_soc_violation_rate_rel_reduction']) * 100, precision=1, pct=True)} & "
            f"{_fmt(row['true_soc_violation_rate_wilcoxon_p'], precision=3)} & "
            f"{_fmt(row['true_soc_violation_severity_p95_baseline_mean'], precision=2)} & "
            f"{_fmt(row['true_soc_violation_severity_p95_dc3s_mean'], precision=2)} & "
            f"{_fmt(float(row['true_soc_violation_severity_p95_rel_reduction']) * 100, precision=1, pct=True)} & "
            f"{_fmt(row['true_soc_violation_severity_p95_wilcoxon_p'], precision=3)} & "
            f"{'yes' if _boolish(row['passes_all_thresholds']) else 'no'} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    return _table_wrapper(title=title, label=token, tabular=lines, size=r"\scriptsize")


def render_table_tex(token: str, csv_path: Path, title: str) -> str:
    if token == "TBL08_FORECAST_BASELINES":
        return _render_tbl08(token, csv_path, title)
    if token == "TBL04_TRANSFER_STRESS":
        return _render_tbl04(token, csv_path, title)
    if token == "TBL02_ABLATIONS":
        return _render_tbl02(token, csv_path, title)
    if token == "TBL03_CQR_GROUP_COVERAGE":
        return _render_tbl03(token, csv_path, title)
    return _render_generic_table(token, csv_path, title)


def write_table_tex(token: str, csv_path: Path, title: str, out_path: Path) -> None:
    content = render_table_tex(token, csv_path, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LaTeX table fragments and manifest lookup for paper assets")
    parser.add_argument("--manifest", default="paper/manifest.yaml")
    parser.add_argument("--paper-dir", default="paper")
    args = parser.parse_args()

    repo_root = Path.cwd()
    manifest_path = repo_root / args.manifest
    paper_dir = repo_root / args.paper_dir

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 1

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}

    figures = manifest.get("figures") or {}
    tables = manifest.get("tables") or {}

    lookup_lines = ["% Auto-generated by scripts/build_paper_table_tex.py"]

    for token, entry in figures.items():
        fig_path = Path(entry["path"])
        rel = repo_to_paper_rel(fig_path, repo_root, paper_dir)
        lookup_lines.append(f"\\DeclarePaperFigure{{{token}}}{{{rel}}}")

    generated_dir = paper_dir / "assets/tables/generated"
    for token, entry in tables.items():
        csv_path = repo_root / entry["path"]
        if not csv_path.exists():
            print(f"ERROR: table csv missing for {token}: {csv_path}")
            return 1
        title = entry.get("title", token)
        out_tex = generated_dir / f"{csv_path.stem}.tex"
        write_table_tex(token, csv_path, title, out_tex)
        rel = repo_to_paper_rel(out_tex, repo_root, paper_dir)
        lookup_lines.append(f"\\DeclarePaperTable{{{token}}}{{{rel}}}")

    lookup_path = paper_dir / "assets/data/manifest_lookup.tex"
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    lookup_path.write_text("\n".join(lookup_lines) + "\n", encoding="utf-8")

    print(f"Wrote table fragments under {generated_dir}")
    print(f"Wrote lookup map: {lookup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
