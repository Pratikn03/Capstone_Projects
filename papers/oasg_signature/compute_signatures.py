"""Compute Battery + AV OASG signatures and emit paper-ready assets."""

from __future__ import annotations

from pathlib import Path

from orius.orius_bench.oasg_metrics import (
    build_submission_domain_surfaces,
    signature_across_domains,
    signature_latex_table,
    signature_report,
)


ROOT = Path(__file__).resolve().parent


def generate_comparison_figure(results: dict, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    names = list(results)
    signatures = [results[name].signature for name in names]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(range(len(names)), signatures, color=["#345995", "#03a678"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel(r"$\Sigma_{\mathrm{OASG}}$")
    ax.set_title("Battery + AV OASG signature comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_csv(results: dict, output_path: Path) -> None:
    lines = ["domain,signature,exposure_rate,severity,blindness,ci_low,ci_high"]
    for name, result in results.items():
        lines.append(
            ",".join(
                [
                    name,
                    f"{result.signature:.6f}",
                    f"{result.exposure_rate:.6f}",
                    f"{result.severity:.6f}",
                    f"{result.blindness:.6f}",
                    f"{result.bootstrap_ci_95[0]:.6f}",
                    f"{result.bootstrap_ci_95[1]:.6f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict:
    results = signature_across_domains(build_submission_domain_surfaces())
    (ROOT / "signature_report.txt").write_text(signature_report(results), encoding="utf-8")
    (ROOT / "signature_table.tex").write_text(signature_latex_table(results), encoding="utf-8")
    generate_csv(results, ROOT / "signature_values.csv")
    generate_comparison_figure(results, ROOT / "signature_figure.pdf")
    return results


if __name__ == "__main__":
    main()
