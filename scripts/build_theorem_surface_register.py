#!/usr/bin/env python3
"""Build an integrated register of theorem-like environments from the active thesis include tree."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MASTER_TEX = REPO_ROOT / "orius_battery_409page_figures_upgraded_main.tex"
APPENDIX_PROOFS = REPO_ROOT / "appendices" / "app_c_full_proofs.tex"
REPORTS_DIR = REPO_ROOT / "reports" / "publication"

SUMMARY_CSV = REPORTS_DIR / "theorem_surface_summary.csv"
REGISTER_CSV = REPORTS_DIR / "theorem_surface_register.csv"
SUMMARY_TEX = REPORTS_DIR / "theorem_surface_summary.tex"
REGISTER_TEX = REPORTS_DIR / "theorem_surface_register.tex"

ENV_ORDER = [
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "remark",
    "example",
    "assumption",
]
SUMMARY_ENV_ORDER = [
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "assumption",
]
GROUP_ORDER = [
    "Defended theorem program",
    "Supporting chapter result",
    "Appendix proof restatement",
    "Assumption register",
]

BEGIN_RE = re.compile(
    r"\\begin\{(theorem|lemma|proposition|corollary|definition|remark|example|assumption)\}"
    r"(?:\[(.*?)\])?"
)
HEADING_RE = re.compile(r"\\(chapter|section|subsection)\{(.+?)\}")

THEOREM_IDS = {
    "existence of the illusion under dropout": "S1",
    "informal safety guarantee of dc3s": "S2",
    "dc3s feasibility guarantee": "S2",
    "oasg existence": "T1",
    "safety preservation": "T2",
    "orius core bound": "T3",
    "no free safety": "T4",
    "certificate validity horizon": "T5",
    "certificate expiration bound": "T6",
    "feasible fallback existence": "T7",
    "graceful degradation dominance": "T8",
    "universal impossibility": "T9",
    "boundary-indistinguishability lower bound": "T10",
    "typed structural transfer theorem": "T11",
}

ASSUMPTION_IDS = {
    "A1: Bounded model error": "A1",
    "A2: Bounded telemetry error": "A2",
    "A3: Feasible safe repair": "A3",
    "A4: Known dynamics": "A4",
    "A5: Monotone bounded uncertainty inflation": "A5",
    "A6: Bounded detector lag": "A6",
    "A7: Causal certificate update": "A7",
    "A8: Admissible fallback policy": "A8",
}
ASSUMPTION_PREFIX_RE = re.compile(r"^(A\d+)\b")


@dataclass
class Record:
    register_id: str
    group: str
    environment: str
    title: str
    context: str
    source: str
    line: int


def chapter_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"ch(\d+)_", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10_000, path.name)


def appendix_sort_key(path: Path) -> tuple[str, str]:
    match = re.search(r"app_([a-z]+)_", path.name)
    if match:
        return (match.group(1), path.name)
    return ("zzzz", path.name)


def clean_text(value: str) -> str:
    return " ".join(value.replace(r"\quad", " ").split())


def tex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


INCLUDE_PATTERN = re.compile(r"\\(?:include|input)\{([^}]+)\}")


def _resolve_include(target: str, base_dir: Path) -> Path | None:
    stem = Path(target)
    candidates = []
    if stem.suffix:
        candidates.append((base_dir / stem).resolve())
    else:
        candidates.append((base_dir / f"{target}.tex").resolve())
        candidates.append((base_dir / target).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _walk_include_tree(path: Path, visited: list[Path], seen: set[Path]) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    visited.append(resolved)
    text = resolved.read_text(encoding="utf-8")
    for match in INCLUDE_PATTERN.finditer(text):
        child = _resolve_include(match.group(1), resolved.parent)
        if child is not None:
            _walk_include_tree(child, visited, seen)


def _normalize_title(value: str) -> str:
    lowered = clean_text(value).lower()
    lowered = lowered.replace("---", " ")
    lowered = lowered.replace("--", " ")
    return " ".join(lowered.split())


def _match_theorem_id(title: str, context: str) -> str:
    haystacks = [_normalize_title(title), _normalize_title(context)]
    for haystack in haystacks:
        for needle, theorem_id in THEOREM_IDS.items():
            if needle in haystack:
                return theorem_id
    return ""


def classify_record(path: Path, environment: str, title: str, context: str) -> tuple[str, str]:
    if path.resolve() == APPENDIX_PROOFS.resolve():
        return ("Appendix proof restatement", "")
    if environment == "assumption":
        assumption_id = ASSUMPTION_IDS.get(title, "")
        if not assumption_id:
            match = ASSUMPTION_PREFIX_RE.match(title)
            assumption_id = match.group(1) if match else ""
        return ("Assumption register", assumption_id)
    if environment == "theorem":
        theorem_id = _match_theorem_id(title, context)
        if theorem_id:
            return ("Defended theorem program", theorem_id)
    return ("Supporting chapter result", "")


def discover_records() -> list[Record]:
    files: list[Path] = []
    _walk_include_tree(MASTER_TEX, files, set())
    records: list[Record] = []
    for path in files:
        context = ""
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            heading = HEADING_RE.search(line)
            if heading:
                context = clean_text(heading.group(2))
            for match in BEGIN_RE.finditer(line):
                environment = match.group(1)
                title = clean_text(match.group(2) or "")
                if title == "restated" and context:
                    title = context
                group, register_id = classify_record(path, environment, title, context)
                records.append(
                    Record(
                        register_id=register_id,
                        group=group,
                        environment=environment,
                        title=title,
                        context=context,
                        source=path.relative_to(REPO_ROOT).as_posix(),
                        line=lineno,
                    )
                )
    return records


def write_summary_csv(records: list[Record]) -> None:
    counts = Counter(record.environment for record in records)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["environment", "count"])
        for environment in SUMMARY_ENV_ORDER:
            writer.writerow([environment, counts.get(environment, 0)])
        writer.writerow(["total", sum(counts.get(environment, 0) for environment in SUMMARY_ENV_ORDER)])


def write_register_csv(records: list[Record]) -> None:
    with REGISTER_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["register_id", "group", "environment", "title", "context", "source", "line"])
        for record in records:
            writer.writerow(
                [
                    record.register_id,
                    record.group,
                    record.environment,
                    record.title,
                    record.context,
                    record.source,
                    record.line,
                ]
            )


def write_summary_tex(records: list[Record]) -> None:
    counts = Counter(record.environment for record in records)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Integrated theorem-like surface counts generated from the active thesis include tree.}",
        r"\label{tab:integrated-theorem-surface-summary}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Environment} & \textbf{Count} \\",
        r"\midrule",
    ]
    for environment in SUMMARY_ENV_ORDER:
        lines.append(rf"{tex_escape(environment.title())} & {counts.get(environment, 0)} \\")
    total = sum(counts.get(environment, 0) for environment in SUMMARY_ENV_ORDER)
    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & \textbf{{{total}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    SUMMARY_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_register_tex(records: list[Record]) -> None:
    grouped_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        grouped_counts[record.group][record.environment] += 1

    lines = [
        r"\begin{small}",
        r"\begin{longtable}{@{}p{1.2cm}p{2.2cm}p{1.7cm}p{3.2cm}p{3.6cm}p{2.1cm}@{}}",
        r"\caption{Integrated theorem-like surface register generated from the active thesis include tree.}",
        r"\label{tab:integrated-theorem-surface-register} \\",
        r"\toprule",
        r"\textbf{ID} & \textbf{Group} & \textbf{Kind} & \textbf{Title} & \textbf{Context} & \textbf{Source} \\",
        r"\midrule",
        r"\endfirsthead",
        r"",
        r"\multicolumn{6}{c}{\tablename\ \thetable{} --- continued} \\",
        r"\toprule",
        r"\textbf{ID} & \textbf{Group} & \textbf{Kind} & \textbf{Title} & \textbf{Context} & \textbf{Source} \\",
        r"\midrule",
        r"\endhead",
        r"",
        r"\midrule",
        r"\multicolumn{6}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for group in GROUP_ORDER:
        if group not in grouped_counts:
            continue
        group_total = sum(grouped_counts[group].values())
        summary_bits = ", ".join(
            f"{environment}:{grouped_counts[group][environment]}"
            for environment in ENV_ORDER
            if grouped_counts[group][environment]
        )
        lines.append(
            rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{tex_escape(group)}}} ({group_total}; {tex_escape(summary_bits)})}} \\"
        )
        lines.append(r"\midrule")
        for record in records:
            if record.group != group:
                continue
            record_id = tex_escape(record.register_id) if record.register_id else "---"
            title = tex_escape(record.title) if record.title else "---"
            context = tex_escape(record.context) if record.context else "---"
            source = rf"\path{{{record.source}}}:{record.line}"
            lines.append(
                rf"{record_id} & {tex_escape(record.group)} & {tex_escape(record.environment)} & "
                rf"{title} & {context} & {source} \\"
            )
        lines.append(r"\addlinespace")

    lines.extend([r"\end{longtable}", r"\end{small}"])
    REGISTER_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    records = discover_records()
    write_summary_csv(records)
    write_register_csv(records)
    write_summary_tex(records)
    write_register_tex(records)
    print(f"Wrote {SUMMARY_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {REGISTER_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {SUMMARY_TEX.relative_to(REPO_ROOT)}")
    print(f"Wrote {REGISTER_TEX.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
