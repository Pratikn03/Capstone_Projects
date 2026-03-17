#!/usr/bin/env python3
"""Analyze ORIUS_PhD_Thesis_300pages.pdf vs paper/paper.pdf for depth, multi-domain, and gaps."""
from __future__ import annotations

from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


def extract_text(pdf_path: Path, max_pages: int = 100) -> str:
    r = PdfReader(str(pdf_path))
    texts = []
    for i in range(min(len(r.pages), max_pages)):
        t = r.pages[i].extract_text()
        texts.append(t or "")
    return " ".join(texts)


def count_keywords(text: str) -> dict[str, int]:
    low = text.lower()
    return {
        "battery": low.count("battery"),
        "multi-domain": low.count("multi-domain") + low.count("multidomain"),
        "vehicle": low.count("vehicle") + low.count(" vehicles ") + low.count("av "),
        "industrial": low.count("industrial"),
        "healthcare": low.count("healthcare") + low.count("vital sign"),
        "dc3s": low.count("dc3s"),
        "theorem": low.count("theorem"),
        "domain": low.count("domain"),
        "orius": low.count("orius"),
        "universal": low.count("universal"),
        "energy": low.count("energy"),
        "soc": low.count("soc") + low.count("state of charge"),
        "conformal": low.count("conformal"),
        "reliability": low.count("reliability"),
        "shield": low.count("shield"),
        "certificate": low.count("certificate"),
    }


def find_sections(text: str) -> list[str]:
    """Find section-like headers (lines that look like section titles)."""
    lines = text.split("\n")
    sections = []
    for line in lines:
        s = line.strip()
        if len(s) > 3 and len(s) < 100:
            if s[0].isdigit() and "." in s[:4]:
                sections.append(s[:80])
            elif s.upper() == s and len(s) > 5:
                sections.append(s[:80])
    return sections[:50]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    thesis_path = repo / "ORIUS_PhD_Thesis_300pages.pdf"
    paper_path = repo / "paper" / "paper.pdf"

    if not thesis_path.exists():
        print(f"Thesis not found: {thesis_path}")
        return
    if not paper_path.exists():
        print(f"Paper not found: {paper_path}")
        return

    thesis_text = extract_text(thesis_path)
    paper_text = extract_text(paper_path)

    thesis_pages = len(PdfReader(str(thesis_path)).pages)
    paper_pages = len(PdfReader(str(paper_path)).pages)

    t_kw = count_keywords(thesis_text)
    p_kw = count_keywords(paper_text)

    print("=" * 60)
    print("ORIUS PhD Thesis (300p) vs Paper — Depth & Multi-Domain Analysis")
    print("=" * 60)
    print()
    print("PAGE COUNTS")
    print("-" * 40)
    print(f"  Thesis: {thesis_pages} pages")
    print(f"  Paper:  {paper_pages} pages")
    print()

    print("KEYWORD COUNTS (first 100 pages each)")
    print("-" * 40)
    for k in sorted(t_kw.keys()):
        t_v = t_kw.get(k, 0)
        p_v = p_kw.get(k, 0)
        diff = "✓" if p_v > 0 or t_v == 0 else "△" if t_v > p_v else " "
        print(f"  {k:20} thesis={t_v:4}  paper={p_v:4}  {diff}")
    print()

    print("MULTI-DOMAIN COVERAGE")
    print("-" * 40)
    domains = ["vehicle", "industrial", "healthcare", "multi-domain", "universal"]
    for d in domains:
        t_v = t_kw.get(d, 0)
        p_v = p_kw.get(d, 0)
        status = "BOTH" if t_v > 0 and p_v > 0 else "THESIS ONLY" if t_v > 0 else "PAPER ONLY" if p_v > 0 else "NEITHER"
        print(f"  {d:20} {status}")
    print()

    print("GAPS: In thesis but absent/weak in paper")
    print("-" * 40)
    gaps = []
    for k, t_v in t_kw.items():
        p_v = p_kw.get(k, 0)
        if t_v >= 5 and p_v < 2:
            gaps.append((k, t_v, p_v))
    for k, t_v, p_v in sorted(gaps, key=lambda x: -x[1]):
        print(f"  {k}: thesis={t_v}, paper={p_v}")
    if not gaps:
        print("  (none significant)")
    print()

    print("IMPLEMENTATION ALIGNMENT")
    print("-" * 40)
    print("  Thesis 300p: Extended battery + multi-domain roadmap")
    print("  Paper:      Battery-only DC3S (per paper.tex title)")
    print("  Code:       Universal framework (energy, av, industrial, healthcare)")
    print("  Gap:        Paper does not yet reflect multi-domain in same depth as thesis")
    print()


if __name__ == "__main__":
    main()
