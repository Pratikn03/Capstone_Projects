"""Artifact consistency tests for Papers 3 and 5.

Guards against staleness drift between the canonical lower-level reports
and the promoted publication copies.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestPaper3ArtifactConsistency:
    """Paper 3: publication summary must match the canonical paper3 surface."""

    def test_graceful_four_policy_metrics_in_sync(self) -> None:
        paper3 = REPO / "reports" / "paper3" / "graceful_four_policy_metrics.csv"
        publication = REPO / "reports" / "publication" / "graceful_four_policy_metrics.csv"

        assert paper3.exists(), f"Missing: {paper3}"
        assert publication.exists(), f"Missing: {publication}"

        paper3_rows = _read_csv(paper3)
        pub_rows = _read_csv(publication)

        assert paper3_rows == pub_rows, (
            "reports/paper3/graceful_four_policy_metrics.csv and "
            "reports/publication/graceful_four_policy_metrics.csv disagree. "
            "Run `python scripts/run_paper3_four_policy_benchmark.py` to regenerate."
        )


class TestPaper5ArtifactConsistency:
    """Paper 5: publication scenario CSV must match the canonical multi_agent surface."""

    def test_multi_agent_scenario_in_sync(self) -> None:
        canonical = REPO / "reports" / "multi_agent" / "multi_agent_transformer_scenario.csv"
        publication = REPO / "reports" / "publication" / "multi_agent_transformer_scenario.csv"

        assert canonical.exists(), f"Missing: {canonical}"
        assert publication.exists(), f"Missing: {publication}"

        canonical_rows = _read_csv(canonical)
        pub_rows = _read_csv(publication)

        assert canonical_rows == pub_rows, (
            "reports/multi_agent/multi_agent_transformer_scenario.csv and "
            "reports/publication/multi_agent_transformer_scenario.csv disagree. "
            "Run `python scripts/run_multi_agent_counterexample.py` to regenerate."
        )

    def test_publication_has_degradation_column(self) -> None:
        """Ensure the publication CSV has the full schema including degradation_allocation_quality."""
        publication = REPO / "reports" / "publication" / "multi_agent_transformer_scenario.csv"
        assert publication.exists()
        rows = _read_csv(publication)
        assert len(rows) > 0
        assert "degradation_allocation_quality" in rows[0], (
            "Publication CSV missing degradation_allocation_quality column. "
            "Run `python scripts/run_multi_agent_counterexample.py` to regenerate."
        )
