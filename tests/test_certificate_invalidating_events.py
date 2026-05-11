from orius.dc3s.temporal_theorems import certificate_invalidating_event


def test_fresh_contradictory_evidence_invalidates_certificate() -> None:
    event = certificate_invalidating_event(
        contradictory_observation=True,
        metadata={"sensor": "soc"},
    )

    assert event["invalidates_certificate"] is True
    assert "contradictory_observation" in event["reasons"]
    assert event["metadata"]["sensor"] == "soc"


def test_no_invalidating_evidence_keeps_certificate_valid() -> None:
    event = certificate_invalidating_event()

    assert event["invalidates_certificate"] is False
    assert event["reasons"] == []
