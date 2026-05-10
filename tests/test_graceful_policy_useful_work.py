from orius.benchmarks.graceful_degradation import (
    BlindPersistencePolicy,
    ImmediateShutdownPolicy,
    ORIUSGracefulPolicy,
    evaluate_policy_frontier,
)


def test_frontier_reports_all_required_policy_classes() -> None:
    rows = evaluate_policy_frontier([False, True, True], [1.0, 1.0, 1.0], lambda_work=0.25)

    assert {row["policy"] for row in rows} == {"Blind", "Shutdown", "Ramp", "ORIUS"}


def test_shutdown_has_zero_work_and_orius_preserves_nontrivial_work() -> None:
    hazards = [True, True, False, True]
    work = [1.0, 1.0, 1.0, 1.0]
    blind = BlindPersistencePolicy().summarize(hazards, work)
    shutdown = ImmediateShutdownPolicy().summarize(hazards, work)
    orius = ORIUSGracefulPolicy().summarize(hazards, work)

    assert shutdown.useful_work == 0.0
    assert orius.useful_work > 0.25 * blind.useful_work
    assert orius.tsvr <= blind.tsvr
