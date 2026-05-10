from orius.benchmarks.graceful_degradation import (
    BlindPersistencePolicy,
    ImmediateShutdownPolicy,
    ORIUSGracefulPolicy,
    RampDownPolicy,
    evaluate_policy_frontier,
    graceful_dominance_with_useful_work,
)


def test_orius_weakly_dominates_blind_persistence_with_useful_work():
    hazards = [False, True, True, False, True]
    work = [1.0, 1.0, 1.0, 1.0, 1.0]
    blind = BlindPersistencePolicy().summarize(hazards, work)
    orius = ORIUSGracefulPolicy().summarize(hazards, work)

    result = graceful_dominance_with_useful_work(orius, blind, lambda_work=0.25)

    assert result["safety_dominates"] is True
    assert result["work_preserved"] is True
    assert result["passes"] is True


def test_immediate_shutdown_is_safe_but_fails_useful_work_threshold():
    hazards = [True, True, False]
    work = [1.0, 1.0, 1.0]
    blind = BlindPersistencePolicy().summarize(hazards, work)
    shutdown = ImmediateShutdownPolicy().summarize(hazards, work)

    result = graceful_dominance_with_useful_work(shutdown, blind, lambda_work=0.25)

    assert result["safety_dominates"] is True
    assert result["work_preserved"] is False
    assert result["passes"] is False


def test_blind_persistence_fails_safety_against_orius_frontier():
    hazards = [False, True, True, False]
    work = [1.0, 1.0, 1.0, 1.0]

    rows = evaluate_policy_frontier(hazards, work, lambda_work=0.25)
    by_policy = {row["policy"]: row for row in rows}

    assert by_policy["Blind"]["tsvr"] > by_policy["ORIUS"]["tsvr"]
    assert by_policy["ORIUS"]["pass"] is True


def test_ramp_down_preserves_some_work_but_not_full_orius_safety():
    hazards = [True, True, True, True]
    work = [1.0, 1.0, 1.0, 1.0]
    ramp = RampDownPolicy().summarize(hazards, work)
    orius = ORIUSGracefulPolicy().summarize(hazards, work)

    assert ramp.useful_work > 0.0
    assert ramp.tsvr > orius.tsvr
