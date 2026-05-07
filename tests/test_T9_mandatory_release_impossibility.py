from orius.universal_theory.ambiguity import (
    build_ambiguity_class,
    compute_common_safe_core,
    find_mandatory_release_counterexample,
    is_empty_safe_core,
)


def test_empty_safe_core_counterexample_exists():
    states = ["x0", "x1"]
    obs = {"x0": "o", "x1": "o"}
    safe = {"x0": {"a"}, "x1": {"b"}}
    amb = build_ambiguity_class(states, lambda x: obs[x], "o")
    assert is_empty_safe_core(amb, lambda x: safe[x])
    assert compute_common_safe_core(amb, lambda x: safe[x]) == set()
    assert find_mandatory_release_counterexample(amb, lambda x: safe[x], "a") == "x1"


def test_nonempty_safe_core_not_impossible():
    states = ["x0", "x1"]
    safe = {"x0": {"a", "b"}, "x1": {"b"}}
    core = compute_common_safe_core(states, lambda x: safe[x])
    assert core == {"b"}
