from orius.dc3s.quality import (
    byzantine_channel_attack,
    median_of_means_reliability,
    trimmed_mean_reliability,
)


def test_byzantine_channel_attack_appends_bounded_channels() -> None:
    attacked = byzantine_channel_attack([0.9, 0.91, 0.92], byzantine_budget=2, attack_value=0.0)

    assert attacked == [0.9, 0.91, 0.92, 0.0, 0.0]


def test_robust_aggregators_reduce_extreme_channel_effect() -> None:
    attacked = byzantine_channel_attack([0.88, 0.9, 0.92, 0.94], byzantine_budget=2, attack_value=0.0)
    naive = sum(attacked) / len(attacked)
    trimmed = trimmed_mean_reliability(attacked, byzantine_budget=2)
    mom = median_of_means_reliability(attacked, num_blocks=3)

    assert trimmed > naive
    assert mom > naive
