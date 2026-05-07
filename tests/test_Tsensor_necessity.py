from orius.universal_theory.sensor_necessity import critical_sensor_test, sensor_ablation


def test_sensor_drop_and_empty_core_trigger():
    obs = {"soc": 0.2, "temp": 20.0}
    assert sensor_ablation(obs, ["soc"]) == {"temp": 20.0}

    states = ["x0", "x1"]
    safe = {"x0": {"charge"}, "x1": {"idle"}}
    assert critical_sensor_test(states, lambda x: safe[x])
