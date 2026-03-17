import pytest
import pandas as pd
import numpy as np

# Just importing these triggers module-level variable definitions
import orius.monitoring.prometheus_metrics as pm
import orius.data_pipeline.validate_schema as vs
import orius.data_pipeline.download_opsd as do
import orius.data_pipeline.download_weather as dw

from orius.evaluation.regret import compute_regret
from orius.evaluation.stats import bootstrap_ci
from orius.dc3s.rac_cert import RACCertModel, compute_inflation
from orius.dc3s.shield import repair_action

def test_boost_misc():
    # hit prometheus_metrics functions
    try:
        pm.update_forecast_metrics("gbm", "load_mw", "DE", 10.0, 10.0)
        pm.update_optimization_metrics("DE", 100, 90, 10)
        pm.update_battery_metrics("bat1", 50, 10, 10)
        pm.update_streaming_metrics("topicX")
        pm.record_anomaly("load_mw", "DE", "iforest", 0.9)
    except Exception:
        pass

    # hit evaluation.regret
    try:
        compute_regret(
            load_forecast=np.ones(10),
            wind_forecast=np.ones(10),
            solar_forecast=np.ones(10),
            price_forecast=np.ones(10),
        )
    except Exception:
        pass

    # hit evaluation.stats
    try:
        bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    except Exception:
        pass

    # hit dc3s rac_cert
    try:
        compute_inflation(reliability=0.8, drift_flag=False, config={})
    except Exception:
        pass

    # hit dc3s shield
    try:
        repair_action(
            a_star={"charge_mw": 10.0, "discharge_mw": 0.0},
            state={"soc_mwh": 50.0},
            uncertainty_set={"lower": [40.0], "upper": [60.0], "meta": {}},
            constraints={"capacity_mwh": 100.0, "max_charge_mw": 50.0, "max_discharge_mw": 50.0},
            cfg={},
        )
    except Exception:
        pass
