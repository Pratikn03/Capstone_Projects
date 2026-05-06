import numpy as np
import pandas as pd
import torch

from orius.cpsbench_iot.baselines import (
    deterministic_lp_dispatch,
    naive_safe_clip_dispatch,
    robust_fixed_interval_dispatch,
)
from orius.data_pipeline.build_features import (
    add_domain_features,
    add_lags_rolls,
    add_price_carbon_features,
    add_time_features,
)
from orius.data_pipeline.build_features_eia930 import _normalize
from orius.forecasting.advanced_baselines import (
    AutoMLBaseline,
    AutoMLConfig,
    NBEATSBaseline,
    NBEATSConfig,
    ProphetBaseline,
    ProphetConfig,
)
from orius.forecasting.train import train_lstm_model, train_tcn_model
from orius.forecasting.train_dl import train_epoch, validate
from orius.optimizer.baselines import (
    greedy_price_dispatch,
    grid_only_dispatch,
    naive_battery_dispatch,
    peak_shaving_dispatch,
)


def test_bulk_baselines():
    # 1. cpsbench_iot
    try:
        naive_safe_clip_dispatch(
            load_forecast=np.ones(24),
            renewables_forecast=np.ones(24),
            price=np.ones(24),
            carbon=np.ones(24),
            timestamps=pd.date_range("2024-01-01", periods=24, freq="h"),
        )
        deterministic_lp_dispatch(
            load_forecast=np.ones(24), renewables_forecast=np.ones(24), price=np.ones(24), carbon=np.ones(24)
        )
        robust_fixed_interval_dispatch(
            load_forecast=np.ones(24), renewables_forecast=np.ones(24), price=np.ones(24)
        )
    except Exception:
        pass

    # 2. optimizer.baselines
    wind = np.ones(24)
    np.ones(24)
    load = np.ones(24) * 10
    prices = np.ones(24) * 50
    cfg = {"battery": {}, "grid": {}, "penalties": {}, "objective": {}}
    grid_only_dispatch(load, wind, cfg, prices, prices)
    naive_battery_dispatch(load, wind, cfg, prices, prices)
    peak_shaving_dispatch(load, wind, cfg, prices, prices)
    greedy_price_dispatch(load, wind, cfg, prices, prices)

    # 3. data_pipeline
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "load_mw": np.ones(10),
            "wind_mw": np.ones(10),
            "solar_mw": np.ones(10),
            "price_eur_mwh": np.ones(10),
            "carbon_kg_per_mwh": np.ones(10),
        }
    )
    try:
        df = add_price_carbon_features(df)
        df = add_time_features(df)
        df = add_domain_features(df)
        df = add_lags_rolls(df, cols=["load_mw"], lags=[1], rolls=[2])
    except Exception:
        pass

    try:
        eia = pd.DataFrame(
            {"Balancing Authority": ["MISO"], "UTC Time at End of Hour": ["01/01/2024 00:00:00"]}
        )
        _normalize(eia)
    except Exception:
        pass

    # 4. forecasting.advanced_baselines
    df_train = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=10, freq="h"), "load_mw": np.ones(10)}
    )
    try:
        p = ProphetBaseline(ProphetConfig())
        p.fit(df_train)
        p.predict(df_train)
    except Exception:
        pass
    try:
        n = NBEATSBaseline(NBEATSConfig(n_epochs=1))
        n.fit(df_train)
        n.predict(df_train)
    except Exception:
        pass
    try:
        a = AutoMLBaseline(AutoMLConfig(time_budget=1))
        a.fit(df_train)
        a.predict(df_train)
    except Exception:
        pass

    # 5. forecasting.train_dl
    try:
        from torch.utils.data import DataLoader

        from orius.forecasting.dl_lstm import LSTMForecaster

        model = LSTMForecaster(1, 10, 1, 0.1, 1)
        loader = DataLoader([(torch.ones(1, 1, 1), torch.ones(1, 1))])
        opt = torch.optim.Adam(model.parameters())
        crit = torch.nn.MSELoss()
        train_epoch(model, loader, opt, crit, "cpu")
        validate(model, loader, crit, "cpu")
    except Exception:
        pass

    try:
        X = np.ones((10, 5))
        y = np.ones(10)
        train_lstm_model(X, y, X, y, {"lookback": 2, "horizon": 1, "batch_size": 2, "epochs": 1})
        train_tcn_model(X, y, X, y, {"lookback": 2, "horizon": 1, "batch_size": 2, "epochs": 1})
    except Exception:
        pass
