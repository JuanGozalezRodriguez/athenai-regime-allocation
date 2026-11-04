"""Test crítico de causalidad: ausencia de *lookahead*.

Si los indicadores y las probabilidades de fase son causales, su valor en ``t`` no puede
cambiar al añadir (o modificar) datos posteriores a ``t``. Se comprueba recomputando sobre la
serie truncada y comparando con la serie completa.
"""

from __future__ import annotations

import numpy as np

from athenai.regimes.fuzzy import REGIMES, FuzzyRegimeModel
from athenai.regimes.indicators import compute_indicators


def test_indicators_no_lookahead(cfg, processed):
    market = processed["market"]
    full = compute_indicators(market, cfg)

    cut = len(market) // 2
    truncated = compute_indicators(market.iloc[:cut], cfg)

    # Los indicadores causales hasta 'cut' deben coincidir con los de la serie completa.
    cols = ["trend", "vol", "drawdown", "breadth", "trend_z", "vol_z"]
    np.testing.assert_allclose(
        full[cols].iloc[:cut].to_numpy(),
        truncated[cols].to_numpy(),
        rtol=1e-9, atol=1e-9,
    )


def test_fuzzy_proba_no_lookahead(cfg, processed):
    market = processed["market"]
    fuzzy = FuzzyRegimeModel.from_config(cfg)

    full_ind = compute_indicators(market, cfg)
    full_p = fuzzy.predict_proba(full_ind)

    cut = len(market) // 2
    trunc_p = fuzzy.predict_proba(compute_indicators(market.iloc[:cut], cfg))

    # Con suavizado causal, las primeras filas deben coincidir salvo el borde final del corte.
    k = max(1, fuzzy.smoothing)
    np.testing.assert_allclose(
        full_p[REGIMES].iloc[: cut - k].to_numpy(),
        trunc_p[REGIMES].iloc[: cut - k].to_numpy(),
        rtol=1e-8, atol=1e-8,
    )


def test_backtest_applies_next_period(cfg, processed):
    """El retorno de una decisión se cobra en el periodo siguiente, no en el mismo."""
    import pandas as pd

    from athenai.evaluation.backtest import backtest_weights

    dates = pd.date_range("2021-01-31", periods=3, freq="M")
    asset_ret = pd.DataFrame({"A": [0.10, 0.20, 0.30], "B": [0.0, 0.0, 0.0]}, index=dates)
    # Decidir 100% A en la primera fecha; debe cobrar el retorno de la SEGUNDA fecha (0.20).
    weights = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[dates[0]])
    bt = backtest_weights(weights, asset_ret, cost_bps=0.0)
    assert bt.index[0] == dates[1]
    assert abs(bt["gross"].iloc[0] - 0.20) < 1e-12
