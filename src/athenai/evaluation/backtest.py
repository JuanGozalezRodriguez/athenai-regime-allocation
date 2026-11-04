"""Motor de backtest causal con costes de transacción.

Convención temporal (sin *lookahead*): los pesos de la fila con fecha ``m`` se deciden con la
información disponible hasta ``m`` y se aplican al retorno del periodo **siguiente** (``m+1``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_monthly_returns(daily_returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compone retornos diarios a retornos mensuales (fin de mes)."""
    return (1.0 + daily_returns.fillna(0.0)).resample("M").prod() - 1.0


def backtest_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Backtest de una cartera con pesos variables y costes de transacción.

    Args:
        weights: pesos por rebalanceo (índice = fechas de decisión, columnas = activos).
        asset_returns: retornos de los activos en la misma frecuencia que el rebalanceo.
        cost_bps: coste por lado en puntos básicos sobre el cambio de pesos.

    Returns:
        DataFrame con columnas ``gross``, ``cost``, ``net`` (retorno neto) y ``turnover``,
        indexado por el periodo al que se aplica cada decisión.
    """
    assets = asset_returns.columns
    w = weights.reindex(columns=assets).fillna(0.0)

    rows = []
    prev_w = pd.Series(0.0, index=assets)
    cost_rate = cost_bps / 1e4
    for date, w_row in w.iterrows():
        # Retorno del periodo siguiente a la decisión.
        future = asset_returns.loc[asset_returns.index > date]
        if future.empty:
            break
        period = future.index[0]
        r = asset_returns.loc[period].reindex(assets).fillna(0.0)
        gross = float((w_row * r).sum())
        l1_change = float((w_row - prev_w).abs().sum())
        cost = cost_rate * l1_change
        rows.append({
            "period": period,
            "gross": gross,
            "cost": cost,
            "net": gross - cost,
            "turnover": 0.5 * l1_change,
        })
        prev_w = w_row
    out = pd.DataFrame(rows).set_index("period")
    out.index.name = "date"
    return out


def static_weights(target: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Repite un vector de pesos fijo en todas las fechas de rebalanceo (para baselines)."""
    return pd.DataFrame(
        np.tile(target.to_numpy(), (len(dates), 1)),
        index=dates, columns=target.index,
    )
