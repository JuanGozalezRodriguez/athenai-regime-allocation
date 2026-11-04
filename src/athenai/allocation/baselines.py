"""Baselines obligatorias de asignación, para comparar con el agente RL.

    - ``equiponderado``: reparte por igual entre los algoritmos invertidos cada mes.
    - ``mejor_algo``: 100% al algoritmo con mejor Sharpe en entrenamiento (constante).
    - ``difusa_sin_rl``: regla fija fase→cartera coincidente (sin aprendizaje).
    - El ``benchmark`` oficial se toma directamente de sus returns (no como pesos).
"""

from __future__ import annotations

import pandas as pd

from ..evaluation import metrics
from ..regimes.fuzzy import REGIMES
from .agent import weights_from_policy


def equal_weight_weights(rebalance_dates: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Cartera equiponderada sobre todos los algoritmos invertidos."""
    w = pd.DataFrame(1.0 / len(columns), index=rebalance_dates, columns=columns)
    w.index.name = "date"
    return w


def best_algo_weights(train_returns: pd.DataFrame, rebalance_dates: pd.DatetimeIndex,
                      periods_per_year: int = 252) -> pd.DataFrame:
    """100% al algoritmo con mejor Sharpe en el periodo de entrenamiento."""
    sharpes = train_returns.apply(lambda r: metrics.sharpe(r[r != 0], periods_per_year))
    best = sharpes.idxmax()
    w = pd.DataFrame(0.0, index=rebalance_dates, columns=train_returns.columns)
    w[best] = 1.0
    w.index.name = "date"
    return w, best


def fuzzy_rule_weights(monthly_regime: pd.Series, portfolios: pd.DataFrame) -> pd.DataFrame:
    """Regla fija sin RL: cada fase usa su cartera homónima (política identidad)."""
    identity = pd.Series(range(len(REGIMES)), index=REGIMES, name="cartera")
    return weights_from_policy(identity, monthly_regime, portfolios)
