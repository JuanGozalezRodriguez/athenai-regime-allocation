"""Métricas de rendimiento y riesgo a partir de una serie de retornos.

Todas las funciones aceptan una serie de retornos periódicos (p. ej. mensuales o diarios) y
reciben los periodos por año para anualizar correctamente. No hay números mágicos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12


def equity_curve(returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Curva de capital a partir de retornos simples."""
    return base * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series, periods_per_year: int) -> float:
    """Tasa de crecimiento anual compuesta (CAGR)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    total = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def annual_vol(returns: pd.Series, periods_per_year: int) -> float:
    """Volatilidad anualizada."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    """Ratio de Sharpe anualizado (tasa libre de riesgo ``rf`` anual)."""
    r = returns.dropna()
    if r.std(ddof=1) == 0 or r.empty:
        return float("nan")
    excess = r - rf / periods_per_year
    return float(excess.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    """Ratio de Sortino (penaliza solo la volatilidad a la baja)."""
    r = returns.dropna()
    excess = r - rf / periods_per_year
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd) or r.empty:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Máximo drawdown (negativo) de la curva de capital asociada."""
    eq = equity_curve(returns)
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def calmar(returns: pd.Series, periods_per_year: int) -> float:
    """Ratio de Calmar: CAGR / |máximo drawdown|."""
    mdd = max_drawdown(returns)
    c = cagr(returns, periods_per_year)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(c / abs(mdd))


def positive_period_pct(returns: pd.Series) -> float:
    """Fracción de periodos con retorno positivo."""
    r = returns.dropna()
    return float((r > 0).mean()) if not r.empty else float("nan")


def turnover(weights: pd.DataFrame) -> float:
    """Rotación media por rebalanceo: 0.5 * suma |Δpeso| promediada en el tiempo."""
    if weights.shape[0] < 2:
        return 0.0
    changes = weights.diff().abs().sum(axis=1) * 0.5
    return float(changes.iloc[1:].mean())


def summary(returns: pd.Series, periods_per_year: int,
            weights: pd.DataFrame | None = None) -> dict[str, float]:
    """Diccionario con el cuadro de métricas estándar de una estrategia."""
    out = {
        "CAGR": cagr(returns, periods_per_year),
        "vol_anual": annual_vol(returns, periods_per_year),
        "Sharpe": sharpe(returns, periods_per_year),
        "Sortino": sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "Calmar": calmar(returns, periods_per_year),
        "pct_periodos_positivos": positive_period_pct(returns),
    }
    if weights is not None:
        out["turnover"] = turnover(weights)
    return out


def yearly_returns(returns: pd.Series) -> pd.Series:
    """Retorno compuesto por año natural."""
    return returns.groupby(returns.index.year).apply(lambda r: (1 + r).prod() - 1.0)
