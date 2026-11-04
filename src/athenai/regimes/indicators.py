"""Indicadores de mercado interpretables y **estrictamente causales**.

Cada indicador en el instante ``t`` usa únicamente información disponible hasta ``t``
(no hay *lookahead*). Las normalizaciones usan estadísticos *expanding* (acumulados desde
el inicio), nunca estadísticos de toda la muestra.

Indicadores:
    - ``trend``: separación relativa entre media móvil rápida y lenta (fuerza de tendencia).
    - ``mom_h``: retorno acumulado a horizonte ``h`` días.
    - ``vol``: volatilidad realizada anualizada (riesgo).
    - ``drawdown``: caída desde el máximo histórico acumulado (positiva).
    - ``breadth``: fracción de algoritmos vivos en positivo (amplitud del mercado).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def causal_zscore(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """Z-score causal usando media y desviación *expanding* (solo pasado y presente)."""
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.fillna(0.0)


def trend_strength(index: pd.Series, fast: int, slow: int) -> pd.Series:
    """Separación relativa entre MA rápida y lenta: ``(MA_fast - MA_slow) / MA_slow``."""
    ma_fast = index.rolling(fast, min_periods=max(2, fast // 2)).mean()
    ma_slow = index.rolling(slow, min_periods=max(2, slow // 2)).mean()
    return (ma_fast - ma_slow) / ma_slow


def realized_vol(ret: pd.Series, window: int, annualization: int) -> pd.Series:
    """Volatilidad realizada anualizada (desviación típica móvil de los retornos)."""
    return ret.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(annualization)


def drawdown(index: pd.Series) -> pd.Series:
    """Drawdown actual (positivo) desde el máximo histórico acumulado hasta ``t``."""
    running_max = index.cummax()
    return (running_max - index) / running_max


def momentum(index: pd.Series, horizon: int) -> pd.Series:
    """Retorno acumulado a ``horizon`` días (momentum)."""
    return index / index.shift(horizon) - 1.0


def compute_indicators(market: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Calcula el panel de indicadores causales a partir del índice de mercado.

    Args:
        market: tabla con ``mkt_index``, ``mkt_ret`` y ``breadth`` (ver ``data.build``).
        cfg: configuración (ventanas en ``cfg['regimes']``).

    Returns:
        DataFrame con indicadores crudos y sus versiones normalizadas (``*_z``), causal.
    """
    r = cfg["regimes"]
    idx = market["mkt_index"]
    ret = market["mkt_ret"]

    out = pd.DataFrame(index=market.index)
    out["trend"] = trend_strength(idx, r["trend_fast"], r["trend_slow"])
    out["vol"] = realized_vol(ret, r["vol_window"], r["annualization"])
    out["drawdown"] = drawdown(idx)
    out["breadth"] = market["breadth"].fillna(0.5)
    for h in r["ret_horizons"]:
        out[f"mom_{h}"] = momentum(idx, h)

    # Normalización causal de tendencia y volatilidad (las usa el motor difuso).
    mp = r["trend_slow"]
    out["trend_z"] = causal_zscore(out["trend"], min_periods=mp)
    out["vol_z"] = causal_zscore(out["vol"], min_periods=mp)
    return out.fillna(0.0)
