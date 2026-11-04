"""Limpieza y normalización de las series de algoritmos.

Decisiones documentadas en ``docs/hallazgos_datos.md``:
    - Un tramo es "plano" (algoritmo fuera de mercado) cuando ``open==high==low==close``.
    - Las velas de 4h caen en dos rejillas horarias desfasadas 1h por el horario de verano
      (DST). Para neutralizarlo se trabaja con el **cierre diario** (último close del día
      natural), robusto al desfase.
    - La "vida" de un algoritmo es el intervalo entre su primer y su último movimiento de
      precio; fuera de ella no se le considera (ni lanzado aún, ni ya congelado).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def flat_mask(df: pd.DataFrame, atol: float = 1e-9) -> pd.Series:
    """Máscara booleana de velas planas (OHLC idénticos → fuera de mercado)."""
    rng = (df[["open", "high", "low", "close"]].max(axis=1)
           - df[["open", "high", "low", "close"]].min(axis=1))
    return rng.abs() <= atol


def daily_close(df: pd.DataFrame) -> pd.Series:
    """Cierre diario (último ``close`` de cada día natural), neutralizando el DST."""
    return df["close"].resample("1D").last().dropna()


def lifespan_bounds(close: pd.Series, atol: float = 1e-9) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Primer y último instante con movimiento de precio.

    Devuelve ``(inicio, fin)`` donde inicio es la primera fecha cuyo cierre difiere del
    anterior y fin la última. Si la serie nunca se mueve, devuelve ``(NaT, NaT)``.
    """
    changed = close.diff().abs() > atol
    if not changed.any():
        return pd.NaT, pd.NaT
    moves = close.index[changed]
    return moves[0], moves[-1]


def daily_returns(close: pd.Series) -> pd.Series:
    """Retornos diarios simples a partir del cierre diario."""
    return (close / close.shift(1) - 1.0).fillna(0.0)


def active_mask(close: pd.Series, atol: float = 1e-9) -> pd.Series:
    """Máscara de días "vivos": dentro del intervalo de vida del algoritmo.

    Fuera del intervalo [primer movimiento, último movimiento] el algoritmo se considera
    no lanzado o ya congelado, y no participa en los agregados.
    """
    start, end = lifespan_bounds(close, atol=atol)
    if pd.isna(start):
        return pd.Series(False, index=close.index)
    within = (close.index >= start) & (close.index <= end)
    return pd.Series(within, index=close.index)


def clip_period(s: pd.Series | pd.DataFrame, start: str, end: str):
    """Recorta una serie/tabla al periodo de análisis [start, end]."""
    return s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


def winsorize(s: pd.Series, limit: float = 0.2) -> pd.Series:
    """Recorta retornos extremos (errores de datos) al rango [-limit, limit].

    Protege los agregados frente a saltos espurios sin alterar la señal normal.
    """
    return s.clip(lower=-limit, upper=limit).replace([np.inf, -np.inf], 0.0)
