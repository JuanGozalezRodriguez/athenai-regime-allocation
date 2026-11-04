"""Carga de los ficheros crudos del concurso.

Formatos esperados (ver ``docs/hallazgos_datos.md``):
    - ``algoritmos/<codigo>.csv``: columnas ``datetime,open,high,low,close`` en velas de 4h.
    - ``trades_benchmark.csv``: operaciones de la cartera de referencia.
    - ``benchmark_monthly_returns.csv`` / ``benchmark_yearly_returns.csv``: rentabilidad oficial.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLC = ["open", "high", "low", "close"]


def algo_dir(raw: Path, algos_subdir: str) -> Path:
    """Ruta a la carpeta de algoritmos dentro de ``data/raw``."""
    return raw / algos_subdir


def list_algo_files(raw: Path, algos_subdir: str) -> list[tuple[str, Path]]:
    """Lista ``(codigo, ruta)`` de todos los CSV de algoritmos, ordenados por código."""
    d = algo_dir(raw, algos_subdir)
    files = sorted(d.glob("*.csv"))
    return [(f.stem, f) for f in files]


def load_algo_csv(path: Path) -> pd.DataFrame:
    """Carga un CSV de algoritmo a OHLC indexado por ``datetime`` (naive), ordenado.

    Garantiza un ``DatetimeIndex`` aunque el fichero esté vacío o tenga fechas corruptas
    (se descartan las filas no parseables). Elimina duplicados de índice (último gana).
    """
    df = pd.read_csv(path)
    empty = pd.DataFrame(columns=OHLC, index=pd.DatetimeIndex([], name="datetime"))
    if df.empty or "datetime" not in df.columns:
        return empty
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    if df.empty:
        return empty
    df = df[~df.index.duplicated(keep="last")]
    return df[OHLC].astype(float)


def load_trades(raw: Path, trades_file: str) -> pd.DataFrame:
    """Carga y tipa ``trades_benchmark.csv``.

    Convierte fechas a UTC. ``dateClose`` vacío indica posición abierta.
    """
    df = pd.read_csv(raw / trades_file)
    df["dateOpen"] = pd.to_datetime(df["dateOpen"], utc=True, errors="coerce")
    df["dateClose"] = pd.to_datetime(df["dateClose"], utc=True, errors="coerce")
    df["is_open"] = df["dateClose"].isna()
    return df


def load_benchmark_monthly(raw: Path, monthly_file: str) -> pd.Series:
    """Serie de rentabilidad mensual oficial del benchmark, indexada por fin de mes."""
    df = pd.read_csv(raw / monthly_file)
    idx = pd.PeriodIndex(df["month"], freq="M").to_timestamp(how="end").normalize()
    return pd.Series(df["monthly_return"].to_numpy(), index=idx, name="benchmark_ret")


def load_benchmark_yearly(raw: Path, yearly_file: str) -> pd.DataFrame:
    """Rentabilidad anual oficial del benchmark, indexada por año."""
    df = pd.read_csv(raw / yearly_file).set_index("year")
    return df
