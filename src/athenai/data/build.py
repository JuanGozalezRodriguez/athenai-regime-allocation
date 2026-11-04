"""Construcción de los datasets procesados (parquets cacheados).

Transforma los datos crudos en tablas limpias y causales que consume el resto del pipeline:

    - ``market.parquet``: índice de mercado equiponderado de algoritmos activos + amplitud.
    - ``returns_panel.parquet`` / ``active_panel.parquet`` / ``close_panel.parquet``: panel
      diario de los algoritmos invertidos por el benchmark.
    - ``trades_clean.parquet``: operaciones del benchmark limpias.
    - ``benchmark.parquet``: curva mensual oficial del benchmark.

El índice de mercado se calcula en *streaming* sobre el universo completo (miles de ficheros)
para no cargar todo en memoria.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from . import clean, loaders

logger = logging.getLogger("athenai")


def _master_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1D")


def build_market_index(
    files: list[tuple[str, Path]], start: str, end: str, winsor: float = 0.2
) -> tuple[pd.DataFrame, dict]:
    """Índice de mercado equiponderado y amplitud, en *streaming* sobre el universo.

    Para cada día acumula, solo sobre algoritmos "vivos": suma de retornos, nº de vivos,
    nº en positivo y nº que realmente operan ese día. El resultado es estrictamente causal.

    Returns:
        ``(market_df, stats)`` donde ``stats`` resume el universo (nº de ficheros, series
        totalmente planas, etc.) para alimentar las cifras del README.
    """
    master = _master_index(start, end)
    n = len(master)
    sum_ret = np.zeros(n)
    n_alive = np.zeros(n)
    n_pos = np.zeros(n)
    n_active = np.zeros(n)
    pos = {d: i for i, d in enumerate(master)}
    n_fully_flat = 0
    n_nonempty = 0

    for k, (_code, path) in enumerate(files):
        df = loaders.load_algo_csv(path)
        close = clean.daily_close(df)
        if close.empty:
            continue
        n_nonempty += 1
        if close.nunique() <= 1:
            n_fully_flat += 1
        amask = clean.active_mask(close)
        rets = clean.winsorize(clean.daily_returns(close), winsor)
        # Nos quedamos solo con el tramo vivo dentro del periodo.
        alive_idx = close.index[amask]
        for d in alive_idx:
            if d not in pos:
                continue
            i = pos[d]
            r = rets.loc[d]
            sum_ret[i] += r
            n_alive[i] += 1
            n_pos[i] += r > 0
            n_active[i] += r != 0
        if (k + 1) % 2000 == 0:
            logger.info("  índice de mercado: %d/%d ficheros procesados", k + 1, len(files))

    with np.errstate(invalid="ignore", divide="ignore"):
        mkt_ret = np.where(n_alive > 0, sum_ret / n_alive, 0.0)
        breadth = np.where(n_alive > 0, n_pos / n_alive, np.nan)
        participation = np.where(n_alive > 0, n_active / n_alive, np.nan)
    out = pd.DataFrame(
        {
            "mkt_ret": mkt_ret,
            "mkt_index": 100.0 * np.cumprod(1.0 + mkt_ret),
            "breadth": breadth,
            "participation": participation,
            "n_alive": n_alive.astype(int),
        },
        index=master,
    )
    out.index.name = "date"
    stats = {
        "n_universe": len(files),
        "n_nonempty": n_nonempty,
        "n_fully_flat": n_fully_flat,
        "pct_fully_flat": round(100.0 * n_fully_flat / max(n_nonempty, 1), 1),
        "n_alive_max": int(n_alive.max()),
        "mean_out_of_market_pct": round(100.0 * float(np.nanmean(1.0 - participation)), 1),
    }
    return out, stats


def build_invested_panels(
    invested_files: list[tuple[str, Path]], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Paneles diarios (retornos, cierres, máscara de actividad) de los algos invertidos."""
    master = _master_index(start, end)
    closes, rets, actives = {}, {}, {}
    for code, path in invested_files:
        df = loaders.load_algo_csv(path)
        close = clean.daily_close(df).reindex(master).ffill()
        amask = pd.Series(clean.active_mask(close.ffill().bfill()), index=master)
        # Fuera del tramo vivo: cierre NaN; retorno 0; inactivo.
        close = close.where(amask)
        r = clean.winsorize((close / close.shift(1) - 1.0).fillna(0.0))
        r = r.where(amask, 0.0)
        closes[code] = close
        rets[code] = r
        actives[code] = amask
    return {
        "close": pd.DataFrame(closes).rename_axis("date"),
        "returns": pd.DataFrame(rets).rename_axis("date"),
        "active": pd.DataFrame(actives).rename_axis("date"),
    }


def build_benchmark(cfg: Config, raw: Path) -> pd.DataFrame:
    """Curva mensual oficial del benchmark (retorno + índice base 100)."""
    monthly = loaders.load_benchmark_monthly(raw, cfg["data"]["monthly_returns_file"])
    monthly = clean.clip_period(monthly, cfg["data"]["start"], cfg["data"]["end"])
    index = 100.0 * (1.0 + monthly).cumprod()
    out = pd.DataFrame({"benchmark_ret": monthly, "benchmark_index": index})
    out.index.name = "date"
    return out


def build_all(cfg: Config) -> dict[str, Path]:
    """Construye y cachea todos los parquets procesados. Devuelve las rutas escritas."""
    raw = cfg.path("raw")
    processed = cfg.path("processed")
    processed.mkdir(parents=True, exist_ok=True)
    start, end = cfg["data"]["start"], cfg["data"]["end"]

    logger.info("Listando universo de algoritmos en %s", raw)
    files = loaders.list_algo_files(raw, cfg["data"]["algos_dir"])
    logger.info("  %d ficheros de algoritmos encontrados", len(files))

    trades = loaders.load_trades(raw, cfg["data"]["trades_file"])
    invested_codes = sorted(trades["productname"].unique())
    code_to_path = dict(files)
    invested_files = [(c, code_to_path[c]) for c in invested_codes if c in code_to_path]
    logger.info("  %d algoritmos invertidos por el benchmark", len(invested_files))

    logger.info("Construyendo índice de mercado (streaming sobre el universo)...")
    market, mkt_stats = build_market_index(files, start, end)

    logger.info("Construyendo paneles de los algoritmos invertidos...")
    panels = build_invested_panels(invested_files, start, end)

    logger.info("Construyendo curva del benchmark...")
    benchmark = build_benchmark(cfg, raw)

    out_paths = {}
    writes = {
        "market": market,
        "returns_panel": panels["returns"],
        "close_panel": panels["close"],
        "active_panel": panels["active"],
        "benchmark": benchmark,
        "trades_clean": trades,
    }
    for name, obj in writes.items():
        p = processed / f"{name}.parquet"
        obj.to_parquet(p)
        out_paths[name] = p
        logger.info("  escrito %s  (%s)", p.name, "x".join(map(str, obj.shape)))

    stats = {
        **mkt_stats,
        "n_invested": len(invested_files),
        "date_start": start,
        "date_end": end,
        "n_days": int(market.shape[0]),
    }
    stats_path = processed / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    out_paths["dataset_stats"] = stats_path
    logger.info("  escrito dataset_stats.json: %s", stats)
    return out_paths


def load_processed(cfg: Config) -> dict[str, pd.DataFrame]:
    """Carga los parquets procesados en un diccionario."""
    processed = cfg.path("processed")
    names = ["market", "returns_panel", "close_panel", "active_panel", "benchmark", "trades_clean"]
    out = {}
    for name in names:
        p = processed / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"No existe {p}. Ejecuta primero `python scripts/build_dataset.py`."
            )
        out[name] = pd.read_parquet(p)
    return out
