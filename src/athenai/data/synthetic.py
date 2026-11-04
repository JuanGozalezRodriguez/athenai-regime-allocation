"""Generador de datos sintéticos con el MISMO formato que los datos del concurso.

Permite ejecutar el pipeline completo, los tests y la CI **sin** los datos del organizador
(que no se redistribuyen). Es determinista dada una semilla.

Modelo:
    - Un "mercado" latente recorre 4 regímenes (alcista, bajista, lateral, crisis) mediante
      una cadena de Markov. Cada régimen define deriva y volatilidad del retorno de mercado.
    - Cada algoritmo tiene un estilo (seguidor de tendencia, reversión a la media, defensivo)
      que determina su beta al mercado y su ruido idiosincrático.
    - Se introducen tramos "fuera de mercado" (velas planas), lanzamientos tardíos y series
      que se congelan antes del final, imitando los sesgos observados en los datos reales.
    - Las velas son de 4h con un desfase horario de 1h en verano (DST).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REGIMES = ["alcista", "bajista", "lateral", "crisis"]
# Deriva y volatilidad diaria (en log-retorno) por régimen.
_REGIME_PARAMS = {
    "alcista": (0.0010, 0.006),
    "bajista": (-0.0010, 0.010),
    "lateral": (0.0000, 0.004),
    "crisis": (-0.0025, 0.022),
}
# Matriz de transición (persistente) entre regímenes.
_TRANS = np.array([
    [0.97, 0.01, 0.015, 0.005],
    [0.02, 0.95, 0.02, 0.01],
    [0.02, 0.02, 0.95, 0.01],
    [0.05, 0.10, 0.05, 0.80],
])
STYLES = ["tendencia", "reversion", "defensivo"]


def _simulate_market(
    days: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simula la senda de regímenes y el log-retorno diario del mercado latente."""
    n = len(days)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(4, p=_TRANS[states[t - 1]])
    rets = np.empty(n)
    for t in range(n):
        mu, sig = _REGIME_PARAMS[REGIMES[states[t]]]
        rets[t] = rng.normal(mu, sig)
    return states, rets


def _simulate_algo(
    market_ret: np.ndarray, style: str, rng: np.random.Generator
) -> np.ndarray:
    """Genera el log-retorno diario de un algoritmo a partir del mercado y su estilo."""
    n = len(market_ret)
    if style == "tendencia":
        beta = rng.uniform(0.6, 1.1)
        idio = rng.normal(0, 0.004, n)
        algo = beta * market_ret + idio
    elif style == "reversion":
        beta = rng.uniform(-0.6, -0.2)
        idio = rng.normal(0.0004, 0.006, n)
        algo = beta * market_ret + idio
    else:  # defensivo
        beta = rng.uniform(0.1, 0.35)
        idio = rng.normal(0.0002, 0.003, n)
        algo = beta * market_ret + idio
    return algo


def _apply_out_of_market(ret: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pone a 0 (fuera de mercado) tramos aleatorios del retorno."""
    n = len(ret)
    out = ret.copy()
    i = 0
    while i < n:
        if rng.random() < 0.02:  # arranca un tramo plano
            length = rng.integers(3, 40)
            out[i:i + length] = 0.0
            i += length
        else:
            i += 1
    return out


def _dst_hour_offset(day: pd.Timestamp) -> int:
    """Desfase de 1h en los meses de verano (aproxima el DST observado)."""
    return 1 if 3 <= day.month <= 10 else 0


def _expand_to_candles(day: pd.Timestamp, day_close: float, prev_close: float,
                       is_flat: bool, rng: np.random.Generator) -> list[dict]:
    """Expande un cierre diario a 6 velas de 4h con OHLC coherente."""
    off = _dst_hour_offset(day)
    rows = []
    if is_flat:
        for h in range(0, 24, 4):
            ts = day + pd.Timedelta(hours=h + off)
            rows.append({"datetime": ts, "open": day_close, "high": day_close,
                         "low": day_close, "close": day_close})
        return rows
    # Interpolación log-lineal del cierre previo al del día, con ruido intradía.
    log_path = np.linspace(np.log(prev_close), np.log(day_close), 7)
    for k, h in enumerate(range(0, 24, 4)):
        ts = day + pd.Timedelta(hours=h + off)
        o = float(np.exp(log_path[k]))
        c = float(np.exp(log_path[k + 1]))
        noise = abs(rng.normal(0, 0.0015)) * c
        hi = max(o, c) + noise
        lo = min(o, c) - noise
        rows.append({"datetime": ts, "open": round(o, 4), "high": round(hi, 4),
                     "low": round(lo, 4), "close": round(c, 4)})
    return rows


def generate_synthetic(
    out_dir: str | Path,
    n_algos: int = 60,
    start: str = "2020-06-01",
    end: str = "2024-12-31",
    n_invested: int = 20,
    seed: int = 42,
) -> dict[str, int]:
    """Genera un dataset sintético completo en ``out_dir`` con el formato del concurso.

    Crea ``out_dir/algoritmos/*.csv``, ``trades_benchmark.csv`` y los returns del benchmark.

    Returns:
        Resumen con el número de ficheros y filas generadas.
    """
    out = Path(out_dir)
    algo_out = out / "algoritmos"
    algo_out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    days = pd.date_range(start, end, freq="1D")
    _, market_ret = _simulate_market(days, rng)

    # Códigos pseudoaleatorios de 5 caracteres, estables por semilla.
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
    codes = {"".join(rng.choice(alphabet, size=5)) for _ in range(n_algos * 2)}
    codes = sorted(codes)[:n_algos]

    algo_rets: dict[str, pd.Series] = {}
    n_rows = 0
    for code in codes:
        style = rng.choice(STYLES)
        ret = _simulate_algo(market_ret, style, rng)
        ret = _apply_out_of_market(ret, rng)
        # Lanzamiento tardío y posible congelación anticipada (sesgos reales).
        launch = int(rng.integers(0, len(days) // 4))
        ret[:launch] = 0.0
        if rng.random() < 0.3:
            freeze = int(rng.integers(len(days) // 2, len(days)))
            ret[freeze:] = 0.0
        base = float(rng.uniform(80, 120))
        close = base * np.exp(np.cumsum(ret))
        close_s = pd.Series(close, index=days)
        algo_rets[code] = close_s.pct_change().fillna(0.0)

        # Expandir a velas de 4h.
        rows: list[dict] = []
        prev = base
        for day, px in close_s.items():
            is_flat = abs(px - prev) < 1e-12
            rows.extend(_expand_to_candles(day, float(px), float(prev), is_flat, rng))
            prev = float(px)
        df = pd.DataFrame(rows)
        df.to_csv(algo_out / f"{code}.csv", index=False,
                  date_format="%Y-%m-%d %H:%M:%S")
        n_rows += len(df)

    # --- Benchmark: cartera equiponderada de un subconjunto invertido ---
    invested = sorted(rng.choice(codes, size=min(n_invested, len(codes)), replace=False))
    _write_benchmark(out, invested, algo_rets, days, rng)

    return {"n_algos": len(codes), "n_invested": len(invested), "n_rows": n_rows}


def _write_benchmark(out: Path, invested: list[str], algo_rets: dict[str, pd.Series],
                     days: pd.DatetimeIndex, rng: np.random.Generator) -> None:
    """Escribe ``trades_benchmark.csv`` y los returns mensual/anual del benchmark."""
    # Curva del benchmark = media equiponderada de los algos invertidos.
    panel = pd.DataFrame({c: algo_rets[c] for c in invested})
    bench_ret = panel.mean(axis=1)
    equity = 100.0 * (1 + bench_ret).cumprod()

    monthly = equity.resample("M").last()
    monthly_ret = monthly.pct_change().dropna()
    start_eq = equity.resample("M").first()
    pd.DataFrame({
        "month": monthly_ret.index.to_period("M").astype(str),
        "start_equity": start_eq.reindex(monthly_ret.index).to_numpy(),
        "end_equity": monthly.reindex(monthly_ret.index).to_numpy(),
        "monthly_return": monthly_ret.to_numpy(),
    }).to_csv(out / "benchmark_monthly_returns.csv", index=False)

    yearly = equity.resample("Y").last()
    yearly_ret = yearly.pct_change().dropna()
    start_y = equity.resample("Y").first()
    pd.DataFrame({
        "year": yearly_ret.index.year,
        "start_equity": start_y.reindex(yearly_ret.index).to_numpy(),
        "end_equity": yearly.reindex(yearly_ret.index).to_numpy(),
        "yearly_return": yearly_ret.to_numpy(),
    }).to_csv(out / "benchmark_yearly_returns.csv", index=False)

    # Trades sintéticos (schema-compatibles) para el bloque de perfilado.
    trades = []
    aum = 100_000.0
    for code in invested:
        r = algo_rets[code]
        active = r[r != 0]
        if active.empty:
            continue
        # Agrupar días activos en operaciones contiguas.
        idx = active.index
        gaps = (idx.to_series().diff() > pd.Timedelta(days=3)).cumsum()
        for _, grp in idx.to_series().groupby(gaps):
            d_open = grp.iloc[0].tz_localize("UTC")
            d_close = grp.iloc[-1].tz_localize("UTC")
            vol = float(rng.uniform(5000, 10000))
            trades.append({
                "volume": round(vol, 1),
                "dateOpen": d_open.isoformat(),
                "dateClose": d_close.isoformat(),
                "total_invested_amount_EOD": round(aum * rng.uniform(0.6, 0.95), 2),
                "equity_EOD": round(aum * rng.uniform(0.1, 0.2), 2),
                "AUM": round(aum, 2),
                "equity_normalized": round(aum * rng.uniform(1.0, 1.1), 2),
                "productname": code,
            })
        aum *= 1.05
    pd.DataFrame(trades).to_csv(out / "trades_benchmark.csv", index=False)
