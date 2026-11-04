"""Perfilado de cada algoritmo a partir únicamente de su serie (y los trades del benchmark).

Se infieren, "a contraluz":
    - Actividad: fracción de tiempo invertido vs fuera de mercado.
    - Rendimiento/riesgo: retorno total, volatilidad, Sharpe, máximo drawdown.
    - Rendimiento condicionado a cada fase de mercado.
    - Correlaciones y agrupamiento (¿qué algoritmos operan parecido?).
    - Etiqueta de estilo inferida (seguidor de tendencia, reversión, defensivo) con evidencia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from ..evaluation import metrics

STYLES = ["tendencia", "reversion", "defensivo"]


def algo_metrics(returns: pd.DataFrame, active: pd.DataFrame,
                 periods_per_year: int = 252) -> pd.DataFrame:
    """Métricas de rendimiento/riesgo y actividad por algoritmo (columnas del panel)."""
    rows = {}
    for code in returns.columns:
        r = returns[code].dropna()
        act = active[code].reindex(r.index).fillna(False)
        r_active = r[act]
        rows[code] = {
            "retorno_total": float((1 + r).prod() - 1.0),
            "vol_anual": metrics.annual_vol(r_active, periods_per_year),
            "sharpe": metrics.sharpe(r_active, periods_per_year),
            "max_drawdown": metrics.max_drawdown(r),
            "frac_en_mercado": float(act.mean()),
            "dias_vivo": int(act.sum()),
        }
    return pd.DataFrame(rows).T


def performance_by_regime(returns: pd.DataFrame, dominant: pd.Series) -> pd.DataFrame:
    """Retorno medio diario (anualizado) de cada algoritmo condicionado a la fase dominante."""
    dom = dominant.reindex(returns.index).ffill()
    out = {}
    for regime in ["alcista", "bajista", "lateral", "crisis"]:
        mask = dom == regime
        out[regime] = returns.loc[mask].mean() * 252
    return pd.DataFrame(out)


def trade_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos de operación por algoritmo a partir de ``trades_benchmark``."""
    df = trades.dropna(subset=["dateClose"]).copy()
    df["dur_dias"] = (df["dateClose"] - df["dateOpen"]).dt.total_seconds() / 86400.0
    g = df.groupby("productname")
    return pd.DataFrame({
        "n_operaciones": g.size(),
        "duracion_media_dias": g["dur_dias"].mean(),
        "volumen_medio": g["volume"].mean(),
    })


def correlation_matrix(returns: pd.DataFrame, min_overlap: int = 60) -> pd.DataFrame:
    """Matriz de correlación entre algoritmos (solo sobre días activos comunes)."""
    r = returns.replace(0.0, np.nan)
    corr = r.corr(min_periods=min_overlap)
    return corr.fillna(0.0)


def cluster_algos(corr: pd.DataFrame, n_clusters: int = 6) -> tuple[np.ndarray, pd.Series]:
    """Agrupamiento jerárquico sobre la distancia ``1 - correlación``.

    Returns:
        ``(linkage, labels)`` donde ``linkage`` es la matriz para el dendrograma y
        ``labels`` asigna un clúster a cada algoritmo.
    """
    dist = 1.0 - corr.to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = dist[np.triu_indices_from(dist, k=1)]
    link = linkage(condensed, method="average")
    labels = pd.Series(fcluster(link, t=n_clusters, criterion="maxclust"),
                       index=corr.index, name="cluster")
    return link, labels


def infer_style(returns: pd.DataFrame, market_ret: pd.Series,
                by_regime: pd.DataFrame) -> pd.DataFrame:
    """Infiere una etiqueta de estilo por algoritmo con la evidencia que la sostiene.

    Heurística interpretable:
        - beta al mercado > 0.4  → seguidor de tendencia.
        - beta al mercado < -0.1 o gana en bajista/crisis → reversión a la media.
        - baja actividad/volatilidad y beta cercana a 0 → defensivo.
    """
    mkt = market_ret.reindex(returns.index).fillna(0.0)
    var_m = float(mkt.var())
    rows = {}
    for code in returns.columns:
        r = returns[code].fillna(0.0)
        beta = float(np.cov(r, mkt)[0, 1] / var_m) if var_m > 0 else 0.0
        defensive_perf = by_regime.loc[code, ["bajista", "crisis"]].mean()
        bull_perf = by_regime.loc[code, "alcista"]
        if beta > 0.4:
            style = "tendencia"
        elif beta < -0.1 or (defensive_perf > 0 and defensive_perf > bull_perf):
            style = "reversion"
        else:
            style = "defensivo"
        rows[code] = {"beta_mercado": beta, "estilo": style,
                      "perf_defensiva": float(defensive_perf),
                      "perf_alcista": float(bull_perf)}
    return pd.DataFrame(rows).T


def build_profiles(returns: pd.DataFrame, active: pd.DataFrame, trades: pd.DataFrame,
                   dominant: pd.Series, market_ret: pd.Series) -> dict[str, pd.DataFrame]:
    """Orquesta todo el perfilado y devuelve las tablas resultantes."""
    met = algo_metrics(returns, active)
    by_regime = performance_by_regime(returns, dominant)
    tstats = trade_stats(trades)
    corr = correlation_matrix(returns)
    link, labels = cluster_algos(corr)
    styles = infer_style(returns, market_ret, by_regime)

    fichas = met.join(styles[["beta_mercado", "estilo"]]).join(tstats)
    fichas["cluster"] = labels
    return {
        "metrics": met,
        "by_regime": by_regime,
        "trade_stats": tstats,
        "correlation": corr,
        "linkage": link,
        "labels": labels,
        "styles": styles,
        "fichas": fichas,
    }
