"""Orquestación de alto nivel: une los tres bloques de forma causal y reproducible.

Convención temporal: la fase dominante se estima a diario (causal); la decisión de cartera se
toma a fin de mes con la fase conocida en ese momento y se aplica al retorno del mes siguiente
(lo gestiona ``evaluation.backtest.backtest_weights``). El periodo de TEST es estrictamente
fuera de muestra.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .allocation import baselines
from .allocation.agent import derive_policy, train_agent, weights_from_policy
from .allocation.env import RegimeAllocationEnv, build_regime_portfolios
from .config import Config, set_seed
from .evaluation import metrics
from .evaluation.backtest import backtest_weights, to_monthly_returns
from .profiling.profile import build_profiles
from .regimes import baseline as regbaseline
from .regimes.fuzzy import REGIMES, FuzzyRegimeModel
from .regimes.indicators import compute_indicators

logger = logging.getLogger("athenai")
MONTHS = 12


def run_regimes(cfg: Config, processed: dict[str, pd.DataFrame]) -> dict:
    """Bloque 1: indicadores causales, probabilidades difusas y comparación con baselines."""
    market = processed["market"]
    indicators = compute_indicators(market, cfg)
    fuzzy = FuzzyRegimeModel.from_config(cfg)
    proba = fuzzy.predict_proba(indicators)
    dominant = fuzzy.dominant(proba)

    # Baselines interpretables de régimen.
    thr = regbaseline.ThresholdRegime().predict(indicators)
    train_mask = indicators.index <= pd.Timestamp(cfg["splits"]["train_end"])
    km = regbaseline.KMeansRegime(seed=cfg.seed).fit(indicators[train_mask])
    km_labels = km.predict(indicators)

    return {
        "indicators": indicators,
        "fuzzy": fuzzy,
        "proba": proba,
        "dominant": dominant,
        "threshold": thr,
        "kmeans": km_labels,
        "agreement_threshold": regbaseline.agreement(dominant, thr),
        "agreement_kmeans": regbaseline.agreement(dominant, km_labels),
    }


def run_profiling(cfg: Config, processed: dict[str, pd.DataFrame], dominant: pd.Series) -> dict:
    """Bloque 2: perfilado de los algoritmos invertidos."""
    return build_profiles(
        returns=processed["returns_panel"],
        active=processed["active_panel"],
        trades=processed["trades_clean"],
        dominant=dominant,
        market_ret=processed["market"]["mkt_ret"],
    )


def _regime_to_idx(regime: pd.Series) -> np.ndarray:
    return regime.map({r: i for i, r in enumerate(REGIMES)}).to_numpy()


def run_allocation(cfg: Config, processed: dict[str, pd.DataFrame], dominant: pd.Series) -> dict:
    """Bloque 3: carteras por fase, agente RL, baselines y backtest por tramos."""
    set_seed(cfg.seed)
    a = cfg["allocation"]
    returns = processed["returns_panel"]
    monthly_asset = to_monthly_returns(returns)
    monthly_regime = dominant.resample("M").last().reindex(monthly_asset.index).ffill()

    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    val_end = pd.Timestamp(cfg["splits"]["val_end"])
    test_end = pd.Timestamp(cfg["splits"]["test_end"])

    # --- Carteras por fase con datos SOLO de entrenamiento ---
    train_mask_daily = returns.index <= train_end
    portfolios = build_regime_portfolios(
        returns[train_mask_daily], dominant[returns.index <= train_end], top_k=a["top_k_algos"]
    )

    # --- Entrenamiento del agente (meses de train, decisión→mes siguiente) ---
    state = monthly_regime
    next_ret = monthly_asset.shift(-1)  # retorno del mes posterior a la decisión
    train_dates = monthly_asset.index[(monthly_asset.index <= train_end)]
    tr_states = _regime_to_idx(state.loc[train_dates]).astype(int)
    tr_next = next_ret.loc[train_dates].fillna(0.0)
    # Descartar la última fila (sin mes siguiente).
    env = RegimeAllocationEnv(
        tr_states[:-1], tr_next.iloc[:-1], portfolios,
        cost_bps=a["transaction_cost_bps"], risk_aversion=a["risk_aversion"],
    )
    ql = a["q_learning"]
    agent = train_agent(env, episodes=ql["episodes"], seed=cfg.seed,
                        alpha=ql["alpha"], gamma=ql["gamma"], epsilon=ql["epsilon"])
    policy = derive_policy(agent)

    # --- Pesos de cada estrategia sobre toda la línea temporal ---
    rebalance_dates = monthly_asset.index
    w_rl = weights_from_policy(policy, monthly_regime, portfolios)
    w_fuzzy = baselines.fuzzy_rule_weights(monthly_regime, portfolios)
    w_eq = baselines.equal_weight_weights(rebalance_dates, returns.columns)
    w_best, best_code = baselines.best_algo_weights(returns[train_mask_daily], rebalance_dates)

    cost = a["transaction_cost_bps"]
    bt = {
        "rl": backtest_weights(w_rl, monthly_asset, cost),
        "difusa_sin_rl": backtest_weights(w_fuzzy, monthly_asset, cost),
        "equiponderado": backtest_weights(w_eq, monthly_asset, cost),
        "mejor_algo": backtest_weights(w_best, monthly_asset, cost),
    }

    # --- Serie del benchmark oficial (ya es retorno mensual) ---
    bench = processed["benchmark"]["benchmark_ret"]

    # --- Métricas por tramo (val y, sobre todo, TEST fuera de muestra) ---
    def _slice(s: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
        return s[(s.index > lo) & (s.index <= hi)]

    splits = {"validacion": (train_end, val_end), "test": (val_end, test_end)}
    results = {}
    series_net = {name: r["net"] for name, r in bt.items()}
    series_net["benchmark"] = bench
    turnovers = {name: r["turnover"] for name, r in bt.items()}

    for split_name, (lo, hi) in splits.items():
        table = {}
        for name, s in series_net.items():
            seg = _slice(s, lo, hi)
            if name in turnovers:
                w_seg = _slice(turnovers[name], lo, hi)
                table[name] = metrics.summary(seg, MONTHS)
                table[name]["turnover"] = float(w_seg.mean()) if not w_seg.empty else 0.0
            else:
                table[name] = metrics.summary(seg, MONTHS)
        results[split_name] = pd.DataFrame(table).T

    return {
        "portfolios": portfolios,
        "agent": agent,
        "policy": policy,
        "best_algo": best_code,
        "monthly_regime": monthly_regime,
        "monthly_asset": monthly_asset,
        "weights": {"rl": w_rl, "difusa_sin_rl": w_fuzzy,
                    "equiponderado": w_eq, "mejor_algo": w_best},
        "backtest": bt,
        "series_net": series_net,
        "results": results,
        "splits": {"train_end": train_end, "val_end": val_end, "test_end": test_end},
    }


# --- Nombres legibles de las estrategias para las tablas del README ---
STRATEGY_LABELS = {
    "rl": "Estrategia RL (fase→cartera)",
    "benchmark": "Benchmark oficial",
    "equiponderado": "Equiponderado (271)",
    "difusa_sin_rl": "Regla difusa sin RL",
    "mejor_algo": "Mejor algoritmo (train)",
}
_METRIC_ORDER = ["CAGR", "vol_anual", "Sharpe", "Sortino", "max_drawdown", "Calmar",
                 "pct_periodos_positivos", "turnover"]
_METRIC_HEADERS = ["CAGR", "Vol.", "Sharpe", "Sortino", "Max DD", "Calmar", "% meses +", "Turnover"]
_PCT_METRICS = {"CAGR", "vol_anual", "max_drawdown", "pct_periodos_positivos"}


def _fmt(metric: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if metric in _PCT_METRICS:
        return f"{value*100:.1f}%"
    return f"{value:.2f}"


def results_markdown(results_table: pd.DataFrame, order: list[str]) -> str:
    """Convierte una tabla de resultados en Markdown legible, filas en orden dado."""
    header = "| Estrategia | " + " | ".join(_METRIC_HEADERS) + " |"
    sep = "|" + "---|" * (len(_METRIC_HEADERS) + 1)
    lines = [header, sep]
    for key in order:
        if key not in results_table.index:
            continue
        row = results_table.loc[key]
        cells = [_fmt(m, row.get(m, np.nan)) for m in _METRIC_ORDER]
        label = STRATEGY_LABELS.get(key, key)
        if key == "rl":
            label = f"**{label}**"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect_readme_numbers(cfg: Config, reg: dict, alloc: dict,
                           processed: dict, dataset_stats: dict) -> dict:
    """Reúne TODAS las cifras que el README mostrará (ninguna se escribe a mano)."""
    market = processed["market"]
    dist = (reg["dominant"].value_counts(normalize=True) * 100).round(1).to_dict()
    order = ["rl", "benchmark", "equiponderado", "difusa_sin_rl", "mejor_algo"]

    test_tbl = alloc["results"]["test"]
    val_tbl = alloc["results"]["validacion"]

    # Rentabilidad anual de cada estrategia vs benchmark.
    yearly = {}
    for key in ["rl", "benchmark", "equiponderado"]:
        s = alloc["series_net"][key]
        yr = metrics.yearly_returns(s)
        yearly[key] = {int(y): round(float(v) * 100, 1) for y, v in yr.items()}

    best_key = test_tbl["CAGR"].idxmax()
    rl_beats_bench = bool(test_tbl.loc["rl", "CAGR"] > test_tbl.loc["benchmark", "CAGR"])

    return {
        "dataset": dataset_stats,
        "market_total_return_pct": round(float(market["mkt_index"].iloc[-1] /
                                               market["mkt_index"].iloc[0] - 1) * 100, 1),
        "breadth_mean_pct": round(float(market["breadth"].mean()) * 100, 1),
        "regime_distribution_pct": dist,
        "agreement_threshold_pct": round(reg["agreement_threshold"] * 100, 1),
        "agreement_kmeans_pct": round(reg["agreement_kmeans"] * 100, 1),
        "n_rules": len(reg["fuzzy"].rules),
        "policy": {r: REGIMES[int(alloc["policy"].loc[r])] for r in REGIMES},
        "best_algo": alloc["best_algo"],
        "splits": {k: v.strftime("%Y-%m-%d") for k, v in alloc["splits"].items()},
        "test_table_md": results_markdown(test_tbl, order),
        "val_table_md": results_markdown(val_tbl, order),
        "test_metrics": {
            k: {m: (None if pd.isna(test_tbl.loc[k, m]) else round(float(test_tbl.loc[k, m]), 4))
                for m in _METRIC_ORDER if m in test_tbl.columns}
            for k in order if k in test_tbl.index
        },
        "yearly_returns_pct": yearly,
        "best_test_strategy": best_key,
        "rl_beats_benchmark_test": rl_beats_bench,
    }


def write_results_summary(numbers: dict, path) -> None:
    """Escribe el resumen de cifras del README en JSON (fuente única de verdad)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, ensure_ascii=False, indent=2)
