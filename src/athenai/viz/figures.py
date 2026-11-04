"""Funciones de figura del proyecto. Todas usan el estilo único de ``viz.style``.

Cada función devuelve la ruta del PNG generado en ``docs/img``. Las cifras provienen siempre
del pipeline; estas funciones solo dibujan.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.cluster.hierarchy import dendrogram

from ..evaluation import metrics
from ..regimes.fuzzy import REGIMES
from . import style as S

VAR_TITLES = {
    "trend": "Tendencia (z-score causal)",
    "vol": "Volatilidad (z-score causal)",
    "drawdown": "Drawdown (fracción)",
    "breadth": "Amplitud (fracción en positivo)",
}
SET_LABELS = {
    "neg": "negativa", "neu": "neutra", "pos": "positiva",
    "low": "baja", "mid": "media", "high": "alta",
    "small": "pequeño", "large": "grande", "weak": "débil", "strong": "fuerte",
}


def _pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))


def fig_banner(proba: pd.DataFrame, out_dir: Path) -> Path:
    """Banner superior del README: título, subtítulo y mini-curva de P(fase) apilada."""
    S.set_style()
    fig = plt.figure(figsize=(11, 2.9))
    fig.patch.set_facecolor(S.NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(S.NAVY)
    ax.axis("off")

    # Mini-curva apilada de probabilidades, suavizada para estilizar la franja inferior.
    p = proba[REGIMES].rolling(15, min_periods=1).mean()
    p = p.div(p.sum(axis=1), axis=0).iloc[::3]
    x = np.arange(len(p))
    bottom = np.zeros(len(p))
    for r in REGIMES:
        top = bottom + p[r].to_numpy()
        ax.fill_between(x, bottom, top, color=S.REGIME_COLORS[r], alpha=0.92, linewidth=0)
        bottom = top
    ax.set_xlim(0, len(p) - 1)
    ax.set_ylim(0, 2.75)  # la franja ocupa el ~36% inferior; el resto queda para el texto

    ax.text(0.035, 0.82, "AthenAI · Régimen & Asignación",
            transform=ax.transAxes, color="white", fontsize=24, fontweight="bold")
    ax.text(0.037, 0.66, "Detección difusa del régimen de mercado + asignación de capital "
            "con aprendizaje por refuerzo — IA explicable",
            transform=ax.transAxes, color="#AEBED0", fontsize=12)
    ax.text(0.037, 0.52, "P(fase) estimada a lo largo de 2020–2024",
            transform=ax.transAxes, color="#6E82A0", fontsize=9.5, style="italic")
    return S.save_fig(fig, "banner", out_dir, tight=False)


def fig_membership(fuzzy, out_dir: Path) -> Path:
    """Funciones de pertenencia difusas (2x2) de las variables de entrada."""
    S.set_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    palette = [S.BLUE, S.GRAY, S.AMBER]
    for ax, var in zip(axes.ravel(), ["trend", "vol", "drawdown", "breadth"], strict=False):
        curves = fuzzy.membership_curves(var)
        for i, (name, (x, y)) in enumerate(curves.items()):
            ax.plot(x, y, color=palette[i % len(palette)], linewidth=2.2,
                    label=SET_LABELS.get(name, name))
            ax.fill_between(x, 0, y, color=palette[i % len(palette)], alpha=0.12)
        ax.set_title(VAR_TITLES[var])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("grado de pertenencia")
        ax.legend(loc="upper center", ncol=3, fontsize=9)
    fig.suptitle("Bloque 1 · Funciones de pertenencia del sistema difuso", y=1.02)
    return S.save_fig(fig, "fuzzy_membership", out_dir)


def fig_phase_areas(proba: pd.DataFrame, market_index: pd.Series, out_dir: Path) -> Path:
    """Área apilada de las 4 probabilidades de fase y, debajo, el índice de mercado."""
    S.set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    p = proba[REGIMES]
    ax1.stackplot(p.index, [p[r] for r in REGIMES],
                  colors=[S.REGIME_COLORS[r] for r in REGIMES],
                  labels=[S.REGIME_LABELS[r] for r in REGIMES], alpha=0.85)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("P(fase)")
    ax1.set_title("Bloque 1 · Probabilidad de cada fase en el tiempo (causal)")
    ax1.legend(loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.28))

    ax2.plot(market_index.index, market_index.to_numpy(), color=S.NAVY, linewidth=1.8)
    ax2.set_ylabel("Índice mercado\n(base 100)")
    ax2.set_xlabel("Fecha")
    _pct_axis(ax2)
    return S.save_fig(fig, "phase_probabilities", out_dir)


def fig_main_capital(series_net: dict[str, pd.Series], monthly_regime: pd.Series,
                     splits: dict, out_dir: Path) -> Path:
    """Figura principal: capital de estrategia vs benchmark vs equiponderado en TEST,
    con el fondo coloreado según la fase dominante."""
    S.set_style()
    lo, hi = splits["val_end"], splits["test_end"]

    def seg(s):
        return s[(s.index > lo) & (s.index <= hi)]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    dom_test = monthly_regime[(monthly_regime.index > lo) & (monthly_regime.index <= hi)]
    S.shade_regimes(ax, dom_test, alpha=0.14)

    order = [("rl", "Estrategia RL (fase→cartera)", "estrategia"),
             ("benchmark", "Benchmark oficial", "benchmark"),
             ("equiponderado", "Equiponderado (271 algos)", "equiponderado")]
    for key, label, ckey in order:
        if key not in series_net:
            continue
        eq = metrics.equity_curve(seg(series_net[key]))
        lw = 2.8 if key == "rl" else 1.9
        ls = "-" if key != "equiponderado" else "--"
        ax.plot(eq.index, eq.to_numpy(), label=label, color=S.SERIES_COLORS[ckey],
                linewidth=lw, linestyle=ls)

    ax.set_title("Capital en el periodo de TEST (2024, fuera de muestra) · base 100")
    ax.set_ylabel("Capital (base 100)")
    ax.set_xlabel("Fecha")
    _pct_axis(ax)
    h1, l1 = ax.get_legend_handles_labels()
    leg1 = ax.legend(h1, l1, loc="upper left", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=S.regime_legend_handles(), loc="lower left", fontsize=8, ncol=2,
              title="Fase dominante")
    return S.save_fig(fig, "capital_test", out_dir)


def fig_regime_heatmap(by_regime: pd.DataFrame, labels: pd.Series, out_dir: Path) -> Path:
    """Heatmap algoritmo × fase (rendimiento anualizado condicionado), ordenado por clúster."""
    S.set_style()
    order = labels.sort_values().index
    data = by_regime.reindex(order)[REGIMES]
    vmax = float(np.nanpercentile(np.abs(data.to_numpy()), 95)) or 1.0

    fig, ax = plt.subplots(figsize=(7.5, 9))
    im = ax.imshow(data.to_numpy(), aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([S.REGIME_LABELS[r] for r in REGIMES], rotation=20, ha="right")
    ax.set_yticks([])
    ax.set_ylabel(f"{len(data)} algoritmos (ordenados por clúster)")
    ax.set_title("Bloque 2 · Rendimiento anualizado\npor algoritmo y fase de mercado")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Retorno anualizado condicionado")
    return S.save_fig(fig, "regime_heatmap", out_dir)


def fig_dendrogram(linkage_mat, labels: pd.Series, out_dir: Path) -> Path:
    """Dendrograma del agrupamiento jerárquico de algoritmos por correlación."""
    S.set_style()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    dendrogram(linkage_mat, no_labels=True, color_threshold=0.7 * linkage_mat[:, 2].max(),
               above_threshold_color=S.GRAY, ax=ax)
    ax.set_title("Bloque 2 · Agrupamiento de algoritmos por correlación (distancia = 1 − ρ)")
    ax.set_ylabel("Distancia")
    ax.set_xlabel(f"{len(labels)} algoritmos")
    return S.save_fig(fig, "dendrogram", out_dir)


def fig_policy(policy: pd.Series, portfolios: pd.DataFrame, styles: pd.DataFrame,
               out_dir: Path) -> Path:
    """Política del agente: fase dominante → composición por estilo de la cartera elegida."""
    S.set_style()
    style_of = styles["estilo"]
    style_names = ["tendencia", "reversion", "defensivo"]
    # Peso por estilo de la cartera elegida en cada fase.
    mat = np.zeros((len(REGIMES), len(style_names)))
    for i, regime in enumerate(REGIMES):
        w = portfolios.iloc[int(policy.loc[regime])]
        for j, st in enumerate(style_names):
            codes = style_of.index[style_of == st]
            mat[i, j] = w.reindex(codes).fillna(0.0).sum()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(style_names)))
    ax.set_xticklabels([s.capitalize() for s in style_names])
    ax.set_yticks(range(len(REGIMES)))
    ax.set_yticklabels([f"{S.REGIME_LABELS[r]}\n→ cartera {int(policy.loc[r])}" for r in REGIMES])
    for i in range(len(REGIMES)):
        for j in range(len(style_names)):
            ax.text(j, i, f"{mat[i, j]*100:.0f}%", ha="center", va="center",
                    color="white" if mat[i, j] > 0.5 else S.INK, fontsize=10)
    ax.set_title("Bloque 3 · Política aprendida: fase dominante → estilo de la cartera")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="peso en la cartera")
    return S.save_fig(fig, "rl_policy", out_dir)


def fig_quantum(mu: np.ndarray, risk: np.ndarray, labels: list[str],
                classical: list[int], qaoa_sel: list[int] | None, out_dir: Path) -> Path:
    """Compara la selección de subcartera clásica exacta y QAOA sobre rentabilidad vs riesgo."""
    S.set_style()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(risk, mu, s=60, color=S.LIGHT_GRAY, zorder=2, label="Candidatos")
    ax.scatter(risk[classical], mu[classical], s=150, facecolors="none",
               edgecolors=S.NAVY, linewidths=2.4, zorder=3, label="Elegidos · clásico exacto")
    if qaoa_sel:
        ax.scatter(risk[qaoa_sel], mu[qaoa_sel], s=55, color=S.AMBER, marker="x",
                   linewidths=2.4, zorder=4, label="Elegidos · QAOA")
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (risk[i], mu[i]), fontsize=8, color=S.GRAY,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Riesgo (volatilidad anualizada)")
    ax.set_ylabel("Rentabilidad esperada (anualizada)")
    ax.set_title("Bloque 4 · Selección de subcartera: QAOA vs solución clásica exacta")
    ax.legend(loc="best", fontsize=9)
    return S.save_fig(fig, "quantum_qaoa", out_dir)


def make_core_figures(cfg, reg, prof, alloc, processed) -> dict[str, Path]:
    """Genera todas las figuras principales del README y devuelve sus rutas."""
    out = cfg.path("figures")
    paths = {}
    paths["banner"] = fig_banner(reg["proba"], out)
    paths["membership"] = fig_membership(reg["fuzzy"], out)
    paths["phase_areas"] = fig_phase_areas(reg["proba"], processed["market"]["mkt_index"], out)
    paths["capital"] = fig_main_capital(alloc["series_net"], alloc["monthly_regime"],
                                        alloc["splits"], out)
    paths["heatmap"] = fig_regime_heatmap(prof["by_regime"], prof["labels"], out)
    paths["dendrogram"] = fig_dendrogram(prof["linkage"], prof["labels"], out)
    paths["policy"] = fig_policy(alloc["policy"], alloc["portfolios"], prof["styles"], out)
    return paths
