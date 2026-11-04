"""Estilo visual único del proyecto: paleta sobria tipo banca y ajustes de matplotlib.

La paleta categórica de fases está validada como distinguible incluso con daltonismo
(separación CVD ΔE suficiente). Todas las figuras comparten tipografía, rejilla recesiva y
resolución, para que el repositorio se lea como un sistema coherente.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Paleta base (azul marino / azul / gris + acento) ---
NAVY = "#0B2545"
BLUE = "#2E6FB0"
TEAL = "#2A9D8F"
GRAY = "#6C757D"
LIGHT_GRAY = "#B8BEC7"
AMBER = "#E8902E"
RED = "#C1121F"
PURPLE = "#6A4C93"
INK = "#1A1A1A"
SURFACE = "#FCFCFB"

# --- Colores de las 4 fases (categóricos, validados) ---
REGIME_COLORS = {
    "alcista": TEAL,
    "bajista": RED,
    "lateral": BLUE,
    "crisis": AMBER,
}
REGIME_LABELS = {
    "alcista": "Alcista",
    "bajista": "Bajista",
    "lateral": "Lateral",
    "crisis": "Crisis / alta volatilidad",
}

# --- Colores de las series de estrategias (contexto distinto a las fases) ---
SERIES_COLORS = {
    "estrategia": NAVY,
    "benchmark": GRAY,
    "equiponderado": AMBER,
    "mejor_algo": PURPLE,
    "difusa_sin_rl": TEAL,
}

DPI = 170


def set_style() -> None:
    """Aplica los ajustes globales de matplotlib (idempotente)."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": GRAY,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E3E6EA",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.titlecolor": NAVY,
        "axes.labelcolor": INK,
        "lines.linewidth": 2.0,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
    })


def shade_regimes(ax, dominant, alpha: float = 0.16) -> None:
    """Colorea el fondo del eje según la fase dominante en cada tramo temporal.

    Args:
        ax: eje de matplotlib (eje x temporal).
        dominant: Series de etiquetas de fase indexada por fecha.
        alpha: opacidad del sombreado.
    """
    if dominant.empty:
        return
    dom = dominant.dropna()
    changes = dom.ne(dom.shift())
    starts = dom.index[changes]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else dom.index[-1]
        color = REGIME_COLORS.get(dom.loc[start], GRAY)
        ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0, zorder=0)


def regime_legend_handles():
    """Parches de leyenda para las 4 fases (identidad nunca solo por color)."""
    from matplotlib.patches import Patch
    return [Patch(facecolor=REGIME_COLORS[r], alpha=0.5, label=REGIME_LABELS[r])
            for r in ["alcista", "bajista", "lateral", "crisis"]]


def save_fig(fig, name: str, out_dir: Path, tight: bool = True) -> Path:
    """Guarda una figura en PNG en ``out_dir`` con la resolución estándar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
