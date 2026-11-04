"""Motor de lógica difusa propio para la detección probabilística del régimen.

Implementación transparente (sin dependencias externas de lógica difusa) para maximizar la
explicabilidad: funciones de pertenencia triangulares legibles y reglas en lenguaje natural.

Flujo de inferencia (tipo Mamdani simplificado a clasificación):
    1. Fuzzificación: cada indicador se traduce a grados de pertenencia a conjuntos difusos
       (p. ej. tendencia = {negativa, neutra, positiva}).
    2. Evaluación de reglas: cada regla "SI ... ENTONCES fase" dispara con una fuerza igual al
       mínimo (operador Y) de las pertenencias de su antecedente.
    3. Agregación: por cada fase se combinan sus reglas (máximo) y se normaliza el vector de
       4 fases para que **sume 1** → probabilidades.

Las 4 fases son: ``alcista``, ``bajista``, ``lateral``, ``crisis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config

REGIMES = ["alcista", "bajista", "lateral", "crisis"]


def trimf(x: np.ndarray, abc: list[float]) -> np.ndarray:
    """Función de pertenencia triangular.

    ``abc = [a, b, c]`` con ``a <= b <= c``. Sube de ``a`` a ``b`` (grado 1 en ``b``) y baja
    de ``b`` a ``c``. Con ``a == b`` es un hombro izquierdo; con ``b == c``, derecho.
    """
    a, b, c = abc
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        left = (x > a) & (x <= b)
        y[left] = (x[left] - a) / (b - a)
    else:
        y[x <= b] = 1.0
    if c > b:
        right = (x > b) & (x < c)
        y[right] = (c - x[right]) / (c - b)
    else:
        y[x >= b] = 1.0
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)


@dataclass(frozen=True)
class Rule:
    """Regla difusa: antecedente (variable→conjunto) y fase consecuente, con texto legible."""

    antecedent: dict[str, str]
    consequent: str
    text: str


def default_rules() -> list[Rule]:
    """Base de reglas por defecto, en lenguaje natural."""
    return [
        Rule({"trend": "pos", "vol": "low"}, "alcista",
             "SI la tendencia es positiva Y la volatilidad es baja ENTONCES alcista"),
        Rule({"trend": "pos", "breadth": "strong"}, "alcista",
             "SI la tendencia es positiva Y la amplitud es fuerte ENTONCES alcista"),
        Rule({"trend": "neg", "breadth": "weak"}, "bajista",
             "SI la tendencia es negativa Y la amplitud es débil ENTONCES bajista"),
        Rule({"trend": "neg", "vol": "mid"}, "bajista",
             "SI la tendencia es negativa Y la volatilidad es media ENTONCES bajista"),
        Rule({"trend": "neu", "vol": "low"}, "lateral",
             "SI la tendencia es neutra Y la volatilidad es baja ENTONCES lateral"),
        Rule({"trend": "neu", "drawdown": "small"}, "lateral",
             "SI la tendencia es neutra Y el drawdown es pequeño ENTONCES lateral"),
        Rule({"vol": "high", "drawdown": "large"}, "crisis",
             "SI la volatilidad es alta Y el drawdown es grande ENTONCES crisis"),
        Rule({"vol": "high", "trend": "neg"}, "crisis",
             "SI la volatilidad es alta Y la tendencia es negativa ENTONCES crisis"),
    ]


@dataclass
class FuzzyRegimeModel:
    """Sistema de inferencia difusa para estimar P(fase) en cada instante.

    Attributes:
        mf: funciones de pertenencia por variable y conjunto (``mf[var][set] = [a,b,c]``).
        rules: base de reglas.
        smoothing: ventana de media móvil causal aplicada a las probabilidades.
    """

    mf: dict[str, dict[str, list[float]]]
    rules: list[Rule] = field(default_factory=default_rules)
    smoothing: int = 1

    @classmethod
    def from_config(cls, cfg: Config) -> FuzzyRegimeModel:
        f = cfg["fuzzy"]
        mf = {
            "trend": f["trend_z"],
            "vol": f["vol_z"],
            "drawdown": f["drawdown"],
            "breadth": f["breadth"],
        }
        return cls(mf=mf, smoothing=int(cfg["regimes"].get("prob_smoothing", 1)))

    # --- fuzzificación ---
    def _memberships(self, indicators: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        """Grados de pertenencia de cada variable a cada conjunto difuso."""
        cols = {"trend": "trend_z", "vol": "vol_z", "drawdown": "drawdown", "breadth": "breadth"}
        deg: dict[str, dict[str, np.ndarray]] = {}
        for var, col in cols.items():
            x = indicators[col].to_numpy()
            deg[var] = {name: trimf(x, abc) for name, abc in self.mf[var].items()}
        return deg

    def predict_proba(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """Probabilidad de cada fase por fila. Las 4 columnas suman 1.

        Es puramente puntual (cada fila depende solo de sus indicadores), por lo que hereda la
        causalidad del panel de indicadores.
        """
        deg = self._memberships(indicators)
        n = len(indicators)
        scores = {reg: np.zeros(n) for reg in REGIMES}
        for rule in self.rules:
            strength = np.ones(n)
            for var, fuzzy_set in rule.antecedent.items():
                strength = np.minimum(strength, deg[var][fuzzy_set])
            scores[rule.consequent] = np.maximum(scores[rule.consequent], strength)

        mat = np.vstack([scores[reg] for reg in REGIMES]).T  # (n, 4)
        row_sum = mat.sum(axis=1, keepdims=True)
        # Si ninguna regla dispara, repartir uniformemente (máxima incertidumbre).
        uniform = np.full_like(mat, 1.0 / len(REGIMES))
        proba = np.where(row_sum > 1e-12, mat / np.where(row_sum == 0, 1, row_sum), uniform)

        out = pd.DataFrame(proba, index=indicators.index, columns=REGIMES)
        if self.smoothing and self.smoothing > 1:
            out = out.rolling(self.smoothing, min_periods=1).mean()
            out = out.div(out.sum(axis=1), axis=0)  # re-normaliza tras suavizar
        return out

    def dominant(self, proba: pd.DataFrame) -> pd.Series:
        """Fase dominante (argmax) por fila."""
        return proba[REGIMES].idxmax(axis=1).rename("regime")

    def membership_curves(self, var: str, n: int = 400) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Curvas (x, grado) de cada conjunto de una variable, para graficar."""
        sets = self.mf[var]
        all_pts = [p for abc in sets.values() for p in abc]
        lo, hi = min(all_pts), max(all_pts)
        pad = 0.1 * (hi - lo + 1e-9)
        x = np.linspace(lo - pad, hi + pad, n)
        return {name: (x, trimf(x, abc)) for name, abc in sets.items()}

    def rules_text(self) -> list[str]:
        """Reglas en lenguaje natural (para el README y la documentación)."""
        return [r.text for r in self.rules]
