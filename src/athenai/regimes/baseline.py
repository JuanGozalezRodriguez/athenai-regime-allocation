"""Métodos alternativos **interpretables** de detección de régimen, para comparar con el difuso.

    - ``ThresholdRegime``: clasificación por umbrales sobre los mismos indicadores (totalmente
      causal y trivialmente auditable). Produce etiquetas duras.
    - ``KMeansRegime``: agrupamiento no supervisado de los indicadores en 4 clústeres,
      etiquetados a posteriori por su tendencia/volatilidad media. Se ajusta solo con datos de
      entrenamiento para no introducir *lookahead*.

Sirven de contraste honesto: mismas entradas, lógica distinta.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

from .fuzzy import REGIMES


class ThresholdRegime:
    """Clasificador por umbrales sobre indicadores normalizados (causal)."""

    def __init__(self, trend_hi: float = 0.5, trend_lo: float = -0.5,
                 vol_hi: float = 1.0, dd_hi: float = 0.1) -> None:
        self.trend_hi = trend_hi
        self.trend_lo = trend_lo
        self.vol_hi = vol_hi
        self.dd_hi = dd_hi

    def predict(self, indicators: pd.DataFrame) -> pd.Series:
        """Etiqueta de fase por fila según reglas duras."""
        t = indicators["trend_z"]
        v = indicators["vol_z"]
        dd = indicators["drawdown"]
        out = pd.Series("lateral", index=indicators.index, dtype=object)
        out[(v > self.vol_hi) & (dd > self.dd_hi)] = "crisis"
        out[(t > self.trend_hi) & (v <= self.vol_hi)] = "alcista"
        out[(t < self.trend_lo) & (out != "crisis")] = "bajista"
        return out.rename("regime")


class KMeansRegime:
    """Agrupamiento de indicadores en 4 clústeres etiquetados por tendencia/volatilidad."""

    FEATURES = ["trend_z", "vol_z", "drawdown", "breadth"]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.km: KMeans | None = None
        self.labels_: dict[int, str] = {}

    def fit(self, indicators: pd.DataFrame) -> KMeansRegime:
        x = indicators[self.FEATURES].to_numpy()
        self.km = KMeans(n_clusters=4, random_state=self.seed, n_init=10).fit(x)
        centers = pd.DataFrame(self.km.cluster_centers_, columns=self.FEATURES)
        # Etiquetado heurístico de cada clúster por su perfil medio.
        self.labels_ = {i: self._label_center(row) for i, row in centers.iterrows()}
        return self

    @staticmethod
    def _label_center(c: pd.Series) -> str:
        if c["vol_z"] > 0.8 and c["drawdown"] > 0.08:
            return "crisis"
        if c["trend_z"] > 0.4:
            return "alcista"
        if c["trend_z"] < -0.4:
            return "bajista"
        return "lateral"

    def predict(self, indicators: pd.DataFrame) -> pd.Series:
        if self.km is None:
            raise RuntimeError("Llama a fit() antes de predict().")
        idx = self.km.predict(indicators[self.FEATURES].to_numpy())
        return pd.Series([self.labels_[i] for i in idx], index=indicators.index,
                         name="regime")


def agreement(a: pd.Series, b: pd.Series) -> float:
    """Fracción de instantes en que dos clasificadores de fase coinciden."""
    common = a.dropna().index.intersection(b.dropna().index)
    return float((a.loc[common] == b.loc[common]).mean())


def confusion(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Matriz de confusión entre dos etiquetadores (filas=a, columnas=b)."""
    common = a.dropna().index.intersection(b.dropna().index)
    return (
        pd.crosstab(a.loc[common], b.loc[common])
        .reindex(index=REGIMES, columns=REGIMES, fill_value=0)
    )
