"""Entorno de asignación de capital condicionado al régimen de mercado.

Para que la política sea **interpretable**, la acción no es un vector continuo de pesos sino la
elección de una de 4 carteras predefinidas (una por fase de mercado). Cada cartera es
equiponderada sobre los ``top_k`` algoritmos con mejor rendimiento en esa fase **durante el
periodo de entrenamiento** (sin usar información futura).

Estado  : fase de mercado dominante en el momento de decidir (0..3).
Acción  : índice de la cartera elegida (0..3).
Recompensa: retorno neto de costes del mes siguiente, con utilidad media-varianza
           ``r - 0.5 * lambda * r^2`` (penaliza el riesgo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..regimes.fuzzy import REGIMES


def build_regime_portfolios(
    train_returns: pd.DataFrame, train_dominant: pd.Series, top_k: int = 15
) -> pd.DataFrame:
    """Construye las 4 carteras (una por fase) con datos SOLO de entrenamiento.

    Para cada fase, selecciona los ``top_k`` algoritmos con mayor retorno medio cuando esa fase
    estuvo activa y los equipondera.

    Returns:
        DataFrame (4 fases × algoritmos) con los pesos de cada cartera (filas suman 1).
    """
    dom = train_dominant.reindex(train_returns.index).ffill()
    weights = pd.DataFrame(0.0, index=REGIMES, columns=train_returns.columns)
    for regime in REGIMES:
        mask = dom == regime
        sub = train_returns.loc[mask] if mask.sum() >= 5 else train_returns
        # Selección por rentabilidad ajustada al riesgo (media / desviación), más estable
        # que la media bruta y menos propensa a elegir algoritmos de alta varianza.
        mean = sub.mean()
        std = sub.std().replace(0.0, np.nan)
        score = (mean / std).fillna(mean)
        top = score.sort_values(ascending=False).head(top_k).index
        weights.loc[regime, top] = 1.0 / len(top)
    return weights


class RegimeAllocationEnv:
    """Entorno episódico determinista sobre la secuencia mensual de fases y retornos."""

    def __init__(
        self,
        regime_idx: np.ndarray,
        asset_returns: pd.DataFrame,
        portfolios: pd.DataFrame,
        cost_bps: float = 10.0,
        risk_aversion: float = 0.5,
    ) -> None:
        if len(regime_idx) != len(asset_returns):
            raise ValueError("regime_idx y asset_returns deben tener la misma longitud.")
        self.regime_idx = np.asarray(regime_idx, dtype=int)
        self.asset_returns = asset_returns.reindex(columns=portfolios.columns).fillna(0.0)
        self.portfolios = portfolios.to_numpy()
        self.cost_rate = cost_bps / 1e4
        self.risk_aversion = risk_aversion
        self.n = len(asset_returns)
        self.n_states = len(REGIMES)
        self.n_actions = portfolios.shape[0]
        self._t = 0
        self._prev_w = np.zeros(portfolios.shape[1])

    def reset(self) -> int:
        self._t = 0
        self._prev_w = np.zeros(self.portfolios.shape[1])
        return int(self.regime_idx[0])

    def step(self, action: int) -> tuple[int | None, float, bool, dict]:
        """Aplica la cartera ``action`` durante el mes actual y avanza."""
        w = self.portfolios[action]
        r = float(self.asset_returns.iloc[self._t].to_numpy() @ w)
        l1_change = float(np.abs(w - self._prev_w).sum())
        cost = self.cost_rate * l1_change
        net = r - cost
        reward = net - 0.5 * self.risk_aversion * r * r
        self._prev_w = w
        info = {"gross": r, "cost": cost, "net": net, "turnover": 0.5 * l1_change}
        self._t += 1
        done = self._t >= self.n
        next_state = None if done else int(self.regime_idx[self._t])
        return next_state, reward, done, info
