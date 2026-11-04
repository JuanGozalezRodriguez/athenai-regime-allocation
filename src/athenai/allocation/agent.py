"""Agente de aprendizaje por refuerzo tabular (Q-learning) sobre la fase de mercado.

Es deliberadamente simple y explicable: una tabla Q de tamaño ``(4 fases × 4 carteras)``.
La política resultante es una regla legible "fase dominante → cartera elegida", que se puede
auditar de un vistazo y comparar con la regla difusa sin RL.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..regimes.fuzzy import REGIMES
from .env import RegimeAllocationEnv

logger = logging.getLogger("athenai")


class QLearningAgent:
    """Q-learning tabular con exploración epsilon-greedy."""

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.2,
                 gamma: float = 0.9, epsilon: float = 0.1, seed: int = 42) -> None:
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def act(self, state: int, greedy: bool = False) -> int:
        """Elige acción epsilon-greedy (o greedy en evaluación)."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))  # desempate aleatorio estable

    def update(self, s: int, a: int, r: float, s_next: int | None) -> None:
        """Actualización de Q-learning."""
        target = r if s_next is None else r + self.gamma * self.q[s_next].max()
        self.q[s, a] += self.alpha * (target - self.q[s, a])


def train_agent(env: RegimeAllocationEnv, episodes: int = 400, seed: int = 42,
                **agent_kwargs) -> QLearningAgent:
    """Entrena un agente recorriendo repetidamente el episodio de entrenamiento."""
    agent = QLearningAgent(env.n_states, env.n_actions, seed=seed, **agent_kwargs)
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state if next_state is not None else state
    logger.info("Agente entrenado (%d episodios). Q=\n%s", episodes,
                np.array2string(agent.q, precision=4))
    return agent


def derive_policy(agent: QLearningAgent) -> pd.Series:
    """Política greedy: fase dominante → índice de cartera elegida."""
    actions = agent.q.argmax(axis=1)
    return pd.Series(actions, index=REGIMES, name="cartera")


def weights_from_policy(policy: pd.Series, monthly_regime: pd.Series,
                        portfolios: pd.DataFrame) -> pd.DataFrame:
    """Construye la serie temporal de pesos aplicando la política a cada mes.

    Args:
        policy: fase → índice de cartera.
        monthly_regime: fase dominante (causal) en cada fecha de rebalanceo.
        portfolios: pesos de las 4 carteras.

    Returns:
        DataFrame (fechas × algoritmos) con los pesos elegidos cada mes.
    """
    rows = {}
    for date, regime in monthly_regime.items():
        action = int(policy.loc[regime])
        rows[date] = portfolios.iloc[action]
    out = pd.DataFrame(rows).T
    out.index.name = "date"
    return out
