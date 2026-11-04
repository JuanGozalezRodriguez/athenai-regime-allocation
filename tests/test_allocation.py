"""Tests del Bloque 3: carteras por fase, entorno y agente de RL."""

from __future__ import annotations

import numpy as np

from athenai.allocation.agent import QLearningAgent, derive_policy, train_agent
from athenai.allocation.env import RegimeAllocationEnv, build_regime_portfolios
from athenai.regimes.fuzzy import REGIMES, FuzzyRegimeModel
from athenai.regimes.indicators import compute_indicators


def _dominant(cfg, processed):
    ind = compute_indicators(processed["market"], cfg)
    fuzzy = FuzzyRegimeModel.from_config(cfg)
    return fuzzy.dominant(fuzzy.predict_proba(ind))


def test_portfolios_sum_to_one(cfg, processed):
    dom = _dominant(cfg, processed)
    pf = build_regime_portfolios(processed["returns_panel"], dom, top_k=5)
    assert list(pf.index) == REGIMES
    # Cada cartera reparte el 100% (filas suman 1).
    np.testing.assert_allclose(pf.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_env_step_and_agent(cfg, processed):
    dom = _dominant(cfg, processed)
    from athenai.evaluation.backtest import to_monthly_returns
    monthly = to_monthly_returns(processed["returns_panel"])
    regime = dom.resample("M").last().reindex(monthly.index).ffill()
    idx = regime.map({r: i for i, r in enumerate(REGIMES)}).to_numpy().astype(int)
    pf = build_regime_portfolios(processed["returns_panel"], dom, top_k=5)

    env = RegimeAllocationEnv(idx[:-1], monthly.iloc[:-1], pf, cost_bps=10, risk_aversion=0.5)
    agent = train_agent(env, episodes=20, seed=cfg.seed)
    policy = derive_policy(agent)
    assert set(policy.index) == set(REGIMES)
    assert policy.between(0, 3).all()


def test_qlearning_updates_q():
    agent = QLearningAgent(4, 4, alpha=0.5, gamma=0.9, seed=0)
    before = agent.q.copy()
    agent.update(0, 1, 1.0, None)
    assert agent.q[0, 1] > before[0, 1]  # recompensa positiva sube el valor
