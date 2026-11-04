"""Tests de las métricas financieras sobre casos conocidos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from athenai.evaluation import metrics


def _series(vals):
    idx = pd.date_range("2020-01-31", periods=len(vals), freq="M")
    return pd.Series(vals, index=idx, dtype=float)


def test_cagr_constant_growth():
    # 12 meses al 1% → (1.01)^12 - 1 anual.
    r = _series([0.01] * 12)
    assert abs(metrics.cagr(r, 12) - ((1.01 ** 12) - 1)) < 1e-9


def test_max_drawdown_monotonic_is_zero():
    r = _series([0.01, 0.02, 0.03])
    assert metrics.max_drawdown(r) == 0.0


def test_max_drawdown_known():
    # +0%, -50%, ... el drawdown mínimo debe ser -0.5.
    r = _series([0.0, -0.5, 0.0])
    assert abs(metrics.max_drawdown(r) - (-0.5)) < 1e-12


def test_sharpe_zero_vol_is_nan():
    r = _series([0.01] * 6)
    assert np.isnan(metrics.sharpe(r, 12))


def test_turnover_full_switch():
    # Pasar de 100% A a 100% B es rotación 1.0; el primer periodo (desde 0) cuenta como 1.0.
    w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]},
                     index=pd.date_range("2020-01-31", periods=2, freq="M"))
    assert abs(metrics.turnover(w) - 1.0) < 1e-12


def test_positive_period_pct():
    r = _series([0.1, -0.1, 0.1, 0.1])
    assert abs(metrics.positive_period_pct(r) - 0.75) < 1e-12
