"""Tests del Bloque 1: pertenencias difusas y probabilidades de fase."""

from __future__ import annotations

import numpy as np

from athenai.regimes.fuzzy import REGIMES, FuzzyRegimeModel, trimf
from athenai.regimes.indicators import compute_indicators


def test_trimf_peak_and_bounds():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    y = trimf(x, [0.0, 1.0, 2.0])
    assert y[0] == 0.0  # fuera por la izquierda
    assert y[2] == 1.0  # en el pico
    assert y[3] == 0.0  # en el borde derecho
    assert np.all((y >= 0) & (y <= 1))


def test_proba_sums_to_one(cfg, processed):
    ind = compute_indicators(processed["market"], cfg)
    fuzzy = FuzzyRegimeModel.from_config(cfg)
    proba = fuzzy.predict_proba(ind)
    sums = proba[REGIMES].sum(axis=1)
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)
    assert (proba.to_numpy() >= -1e-12).all()


def test_dominant_is_argmax(cfg, processed):
    ind = compute_indicators(processed["market"], cfg)
    fuzzy = FuzzyRegimeModel.from_config(cfg)
    proba = fuzzy.predict_proba(ind)
    dom = fuzzy.dominant(proba)
    assert set(dom.unique()).issubset(set(REGIMES))
