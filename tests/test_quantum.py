"""Tests del Bloque 4 (cuántico): QUBO clásico siempre; QAOA solo si Qiskit está."""

from __future__ import annotations

import numpy as np
import pytest

from athenai.quantum import qaoa


def _toy_problem():
    rng = np.random.default_rng(0)
    mu = np.array([0.10, 0.08, 0.02, 0.05, 0.01, 0.09])
    cov = rng.normal(0, 0.01, (6, 6))
    cov = cov @ cov.T  # semidefinida positiva
    return mu, cov


def test_classical_respects_cardinality():
    mu, cov = _toy_problem()
    q = qaoa.build_qubo(mu, cov, k=3, risk_lambda=0.5)
    sol = qaoa.solve_classical_exact(q)
    # La penalización de cardinalidad debe llevar a seleccionar ~k activos.
    assert len(sol["selected"]) == 3


def test_qubo_value_matches_bruteforce_min():
    mu, cov = _toy_problem()
    q = qaoa.build_qubo(mu, cov, k=2, risk_lambda=0.5)
    sol = qaoa.solve_classical_exact(q)
    # Ningún vector binario debe tener coste menor que el óptimo encontrado.
    from itertools import product
    best = min(qaoa.qubo_value(q, np.array(b, float)) for b in product([0, 1], repeat=len(mu)))
    assert abs(best - sol["value"]) < 1e-9


@pytest.mark.skipif(not qaoa.qiskit_available(), reason="Qiskit no instalado")
def test_qaoa_runs_and_selects():
    mu, cov = _toy_problem()
    q = qaoa.build_qubo(mu, cov, k=3, risk_lambda=0.5)
    res = qaoa.solve_qaoa(q, reps=1, seed=0, maxiter=20)
    assert res["n_qubits"] == len(mu)
    assert 1 <= len(res["selected"]) <= len(mu)
