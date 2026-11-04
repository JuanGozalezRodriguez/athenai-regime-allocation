"""Selección de subcartera como problema QUBO, resuelto de forma clásica y con QAOA.

**Explicación para quien no sabe computación cuántica.** Elegir qué ``k`` algoritmos incluir en
una cartera, maximizando rentabilidad esperada y minimizando riesgo/correlación, es un problema
combinatorio: hay que probar muchas combinaciones de "dentro/fuera". Se formula como un QUBO
(*Quadratic Unconstrained Binary Optimization*): una variable binaria por algoritmo. QAOA es un
algoritmo cuántico que prepara un estado cuyas mediciones tienden a dar las mejores combinaciones.

Aquí lo resolvemos de dos formas y las comparamos:
    1. **Clásica exacta**: se prueban todas las combinaciones (viable con pocos algoritmos).
    2. **QAOA** (Qiskit, simulador por vector de estado): se compara su mejor solución con la
       óptima exacta. Es un experimento ilustrativo, sin pretensión de ventaja cuántica.

Si Qiskit no está instalado, la parte QAOA se omite con un aviso; la clásica siempre funciona.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product

import numpy as np

logger = logging.getLogger("athenai")


@dataclass
class QUBO:
    """Matriz QUBO simétrica tal que el coste es ``x^T Q x + const`` (a minimizar)."""

    Q: np.ndarray
    const: float
    k: int


def build_qubo(mu: np.ndarray, cov: np.ndarray, k: int, risk_lambda: float = 0.5,
               penalty: float | None = None) -> QUBO:
    """Construye el QUBO de selección de ``k`` activos.

    Minimiza ``-rentabilidad + lambda·riesgo + penalización_cardinalidad``:
        ``f(x) = -Σ mu_i x_i + lambda Σ cov_ij x_i x_j + P (Σ x_i - k)^2``.
    """
    n = len(mu)
    if penalty is None:
        penalty = float(abs(mu).sum() + risk_lambda * abs(cov).sum() + 1.0)
    Q = risk_lambda * cov.astype(float).copy()
    # Término de cardinalidad P(Σx - k)^2 → cuadrático fuera de diagonal 2P, diagonal P(1-2k).
    Q += penalty * (np.ones((n, n)) - np.eye(n))  # 2P en cada par (i,j) i≠j al simetrizar
    diag = -mu + penalty * (1.0 - 2.0 * k)
    np.fill_diagonal(Q, Q.diagonal() + diag)
    const = penalty * (k ** 2)
    return QUBO(Q=Q, const=const, k=k)


def qubo_value(qubo: QUBO, x: np.ndarray) -> float:
    """Evalúa el coste del QUBO para un vector binario ``x``."""
    return float(x @ qubo.Q @ x + qubo.const)


def solve_classical_exact(qubo: QUBO) -> dict:
    """Resuelve el QUBO por fuerza bruta (óptimo exacto). Viable con pocos activos."""
    n = qubo.Q.shape[0]
    best_x, best_val = None, np.inf
    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        val = qubo_value(qubo, x)
        if val < best_val:
            best_val, best_x = val, x
    return {"x": best_x, "value": best_val, "selected": list(np.flatnonzero(best_x))}


def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        return True
    except Exception:
        return False


def _qubo_to_ising(qubo: QUBO):
    """Convierte el QUBO a un hamiltoniano de Ising (SparsePauliOp) para QAOA."""
    from qiskit.quantum_info import SparsePauliOp

    Q = qubo.Q
    n = Q.shape[0]
    # x_i = (1 - z_i)/2, con z_i ∈ {+1,-1}. Expandimos x^T Q x en términos de Z.
    pauli_terms: dict[str, float] = {}

    def add(label: str, coeff: float) -> None:
        pauli_terms[label] = pauli_terms.get(label, 0.0) + coeff

    ident = "I" * n
    for i in range(n):
        for j in range(n):
            q = Q[i, j]
            if q == 0:
                continue
            # (1-z_i)/2 * (1-z_j)/2 = 1/4 (1 - z_i - z_j + z_i z_j)
            add(ident, q / 4.0)
            zi = list(ident)
            zi[i] = "Z"
            add("".join(zi), -q / 4.0)
            zj = list(ident)
            zj[j] = "Z"
            add("".join(zj), -q / 4.0)
            zz = list(ident)
            zz[i] = "Z"
            zz[j] = "Z"
            add("".join(zz), q / 4.0)
    labels = [(lbl, c) for lbl, c in pauli_terms.items() if abs(c) > 1e-9]
    return SparsePauliOp.from_list(labels)


def solve_qaoa(qubo: QUBO, reps: int = 2, seed: int = 42, maxiter: int = 100) -> dict:
    """Resuelve el QUBO con QAOA sobre el simulador de vector de estado de Qiskit.

    Devuelve la mejor combinación muestreada y su coste, para comparar con la exacta.
    """
    if not qiskit_available():
        raise RuntimeError("Qiskit no está instalado; instala el extra 'quantum'.")
    import numpy as _np
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.primitives import Sampler
    from scipy.optimize import minimize

    hamiltonian = _qubo_to_ising(qubo)
    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)
    ansatz = ansatz.measure_all(inplace=False)
    sampler = Sampler(options={"seed": seed})
    n = qubo.Q.shape[0]

    def expected_cost(params) -> float:
        job = sampler.run(ansatz, parameter_values=[params])
        quasi = job.result().quasi_dists[0]
        total = 0.0
        for state, prob in quasi.items():
            bits = _np.array([(state >> b) & 1 for b in range(n)], dtype=float)
            total += prob * qubo_value(qubo, bits)
        return total

    rng = _np.random.default_rng(seed)
    x0 = rng.uniform(0, _np.pi, ansatz.num_parameters)
    res = minimize(expected_cost, x0, method="COBYLA", options={"maxiter": maxiter})

    job = sampler.run(ansatz, parameter_values=[res.x])
    quasi = job.result().quasi_dists[0]
    # Post-proceso estándar de QUBO con restricción: entre los estados más muestreados,
    # se prefieren los FACTIBLES (cardinalidad = k) y, de esos, el de menor coste. Si ninguno
    # es factible, se toma el de menor coste global.
    top = sorted(quasi.items(), key=lambda kv: kv[1], reverse=True)[:32]
    cand = [_np.array([(s >> b) & 1 for b in range(n)], dtype=float) for s, _ in top]
    feasible = [x for x in cand if int(x.sum()) == qubo.k]
    pool = feasible if feasible else cand
    best_x = min(pool, key=lambda x: qubo_value(qubo, x))
    return {
        "x": best_x,
        "value": qubo_value(qubo, best_x),
        "selected": list(_np.flatnonzero(best_x)),
        "n_qubits": n,
        "reps": reps,
    }
