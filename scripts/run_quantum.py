"""Experimento cuántico (EXPERIMENTAL): selección de subcartera como QUBO.

Elige ``n_assets`` algoritmos candidatos (mejor Sharpe en entrenamiento), plantea el QUBO
rentabilidad-vs-riesgo y lo resuelve de forma clásica exacta y con QAOA (si Qiskit está),
comparando ambas soluciones. Genera ``docs/img/quantum_qaoa.png`` y un JSON con el resultado.

Uso:
    python scripts/run_quantum.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

import _bootstrap  # noqa: F401
from athenai.config import load_config, set_seed, setup_logging
from athenai.data.build import load_processed
from athenai.evaluation import metrics
from athenai.quantum import qaoa
from athenai.viz.figures import fig_quantum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    log = setup_logging(logging.INFO)
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    q = cfg["quantum"]
    if not q.get("enabled", True):
        log.info("Experimento cuántico deshabilitado en la configuración.")
        return

    processed = load_processed(cfg)
    returns = processed["returns_panel"]
    train = returns[returns.index <= cfg["splits"]["train_end"]]

    # Candidatos: mejor Sharpe en entrenamiento, entre los suficientemente activos.
    active_enough = train.columns[(train != 0).mean() > 0.2]
    sharpes = train[active_enough].apply(lambda r: metrics.sharpe(r[r != 0], 252))
    candidates = sharpes.sort_values(ascending=False).head(q["n_assets"]).index.tolist()

    sub = train[candidates]
    mu = (sub.mean() * 252).to_numpy()
    cov = (sub.cov() * 252).to_numpy()
    risk = np.sqrt(np.diag(cov))

    qubo = qaoa.build_qubo(mu, cov, k=q["target_cardinality"], risk_lambda=q["risk_lambda"])
    classical = qaoa.solve_classical_exact(qubo)
    log.info("Clásico exacto: selecciona %s (coste %.4f)",
             [candidates[i] for i in classical["selected"]], classical["value"])

    qaoa_res = None
    if qaoa.qiskit_available():
        qaoa_res = qaoa.solve_qaoa(qubo, reps=q["qaoa_reps"], seed=cfg.seed)
        log.info("QAOA: selecciona %s (coste %.4f)",
                 [candidates[i] for i in qaoa_res["selected"]], qaoa_res["value"])
        match = set(classical["selected"]) == set(qaoa_res["selected"])
        log.info("¿QAOA coincide con la solución exacta? %s", match)
    else:
        log.warning("Qiskit no instalado: se omite QAOA (instala el extra 'quantum').")

    fig_quantum(mu, risk, candidates, classical["selected"],
                qaoa_res["selected"] if qaoa_res else None, cfg.path("figures"))

    out = {
        "candidates": candidates,
        "mu": [round(float(x), 4) for x in mu],
        "risk": [round(float(x), 4) for x in risk],
        "classical_selected": [candidates[i] for i in classical["selected"]],
        "classical_value": round(float(classical["value"]), 4),
        "qaoa_selected": [candidates[i] for i in qaoa_res["selected"]] if qaoa_res else None,
        "qaoa_value": round(float(qaoa_res["value"]), 4) if qaoa_res else None,
        "qaoa_matches_exact": (set(classical["selected"]) == set(qaoa_res["selected"]))
        if qaoa_res else None,
    }
    path = cfg.path("processed") / "quantum_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Resultado cuántico escrito en %s", path)


if __name__ == "__main__":
    main()
