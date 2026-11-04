"""Ejecuta el pipeline completo: regímenes → perfilado → asignación → figuras → cifras.

Produce las figuras del README en ``docs/img`` y el resumen de cifras en
``data/processed/results_summary.json`` (fuente única de todos los números del README).

Uso:
    python scripts/run_pipeline.py [--config configs/default.yaml] [--no-figures]
"""

from __future__ import annotations

import argparse
import json
import logging

import _bootstrap  # noqa: F401
from athenai.config import load_config, set_seed, setup_logging
from athenai.data.build import load_processed
from athenai.pipeline import (
    collect_readme_numbers,
    run_allocation,
    run_profiling,
    run_regimes,
    write_results_summary,
)
from athenai.viz.figures import make_core_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    log = setup_logging(logging.INFO)
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    processed = load_processed(cfg)
    stats_path = cfg.path("processed") / "dataset_stats.json"
    dataset_stats = (json.loads(stats_path.read_text(encoding="utf-8"))
                     if stats_path.exists() else {})

    log.info("Bloque 1: detección difusa del régimen...")
    reg = run_regimes(cfg, processed)
    log.info("Bloque 2: perfilado de algoritmos...")
    prof = run_profiling(cfg, processed, reg["dominant"])
    log.info("Bloque 3: asignación con RL...")
    alloc = run_allocation(cfg, processed, reg["dominant"])

    if not args.no_figures:
        log.info("Generando figuras...")
        paths = make_core_figures(cfg, reg, prof, alloc, processed)
        log.info("  %d figuras en %s", len(paths), cfg.path("figures"))

    numbers = collect_readme_numbers(cfg, reg, alloc, processed, dataset_stats)
    out = cfg.path("processed") / "results_summary.json"
    write_results_summary(numbers, out)
    log.info("Resumen de cifras escrito en %s", out)

    # Guardar fichas de perfilado y tablas de resultados como CSV para consulta.
    prof["fichas"].to_csv(cfg.path("processed") / "fichas_algoritmos.csv")
    alloc["results"]["test"].to_csv(cfg.path("processed") / "resultados_test.csv")
    log.info("RL bate al benchmark en test: %s", numbers["rl_beats_benchmark_test"])


if __name__ == "__main__":
    main()
