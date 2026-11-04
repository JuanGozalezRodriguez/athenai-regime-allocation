"""Genera un dataset sintético con el formato del concurso en ``data/raw``.

Permite ejecutar todo el pipeline, los tests y la CI sin los datos del organizador.

Uso:
    python scripts/make_synthetic.py [--out data/raw] [--n-algos 60] [--n-invested 20]
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401
from athenai.config import load_config, setup_logging
from athenai.data.synthetic import generate_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="Directorio de salida (por defecto data/raw).")
    parser.add_argument("--n-algos", type=int, default=60)
    parser.add_argument("--n-invested", type=int, default=20)
    args = parser.parse_args()

    log = setup_logging(logging.INFO)
    cfg = load_config(args.config)
    out = args.out if args.out is not None else str(cfg.path("raw"))

    summary = generate_synthetic(
        out,
        n_algos=args.n_algos,
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        n_invested=args.n_invested,
        seed=cfg.seed,
    )
    log.info("Dataset sintético generado en %s: %s", out, summary)


if __name__ == "__main__":
    main()
