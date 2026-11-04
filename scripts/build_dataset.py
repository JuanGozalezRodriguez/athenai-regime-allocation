"""Construye los datasets procesados a partir de los datos crudos del concurso.

Uso:
    python scripts/build_dataset.py [--raw RUTA] [--config configs/default.yaml]

Si no se dispone de los datos del organizador, use ``scripts/make_synthetic.py`` para
generar un dataset sintético con el mismo formato en ``data/raw``.
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401  (añade src al path)
from athenai.config import load_config, set_seed, setup_logging
from athenai.data import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Ruta al YAML de configuración.")
    parser.add_argument("--raw", default=None,
                        help="Ruta a los datos crudos (sobrescribe el config).")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    if args.raw is not None:
        # Sobrescribe la ruta de datos crudos en caliente (sin tocar el YAML).
        cfg.raw["paths"]["raw"] = args.raw

    build.build_all(cfg)


if __name__ == "__main__":
    main()
