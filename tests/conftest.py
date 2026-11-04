"""Fixtures de test: generan un dataset sintético y construyen los parquets en un tmp.

Permite que toda la batería (y la CI) corra sin los datos del organizador.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from athenai.config import load_config  # noqa: E402
from athenai.data import build  # noqa: E402
from athenai.data.synthetic import generate_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """Config de test con rutas redirigidas a un directorio temporal."""
    base = tmp_path_factory.mktemp("athenai_data")
    c = load_config()
    c.raw["paths"]["raw"] = str(base / "raw")
    c.raw["paths"]["processed"] = str(base / "processed")
    # Universo pequeño y rápido para los tests.
    generate_synthetic(c.path("raw"), n_algos=40, start=c["data"]["start"],
                       end=c["data"]["end"], n_invested=12, seed=c.seed)
    return c


@pytest.fixture(scope="session")
def processed(cfg):
    """Parquets procesados a partir del dataset sintético."""
    build.build_all(cfg)
    return build.load_processed(cfg)
