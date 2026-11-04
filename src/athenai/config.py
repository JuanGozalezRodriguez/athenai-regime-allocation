"""Carga de configuración y utilidades de rutas/semilla.

La configuración vive en ``configs/default.yaml``. El acceso se hace mediante un
objeto :class:`Config` que expone los apartados como atributos de solo lectura y
resuelve las rutas relativas respecto a la raíz del repositorio.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("athenai")


def repo_root() -> Path:
    """Devuelve la raíz del repositorio (dos niveles sobre este fichero)."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """Configuración del pipeline, cargada de un YAML.

    Attributes:
        raw: diccionario con todo el contenido del YAML.
        root: raíz del repositorio para resolver rutas relativas.
    """

    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # --- rutas resueltas ---
    def path(self, kind: str) -> Path:
        """Resuelve una ruta declarada en el apartado ``paths`` del YAML."""
        return (self.root / self.raw["paths"][kind]).resolve()

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Carga la configuración desde un YAML (por defecto ``configs/default.yaml``)."""
    root = repo_root()
    cfg_path = Path(path) if path is not None else root / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=root)


def set_seed(seed: int) -> None:
    """Fija las semillas de ``random`` y ``numpy`` para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura el logger ``athenai`` (idempotente)."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
