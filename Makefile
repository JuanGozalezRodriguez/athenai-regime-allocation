# Makefile del proyecto AthenAI Regime Allocation.
# En Windows puede usarse con `make` de Git Bash / MSYS, o ejecutar los comandos python directamente.

PYTHON ?= python
RAW ?= data/raw

.PHONY: help setup synthetic data-check build pipeline figures readme all test lint clean

help:
	@echo "Targets disponibles:"
	@echo "  setup      Instala el paquete y las dependencias (modo editable)"
	@echo "  synthetic  Genera datos sinteticos en data/raw (sin datos del organizador)"
	@echo "  build      Construye los parquets procesados desde RAW (por defecto data/raw)"
	@echo "  pipeline   Ejecuta regimenes + perfilado + asignacion + figuras + cifras"
	@echo "  readme     Inyecta las cifras del pipeline en el README"
	@echo "  all        build + pipeline + quantum + readme"
	@echo "  quantum    Ejecuta el experimento cuantico (QAOA vs clasico)"
	@echo "  figures    Regenera solo las figuras"
	@echo "  test       Ejecuta la bateria de tests (pytest)"
	@echo "  lint       Ejecuta ruff"
	@echo "  data-check Comprueba que los parquets procesados existen"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

synthetic:
	$(PYTHON) scripts/make_synthetic.py --out $(RAW)

build:
	$(PYTHON) scripts/build_dataset.py --raw $(RAW)

data-check:
	$(PYTHON) -c "from pathlib import Path; import sys; p=Path('data/processed/market.parquet'); sys.exit(0 if p.exists() else 'Faltan parquets: ejecuta make build')"

pipeline:
	$(PYTHON) scripts/run_pipeline.py

quantum:
	$(PYTHON) scripts/run_quantum.py

figures:
	$(PYTHON) scripts/run_pipeline.py

readme:
	$(PYTHON) scripts/update_readme.py

all: build pipeline quantum readme
	@echo "Pipeline completo ejecutado."

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests

clean:
	rm -rf data/processed/*.parquet data/processed/*.json data/processed/*.csv
