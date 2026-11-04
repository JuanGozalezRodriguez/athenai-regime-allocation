# Datos

Los datos del concurso de **AthenAI** **no se incluyen** en este repositorio (no está claro si
son redistribuibles) y están excluidos en `.gitignore`. Aquí se explica cómo colocarlos.

## Opción A — Datos reales del concurso

Coloca los ficheros del organizador en `data/raw/` con esta estructura:

```
data/raw/
├── algoritmos/
│   ├── 00eyz.csv          # datetime,open,high,low,close  (velas de 4h)
│   ├── 00kZg.csv
│   └── ...                # ~14.761 ficheros
├── trades_benchmark.csv
├── benchmark_monthly_returns.csv
└── benchmark_yearly_returns.csv
```

Luego:

```bash
python scripts/build_dataset.py          # usa data/raw por defecto
# o, sin copiar los datos, apuntando a otra carpeta:
python scripts/build_dataset.py --raw "/ruta/a/los/datos"
```

## Opción B — Datos sintéticos (sin los datos del organizador)

Para probar el pipeline, los tests y la CI sin los datos reales, se incluye un **generador
sintético** con el mismo formato (regímenes de mercado, tramos planos, DST, benchmark):

```bash
python scripts/make_synthetic.py --out data/raw
python scripts/build_dataset.py
```

## Salida

El build escribe parquets cacheados en `data/processed/` (tampoco versionados):
`market.parquet`, `returns_panel.parquet`, `close_panel.parquet`, `active_panel.parquet`,
`benchmark.parquet`, `trades_clean.parquet` y `dataset_stats.json`.
