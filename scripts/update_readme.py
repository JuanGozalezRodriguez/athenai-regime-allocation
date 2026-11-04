"""Reescribe las cifras del README entre marcadores a partir de ``results_summary.json``.

Ningún número del README se escribe a mano: este script sustituye el contenido entre
``<!--AUTO:clave-->`` y ``<!--/AUTO:clave-->`` por el valor calculado por el pipeline.

Uso:
    python scripts/update_readme.py
"""

from __future__ import annotations

import json
import logging
import re

import _bootstrap  # noqa: F401
from athenai.config import load_config, setup_logging

log = setup_logging(logging.INFO)


def _bullet_distribution(dist: dict) -> str:
    order = ["alcista", "bajista", "lateral", "crisis"]
    labels = {"alcista": "Alcista", "bajista": "Bajista", "lateral": "Lateral", "crisis": "Crisis"}
    return " · ".join(f"{labels[k]} {dist.get(k, 0)}%" for k in order if k in dist)


def _yearly_table(yearly: dict) -> str:
    years = sorted({y for d in yearly.values() for y in d})
    labels = {"rl": "Estrategia RL", "benchmark": "Benchmark", "equiponderado": "Equiponderado"}
    header = "| Estrategia | " + " | ".join(str(y) for y in years) + " |"
    sep = "|" + "---|" * (len(years) + 1)
    lines = [header, sep]
    for key in ["rl", "benchmark", "equiponderado"]:
        cells = [f"{yearly[key].get(y, '—')}%" if y in yearly[key] else "—" for y in years]
        lines.append(f"| {labels[key]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_values(n: dict) -> dict[str, str]:
    """Construye el texto de cada marcador a partir del resumen de cifras."""
    ds = n["dataset"]
    pol = n["policy"]
    verdict = ("**sí**" if n["rl_beats_benchmark_test"]
               else "**no** (resultado honesto, analizado en Limitaciones)")
    return {
        "universe": f"{ds.get('n_universe', '—'):,}".replace(",", "."),
        "invested": str(ds.get("n_invested", "—")),
        "flat": f"{ds.get('pct_fully_flat', '—')}",
        "out_of_market": f"{ds.get('mean_out_of_market_pct', '—')}",
        "period": f"{ds.get('date_start', '')} → {ds.get('date_end', '')}",
        "market_return": f"{n['market_total_return_pct']}",
        "regime_dist": _bullet_distribution(n["regime_distribution_pct"]),
        "n_rules": str(n["n_rules"]),
        "agreement_threshold": f"{n['agreement_threshold_pct']}",
        "test_table": n["test_table_md"],
        "val_table": n["val_table_md"],
        "yearly_table": _yearly_table(n["yearly_returns_pct"]),
        "policy": " · ".join(f"{k.capitalize()}→{v}" for k, v in pol.items()),
        "rl_verdict": verdict,
        "test_period": n["splits"]["val_end"][:4] + "→" + n["splits"]["test_end"][:4],
    }


def apply_markers(text: str, values: dict[str, str]) -> tuple[str, int]:
    """Sustituye cada bloque ``<!--AUTO:k-->...<!--/AUTO:k-->`` por ``values[k]``."""
    count = 0
    for key, val in values.items():
        pattern = re.compile(
            rf"(<!--AUTO:{re.escape(key)}-->)(.*?)(<!--/AUTO:{re.escape(key)}-->)",
            re.DOTALL,
        )
        new_text, k = pattern.subn(rf"\g<1>{val}\g<3>", text)
        if k:
            text = new_text
            count += k
        else:
            log.warning("Marcador no encontrado en el README: %s", key)
    return text, count


def _quantum_summary(cfg) -> str:
    """Texto del experimento cuántico a partir de ``quantum_results.json`` (si existe)."""
    path = cfg.path("processed") / "quantum_results.json"
    if not path.exists():
        return "Experimento no ejecutado (ejecuta `python scripts/run_quantum.py`)."
    q = json.loads(path.read_text(encoding="utf-8"))
    cls = ", ".join(q["classical_selected"])
    if q.get("qaoa_selected") is None:
        return (f"Solución clásica exacta: selecciona **{cls}**. "
                "(QAOA omitido: Qiskit no instalado.)")
    overlap = len(set(q["classical_selected"]) & set(q["qaoa_selected"]))
    match = "coinciden exactamente" if q["qaoa_matches_exact"] else (
        f"QAOA recupera {overlap} de los {len(q['classical_selected'])} activos óptimos")
    return (f"Sobre {len(q['candidates'])} algoritmos candidatos, la solución clásica exacta "
            f"selecciona **{cls}**; {match}. Experimento en simulador, sin ventaja cuántica.")


def main() -> None:
    cfg = load_config()
    summary_path = cfg.path("processed") / "results_summary.json"
    numbers = json.loads(summary_path.read_text(encoding="utf-8"))
    values = build_values(numbers)
    values["quantum_summary"] = _quantum_summary(cfg)

    readme = cfg.root / "README.md"
    text = readme.read_text(encoding="utf-8")
    new_text, count = apply_markers(text, values)
    readme.write_text(new_text, encoding="utf-8")
    log.info("README actualizado: %d marcadores sustituidos.", count)


if __name__ == "__main__":
    main()
