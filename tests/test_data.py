"""Tests de la capa de datos: formato sintético, detección de planos y paneles."""

from __future__ import annotations

import pandas as pd

from athenai.data import clean


def test_synthetic_format(cfg):
    files = list(cfg.path("raw").joinpath("algoritmos").glob("*.csv"))
    assert len(files) == 40
    df = pd.read_csv(files[0])
    assert list(df.columns) == ["datetime", "open", "high", "low", "close"]
    # Debe haber datos de velas de 4h (varias por día).
    assert len(df) > 1000
    assert (cfg.path("raw") / "trades_benchmark.csv").exists()
    assert (cfg.path("raw") / "benchmark_monthly_returns.csv").exists()


def test_flat_mask_detects_constant():
    idx = pd.date_range("2021-01-01", periods=4, freq="4h")
    df = pd.DataFrame({"open": [1, 1, 2, 2], "high": [1, 1, 2, 2],
                       "low": [1, 1, 2, 2], "close": [1, 1, 2, 2]}, index=idx)
    mask = clean.flat_mask(df)
    assert mask.all()  # todas las velas son planas (OHLC idénticos)


def test_processed_panels_shapes(processed):
    r = processed["returns_panel"]
    assert r.shape[1] == processed["active_panel"].shape[1]
    assert r.shape[0] == processed["close_panel"].shape[0]
    # El índice de mercado existe y es positivo.
    assert (processed["market"]["mkt_index"] > 0).all()


def test_active_mask_within_lifespan():
    idx = pd.date_range("2021-01-01", periods=10, freq="1D")
    close = pd.Series([100] * 3 + [101, 102, 103] + [103] * 4, index=idx, dtype=float)
    mask = clean.active_mask(close)
    # Antes del primer movimiento (día 3) no está vivo; después del último sí hasta el final.
    assert not mask.iloc[0]
    assert mask.iloc[3]
