# Hallazgos de la exploración de datos (Fase 0)

Resumen de la exploración previa a la implementación. Todas las cifras provienen del análisis
directo de los ficheros del concurso. Las cifras marcadas 🔢 se recalculan en cada ejecución del
pipeline y se insertan automáticamente en el README.

## 1. Inventario

| Fichero | Contenido | Observaciones |
|---|---|---|
| `algoritmos/*.csv` | **14.761** curvas de algoritmos | `datetime,open,high,low,close`, velas de 4h, base ~100 |
| `trades_benchmark.csv` | 5.394 operaciones del benchmark | 271 `productname` distintos; 31 posiciones abiertas |
| `benchmark_monthly_returns.csv` | Rentabilidad mensual 2020-06 → 2024-12 | — |
| `benchmark_yearly_returns.csv` | Rentabilidad anual | ≈ 0,5 / 5,0 / 7,1 / 4,1 / 7,0 % |
| `datos_competicion.zip` | Solo la carpeta `algoritmos/` | **No contiene documentación del organizador** |

## 2. Validación de hipótesis

- ✅ **Los 271 `productname` del benchmark coinciden al 100% con ficheros CSV.** El benchmark
  invierte, por tanto, en **271 de 14.761** algoritmos: su valor está en la **selección**.
- ✅ **Tramos planos = fuera de mercado.** Cuando `open=high=low=close`, el algoritmo no opera.
  🔢 ~24% de las series están **completamente planas** (constantes de principio a fin) y, de media,
  los algoritmos activos están **fuera de mercado ~34%** del tiempo (sobre su tramo de vida).
- ✅ **Zona horaria / DST.** Las velas caen en dos rejillas de 4h **desfasadas 1h** (horas
  1,5,9,13,17,21 en invierno vs 2,6,10,14,18,22 en verano). Se neutraliza trabajando con el
  **cierre diario**.
- ⚠️ **Sesgo de supervivencia / inicio.** Las fechas de lanzamiento son muy dispares y muchas
  series se **congelan** antes del final (≈72% terminan sin movimiento). El universo observado no
  es neutral.
- ⚠️ **~1.100 ficheros vacíos o no parseables** (de 14.761). Se descartan con robustez en la carga.

## 3. Reconstrucción del benchmark

- Los **saltos de escala en `AUM`** (de ~130k en 2020 a ~30M en 2024) y en `total_invested`
  (hasta ~14,7M) **no son rentabilidad ni errores**: son **aportaciones de capital** (suscripciones).
  La rentabilidad del benchmark es de un dígito anual, incompatible con un crecimiento ×100 de AUM.
- Un intento de reconstruir la curva desde `AUM`/`equity_normalized` da valores absurdos
  (retornos mensuales de +88% / −48%), porque esos niveles están contaminados por el capital que
  entra y sale. **Conclusión:** la curva del benchmark se toma de los ficheros oficiales de
  returns, y la rentabilidad de cada algoritmo se calcula de su serie `close`, no de los importes.
- Dato adicional: los `trades` llegan hasta **2025-10**, más allá de los returns oficiales
  (2024-12). Ese tramo queda **fuera de alcance**.

## 4. El "mercado" medio pierde dinero

🔢 El índice **equiponderado de algoritmos activos** cae ~**−20%** en 2020–2024, mientras el
benchmark curado sube a ~123 (base 100). Es el hallazgo más relevante: **batir al benchmark es
difícil porque su mérito es la selección**, no el mercado medio. Esto condiciona (y explica) los
resultados del Bloque 3.

## 5. Decisiones propuestas (confirmadas)

1. Universo de perfilado/RL = **271 invertidos**; universo completo solo para la **amplitud**.
2. 4 fases = **Alcista / Bajista / Lateral / Crisis**.
3. Horizonte = **2020-06 → 2024-12**.
4. Mercado de referencia = **agregado equiponderado de activos**, causal.
