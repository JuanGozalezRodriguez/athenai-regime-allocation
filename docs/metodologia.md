# Metodología

Detalle técnico de cada bloque. Principio transversal: **causalidad estricta** (toda cantidad en
`t` usa solo datos hasta `t`), parámetros en `configs/default.yaml`, semillas fijas y resultados
reproducibles.

## Capa de datos

1. **Carga** robusta de cada CSV a OHLC con `DatetimeIndex` garantizado (se descartan ficheros
   vacíos o con fechas corruptas).
2. **Cierre diario**: se toma el último `close` de cada día natural, lo que neutraliza el desfase
   DST de 1h de las velas de 4h.
3. **Tramos planos**: `open=high=low=close` ⇒ fuera de mercado.
4. **Vida del algoritmo**: intervalo entre su primer y último movimiento de precio. Fuera de él no
   participa en los agregados (ni "no lanzado" ni "congelado").
5. **Índice de mercado** (en *streaming* sobre los 14.761): media equiponderada de los retornos
   diarios de los algoritmos vivos; **amplitud** = fracción de vivos en positivo. Los retornos se
   *winsorizan* a ±20% para evitar que errores de datos contaminen el agregado.

## Bloque 1 · Régimen difuso

**Indicadores causales** (sobre el índice de mercado): tendencia (separación relativa MA rápida/
lenta), volatilidad realizada anualizada, drawdown desde máximo histórico, amplitud y momentum a
varios horizontes. La tendencia y la volatilidad se normalizan con **z-score *expanding*** (media
y desviación acumuladas, nunca de toda la muestra).

**Motor difuso** (implementación propia, sin scikit-fuzzy, para máxima transparencia):
- Fuzzificación con **funciones de pertenencia triangulares** legibles (ver figura del README).
- **8 reglas** "SI … ENTONCES fase" en lenguaje natural, con operador Y = mínimo.
- Agregación por fase = máximo de sus reglas; normalización ⇒ **4 probabilidades que suman 1**.
- Suavizado causal (media móvil corta) para reducir el *whipsaw*; se re-normaliza tras suavizar.

**Comparación**: un clasificador por **umbrales** y un **k-means** (ajustado solo en train) sobre
los mismos indicadores. Se reporta el % de coincidencia con el difuso.

## Bloque 2 · Perfilado "a contraluz"

Por algoritmo: fracción de tiempo invertido, operaciones (de los trades), retorno/volatilidad/
Sharpe/drawdown, **rendimiento condicionado a cada fase**, correlaciones y **agrupamiento
jerárquico** (distancia = 1 − ρ). La **etiqueta de estilo** (tendencia / reversión / defensivo) se
infiere de la **beta al mercado** y del comportamiento por fase, con la evidencia que la sostiene.

## Bloque 3 · Asignación con RL

- **Carteras por fase** (4): equiponderadas sobre los `top_k` algoritmos con mejor rentabilidad
  **ajustada al riesgo** (media/desviación) en cada fase, calculadas **solo con datos de train**.
- **Entorno**: estado = fase dominante (0..3); acción = elegir una de las 4 carteras; recompensa =
  retorno neto de costes del mes siguiente con utilidad media-varianza `r − ½·λ·r²`.
- **Agente**: **Q-learning tabular** (tabla 4×4), ε-greedy. La política resultante es una regla
  legible *fase → cartera*.
- **Costes de transacción**: configurables (por defecto 10 pb por lado sobre el cambio de pesos).
- **Validación walk-forward**: train (→2022) · validación (2023) · **test 2024 fuera de muestra**,
  reportado aparte.
- **Baselines**: benchmark oficial, equiponderado (271), mejor algoritmo en train, y **regla
  difusa sin RL** (política identidad fase→cartera homónima).

### Resultado y honestidad
La estrategia RL **no bate al benchmark en test**. Aprende una política **defensiva** que protege
capital (drawdown bajo) pero pierde parte de las subidas. Con ~30 observaciones mensuales de
entrenamiento y 4 estados, el riesgo de **sobreajuste** es real; el equiponderado simple resulta un
rival muy fuerte. Se reporta tal cual, sin maquillar (ver README → Limitaciones).

## Bloque 4 · Cuántica (experimental)

Selección de `k` de `n` algoritmos como **QUBO**: minimizar `−rentabilidad + λ·riesgo +
P·(Σx − k)²`. Se resuelve de dos formas y se comparan:
- **Clásica exacta** (fuerza bruta, viable con `n` pequeño).
- **QAOA** (Qiskit, primitivas de **vector de estado**; no requiere `qiskit-aer`): se construye el
  hamiltoniano de Ising, se optimizan los parámetros del ansatz (COBYLA) y se muestrea. Entre los
  estados más probables se prefiere uno **factible** (cardinalidad = k) de menor coste.

Es un experimento **ilustrativo**: con pocas capas en simulador, QAOA recupera parte de la solución
óptima, **sin pretensión de ventaja cuántica**.

## Reproducibilidad

- Todos los parámetros en `configs/default.yaml`; semilla global fija.
- Todas las cifras del README se generan con `scripts/run_pipeline.py` y se inyectan con
  `scripts/update_readme.py` entre marcadores `<!--AUTO:…-->`.
- Tests en `tests/` verifican causalidad (sin lookahead), que P(fase) suma 1, las métricas sobre
  casos conocidos y el formato del generador sintético.
