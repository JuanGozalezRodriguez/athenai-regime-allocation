# Enunciado del reto (reconstruido)

> ⚠️ **Nota importante.** El enunciado original del concurso de AthenAI **se perdió** junto con el
> repositorio. En la carpeta de datos **no hay ningún documento del organizador** (ni PDF, ni
> notebook, ni especificación): `datos_competicion.zip` solo contiene la carpeta `algoritmos/`.
> Este documento es una **reconstrucción de memoria** de los conceptos clave del reto, con mis
> palabras. Donde el enunciado real pudiera diferir, **mandaría el enunciado**; al no disponer de
> él, las definiciones operativas (las 4 fases, la métrica, el formato) son **decisiones propias
> justificadas**, señaladas como tales.

## Conceptos clave del reto (según recuerdo)

- **Aprendizaje por refuerzo** para la toma de decisiones de inversión.
- **Lógica difusa** como motor de razonamiento.
- **IA no generativa**: nada de LLMs ni modelos generativos; todo debe ser **explicable**.
- El mercado se mueve en **4 fases o "movimientos"**, y nunca se sabe con certeza en cuál se está.
- Objetivo: **estimar en qué fase se está asignando una probabilidad a cada una**.
- Inferir **"a contraluz"** cómo se comporta / en qué invierte cada algoritmo a partir solo de su
  serie de precios.
- **Computación cuántica** como posible herramienta exploratoria.

## Decisiones propias (al no existir enunciado)

| Aspecto | Decisión adoptada | Justificación |
|---|---|---|
| Las 4 fases | Alcista / Bajista / Lateral / Crisis (alta volatilidad) | Régimenes clásicos, legibles para banca y fáciles de mapear a reglas difusas. Alternativa considerada: fases de Wyckoff (acumulación/subida/distribución/bajada). |
| Universo para perfilado y RL | Los **271** algoritmos invertidos por el benchmark | Tienen *ground-truth* de asignación y son tratables; el universo completo (14.761) está dominado por series planas/muertas. |
| Referencia de "mercado" | Agregado **equiponderado de algoritmos activos**, causal | Refleja el mercado que ven los algoritmos y alimenta de forma natural el indicador de amplitud. |
| Horizonte | 2020-06 → 2024-12 | Periodo con returns oficiales del benchmark, para comparar de forma honesta. |
| Métrica de evaluación | CAGR, Sharpe, Sortino, Calmar, max DD, etc., con **test fuera de muestra** | Estándar de la industria; no hay métrica oficial del concurso. |

El detalle de la validación empírica de estas decisiones está en
[`hallazgos_datos.md`](hallazgos_datos.md) y la metodología técnica en
[`metodologia.md`](metodologia.md).
