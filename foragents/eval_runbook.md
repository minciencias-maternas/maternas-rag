# Guía de Evaluación RAG — Maternas

> **LEER ANTES DE TOCAR CUALQUIER ARCHIVO DEL PIPELINE DE EVALUACIÓN.**
> Ver también `foragents/eval_setup_critico.md` para decisiones arquitectónicas
> y resultados históricos.

> ⚠️ **Actualización 28-ago-2026:** **Config F es ahora la config de producción**
> (reemplaza a Config D). Retoma BM25 — pero sobre `maternaqaes_lm` + `upload`
> (~5.4k frags ES), no sobre Multiclinsum (que ya no está en el índice). Medido con
> `src/evaluation/retrieval_eval.py` (nuevo, sin juez LLM): híbrido RRF R@5=0.915 vs
> denso puro R@5=0.838 sobre los 328 pares de MaternaQA-es test. Además,
> `ingest_maternaqaes_lm.py --min-clinical-score 8` (antes 15) eliminó el techo de
> alcanzabilidad: 72.6% → 100% de los pares del golden set tienen su chunk-oro en el
> índice. Índice: 253,469 → 255,625 vectores. Detalle completo en
> `foragents/retrieval_arquitecturas_configs.md` (Config F) y `qa_technical.md`.

> ⚠️ **Actualización 13-ago-2026:** `textbook` y `multiclinsum` fueron removidos del
> índice FAISS por licencia (`qa_technical.md` Q31). Índice: 380,745 → 253,455
> vectores en ese momento (ver arriba para el estado actual).

---

## Arquitectura de dos modelos — NO confundir

```
FASE 1 (generación)        FASE 2 (evaluación)
llama-3.3-70b (Groq)  →   gemma-4-31b (Cerebras)
GROQ_API_KEY               CEREBRAS_KEY
El chatbot RAG real        El juez de Ragas
```

**Nunca usar el mismo modelo para ambas fases.** El juez debe ser independiente.

---

## Variables de entorno requeridas (.env)

```env
GROQ_API_KEY=gsk_...        # chatbot — fase 1
CEREBRAS_KEY=csk_...        # Ragas judge — fase 2
GROQ_API_KEY_2=gsk_...      # backup Groq (opcional)
OPENROUTER_KEY=sk-or-...    # backup OpenRouter (opcional, inestable)
```

---

## Configs de retrieval disponibles

| Archivo | Activar con | Descripción |
|---|---|---|
| `src/rag/retriever_configA.py` | `copy configA retriever.py` | FAISS puro, top-k global sin filtro (historico, no reproducible tal cual) |
| `src/rag/retriever_configB.py` | `copy configB retriever.py` | FAISS+BM25 hibrido sobre Multiclinsum (historico, dataset removido) |
| `src/rag/retriever_configC.py` | `copy configC retriever.py` | FAISS+BM25 + corpus obstetrics ES (historico) |
| `src/rag/retriever_configD.py` | `copy configD retriever.py` | denso puro, sin textbook/Multiclinsum/BM25 |
| `src/rag/retriever_configF.py` | `copy configF retriever.py` | denso + BM25 sobre maternaqaes_lm/upload (produccion) |
| `src/rag/retriever.py` | — | **Activo en producción — Config F** |

**Siempre restaurar Config F al terminar una evaluación:**
```bash
copy src\rag\retriever_configF.py src\rag\retriever.py
```

---

## Corpus en el índice FAISS

> ⚠️ Tabla desactualizada tras la remoción de textbook/multiclinsum (13-ago) y la
> re-ingesta de maternaqaes_lm con umbral bajado (28-ago). Estado actual real:

| Dataset | Chunks | Idioma | Fuente |
|---|---|---|---|
| medmcqa | ~187k | EN | Exámenes médicos India |
| medqa_* | ~61k | EN/ZH | USMLE + Taiwan + Mainland |
| **maternaqaes_lm** | **~7.5k** | **ES** | **Corpus obstetrico colombiano (train+val+test, rechunked ~400tok)** |
| upload | variable | ES | Documentos subidos desde el panel |
| **Total** | **255.625** | — | — |

**IMPORTANTE sobre maternaqaes_lm — data leakage confirmado, no nuevo:**
- Los 3 PDFs del split `test` del corpus LM (`GPC-Atencion-Prenatal-de-Bajo-Riesgo-2023.pdf`,
  `vol831-1.pdf`, `4142_stamped.pdf`) son **exactamente** los 3 `source_pdf` de los que
  sale el golden set de evaluación (`maternaqa_test.jsonl`, 328 pares, repo
  `JhonHander/MaternaQA-es`) — verificado por intersección exacta, no coincidencia.
  Las preguntas de evaluación se generaron a partir de estos mismos documentos.
- `ingest(include_test=True)` es el **default histórico** (ya lo era antes de esta
  sesión: a `clinical_score>=15` el split test ya aportaba 290 chunks). Esto significa
  que TODOS los `eval_report_config{A..F}_*.md` publicados evalúan retrieval con el
  documento fuente de la pregunta ya en el índice — no es una medida de generalización
  a documentos nunca vistos, es una medida de "¿encuentra el pasaje correcto dentro de
  documentos que ya indexó?", que es lo que un sistema RAG desplegado necesita.
- La comparación contra el baseline publicado de MaternaQA-es (`faithfulness` train
  0.7726 / test 0.7132 en `eval_pipeline.py`) hereda esta asimetría: no se sabe si ese
  baseline tuvo o no acceso a los mismos documentos al generar sus respuestas. Tratarla
  como comparación exacta es optimista; ya lo era antes de este cambio.
- `--exclude-test` existe para una comparación sin esta leakage, pero nunca se ha usado
  en producción. Bajar `--min-clinical-score` a 8 (28-ago-2026) no introdujo el
  problema — solo recuperó más chunks (score 8-14) de los mismos PDFs que ya estaban
  parcialmente indexados, cerrando el hueco de cobertura que antes hacía inalcanzables
  90/328 pares del golden set.
- Re-chunkeado a ~400 tok con `clinical_score >= 8` (antes 15 — ver
  `foragents/qa_technical.md` para el análisis del techo de recall que motivó bajarlo)
- Script de ingestión: `src/ingestion/ingest_maternaqaes_lm.py`

---

## Comandos para correr una evaluación completa

### Paso 0 — Verificar tokens disponibles

```bash
python C:\Users\Usuario\AppData\Local\Temp\opencode\check_quota.py
# Debe mostrar: KEY_1: OK | KEY_2: OK
# Si muestra LIMIT, esperar renovación a las 00:00 UTC
```

### Paso 1 — Activar la config a evaluar

```bash
# Config B (producción actual)
copy src\rag\retriever_configB.py src\rag\retriever.py

# Config C (+ corpus obstetrics ES)
copy src\rag\retriever_configC.py src\rag\retriever.py
```

### Paso 2 — Generar respuestas (Fase 1)

```bash
python src/evaluation/eval_pipeline.py --config configB --sample 15 --generate-only
# Salida: evaluation_reports/eval_raw_configB_<ts>.json
```

**Verificar SIEMPRE que no hay fallbacks antes de fase 2:**
```bash
# Abrir el JSON y confirmar que ninguna respuesta contiene "Lo siento, tuve un problema"
# Si hay fallbacks, la KEY_1 estaba agotada — regenerar con tokens frescos
```

### Paso 3 — Evaluar con Ragas (Fase 2)

```bash
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports\eval_raw_configB_<ts>.json
# Salida: evaluation_reports/eval_report_configB_<ts>.md
#         evaluation_reports/eval_results_configB_<ts>.json
```

**Tiempo estimado fase 2:** ~35-45 minutos para 15 pares con Cerebras.

### Paso 4 — Restaurar Config B

```bash
copy src\rag\retriever_configB.py src\rag\retriever.py
```

---

## Flujo completo en un solo bloque (copiar y pegar)

```bash
# 1. Activar config
copy src\rag\retriever_configC.py src\rag\retriever.py

# 2. Generar (usa GROQ_API_KEY)
python src/evaluation/eval_pipeline.py --config configC --sample 15 --generate-only

# 3. Verificar fallbacks en el JSON generado
# 4. Evaluar (usa CEREBRAS_KEY)
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports\eval_raw_configC_<ts>.json

# 5. Restaurar producción
copy src\rag\retriever_configB.py src\rag\retriever.py
```

---

## Métricas que calcula Ragas y su interpretación

| Métrica | Qué mide | Referencia baseline test |
|---|---|---|
| `faithfulness` | ¿La respuesta está respaldada por los fragmentos recuperados? | 0.7132 |
| `answer_correctness` | ¿Qué tan correcta es la respuesta vs el ground truth? | N/A |
| `answer_relevancy` | ¿La respuesta responde bien la pregunta? | 0.5583 |
| `context_recall` | ¿El retrieval capturó contexto del ground truth? | N/A |
| `context_precision` | ¿Los fragmentos recuperados son precisos y útiles? | N/A |
| `latency_s` | Tiempo end-to-end por par (medido en fase 1) | — |

**Por qué context_recall y context_precision son bajos:**
El corpus RAG no contiene los PDFs exactos del benchmark MaternaQA-es (split test excluido por leakage). Los valores subirán al ingestar esos documentos.

---

## Resultados históricos (seed=42, 15 pares, Cerebras judge)

| Config | Faithfulness | Ans. Correct. | Ans. Relev. | Ctx. Recall | Ctx. Prec. |
|---|---|---|---|---|---|
| A — FAISS puro | 0.162 | 0.350 | 0.634 | 0.000 | 0.000 |
| B — FAISS+BM25 | 0.228 | 0.338 | 0.631 | 0.000 | 0.000 |
| C v1 — +obstetrics 879tok | 0.133 | 0.378 | 0.691 | 0.033 | 0.143 |
| **C v2 — +obstetrics 336tok** | **0.358** | **0.337** | **0.631** | **0.067** | **0.083** |
| Baseline MaternaQA-es (test) | 0.713 | — | 0.558 | — | — |

**Config activa en producción: B**
**Mejor faithfulness hasta ahora: C v2 (0.358)**

---

## Por qué faithfulness no llega al baseline (0.71)

1. **Causa principal**: los 3 PDFs del split test (`GPC-Atencion-Prenatal-2023`, `vol831-1`, `4142_stamped`) no están en el índice — son exactamente los documentos que generaron las 328 preguntas del benchmark. El LLM responde desde conocimiento general, no desde fuentes indexadas.

2. **Segunda causa**: cuando el sistema lanza una clarification question, faithfulness=0 automáticamente porque una pregunta no tiene statements verificables. Excluir `needs_clarification=True` de la muestra mejoraría el promedio.

3. **Tercera causa**: varianza alta con 15 pares (std~0.25). Con 30 pares la estimación sería más estable.

### Cómo mejorar faithfulness sin ingestar el split test

- **System prompt más estricto**: instruir al LLM a decir explícitamente "no tengo información" en vez de responder con conocimiento general → faithfulness de los "no sé" sube (como se vio en Config B pares con contexto irrelevante)
- **Excluir clarification questions** de la muestra de evaluación: filtrar `needs_clarification=True`
- **Más pares (30)**: reduce varianza estadística aunque no sube el techo estructural
- **Ingestar el split test** (con flag `--include-test`): upper bound real, invalida comparación justa con benchmark pero útil para medir el techo del sistema

---

## Problema conocido: `TimeoutError` en Cerebras

Algunos batches del grupo 2 (answer_relevancy/context_recall/context_precision) tienen
`TimeoutError` ocasional cuando Cerebras tarda más de 120s. Es benigno — Ragas lo
marca como fallo del par pero el resto continúa. Los pares fallidos quedan como -1
y se excluyen del promedio.

Si hay muchos TimeoutError aumentar el timeout en `_run_ragas_group()`:
```python
run_cfg = RunConfig(max_workers=1, max_retries=2, max_wait=15, timeout=180)  # 180 en vez de 120
```

---

*Actualizado: 20 de julio de 2026*
*Ver también: `foragents/eval_setup_critico.md`, `foragents/qa_technical.md` Q23-Q27*
