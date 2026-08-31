# Configuraciones de Arquitectura de Retrieval — Maternas RAG

Documento de referencia para comparar y replicar las dos configuraciones de retrieval
evaluadas en el sistema. Cada configuración puede restaurarse siguiendo los pasos de código
indicados. Los resultados de Ragas quedan registrados en `evaluation_reports/`.

> ⚠️ **Actualización 28-ago-2026:** **Config F** (denso + BM25 sobre `maternaqaes_lm` +
> `upload`) es la config activa en producción — ver `src/rag/retriever_configF.py`.
> BM25 vuelve, corrigiendo la premisa de Config B: ya no filtra ruido de Multiclinsum
> (removido por licencia el 13-ago), sino que desempata léxicamente dentro del corpus
> en español. Medido con `src/evaluation/retrieval_eval.py` (sin juez LLM, 328 pares):
> Recall@5 sube de 0.838 (denso, Config D) a 0.915 (híbrido RRF). Además,
> `ingest_maternaqaes_lm.py --min-clinical-score 8` (antes 15) llevó la alcanzabilidad
> del golden set de 72.6% a 100% — ver la sección Config F más abajo para el detalle
> completo, incluyendo un data leakage estructural preexistente que hay que tener en
> cuenta al leer estos números. Config A/B/C ya no son reproducibles tal cual: dependen
> de fuentes que ya no están en el índice. Índice actual: 255.625 vectores.

---

## Config A — FAISS Puro (baseline)

**Commit de referencia:** `6c2c406`
**Fecha:** pre julio 2026

### Descripción

Búsqueda densa única sobre el índice FAISS completo (375.392 vectores).
No hay distinción de fuente: `multiclinsum`, `textbook`, `medmcqa` y `medqa_*`
compiten en el mismo ranking por similitud coseno (IndexFlatIP).

El sistema devuelve los top-k globales sin importar de qué dataset provienen.
Dado que Multiclinsum representa ~14% del índice (51.804 fragmentos) y sus
casos clínicos en español tienden a tener scores de similitud relativamente
altos por el idioma, en la práctica muchos de los top-5 provenían de casos
clínicos (p.ej. liposarcoma, tromboembolismo) aunque la pregunta fuese sobre
anticoncepción o clarificación de vacunas, contaminando el contexto del LLM.

### Flujo de retrieve()

```
query → embed("query: " + query) → FAISS.search(k=5)
      → top-5 sin filtro de fuente → LLM
```

### Archivos modificados respecto al estado actual

#### `src/rag/retriever.py` — reemplazar con la versión del commit 6c2c406

Cambios clave a revertir:

1. **Eliminar** las constantes `DENSE_SOURCES` y `MULTICLINSUM_SOURCES`
2. **Eliminar** las funciones `_retrieve_dense()` y `_retrieve_bm25()`
3. **Reemplazar** `retrieve()` con la versión simple:

```python
def retrieve(
    query: str,
    k: int | None = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    if not query or not query.strip():
        return []
    if k is None:
        k = settings.rag_top_k
    store   = _get_store()
    results = store.search(query, k=k)
    if min_score > 0.0:
        results = [r for r in results if r.get("score", 0.0) >= min_score]
    return results
```

4. **Reemplazar** `format_context()` — sin el tag `[caso clínico]`:

```python
def format_context(docs: list[dict[str, Any]], max_chars: int = 4000) -> str:
    if not docs:
        return "No se encontraron fragmentos relevantes en la base de conocimiento."
    fragments: list[str] = []
    total_chars = 0
    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "").strip()
        fragment = f"--- Fragmento [{i}] ---\n{text}"
        if total_chars + len(fragment) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                fragments.append(fragment[:remaining] + "...")
            break
        fragments.append(fragment)
        total_chars += len(fragment)
    return "\n\n".join(fragments)
```

5. **No se usa** `src/rag/bm25_index.py` en esta config (puede dejarse en disco,
   simplemente no se importa).

#### `src/rag/chain.py` — sin cambios necesarios

`chain.py` llama `retrieve()` y `format_context()` de forma agnóstica;
no necesita modificación al cambiar de config.

### Parámetros clave

| Parámetro | Valor |
|---|---|
| Índice | FAISS IndexFlatIP, 375.392 vectores |
| Fuentes en retrieval | todas (textbook + medmcqa + medqa_* + multiclinsum) |
| k devuelto | `settings.rag_top_k` = 5 |
| Filtro por fuente | ninguno |
| BM25 | no |
| Candidatos intermedios | 5 (= k final) |

### Comportamiento esperado en evaluación

- **Faithfulness**: probablemente moderado. El LLM recibe contexto irrelevante
  (casos clínicos de oncología, traumatología) y tiende a ignorarlo o
  responder desde conocimiento general → menos anclaje en fuentes.
- **Answer relevancy**: puede ser aceptable si el LLM compensa con conocimiento
  general, pero sin respaldo documental.
- **Context recall**: muy bajo. Los fragmentos recuperados rara vez corresponden
  a los documentos de MaternaQA-es (GPC Colombia, revistas obstétricas).

---

## Config B — Retrieval Híbrido FAISS + BM25 (actual)

**Commit de referencia:** `a6baf49`
**Fecha:** julio 2026
**Archivo activo:** `src/rag/retriever.py` (estado actual en rama main)

### Descripción

Búsqueda híbrida con dos capas independientes según el tipo de fuente:

- **Capa densa (FAISS):** opera exclusivamente sobre `textbook`, `medmcqa`,
  `medqa_us`, `medqa_taiwan`, `medqa_mainland`. Pide k×10 candidatos al índice
  para tener margen de filtrado post-proceso de Multiclinsum (~14% del índice).
  Devuelve los primeros k que pertenezcan a `DENSE_SOURCES`.

- **Capa léxica (BM25):** opera exclusivamente sobre los 51.804 fragmentos de
  `multiclinsum_summary` y `multiclinsum_fulltext`. Usa `rank_bm25` con
  tokenizador ES/EN con stopwords. Solo devuelve fragmentos si el score BM25
  supera el umbral `min_score=0.5`, garantizando que Multiclinsum solo aparece
  cuando hay coincidencia léxica real con la query.

El resultado final combina: hasta 5 fragmentos densos + hasta 2 BM25.
Los fragmentos BM25 se etiquetan `[caso clínico]` en el contexto.

### Flujo de retrieve()

```
query ──┬──► embed("query: " + query) → FAISS.search(k×10)
        │    → filtrar solo DENSE_SOURCES → top-5 densos
        │
        └──► BM25.search(query, k=2, min_score=0.5)
             → solo fragmentos Multiclinsum con match léxico real
             → top-2 BM25 (puede ser 0 si no hay match)

top-5 densos + top-2 BM25 → LLM
```

### Archivos activos (sin cambio)

#### `src/rag/retriever.py`
- Constantes: `DENSE_SOURCES`, `MULTICLINSUM_SOURCES`
- Funciones: `_retrieve_dense(query, k)`, `_retrieve_bm25(query, k)`
- `retrieve(query, k=None, k_bm25=2)` — merge dense + BM25
- `format_context()` — añade tag `[caso clínico]` a fragmentos BM25

#### `src/rag/bm25_index.py`
- Singleton `BM25Okapi` construido en memoria al primer uso (~10-20s, ~150 MB RAM)
- Función pública: `search_bm25(query, k, min_score) -> list[dict]`
- Tokenizador con stopwords ES+EN

### Parámetros clave

| Parámetro | Valor |
|---|---|
| Índice FAISS | IndexFlatIP, 375.392 vectores |
| Fuentes capa densa | textbook, medmcqa, medqa_us, medqa_taiwan, medqa_mainland |
| Candidatos FAISS intermedios | k × 10 = 50 (para garantizar 5 no-Multiclinsum) |
| k densa final | `settings.rag_top_k` = 5 |
| Corpus BM25 | multiclinsum_summary + multiclinsum_fulltext (51.804 fragmentos) |
| k BM25 | 2 |
| min_score BM25 | 0.5 |
| Total máximo fragmentos | 7 (5 densos + 2 BM25) |

### Comportamiento esperado en evaluación

- **Faithfulness**: más alto que Config A. El LLM recibe contexto semánticamente
  más relevante de textbooks y MedMCQA, con menos "ruido" de casos clínicos
  irrelevantes. Cuando BM25 aporta un caso clínico, es porque hay match léxico real.
- **Answer relevancy**: igual o mejor que Config A por la misma razón.
- **Context recall**: sigue siendo bajo (el corpus no contiene los PDFs de MaternaQA-es),
  pero la capa BM25 puede mejorar marginalmente cuando la query usa términos
  específicos que aparecen en Multiclinsum.

---

## Config F — Híbrido denso + BM25 sobre el corpus en español (producción actual)

**Commit de referencia:** ver git log de `src/rag/retriever_configF.py`
**Fecha:** 28-ago-2026
**Archivo activo:** `src/rag/retriever.py`

### Descripción

Retoma el híbrido de Config B, pero corrigiendo su premisa. Config B usaba BM25 para
*filtrar ruido*: Multiclinsum (casos clínicos) contaminaba el top-k denso con
fragmentos irrelevantes. Ese problema ya no existe — Multiclinsum y textbook salieron
del índice el 13-ago-2026 (Config D). Medido sobre el índice actual: en el último eval
de producción, el 100% de los fragmentos recuperados ya venían de `maternaqaes_lm`
(0 de MedMCQA/MedQA, que son el 97,8% del índice). `multilingual-e5-base` ya separa
bien las fuentes en español de las de inglés/chino sin ayuda de BM25.

Lo que BM25 aporta aquí es distinto: **desempate léxico dentro del corpus correcto**.
El corpus en español son solo ~5.400 fragmentos (`maternaqaes_lm` + `upload`). Ahí BM25
gana en fármacos, dosis, cifras y siglas exactas — donde el embedding denso difumina
sinónimos y pierde precisión numérica.

### Flujo de retrieve()

```
query ──┬──► embed("query: " + query) → FAISS.search(pool=20) → filtrar DENSE_SOURCES
        │
        └──► BM25.search(query, pool=20) sobre maternaqaes_lm + upload (~5.400 frags)

fusión (RRF k=10 por defecto, o ponderada min-max) → top-k final
```

`DENSE_SOURCES` hoy coincide con el 100% de las fuentes del índice (no hay textbook ni
Multiclinsum) — el filtro se mantiene como red de seguridad, no porque haga algo hoy.

### Archivos

- `src/rag/bm25_index.py` — singleton BM25 sobre `LEXICAL_SOURCES = {maternaqaes_lm, upload}`.
  No relee `metadata.pkl`: reusa `FAISSStore.metadata` ya cargado. Se reconstruye
  automáticamente si `store.mutation_seq` cambia (documento subido/activado/desactivado
  desde el panel) — a diferencia de la versión de Config B, si respeta el flag `active`.
- `src/rag/retriever.py` — `_retrieve_dense`, `_retrieve_bm25`, `_fuse_rrf`,
  `_fuse_weighted`, `_doc_identity` (deduplicación por `(source_dataset, doc_id, chunk_id)`,
  no por `chunk_id` solo — no está garantizado único entre datasets).
- `src/settings.py` — `rag_fusion_strategy` (`"rrf"` por defecto), `rag_fusion_rrf_k`,
  `rag_fusion_dense_weight`, `rag_dense_pool`, `rag_bm25_pool`.
- `src/evaluation/retrieval_eval.py` — harness determinista (sin juez LLM) usado para
  elegir estrategia de fusión y validar el impacto del umbral de ingestión (ver abajo).

### Resultados medidos (retrieval_eval.py, 328 pares, sin juez LLM)

| Estrategia | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Denso (Config D) | 0.838 | 0.902 | 0.647 | 0.710 |
| BM25 solo | 0.860 | 0.930 | 0.717 | 0.769 |
| **Híbrido RRF k=10 (default)** | **0.915** | **0.960** | 0.729 | 0.786 |
| Híbrido ponderado (denso=0.5) | 0.909 | 0.960 | **0.755** | **0.805** |

RRF y ponderada quedan a un pelo — RRF gana en R@5, ponderada en MRR/nDCG. Se dejan
ambas implementadas y configurables; el default es RRF por robustez (no depende de que
los scores de FAISS-IP y BM25 sean comparables en escala, cosa que no está garantizada).

Reporte completo: `evaluation_reports/retrieval_eval_configF_vs_configD_*.md`.

### Resultados Ragas (cadena completa, LLM judge) — confirman el efecto medido arriba

| Métrica | Config D (19-ago, n=15) | **Config F (28-ago, n=39)** | Δ |
|---|---:|---:|---:|
| `faithfulness` | 0.3915 | **0.6173** | +0.226 |
| `answer_correctness` | 0.6362 | **0.7876** | +0.151 |
| `answer_relevancy` | 0.8101 | **0.9396** | +0.130 |
| `context_recall` | 0.4222 | **0.6368** | +0.215 |
| `context_precision` | 0.3359 | **0.5608** | +0.225 |

Mismo generador (`gpt-oss-120b`) y mismo juez (Cerebras `gemma-4-31b`) en ambas corridas —
la diferencia es atribuible al retrieval híbrido + la eliminación del techo de
alcanzabilidad, no a un cambio de modelo. Reporte completo:
`evaluation_reports/eval_report_configF_20260828_173548.md`.

### Cambio acompañante — techo de alcanzabilidad eliminado

Independiente del retrieval, `ingest_maternaqaes_lm.py --min-clinical-score 8` (antes
15, hardcodeado) recuperó 2.156 chunks nuevos vía re-ingesta incremental (dedup por
`chunk_id`, sin reconstruir el índice). Efecto: 90 de los 328 pares del golden set
tenían su chunk-oro descartado por el umbral (score 8-14) y ahora están en el índice.
Alcanzabilidad: 72.6% → 100%. Índice: 253.469 → 255.625 vectores.

### ⚠️ Data leakage estructural — preexistente, no introducido por este cambio

Los 3 PDFs del split `test` del corpus LM (`GPC-Atencion-Prenatal-de-Bajo-Riesgo-2023.pdf`,
`vol831-1.pdf`, `4142_stamped.pdf`) son **exactamente** los 3 `source_pdf` de los que sale
el golden set de evaluación (`maternaqa_test.jsonl`) — verificado por intersección exacta.
`ingest_maternaqaes_lm.py` incluye ese split por default (`include_test=True`), y ya lo
hacía antes de este cambio (a `clinical_score>=15` el split test ya aportaba 290 chunks).

Esto significa que **todos** los `eval_report_config{A..F}_*.md` publicados evalúan
retrieval con el documento fuente de cada pregunta ya en el índice. Es una medida válida
de "¿encuentra el pasaje correcto dentro de documentos que ya indexó?" (lo relevante para
producción), pero **no** de generalización a documentos nunca vistos, y la comparación
contra el baseline publicado de MaternaQA-es (`eval_pipeline.py`, `faithfulness` train
0.7726 / test 0.7132) hereda esa asimetría — no se sabe si ese baseline tuvo acceso a los
mismos documentos. `--exclude-test` existe para una comparación sin esta leakage pero
nunca se ha usado en producción.

### Qué NO arregla

`faithfulness` (0.3915 en el último reporte de producción) es el peor número medido y es
un problema de **generación**, no de retrieval: el LLM responde desde conocimiento general
en vez de anclarse en los fragmentos entregados. Mejor contexto ayuda pero no lo resuelve
— es el siguiente candidato a atacar, en el prompt de `src/rag/chain.py`.

---

## Comparación directa

| Aspecto | Config A (FAISS puro) | Config B (Híbrido) |
|---|---|---|
| Fuentes en capa densa | todas | solo textbook/medmcqa/medqa_* |
| Multiclinsum en retrieval | compite en ranking FAISS | solo por BM25 con umbral |
| Candidatos FAISS | k = 5 | k × 10 = 50 |
| BM25 | no | sí, sobre 51.804 fragmentos |
| Riesgo contaminación contexto | alto (casos clínicos irrelevantes) | bajo |
| Fragmentos máximos al LLM | 5 | 7 (5+2) |
| Overhead de memoria extra | ninguno | ~150 MB RAM (BM25 index) |
| Overhead de latencia extra | ninguno | ~10-20s al primer uso (build BM25) |

---

## Plan de evaluación con Ragas — uso eficiente de API

### Problema

Ragas necesita un LLM judge (Groq) para calcular `faithfulness` y `context_recall`.
Con 50 pares × 2 configuraciones = 100 generaciones + múltiples llamadas de evaluación,
el límite de 100k tokens/día del tier gratuito de Groq se agota fácilmente.

### Estrategia propuesta

#### 1. Separar claves por función

- `GROQ_API_KEY` → generación de respuestas del chatbot (fase 1, el sistema RAG)
- `GROQ_API_KEY_2` → LLM judge de Ragas (fase 2, evaluación)

Al tener claves independientes, cada una consume su propio cupo de 100k tokens/día,
duplicando el presupuesto disponible.

#### 2. Reducir la muestra a 20 pares estratificados (en vez de 50)

Con 20 pares bien distribuidos (proporción tipos/dificultades del dataset):

| Tipo | N en 328 | N en muestra 20 |
|---|---|---|
| factual | 101 (31%) | 6 |
| aplicacion | 94 (29%) | 6 |
| razonamiento | 85 (26%) | 5 |
| definicion | 28 (9%) | 2 |
| hipotetico | 15 (5%) | 1 |
| comparacion | 5 (2%) | 0 |

20 pares × 2 configs = 40 generaciones totales + evaluaciones Ragas.
Estimado: ~30-40k tokens por ejecución completa → dentro del límite diario.

#### 3. Reutilizar la fase 1 entre runs

La fase 1 (generación de respuestas) es determinista dado el mismo seed.
Si ambas configs usan el mismo set de preguntas (`seed=42`, `sample=20`),
los `eval_raw_*.json` de cada config pueden guardarse con nombre descriptivo
y la fase 2 puede ejecutarse sobre ellos en cualquier momento posterior,
incluso al día siguiente cuando los tokens se renuevan.

Comando sugerido:
```bash
# Config A: generar respuestas (usa retriever revertido a FAISS puro)
python src/evaluation/eval_pipeline.py --sample 20 --generate-only
# renombrar: eval_raw_configA_<ts>.json

# Config B: generar respuestas (retriever híbrido actual)
python src/evaluation/eval_pipeline.py --sample 20 --generate-only
# renombrar: eval_raw_configB_<ts>.json

# Evaluar ambos con Ragas (usa GROQ_API_KEY_2)
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_configA_<ts>.json
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_configB_<ts>.json
```

#### 4. Deshabilitar `answer_relevancy` en la primera pasada (opcional)

`answer_relevancy` consume tokens adicionales de embeddings.
En una primera pasada comparativa, `faithfulness` y `context_recall`
son las métricas que más diferencian las configs.
Si los tokens son escasos, evaluar solo esas dos primero.

#### 5. Guardar eval_raw reutilizable

Los `eval_raw_*.json` contienen las respuestas generadas y los contextos recuperados.
Si Ragas falla a mitad por rate limit, se puede relanzar `--evaluate-only` con el
mismo archivo sin repetir las generaciones (que son la parte más costosa).

### Resumen del plan

```
Día 1:
  09:00  Fase 1 Config A: 20 generaciones (~10k tokens GROQ_API_KEY)
  09:05  Fase 1 Config B: 20 generaciones (~10k tokens GROQ_API_KEY)
  09:10  Fase 2 Config A: Ragas 20 pares (~15k tokens GROQ_API_KEY_2)
  09:30  Fase 2 Config B: Ragas 20 pares (~15k tokens GROQ_API_KEY_2)
  Total: ~20k tokens en KEY_1 + ~30k tokens en KEY_2 → dentro del límite
```

---

## Archivos relevantes

| Archivo | Rol |
|---|---|
| `src/rag/retriever.py` | Lógica de retrieval — cambiar para alternar configs |
| `src/rag/bm25_index.py` | Singleton BM25 — solo usado en Config B |
| `src/rag/chain.py` | Agnóstico a la config de retrieval |
| `src/evaluation/eval_pipeline.py` | Pipeline de evaluación (fases 1 y 2) |
| `src/evaluation/sampler.py` | Muestreo estratificado de MaternaQA-es |
| `evaluation_reports/` | Raw JSONs y reportes Markdown (gitignored) |
| `.env` | `GROQ_API_KEY` (chatbot) y `GROQ_API_KEY_2` (Ragas judge) |

---

*Documento generado durante sesión de desarrollo — julio 2026*
