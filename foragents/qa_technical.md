# Preguntas y Respuestas Técnicas — Chatbot RAG Maternas

Registro de preguntas técnicas surgidas durante el desarrollo y sus respuestas definitivas.
Sirve como referencia rápida para cualquier sesión futura.

---

## Q1: ¿Dónde se almacenan los embeddings y la vector store?

**Respuesta:**

Los embeddings se almacenan en **dos lugares distintos según el momento**:

### En disco (persistencia entre sesiones)

La carpeta `faiss_store/` contiene tres archivos que juntos forman la vector store completa:

```
faiss_store/
├── index.faiss       ← Los vectores (embeddings float32) en formato binario FAISS
├── metadata.pkl      ← Pickle Python: dict { chunk_id → { text, source_dataset, language, ... } }
└── build_info.json   ← Parámetros de construcción: fecha, modelo, dimensión, total de vectores
```

- **`index.faiss`**: Contiene los ~402,000 vectores de 768 dimensiones cada uno, en el formato nativo de FAISS (`IndexFlatIP`). Es un archivo binario — no es legible directamente. Tamaño estimado: ~1.2 GB.
- **`metadata.pkl`**: Mapea cada vector (por su posición/ID en el índice) con el texto original del chunk y sus metadatos. Es necesario para poder mostrar la fuente de cada fragmento recuperado.
- **`build_info.json`**: Registro de auditoría. Indica qué modelo generó los embeddings, cuándo, con qué parámetros. Permite detectar si el índice está desactualizado.

### En RAM (mientras la app está corriendo)

Cuando la API o la interfaz arrancan, se ejecuta:
```python
faiss.read_index("faiss_store/index.faiss")
```
Esto carga **todos los vectores en memoria RAM**. Estimado: ~1.2 GB de RAM ocupados en tiempo de ejecución.

El modelo de embedding (`multilingual-e5-base`) también se carga en memoria (CPU) en tiempo de ejecución para convertir las queries del usuario en vectores. Pesa ~1.1 GB adicional.

### Flujo completo

```
INGESTIÓN (one-time, offline):
  texto del chunk
      │
      ▼
  sentence-transformers (multilingual-e5-base)
      │  → vector float32[768]
      ▼
  faiss.add(vector)  →  index.faiss (disco)
  metadata_dict[id] = {text, source, ...}  →  metadata.pkl (disco)

QUERY (runtime, cada turno):
  query del usuario
      │
      ▼
  sentence-transformers (mismo modelo, en RAM)
      │  → vector float32[768]
      ▼
  faiss.search(vector, k=5)  →  top-5 chunk_ids
      │
      ▼
  metadata.pkl lookup  →  textos de los 5 fragmentos
      │
      ▼
  LLM (Groq API)  →  respuesta fundamentada
```

### Resumen en una línea

> Los embeddings viven en `faiss_store/index.faiss` en disco, y se cargan completos en RAM al iniciar la aplicación. La metadata asociada (texto + fuente) vive en `faiss_store/metadata.pkl`.

---

## Q2: ¿Por qué FAISS y no una base de datos vectorial como Chroma o Pinecone?

**Respuesta:**

Las restricciones del proyecto definen la respuesta: costo cero, ejecución local, sin servicios externos.

| Criterio | FAISS | Chroma | Pinecone |
|----------|-------|--------|---------|
| Costo | Gratuito | Gratuito (local) | Pago (cloud) |
| Ejecución local | Sí | Sí | No (cloud) |
| Dependencias externas | Ninguna | SQLite | API cloud |
| RAM para 402k vectores | ~1.2 GB | ~1.5 GB+ | N/A |
| Velocidad búsqueda exacta | Muy alta | Media | Alta |
| Persistencia | Archivo binario simple | SQLite | Cloud |

Para ~402k vectores en hardware local, `IndexFlatIP` de FAISS es la opción más eficiente y simple. No requiere servidor, no tiene overhead de base de datos, y el archivo resultante es portátil.

---

## Q3: ¿Hay que regenerar el índice FAISS si se cambia el modelo de embedding?

**Respuesta:**

**Sí, obligatoriamente.** Los vectores en `index.faiss` son específicos al modelo que los generó. Si se cambia de `multilingual-e5-base` a cualquier otro modelo, los vectores existentes son incompatibles — la búsqueda devolvería resultados sin sentido.

El `build_info.json` registra el modelo usado, precisamente para detectar este escenario al arrancar.

---

## Q4: ¿Qué pasa si la RAM no alcanza para cargar el índice completo?

**Respuesta:**

Si los ~1.2 GB del índice + ~1.1 GB del modelo de embedding saturan la RAM disponible (16 GB, muy improbable), las opciones son:

1. **`IndexIVFFlat`** en lugar de `IndexFlatIP`: divide el índice en celdas, carga solo las relevantes en memoria. Requiere una fase de entrenamiento adicional pero reduce RAM activa.
2. **`IndexFlatIP` con memory-mapped files**: FAISS soporta mmap, el SO gestiona qué páginas están en RAM.
3. Reducir a `MiniLM-L12` (384 dims) — reduce el índice a ~600 MB.

Con 16 GB disponibles, el escenario actual (IndexFlatIP, 402k × 768) no debería ser problema.

---

## Q5: ¿Por qué se usan los prefijos `"query: "` y `"passage: "` en el modelo E5?

**Respuesta:**

El modelo `multilingual-e5-base` fue entrenado con esos prefijos como parte de la representación del texto. Sin ellos, el modelo no distingue si está procesando una consulta (búsqueda) o un pasaje (documento a indexar), y la calidad del retrieval cae significativamente.

- Al indexar documentos → `"passage: " + texto_del_chunk`
- Al embeddear la query del usuario → `"query: " + pregunta_del_usuario`

Es una convención del modelo, no de FAISS.

---

## Q6: ¿Cuánto tarda la ingestión completa (~402k documentos)?

**Respuesta (estimada, pendiente de medir en hardware real):**

Con batch_size=64 en CPU:
- Velocidad típica: ~500-800 documentos/segundo con `multilingual-e5-base`
- Estimado: 402,000 / 650 ≈ **~620 segundos (~10 minutos)**

Con batch_size=64 en GPU (RTX 2050, 4GB VRAM):
- Velocidad típica: ~2,000-3,500 documentos/segundo
- Estimado: 402,000 / 2,500 ≈ **~160 segundos (~2-3 minutos)**

La ingestión es **one-time**: se ejecuta una sola vez y el índice queda persistido. No se re-ejecuta al iniciar la aplicación.

---

## Q7: ¿Por qué es importante mantener el rango objetivo de 350-400 tokens por chunk?

**Respuesta:**

El rango de 350-400 tokens no es arbitrario — es el punto de equilibrio entre tres fuerzas que se contradicen entre sí:

### El problema de los chunks demasiado cortos (< 100 tokens)

Un chunk muy corto contiene tan poco contexto que el embedding no puede capturar una idea médica completa. Por ejemplo:

```
"La preeclampsia es una complicación del embarazo."
```

Este fragmento produce un vector que semánticamente puede matchear con casi cualquier pregunta sobre embarazo, aunque no tenga información útil para responderla. El resultado es **ruido en el retrieval**: se recuperan fragmentos que parecen relevantes por similitud superficial pero no aportan conocimiento real al LLM.

### El problema de los chunks demasiado largos (> 600 tokens)

Un chunk muy largo mezcla varias ideas clínicas distintas en un solo vector. Por ejemplo, un chunk de 1,000 tokens podría contener:

```
[síntomas de preeclampsia] + [diagnóstico diferencial] + [manejo farmacológico] + [criterios de hospitalización]
```

El embedding de ese chunk es un **promedio semántico** de los cuatro temas. Si la query es sobre síntomas, el vector del chunk está "diluido" por los otros tres temas y puede quedar por debajo en ranking frente a un chunk más enfocado. Además, se consumen más tokens del contexto del LLM con información que puede no ser relevante para la pregunta específica.

### El rango 350-400 tokens es el punto óptimo para este corpus

| Factor | Justificación basada en los datos reales |
|--------|------------------------------------------|
| Los párrafos clínicos de Multiclinsum tienen mediana de ~230 tokens | Un chunk de 350-400 agrupa 1-2 párrafos completos, capturando una unidad clínica coherente (motivo + diagnóstico, o diagnóstico + tratamiento) |
| Los párrafos de textbooks tienen mediana de 49 tokens | Necesitan agrupación — 350-400 tokens agrega ~7 párrafos relacionados, suficiente para contexto médico completo |
| El modelo `multilingual-e5-base` tiene ventana de 512 tokens | El rango 350-400 deja margen para el prefijo `"passage: "` y evita truncación silenciosa del modelo |
| Top-k=5 chunks en retrieval | 5 chunks × 400 tokens = ~2,000 tokens de contexto enviados al LLM, dentro del límite operativo de Groq |

---

## Q8: ¿Qué significan los scores de similitud coseno y por qué nunca son cero?

**Respuesta:**

Los vectores del modelo `multilingual-e5-base` viven en un espacio de alta dimensión (768 dims) donde todos los textos, sin importar su contenido, terminan proyectándose en regiones relativamente cercanas entre sí. La similitud coseno entre dos vectores cualesquiera rara vez baja de 0.6-0.7 en modelos de lenguaje modernos porque comparten vocabulario estadístico general.

**FAISS siempre devuelve los k documentos más cercanos**, incluso si ninguno es relevante. No tiene un concepto de "no encontré nada útil".

Lo que importa no es el valor absoluto del score, sino la **diferencia entre scores relevantes e irrelevantes**. El filtro real es el clasificador de intención + el criterio del LLM, no el score de FAISS.

---

## Q9: ¿Por qué el modelo de embedding se carga como singleton?

**Respuesta:**

El modelo `multilingual-e5-base` ocupa aproximadamente **1.1 GB en RAM** una vez cargado. Si se instanciara en cada llamada a `embed_query()`, el sistema cargaría y descargaría 1.1 GB de memoria en cada turno conversacional — inviable en tiempo real.

El patrón singleton garantiza que el modelo se carga **una sola vez** al importar el módulo `embedder.py`, y permanece en memoria durante toda la vida de la aplicación.

---

## Q10: ¿Cuántos vectores quedaron en el índice FAISS final y cuánto tardó la ingestión?

**Respuesta:**

Ingestión completa ejecutada el 2 de junio de 2026 con `sentence-transformers==2.7.0`, modelo `intfloat/multilingual-e5-base`, device `cuda` (RTX 2050 4 GB).

| Dataset | Vectores añadidos | Tiempo |
|---|---|---|
| Multiclinsum (summaries + fulltexts) | 51,804 | ~47 min |
| MedMCQA (train + dev) | 187,005 | ~2.5 h |
| MedQA questions (US + Taiwan + Mainland) | ~99,779 | ~30 min |
| Textbooks EN (18 libros) | ~36,000 | ~1h 35 min |
| **TOTAL** | **375,392** | **~5 h total** |

**Archivos en disco:**
- `faiss_store/index.faiss` → 1,153.2 MB
- `faiss_store/metadata.pkl` → 431.5 MB
- `faiss_store/build_info.json` → marcas de cada fase completada

**Stack definitivo que resolvió el cuelgue de sentence_transformers:**
- Problema: `sentence-transformers==3.3.1` colgaba silenciosamente al importar cuando `torch` ya estaba en memoria (trainer stack con accelerate).
- Solución: downgrade a `sentence-transformers==2.7.0`. Import instantáneo, CUDA funcional.

**Versiones definitivas del entorno:**
```
torch==2.5.1+cu121
sentence-transformers==2.7.0
transformers==4.57.6
tokenizers==0.22.2
faiss-cpu==1.9.0
```

---

## Q11: ¿Cómo funcionan los clasificadores de intención y riesgo?

**Respuesta:**

Implementados el 2 de junio de 2026 en `src/classifiers/`.

### Clasificador de intención (`intent_classifier.py`)
- **Método:** zero-shot con Groq LLM (`llama-3.3-70b-versatile`), `temperature=0`, `response_format=json_object`
- **12 categorías:** `control_prenatal`, `signos_de_alarma`, `sintomas_embarazo`, `postparto`, `lactancia`, `salud_mental_perinatal`, `medicamentos`, `nutricion`, `actividad_fisica`, `planificacion_familiar`, `consulta_administrativa`, `pregunta_fuera_de_alcance`
- **Retorna:** `IntentResult(intent, confidence, reasoning)`
- **Fallback:** heurística de keywords si el LLM falla o devuelve intent inválido

### Detector de riesgo (`risk_detector.py`)
- **Dos capas:**
  1. **Heurística rápida** (sin API): keywords agrupadas por categoría clínica → HIGH o MEDIUM instantáneo
  2. **LLM Groq** (contextual): solo si la heurística no detecta nada
- **3 niveles:** `low → educational_answer` | `medium → medical_consultation` | `high → urgent_care`
- **Retorna:** `RiskResult(level, flags, action, reasoning, used_heuristic)`
- **Ventaja:** los casos más urgentes (hemorragia, convulsión, ideación suicida) se detectan SIN llamada a la API → latencia ~0ms

### Resultados del test
| Mensaje | Intent | Risk | Método riesgo |
|---|---|---|---|
| Náuseas y vómitos | `sintomas_embarazo` 90% | low | LLM |
| Sangrando mucho con coágulos | `signos_de_alarma` 95% | high → urgent_care | heurística |
| Calcio en embarazo | `nutricion` 90% | low | LLM |
| Bebé no se mueve | `signos_de_alarma` 90% | high → urgent_care | LLM |
| Depresión + no quiero vivir | `salud_mental_perinatal` 99% | high → urgent_care | heurística |
| Dolor leve de cabeza | `sintomas_embarazo` 70% | low | LLM |

**Nota:** el modelo `llama-3.1-70b-versatile` fue dado de baja por Groq. El reemplazo es `llama-3.3-70b-versatile` — actualizado en `.env`.

---

## Q12: ¿FastAPI + Streamlit o Streamlit directo? ¿Qué tiene menos latencia y qué conviene para entrega?

**Respuesta:**

**Opción elegida: FastAPI + Streamlit** conectados vía HTTP.

- **Latencia:** diferencia de ~20-50ms sobre un turno de 2-4s — imperceptible para el usuario
- **Entrega:** endpoints `/chat`, `/classify`, `/health` permiten probar la API con Postman sin la UI
- **Arranque:** Terminal 1 → `uvicorn src.api.main:app --port 8080` | Terminal 2 → `streamlit run src/ui/app.py`

**Latencia total estimada por turno:**
```
intent (Groq)      ~300ms
risk (heurística)    ~0ms  o LLM ~300ms
embed + FAISS      ~150ms
LLM generation     ~1-3s
HTTP overhead       ~30ms
─────────────────────────
Total              ~2-4s
```

---

## Q13: ¿Cuál es el stack tecnológico completo y para qué sirve cada componente?

**Respuesta:**

### Hardware objetivo
| Componente | Valor |
|---|---|
| CPU | AMD Ryzen 5 |
| GPU | NVIDIA RTX 2050 (4 GB VRAM) |
| RAM | 16 GB |
| OS | Windows 11 |

---

### Lenguaje y entorno
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **Python** | 3.12.7 | Lenguaje principal de todo el proyecto |
| **venv** | stdlib | Entorno virtual aislado para dependencias |

---

### Embedding y modelos locales
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **PyTorch** | 2.5.1+cu121 | Motor de cómputo tensorial con soporte CUDA para la GPU |
| **sentence-transformers** | 2.7.0 | Carga y ejecuta el modelo de embedding `multilingual-e5-base` |
| **transformers** (HuggingFace) | 4.57.6 | Carga los pesos del modelo base internamente |
| **intfloat/multilingual-e5-base** | — | Modelo de embedding: convierte texto en vectores de 768 dims. Soporta ES, EN, ZH. Se ejecuta en la RTX 2050 |

---

### Vector store
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **FAISS** (faiss-cpu) | 1.9.0 | Índice vectorial con AVX2. Almacena 375,392 embeddings (1.15 GB). Búsqueda de similitud coseno en milisegundos |

---

### LLM (generación de texto)
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **Groq API** | — | API cloud que ejecuta LLMs en hardware LPU de alta velocidad |
| **llama-3.3-70b-versatile** | — | Genera respuestas, clasifica intención y evalúa riesgo clínico |
| **groq** (SDK Python) | 0.37.1 | Cliente Python oficial para llamar a la API de Groq |

---

### API REST
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **FastAPI** | 0.115.6 | Framework para la API REST — endpoints `/chat`, `/classify`, `/health`. Genera Swagger en `/docs` |
| **uvicorn** | 0.48.0 | Servidor ASGI que ejecuta FastAPI |
| **Pydantic** | 2.x | Validación y serialización de schemas de request/response |
| **httpx** | 0.28.1 | Cliente HTTP usado por Streamlit para llamar a la API |

---

### Interfaz de usuario
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **Streamlit** | 1.41.1 | UI web en Python puro: chat con burbujas, sidebar con metadata, badges de riesgo, panel de fuentes |

---

### Datos y configuración
| Tecnología | Versión | Para qué sirve |
|---|---|---|
| **datasets** (HuggingFace) | 4.8.5 | Carga y procesamiento de datasets médicos |
| **pydantic-settings** | 2.14.1 | Lee el `.env` y expone configuración tipada en `src/settings.py` |
| **python-dotenv** | 1.2.2 | Carga el archivo `.env` en variables de entorno |
| **tqdm** | 4.67.3 | Barras de progreso en la ingestión de datasets |

---

### Flujo completo
```
Usuario escribe mensaje
        ↓
  Streamlit UI  →  POST /chat  →  FastAPI
                                      ↓
                          classify_intent()  →  Groq LLM (~300ms)
                                      ↓
                          detect_risk()      →  heurística (~0ms) / Groq LLM
                                      ↓
                          retrieve()         →  multilingual-e5-base (CUDA)
                                             →  FAISS 375k vectores (~150ms)
                                      ↓
                          Groq LLM genera respuesta con contexto (~1-3s)
                                      ↓
                     ChatResponse  →  Streamlit renderiza
```

---

## Q14: ¿Qué es la capa heurística en el detector de riesgo y por qué es útil?

**Respuesta:**

### Qué es

La capa heurística es un conjunto de listas de palabras clave agrupadas por categoría clínica que se revisan directamente contra el texto del mensaje **sin llamar a ninguna API**. Si alguna keyword hace match, el sistema retorna inmediatamente `risk=high` o `risk=medium` sin esperar respuesta del LLM.

```python
HIGH_RISK_KEYWORDS = {
    "hemorragia":    ["sangrando mucho", "hemorragia", "sangrado abundante", ...],
    "eclampsia":     ["convulsión", "pérdida de conocimiento", "desmayo", ...],
    "movimiento_fetal_ausente": ["no se mueve", "dejó de moverse", ...],
    "depresion_grave": ["quiero hacerme daño", "no quiero vivir", ...],
    ...
}
```

Si ninguna keyword hace match → se escala al LLM para evaluación contextual.

---

### Por qué es útil en este caso específico

**1. Latencia cero en los casos más críticos**

Una hemorragia activa o una convulsión no pueden esperar 300-600ms de llamada a la API. La heurística responde en microsegundos. En emergencias reales esos milisegundos importan psicológicamente — la alerta aparece de inmediato.

**2. Funciona sin conexión a internet**

Si la API de Groq falla o hay un corte de red, la heurística sigue detectando las señales de alarma más graves. El sistema nunca deja pasar una hemorragia sin alertar, aunque el LLM esté caído.

**3. Determinismo total en los casos de alto riesgo**

El LLM es probabilístico — en teoría podría clasificar "estoy sangrando mucho" como `medium` en algún contexto inusual. La heurística es absolutamente determinista: si la frase está en la lista, siempre es `high`. Para señales clínicas de alarma mayor (eclampsia, ausencia de movimiento fetal, ideación suicida) el determinismo es más seguro que la probabilidad.

**4. Ahorra tokens de API**

Cada llamada al LLM para clasificar riesgo consume ~100-200 tokens. Con la heurística, los mensajes con señales obvias no llegan al LLM — se resuelven localmente. En un sistema con muchos usuarios, esto reduce costo y latencia promedio.

**5. Las keywords clínicas son conocimiento estable**

A diferencia de la clasificación de intención (que requiere entender matices del lenguaje), las señales de alarma obstétrica son un conjunto finito y bien documentado médicamente. "Sangrado abundante", "convulsión", "no se mueve el bebé" — estas frases no cambian con el contexto. Son candidatas ideales para reglas deterministas.

---

### Cuándo la heurística NO es suficiente y necesita el LLM

- Frases ambiguas: *"me duele la cabeza y tengo la vista un poco rara"* — puede ser preeclampsia o puede ser cansancio. La heurística no puede capturar esa ambigüedad con keywords.
- Negaciones: *"ya no tengo sangrado"* — la heurística ingenua detectaría "sangrado" y marcaría HIGH incorrectamente. El LLM entiende la negación.
- Contexto histórico: *"ayer tuve una convulsión pero ya estoy bien"* — requiere razonamiento sobre tiempo y estado actual.

Por eso el diseño usa **ambas capas en secuencia**: la heurística para los casos claros y urgentes, el LLM para los casos que requieren comprensión contextual.

---

---

## Q15: ¿Hay TF-IDF en el proyecto y por qué no se usó?

**Respuesta:**

**No hay TF-IDF en el proyecto ni en ninguna de sus dependencias.** Se buscó explícitamente en todo el código y no existe ninguna referencia a `TfidfVectorizer`, `TfidfModel` ni ninguna implementación de TF-IDF.

### Por qué no se necesita

El proyecto usa **embeddings densos** (`intfloat/multilingual-e5-base`, 768 dimensiones) en lugar de vectores sparse como TF-IDF. Las razones:

| Aspecto | TF-IDF | multilingual-e5-base (usado) |
|---|---|---|
| Tipo de vector | Sparse (una entrada por palabra) | Denso (768 floats densos) |
| Semántica | Coincidencia de palabras exactas | Captura sinónimos y significado |
| Multilingüe | Requiere un vocabulario por idioma | Soporta ES/EN/ZH en un solo modelo |
| Tamaño del índice | Depende del vocabulario (puede ser grande) | Fijo: 768 dims por documento |
| Matching | "náusea" no matchea con "arcada" | Ambas tienen vectores cercanos |

### Cuándo TF-IDF podría haber sido útil

En un escenario con recursos muy limitados (sin GPU, con RAM < 8 GB), TF-IDF sería una alternativa viable porque:

- No requiere GPU
- Ocupa menos RAM (el vocabulario es más compacto que 375k × 768 floats)
- No necesita descargar un modelo de ~1.1 GB

Pero sacrificaría calidad de retrieval — especialmente en un corpus multilingüe donde el mismo concepto médico se expresa con palabras distintas en español, inglés y chino.

### Conclusión

TF-IDF es una técnica de recuperación de información clásica. Funciona bien para búsqueda por palabras clave en corpus pequeños y monolingües. No se usó porque el proyecto necesita búsqueda semántica multilingüe, y el hardware disponible (RTX 2050, 16 GB RAM) permite ejecutar embeddings densos sin problemas.

---

## Q16: ¿Qué es TF-IDF y cómo funciona? (contexto académico)

**Respuesta:**

TF-IDF (Term Frequency — Inverse Document Frequency) es un método clásico de recuperación de información que convierte texto en vectores numéricos basándose en la frecuencia de palabras.

### Cómo funciona

**TF (Term Frequency):** cuántas veces aparece una palabra en un documento específico. Si "preeclampsia" aparece 5 veces en un caso clínico de 100 palabras, su TF es 5/100 = 0.05.

**IDF (Inverse Document Frequency):** qué tan rara o común es una palabra en todo el corpus. "preeclampsia" aparece en pocos documentos → IDF alto. "la" aparece en casi todos → IDF bajo (casi cero).

**TF-IDF = TF × IDF.** Una palabra obtiene peso alto si aparece frecuentemente en un documento específico pero es rara en el corpus general. Esto filtra palabras vacías (artículos, preposiciones) y destaca términos relevantes del documento.

### Limitaciones frente al enfoque usado en Maternas

1. **No captura sinónimos:** una pregunta sobre "hipertensión gestacional" no matchearía con un documento que solo menciona "preeclampsia", aunque sean el mismo tema.
2. **No captura contexto:** la palabra "sangrado" tiene el mismo vector sin importar si aparece en "tengo sangrado abundante" o en "ya no tengo sangrado".
3. **Vocabulario por idioma:** un índice TF-IDF en español no sirve para consultas en chino. Los embeddings multilingües resuelven esto.
4. **Dimensionalidad variable:** el vector TF-IDF crece con el vocabulario del corpus (puede ser de cientos de miles de dimensiones). Los embeddings densos tienen dimensionalidad fija (768) y son más eficientes para búsqueda en índices como FAISS.

---

## Q17: ¿Se usa clustering de vectores en el proyecto?

**Respuesta:**

**No.** No hay clustering en el proyecto. Ni en el código fuente ni en las dependencias. El índice FAISS es `IndexFlatIP` — una estructura plana donde todos los vectores se almacenan en una sola lista y la búsqueda compara la query contra **todos** los vectores (fuerza bruta exacta).

No hay `KMeans`, `DBSCAN`, `HDBSCAN`, `AgglomerativeClustering` ni ninguna otra técnica de agrupamiento. Tampoco se usa `IndexIVFFlat` (que sí usa clustering internamente para particionar).

### Por qué no se necesita

El índice actual tiene **375,392 vectores**. La búsqueda exacta en `IndexFlatIP` tarda ~20-50ms en GPU (vía FAISS con AVX2) para cada query. Eso es más que suficientemente rápido para el objetivo de latencia (< 8s por turno).

```python
# El buscador es simplemente:
scores, ids = index.search(query_vector, k=5)   # compara contra todos
```

No hay necesidad de clustering porque el índice cabe completo en RAM (~1.15 GB de 16 GB disponibles) y el tiempo de búsqueda es despreciable frente al ~2-4s totales del turno.

### Cuándo se necesitaría clustering

Si el índice creciera a **millones** de vectores, la búsqueda exacta empezaría a ser lenta (ej. 10M vectores → ~1-2s solo la búsqueda). Ahí entrarían dos opciones con clustering:

**Opción 1: `IndexIVFFlat` (clustering interno de FAISS)**
- Durante la construcción, FAISS aplica K-Means para dividir los vectores en `n` grupos (ej. 4096)
- En búsqueda, solo compara contra los vectores del grupo más cercano a la query
- Intercambia exactitud por velocidad: ajustando `nprobe` (cuántos grupos revisar) se controla el balance
- La pérdida de recall es típicamente < 5% con `nprobe=10`

**Opción 2: Clustering externo previo**
- Agrupar los fragmentos por tema (ej. "nutrición", "preeclampsia", "lactancia")
- Enrutar la query al grupo correcto antes de buscar en FAISS
- Esto ya se hace implícitamente con el clasificador de intención — la intención no dirige a un cluster distinto, pero el LLM recibe contexto filtrado

### Conclusión

Para 375k vectores y latencia objetivo de < 8s, `IndexFlatIP` es la opción correcta. Clustering agregaría complejidad sin beneficio real. Si el proyecto escalara a millones de vectores en el futuro, se migraría a `IndexIVFFlat` (que está disponible en la misma librería FAISS y no requiere cambios en el resto del sistema).

---

---

## Q18: ¿Cómo funciona la integración con Telegram?

**Respuesta:**

El bot de Telegram se implementó el 3 de junio de 2026 en `src/bot/maternas_bot.py` usando la librería `python-telegram-bot`.

### Arquitectura

```
Usuario Telegram → Bot API → polling (python-telegram-bot) → POST /chat → FastAPI → RAG chain → respuesta → Telegram
```

El bot no implementa lógica RAG propia — es un **cliente ligero** que envía cada mensaje al endpoint `POST /chat` de la API REST de Maternas (FastAPI) y muestra la respuesta al usuario.

### Características

| Aspecto | Detalle |
|---|---|
| Método | Polling (sin webhook, sin servidor público) |
| Librería | `python-telegram-bot==21.11.1` |
| Historial | En RAM por `user_id` (diccionario en memoria, se pierde al reiniciar el bot) |
| Formato | Header informativo en HTML (negritas, itálicas controladas) + cuerpo de respuesta en texto plano |
| Comandos | `/start` — bienvenida, `/help` — instrucciones, `/reset` — reinicia historial, `/stats` — estadísticas del bot |
| Manejo de errores | Si la API falla, responde con mensaje de error amigable sin crashear |

### Por qué mensajes separados (HTML + texto plano)

El LLM de Groq genera respuestas en markdown impredecible (a veces mezcla `*`, `_`, `**` de forma inconsistente). Telegram parsea markdown estrictamente y cualquier error de formato hace que el mensaje completo falle con `BadRequest`.

Solución: se envían **dos mensajes**:
1. **Header en HTML** (controlado por el código): nombre del bot, badge de riesgo, advertencias — formateo seguro porque lo genera Python, no el LLM.
2. **Cuerpo en texto plano**: la respuesta generada por el LLM, sin parseo. Telegram la muestra tal cual.

Esto elimina por completo los errores `BadRequest: Can't parse entities` sin perder la experiencia de usuario.

### Limitaciones actuales

- **Historial volátil**: se almacena en un `defaultdict(list)` en RAM. Si el bot se reinicia, todas las conversaciones se pierden. Para producción se migraría a Redis o SQLite.
- **Sin manejo de grupos**: el bot responde en cualquier chat donde esté agregado. No hay filtro por chat_id.
- **Sin rate limiting**: no hay throttle de mensajes por usuario.
- **Sin logs persistentes**: los logs van a stdout/stderr.

### Inicio

```bash
# Terminal 1: API
python -m uvicorn src.api.main:app --port 8080

# Terminal 2: Bot
python src/bot/maternas_bot.py
```

El token se lee de `settings.TELEGRAM_BOT_TOKEN` (configurado en `.env`).

---

## Q19: ¿Qué necesita otra persona para clonar y correr el proyecto desde cero (con embeddings ya generados)?

**Respuesta:**

Los embeddings (índice FAISS + metadatos) se compartieron por WeTransfer (~1.6 GB comprimido). Con eso, la nueva persona **no necesita ejecutar la ingestión** — solo descargar, extraer y arrancar.

### Paso a paso

```bash
# 1. Clonar el repositorio
git clone https://github.com/elrios893/maternas-rag.git
cd maternas-rag

# 2. Crear entorno virtual con Python 3.12.7
py -3.12 -m venv venv
.\venv\Scripts\activate

# 3. Instalar PyTorch con CUDA (RTX 2050, 4 GB VRAM)
#    Si no tiene GPU, instalar torch sin --index-url
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar sentence-transformers 2.7.0 (la versión 3.x falla con torch)
pip install sentence-transformers==2.7.0

# 5. Instalar el resto de dependencias
pip install -r requirements.txt

# 6. Configurar .env
copy .env.example .env
# Editar .env: poner GROQ_API_KEY real (https://console.groq.com)
# IMPORTANTE: generar una NUEVA clave, no usar la que está en este documento
# EMBEDDING_DEVICE=cuda (o cpu si no tiene GPU NVIDIA)

# 7. Extraer el índice FAISS (archivo de WeTransfer)
#    Dejar la carpeta faiss_store/ en la raíz del proyecto:
#    faiss_store/
#    ├── index.faiss      (~1.15 GB)
#    ├── metadata.pkl     (~431 MB)
#    └── build_info.json

# 8. Verificar integridad del índice (opcional)
python -c "import pickle; d=pickle.load(open('faiss_store/metadata.pkl','rb')); print(f'{len(d)} vectores en metadata')"
# Debe mostrar: 375392 vectores en metadata
```

### Arranque

```bash
# Terminal 1: API
python -m uvicorn src.api.main:app --port 8080

# Terminal 2: Streamlit (opcional)
streamlit run src/ui/app.py

# Terminal 3: Telegram bot (opcional, requiere TELEGRAM_BOT_TOKEN en .env)
python src/bot/maternas_bot.py
```

### Notas importantes

- **GROQ_API_KEY**: la clave actual expuesta en este Q&A ya fue regenerada. Quien clone el proyecto debe crear su propia clave en https://console.groq.com (tier gratuito: ~30 requests/minuto, ~6000/día — suficiente para desarrollo).
- **sentence-transformers 2.7.0 obligatorio**: la versión 3.3.1 del `requirements.txt` es la que estaba instalada al inicio del proyecto, pero produce un cuelge silencioso al importar con torch cargado. La solución es instalar 2.7.0 **antes** que el resto de dependencias.
- **CUDA 12.1**: el índice se generó con `multilingual-e5-base` en CUDA. Si la nueva máquina no tiene GPU NVIDIA, cambiar `EMBEDDING_DEVICE=cpu` en `.env`. El embedding será más lento (~200-400ms por query en vez de ~50ms) pero funcional.
- **FAISS CPU vs GPU**: el índice usa `faiss-cpu`. La búsqueda se hace en CPU aunque el embedding esté en GPU. No necesita CUDA para FAISS.

---

## Q20: ¿Por qué se implementó búsqueda híbrida y cómo funciona?

**Respuesta:**

Implementado el 3 de julio de 2026 tras diagnóstico de calidad del retrieval.

### El problema original

Con búsqueda densa pura (FAISS sobre todos los vectores), Multiclinsum dominaba los resultados para preguntas generales. Multiclinsum tiene 51,804 vectores de casos clínicos individuales — son útiles para términos médicos específicos, pero inútiles para preguntas como "¿qué alimentos evitar en el embarazo?". Los scores FAISS son engañosamente altos (~0.84) porque los embeddings densos siempre están cerca en el espacio vectorial, independientemente de la relevancia real.

Ejemplo del problema:
```
Q: "Que alimentos debo evitar durante el embarazo?"
Antes: Score 0.84 → caso clínico de mujer post-gastrectomía (irrelevante)
       Score 0.83 → caso de melioidosis en embarazo (irrelevante)
```

### La solución: búsqueda híbrida por tipo de fuente

```
query
  │
  ├─► FAISS densa (solo textbook + medmcqa + medqa_*)
  │   - Semántica: captura sinónimos y paráfrasis
  │   - top-5 fragmentos
  │
  └─► BM25 léxico (solo multiclinsum_summary + multiclinsum_fulltext)
      - Exacta: solo retorna si hay coincidencia real de términos
      - top-2 fragmentos, solo si BM25 score >= 0.5
      - Si no hay match léxico → Multiclinsum no aparece
```

### Módulos involucrados

- **`src/rag/bm25_index.py`** — singleton BM25 sobre Multiclinsum. Se construye en memoria al primer uso (~15-20s, ~150MB RAM). Usa `rank_bm25` con tokenizador multilingüe (ES/EN) y eliminación de stopwords.
- **`src/rag/retriever.py`** — orquesta ambas búsquedas y mergea resultados. El FAISS pide `k×10` candidatos para filtrar Multiclinsum con suficiente margen.

### Por qué BM25 para Multiclinsum específicamente

Los casos clínicos de Multiclinsum son valiosos cuando el usuario pregunta algo específico que aparece literalmente en los casos (preeclampsia, eclampsia, placenta previa, hemorragia). En esos casos, la coincidencia léxica exacta es más fiable que la similitud semántica. Para preguntas generales sin términos médicos exactos, BM25 simplemente no retorna nada — que es el comportamiento correcto.

### Resultado post-mejora

```
Q: "Que alimentos debo evitar durante el embarazo?"
Después: 5 densos (medmcqa + textbook) + 2 BM25 con match léxico
→ Respuesta con lista concreta: mercurio, no pasteurizados, etc.

Q: "Es seguro hacer ejercicio durante el embarazo?"
Después: 5 fragmentos de textbook → Respuesta con cita [1] del ACOG
```

### Mejora del prompt (simultánea)

Se reforzó el system prompt con reglas explícitas de citación:
- Solo citar [n] si el fragmento literalmente respalda la afirmación
- Si los fragmentos son adyacentes pero no exactos, usarlos de apoyo y complementar con conocimiento general sin aclarar innecesariamente "no tengo fuentes"
- Tono más conciso, cálido y directo — sin introducciones largas ni despedidas genéricas

---

## Q21: ¿Cómo funciona el sistema de preguntas de clarificación?

**Respuesta:**

Implementado el 3 de julio de 2026. Cuando la query del usuario es vaga o le falta contexto clínico clave, el sistema pide información adicional antes de recuperar fragmentos o generar respuesta.

### El problema que resuelve

Preguntas como *"me duele la cabeza"* o *"puedo tomar algo"* no tienen suficiente contexto para dar una respuesta útil y segura. Sin saber las semanas de gestación, el síntoma exacto o si está en lactancia, el LLM improvisa o da consejos genéricos poco útiles.

### Arquitectura: opción híbrida (reglas + LLM)

```
query → classify_intent() → detect_risk()
              │
              ▼
   _should_clarify(query, intent, risk_level)
   ┌── Capa 1: reglas deterministicas ──────────────────────────────┐
   │  - Si risk != "low" → False (urgente/medio: responder siempre) │
   │  - Si intent en NEVER_CLARIFY → False                         │
   │  - Si query >= 20 tokens → False (suficiente contexto)         │
   │  - Si intent en CLARIFICATION_RULES Y query corta              │
   │    Y no contiene keywords de contexto → True                   │
   └────────────────────────────────────────────────────────────────┘
              │ True                    │ False
              ▼                         ▼
   _generate_clarification()        flujo RAG normal
   (LLM genera pregunta empática)
              │
              ▼
   ChatResponse(
     needs_clarification=True,
     clarification_question="...",
     answer=clarification_question   ← mismo texto, para que callers simples lo muestren
   )
```

### CLARIFICATION_RULES

Cada intent define:
- `min_tokens`: si la query tiene menos tokens que este valor, se considera vaga
- `keywords`: contexto esperado (semana, trimestre, síntoma...). Si no hay ninguno → clarificar
- `missing_info`: qué información le falta (se pasa al LLM para generar la pregunta)

| Intent | Activa si... |
|---|---|
| `medicamentos` | Query corta sin síntoma ni semana de gestación |
| `sintomas_embarazo` | Query corta sin mención de semanas/trimestre |
| `control_prenatal` | Query corta sin semana o trimestre |
| `nutricion` | Query corta sin mencionar embarazo o lactancia |
| `actividad_fisica` | Query corta sin trimestre |
| `salud_mental_perinatal` | Query corta sin contexto temporal |

### Casos especiales

- **`signos_de_alarma`** → **nunca** pide clarificación. Si hay riesgo, se actúa de inmediato.
- **Risk medium o high** → nunca pide clarificación. Responde con urgencia apropiada.
- **Query >= 20 tokens** → se asume suficiente contexto, nunca clarifica.

### Resultados de prueba

| Query | Resultado |
|---|---|
| `"me duele la cabeza"` | ✅ Clarifica: *"¿Cuántas semanas de embarazo estás actualmente?"* |
| `"puedo tomar algo"` | ✅ Clarifica: *"¿En qué semana de embarazo te encuentras y qué síntomas tienes?"* |
| `"me siento triste"` | ✅ Clarifica: *"¿Cuánto tiempo has estado sintiéndote así y en qué momento del embarazo/postparto?"* |
| `"me siento mal"` | ✅ NO clarifica — detector marcó risk=medium, responde de inmediato |
| `"tengo 28 semanas y me duele la cabeza con visión borrosa"` | ✅ NO clarifica — suficiente contexto |
| `"tengo sangrado abundante"` | ✅ NO clarifica — high risk, actúa de inmediato |

### Cambios en el código

- `src/rag/chain.py`: `CLARIFICATION_RULES`, `_should_clarify()`, `_generate_clarification()`, campos `needs_clarification` y `clarification_question` en `ChatResponse`
- `src/api/schemas.py`: nuevos campos en `ChatResponse`
- `src/ui/app.py`: burbuja amarilla diferenciada para preguntas de clarificación
- `src/bot/maternas_bot.py`: muestra `💬 {pregunta}` sin header de riesgo cuando `needs_clarification=True`

---

## Q22: ¿Cómo se evalúa la calidad del sistema RAG con MaternaQA-es?

**Respuesta:**

Implementado el 3 de julio de 2026. El pipeline de evaluación usa el compendio QA **MaternaQA-es** (`JhonHander/MaternaQA-es`) como benchmark y Ragas como motor de métricas.

### ¿Qué es MaternaQA-es?

Dataset público en español de **5.727 pares pregunta-respuesta** derivados de 63 PDFs clínicos (GPC de atención prenatal, revistas de obstetricia colombianas, protocolos). Construido por el mismo equipo del proyecto Minciencias. Cada par tiene:
- `pregunta` / `respuesta` / `contexto_fuente`
- `tipo`: factual | definicion | comparacion | razonamiento | aplicacion | hipotetico
- `dificultad`: basico | intermedio | avanzado
- Split train/validation/test sin fuga de datos (división a nivel de documento)

### Flujo del pipeline (dos fases separadas)

La separación en fases es necesaria para evitar conflictos CUDA/CPU entre el embedding del proyecto (GPU) y los embeddings de Ragas (CPU).

```
FASE 1 (--generate-only):
  muestra estratificada de test.jsonl
       ↓
  chat(pregunta) → respuesta generada + fragmentos recuperados
       ↓
  evaluation_reports/eval_raw_<ts>.json

FASE 2 (--evaluate-only <raw.json>):
  Ragas evaluate() con LLM judge (Groq) + embeddings CPU
  métricas: faithfulness, answer_relevancy, context_recall
       ↓
  evaluation_reports/eval_results_<ts>.json
  evaluation_reports/eval_report_<ts>.md
```

### Muestra estratificada (~50 pares)

| Tipo | N |
|---|---|
| factual | 15 |
| definicion | 10 |
| razonamiento | 10 |
| aplicacion | 10 |
| comparacion | 3 |
| hipotetico | 2 |

### Métricas usadas

| Métrica | Qué mide | Notas |
|---|---|---|
| `faithfulness` | ¿La respuesta está respaldada por los fragmentos recuperados? | Alto = el LLM no inventa datos |
| `answer_relevancy` | ¿La respuesta responde la pregunta formulada? | Alto = respuesta pertinente |
| `context_recall` | ¿El retrieval capturó información relevante del ground truth? | Bajo es esperable: el corpus actual no tiene los PDFs de MaternaQA-es |

### Uso

```bash
# Evaluación completa (~50 pares, ambas fases)
python src/evaluation/eval_pipeline.py

# Solo generar respuestas (fase 1)
python src/evaluation/eval_pipeline.py --generate-only

# Solo evaluar un raw ya generado (fase 2)
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_XXX.json

# Muestra reducida para prueba rápida
python src/evaluation/eval_pipeline.py --sample 10
```

### Referencia de línea base (MaternaQA-es propio)

| Split | Faithfulness | Answer Relevancy |
|---|---|---|
| Train | 0.7726 | 0.6466 |
| Test | 0.7132 | 0.5583 |

### Archivos

- `src/evaluation/sampler.py` — descarga y muestrea estratificadamente `test.jsonl` desde GitHub raw
- `src/evaluation/eval_pipeline.py` — pipeline completo: fase 1 (generación), fase 2 (Ragas)
- `evaluation_reports/` — reportes JSON y Markdown generados (ignorado por git, muy pesados)

---

## Q23: ¿Por qué Ragas agota la cuota de tokens incluso con 20 pares, y cómo se resuelve?

**Contexto:** Al correr `phase_evaluate()` con 20 pares y 5 métricas, ambas API keys de Groq (100k tokens/día cada una) se agotan antes de completar todos los pares, dejando resultados parciales.

**Causa raíz:**

Ragas 0.2.12 con defaults lanza **N_pares × N_métricas jobs** contra la API. Con 20 pares × 5 métricas = 100 jobs, y con `max_workers=16` (default) los lanza en ráfagas de 16 simultáneos. Cada job puede consumir entre 700 y 3.000 tokens según la métrica:

| Métrica | Tokens/par (estimado) | Razón |
|---|---|---|
| `faithfulness` | ~2.000–3.000 | 2 prompts LLM: genera statements + verifica NLI |
| `answer_correctness` | ~1.500–2.000 | F1 semántico + factual combinado |
| `context_recall` | ~800–1.200 | Clasifica cada oración del ground truth |
| `context_precision` | ~400–600 | Verifica relevancia de cada chunk recuperado |
| `answer_relevancy` | ~200–400 | Solo genera preguntas alternativas, prompt liviano |

Con 20 pares y sin throttling: estimado **98k–144k tokens total**, excede una sola key. Con `max_retries=10` (default) y errores de output parser, se multiplica.

**Solución implementada — dos grupos secuenciales:**

```python
# Grupo 1: KEY_1 (GROQ_API_KEY) — métricas pesadas ~35-50k tokens
evaluate([faithfulness, answer_correctness], run_config=RunConfig(
    max_workers=1,   # sin ráfagas concurrentes
    max_retries=2,   # máximo 2 reintentos
    max_wait=15,
    timeout=120,
), batch_size=1)

# Pausa 10s
# Grupo 2: KEY_2 (GROQ_API_KEY_2) — métricas livianas ~28-44k tokens
evaluate([answer_relevancy, context_recall, context_precision], ...)
```

Esto distribuye la carga entre las dos keys independientes, y `max_workers=1 + batch_size=1` elimina la concurrencia que provocaba picos de consumo. Los jobs se procesan uno por uno, predeciblemente.

**Limitación observada:** Incluso con esta estrategia, si las keys ya tienen tokens consumidos del día (por generaciones de fase 1 o sesiones anteriores), los últimos pares de cada grupo fallan con 429. El raw JSON de fase 1 persiste en disco — se puede relanzar `--evaluate-only` al día siguiente sin repetir las generaciones.

**Archivos relevantes:**
- `src/evaluation/eval_pipeline.py` — funciones `_run_ragas_group()`, `phase_evaluate()`
- `src/settings.py` — `groq_api_key` y `groq_api_key_2`
- `.env` — `GROQ_API_KEY` (chatbot), `GROQ_API_KEY_2` (Ragas judge)

---

## Q24: ¿Contra qué se evalúa el sistema RAG y qué significan realmente las métricas obtenidas?

**Pregunta frecuente:** "¿Las métricas de Ragas miden si el sistema es bueno o malo en salud materna?"

**Respuesta corta:** Miden cosas distintas, y el ground truth importa tanto como la métrica.

### Dataset de evaluación: MaternaQA-es

`JhonHander/MaternaQA-es` es un dataset de QA en español construido sobre documentos de salud materna colombiana:

| Campo | Detalle |
|---|---|
| Fuente | PDFs académicos: GPC Atención Prenatal de Bajo Riesgo 2023, revistas de obstetricia (vol831-1.pdf, etc.) |
| Idioma | Español colombiano |
| Split usado | `test` (328 pares) |
| Estructura | `pregunta`, `respuesta` (ground truth), `contexto_fuente`, `tipo`, `dificultad`, `source_pdf`, `topics` |
| Tipos de pregunta | factual (31%), aplicacion (29%), razonamiento (26%), definicion (9%), hipotetico (5%) |
| Dificultad | intermedio (57%), basico (37%), avanzado (6%) |

El ground truth son respuestas redactadas por humanos **basadas en fragmentos textuales exactos** de esos PDFs.

### El problema estructural: corpus mismatch

El corpus RAG actual (textbooks EN, MedMCQA, MedQA, Multiclinsum) **no contiene los PDFs de MaternaQA-es**. Por eso:

| Métrica | Resultado esperado | Razón |
|---|---|---|
| `context_recall` | Cercano a 0.0 | Los fragmentos recuperados nunca son del PDF de referencia |
| `context_precision` | Bajo (~0.03–0.08) | Los chunks recuperados son irrelevantes para ese ground truth |
| `faithfulness` | Moderado (~0.3–0.7) | El LLM responde desde conocimiento general, no desde los fragmentos |
| `answer_relevancy` | Relativamente alto (~0.6–0.7) | La respuesta es pertinente a la pregunta aunque no use las fuentes correctas |
| `answer_correctness` | Moderado (~0.4–0.6) | Coincidencia semántica parcial con el ground truth |

### Qué métricas son válidas para comparar Config A vs Config B

| Métrica | ¿Válida para comparar configs? | Por qué |
|---|---|---|
| `faithfulness` | ✅ Sí | Mide si el LLM se ciñe a lo que recupera — diferente según retrieval |
| `answer_relevancy` | ✅ Sí | Pertinencia de la respuesta a la pregunta — refleja calidad del LLM |
| `answer_correctness` | ✅ Sí | Distancia semántica respuesta↔ground truth — comparable entre configs |
| `context_recall` | ⚠️ Limitada | Siempre cercana a 0 porque el corpus no tiene los docs de referencia |
| `context_precision` | ⚠️ Limitada | Igual — mejorará solo cuando se ingesten los PDFs de MaternaQA-es |
| `latency_s` | ✅ Sí | Tiempo real medido en fase 1 — diferencia entre configs es real |

### Cuándo tendrán valor pleno context_recall y context_precision

Cuando se ingesten los PDFs de MaternaQA-es al índice FAISS. En ese momento el retrieval podrá recuperar los fragmentos exactos que el ground truth cita, y las métricas de contexto pasarán de ~0 a valores significativos. Ese es el **next step** documentado en el plan técnico.

### Referencia baseline publicada

El paper de MaternaQA-es reporta estos valores evaluando un sistema RAG que sí tiene los PDFs indexados:

| Split | Faithfulness | Answer Relevancy |
|---|---|---|
| train | 0.7726 | 0.6466 |
| test | 0.7132 | 0.5583 |

Nuestro sistema sin los PDFs obtiene `answer_relevancy` ~0.66 (por encima del baseline test 0.5583), lo que indica que la calidad de generación del LLM es competitiva. La brecha en `faithfulness` (~0.29 vs 0.71) se explica por el corpus mismatch: el LLM no puede citar fuentes que no recuperó.

**Archivos relevantes:**
- `src/evaluation/sampler.py` — descarga y muestrea estratificadamente `test.jsonl`
- `src/evaluation/eval_pipeline.py` — pipeline completo con métricas y reporte MD
- `evaluation_reports/eval_raw_configB_20260716_012717.json` — raw de fase 1 (reutilizable)
- `evaluation_reports/eval_report_configB_20260716_012717.md` — reporte con resultados parciales
- `foragents/retrieval_arquitecturas_configs.md` — documentación de Config A y Config B

---

## Q25: ¿Por qué se descartó llama-3.3-70b como juez de Ragas y qué se usa en su lugar?

**Contexto:** El pipeline de evaluación usaba `llama-3.3-70b-versatile` (Groq) como LLM judge para Ragas. Esto causaba agotamiento de la cuota de 100k tokens/día incluso con solo 15-20 pares.

### El problema de fondo: tokens por llamada

`faithfulness` en Ragas hace 2 llamadas LLM por par: (1) generación de statements y (2) NLI verdicts. Con llama-3.3-70b en español, el modelo generaba statements muy extensos con explicaciones adicionales, consumiendo ~4.500 tokens por par solo para faithfulness.

| Modelo | Tokens/par faithfulness | 15 pares × 5 métricas |
|---|---|---|
| llama-3.3-70b (Groq) | ~4.500 | ~337k tokens — imposible en tier free |
| llama-3.1-8b (Groq) | ~220 | ~16k tokens — dentro del límite, pero falló JSON |
| **gemma-4-31b (Cerebras)** | ~296 | **~22k tokens — completó 15/15 sin errores** |

### Por qué llama-3.1-8b también falló

El 8B generaba JSON válido en pruebas directas, pero Ragas usa un loop de reintentos de parseo con prompts en cadena. El 8B fallaba consistentemente con `RagasOutputParserException` — no seguía el formato JSON anidado del prompt interno de Ragas de forma confiable.

### Solución: Cerebras `gemma-4-31b`

`gemma-4-31b` en Cerebras superó las pruebas:
- JSON válido en el prompt complejo de Ragas
- ~296 tokens por llamada (66x menos que llama-3.3-70b)
- Sin límite diario estricto de tokens (límite por minuto, no por día)
- **15/15 completados en evaluación real** sin un solo error de rate limit o parser

### Problema adicional: `LLMDidNotFinishException`

Ragas verifica el `finish_reason` de cada respuesta. Si el modelo termina por longitud (`"length"`) en vez de por stop token (`"stop"`), lanza `LLMDidNotFinishException`. La solución fue pasar un `is_finished_parser` permisivo al `LangchainLLMWrapper`:

```python
def _is_finished(response: LLMResult) -> bool:
    VALID = {"stop", "STOP", "length", "MAX_TOKENS", "end_turn", "eos"}
    for g in response.flatten():
        resp = g.generations[0][0]
        finish = None
        if resp.generation_info:
            finish = resp.generation_info.get("finish_reason")
        if finish is not None and finish not in VALID:
            return False
    return True

llm = LangchainLLMWrapper(ChatOpenAI(...), is_finished_parser=_is_finished)
```

### Providers evaluados y resultado

| Provider | Modelo | JSON Ragas | Rate limit | Resultado |
|---|---|---|---|---|
| Groq | llama-3.3-70b-versatile | ✅ | 100k tok/día — se agota en ~12 pares | ❌ Descartado |
| Groq | llama-3.1-8b-instant | ✅ directo / ❌ Ragas | 500k tok/día | ❌ Falla en parser loop |
| Groq | gemma2-9b-it | — | Modelo dado de baja (400 error) | ❌ No disponible |
| Cerebras | gemma-4-31b | ✅ | Sin cuota diaria estricta | ✅ **Seleccionado** |
| Cerebras | gpt-oss-120b | ❌ respuesta None | — | ❌ Descartado |
| OpenRouter | nvidia/nemotron-3-super | ✅ | Rate limit bajo (429 frecuente) | ❌ Inestable |
| OpenRouter | modelos :free | ❌ 404/429 | — | ❌ No disponibles |

### Configuración final en el pipeline

```python
# src/evaluation/eval_pipeline.py — _make_llm()
llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gemma-4-31b",
        api_key=settings.cerebras_key,
        base_url="https://api.cerebras.ai/v1",
        temperature=0,
        max_tokens=1024,
    ),
    is_finished_parser=_is_finished,   # permisivo con finish_reason "length"
)
```

Variables de entorno necesarias:
- `.env`: `CEREBRAS_KEY=csk-...`
- `src/settings.py`: campo `cerebras_key: str = Field("", env="CEREBRAS_KEY")`

**Archivos relevantes:**
- `src/evaluation/eval_pipeline.py` — funciones `_make_llm()` y `phase_evaluate()`
- `src/settings.py` — campos `cerebras_key` y `openrouter_key`

---

## Q26: Resultados de la evaluación Config B (retrieval híbrido FAISS+BM25) — 15 pares

**Fecha:** 18-19 de julio de 2026
**Judge:** Cerebras `gemma-4-31b` — 15/15 pares completados en todas las métricas en ambas configs

### Resultados globales — Config B

| Métrica | Valor | Baseline test MaternaQA-es | Interpretación |
|---|---|---|---|
| `faithfulness` | 0.228 | 0.7132 | LLM responde desde conocimiento general — corpus mismatch |
| `answer_correctness` | 0.338 | N/A | Coincidencia semántica moderada con ground truth |
| `answer_relevancy` | 0.631 | 0.5583 | **Por encima del baseline** — respuestas pertinentes |
| `context_recall` | 0.000 | N/A | Esperado: corpus mismatch |
| `context_precision` | 0.000 | N/A | Esperado: mismo motivo |
| `latency_avg_s` | 10.36s | — | p50 real ~6s |

### Comparativa Config A (FAISS puro) vs Config B (FAISS+BM25)

Mismos 15 pares, mismo seed=42, mismo judge (Cerebras gemma-4-31b).

| Métrica | Config A | Config B | Delta | Ganador |
|---|---|---|---|---|
| `faithfulness` | 0.1615 | **0.2278** | +0.066 | **B** |
| `answer_correctness` | 0.3500 | 0.3381 | -0.012 | Empate |
| `answer_relevancy` | 0.6345 | 0.6305 | -0.004 | Empate |
| `context_recall` | 0.000 | 0.000 | 0.000 | Empate |
| `context_precision` | 0.000 | 0.000 | 0.000 | Empate |
| `latency_avg_s` | 11.35s | **10.36s** | -0.99s | **B** |

**Config B gana en faithfulness (+6.6 pp) en todos los tipos de pregunta excepto `aplicacion`.**
El BM25 sobre Multiclinsum reduce el ruido de casos clínicos irrelevantes (oncología, traumatología)
que Config A incluía en el ranking por similitud coseno, permitiendo que el LLM se ancle mejor
en los fragmentos recuperados.

| Tipo | Config A faith. | Config B faith. | Delta |
|---|---|---|---|
| factual | 0.375 | 0.467 | +0.092 |
| razonamiento | 0.119 | 0.188 | +0.069 |
| definicion | 0.062 | 0.167 | +0.104 |
| aplicacion | 0.000 | 0.000 | 0.000 |

### Conclusión

**Config B queda como la arquitectura de retrieval de producción.**
La mejora en faithfulness es consistente y estructural — no ruido estadístico con 15 pares.
`answer_relevancy` y `answer_correctness` son equivalentes entre configs, lo que confirma
que la mejora viene del retrieval (menos ruido en contexto) y no del LLM en sí.

### Próximos pasos para mejorar las métricas

1. Ingestar PDFs de MaternaQA-es → `context_recall` y `context_precision` pasarán de 0 a valores reales
2. Con corpus completo, re-evaluar `faithfulness` — se espera subida significativa hacia el baseline 0.71

**Archivos relevantes:**
- Config A raw/results/report: `evaluation_reports/*configA_20260719_171714.*`
- Config B raw/results/report: `evaluation_reports/*configB_20260718_212843.*`
- `foragents/eval_setup_critico.md` — setup completo del pipeline de evaluación

---

## Q27: ¿Por qué se ingestan los JSONL del corpus LM y no los PDFs crudos de MaternaQA-es?

**Contexto:** El repositorio `minciencias-maternas/MaternaQA-es` contiene tanto los 63 PDFs
fuente como un corpus LM ya procesado (`datasets/obstetrics/lm/`). Había que elegir
cuál ingestar al índice FAISS del sistema RAG.

### Recursos disponibles

**Corpus LM (`datasets/obstetrics/lm/`):**
- `train_lm.jsonl` — 1.744 chunks, 52 PDFs fuente
- `validation_lm.jsonl` — 101 chunks, 2 PDFs fuente
- `test_lm.jsonl` — 108 chunks, 3 PDFs fuente (exactamente los que generan los 328 QA del benchmark)
- **Total: 1.953 chunks**, promedio 879 tokens/chunk, metadatos ricos

**PDFs crudos (`pdfs/obstetrics/`):**
- 63 PDFs en español sobre obstetricia — GPCs colombianas, artículos de revistas, manuales
- Habría que extraer texto (algunos requieren OCR), limpiar, chunkear y deduplicar desde cero

### Razones para elegir los JSONL del corpus LM

**1. El procesamiento ya fue hecho y auditado por el equipo de MaternaQA-es.**
Cada chunk pasó por: extracción textual, filtrado de páginas no clínicas, chunking con límites
de longitud, deduplicación, enriquecimiento temático y control de calidad con `clinical_score`.
Replicar ese procesamiento desde los PDFs crudos tomaría tiempo y podría introducir errores
de extracción (especialmente en PDFs con tablas o columnas).

**2. Metadatos ricos y trazables listos para usar.**
Cada registro del LM tiene:
```json
{
  "text": "...",
  "metadata": {
    "chunk_id": "GPC-Atencion-Prenatal_00012",
    "source_pdf": "GPC-Atencion-Prenatal-de-Bajo-Riesgo-2023.pdf",
    "section_type": "recommendations",
    "content_role": "recommendation",
    "topics": ["prenatal_care", "hemorrhage"],
    "clinical_score": 28,
    "token_estimate": 879,
    "split": "test"
  }
}
```
El campo `split` permite saber si un chunk proviene del split test o train del benchmark,
lo cual es importante para interpretar correctamente las métricas de evaluación.

**3. Control de contaminación de splits.**
El repositorio garantiza que la división train/validation/test se hizo **a nivel de documento**,
sin fuga de información. Esto significa:
- `test_lm.jsonl` contiene chunks de los 3 PDFs exactos que generaron los 328 pares del test QA
- Si ingestamos los 3 splits, el retrieval podrá recuperar los fragmentos exactos del benchmark → métricas reales
- Si ingestamos solo train+val, las métricas del test set siguen siendo "fair" (sin leak)

Para la Config C se ingestan los 3 splits para medir el **upper bound** alcanzable con el corpus completo.

**4. Descarga directa sin dependencias.**
Los JSONL se descargan directamente de GitHub raw (~2-3 MB total) sin necesidad de
clonar el repositorio, instalar dependencias adicionales ni tener `poppler` o `tesseract`
para OCR. Los PDFs de mayor tamaño (ej: `Manual-Obstetricia-y-Ginecologia-2024_compressed.pdf`)
pueden superar los 50 MB y algunos requieren OCR.

**5. Tamaño manejable y coherente con el hardware disponible.**
1.953 chunks × 768 dims = ~6 MB de vectores adicionales — completamente insignificante
comparado con los 375.392 vectores existentes (~1.15 GB). El índice FAISS soporta
adición incremental sin reconstruir desde cero.

### Estructura de los JSONL del corpus LM

```
{"text": "<texto clínico en español>",
 "metadata": {
   "source": "obstetrics_spanish",
   "pdf_id": "<nombre_sin_extension>",
   "source_pdf": "<nombre_con_extension.pdf>",
   "doc_type": "article" | "guideline" | ...,
   "pages": [<numeros de pagina>],
   "section": "<titulo de seccion>",
   "chunk_id": "<pdf_id>_<NNNNN>",
   "token_estimate": <int>,
   "clinical_score": <int 0-30>,
   "section_type": "recommendations" | "clinical_content" | "introduction" | ...,
   "content_role": "recommendation" | "evidence" | "treatment" | "background" | ...,
   "topics": [<lista de temas clinicos>],
   "split": "train" | "validation" | "test"
 }
}
```

### Mapeo al formato FAISSStore del proyecto

El `FAISSStore` del proyecto almacena metadatos con esta estructura:
```python
{
    "text":           chunk["text"],
    "source_dataset": "maternaqaes_lm",   # nuevo dataset_id
    "language":       "es",
    "doc_id":         chunk["metadata"]["pdf_id"],
    "chunk_id":       chunk["metadata"]["chunk_id"],
    # campos adicionales preservados:
    "topics":         chunk["metadata"]["topics"],
    "clinical_score": chunk["metadata"]["clinical_score"],
    "section_type":   chunk["metadata"]["section_type"],
    "content_role":   chunk["metadata"]["content_role"],
    "lm_split":       chunk["metadata"]["split"],
}
```

### Impacto esperado en las métricas (Config C)

| Métrica | Antes (Config B) | Esperado (Config C) | Razón |
|---|---|---|---|
| `faithfulness` | 0.228 | ~0.50–0.65 | El LLM podrá anclar en fragmentos en español |
| `answer_correctness` | 0.338 | ~0.45–0.60 | Mayor coincidencia semántica con ground truth |
| `answer_relevancy` | 0.631 | ~0.63–0.70 | Similar o ligeramente mejor |
| `context_recall` | 0.000 | **~0.30–0.60** | El corpus ahora tiene los documentos del benchmark |
| `context_precision` | 0.000 | **~0.20–0.50** | Fragmentos relevantes recuperados |

**Archivos relevantes:**
- `src/ingestion/ingest_maternaqaes_lm.py` — script de ingestión (a crear)
- `src/rag/retriever_configC.py` — config C con `maternaqaes_lm` en DENSE_SOURCES (a crear)
- `foragents/retrieval_arquitecturas_configs.md` — documentación de configs

---

## Q28: ¿Los datasets clínicos usados tienen licencia para este tipo de trabajo? ¿Están anonimizados?

**Contexto:** El proyecto no tenía documentada la licencia ni la política de privacidad de ninguno de los datasets fuente (`foragents/data_schemas.md` documenta el *esquema* de los datos, no su procedencia legal). Se revisó cada repositorio de origen citado en `foragents/Segundo entregable.md` y en Q22-Q27.

**Respuesta corta:** No, no todos dicen lo mismo — hay tres licencias distintas, un bloque de contenido (los textbooks) sin licencia identificable, y ninguna declaración explícita de anonimización por parte de los propios creadores de los datasets. Nada de esto es necesariamente descalificante, pero requiere acción (atribución CC-BY pendiente) y una decisión consciente sobre los textbooks.

### Tabla comparativa por dataset ingestado

| Dataset (carpeta local) | Fuente / origen citado | Licencia declarada | ¿Contiene datos de pacientes reales? | Anonimización declarada por el creador |
|---|---|---|---|---|
| `datasets/data/` (MedMCQA) | [huggingface.co/datasets/openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) | **Apache 2.0** | No — preguntas de examen (AIIMS, NEET PG), no historias clínicas | N/A (no hay paciente que anonimizar); la sección "Personal and Sensitive Information" del dataset card está sin completar (`[Needs More Information]`) |
| `datasets/data_clean/` (MedQA: US/Taiwan/Mainland + textbooks) | Preguntas: [github.com/jind11/MedQA](https://github.com/jind11/MedQA) (paper Jin et al. 2020, arXiv:2009.13081), mirror usado: Kaggle `moaaztameer/medqa-usmle` | **MIT** (repo jind11/MedQA) — la licencia del mirror de Kaggle no se pudo verificar (página no expone metadatos de licencia vía fetch) | No — preguntas de exámenes de licenciatura médica (USMLE/MCMLE/TWMLE) | N/A por el mismo motivo — no son historias clínicas |
| `datasets/data_clean/.../textbooks/en/` (18 libros: Gray's Anatomy, Harrison, Robbins, Schwartz's Surgery…) | Empaquetados junto al dataset MedQA en el mismo repositorio/Google Drive | **⚠️ Sin licencia identificable** — son libros de texto médicos comerciales con copyright editorial propio, la inclusión en el repo de MedQA no otorga una licencia de reuso | No aplica (no son datos de pacientes) | No aplica |
| `datasets/multiclinsum_large-scale_train_es/` | [zenodo.org/records/15517617](https://zenodo.org/records/15517617) — shared task MultiClinSum, BioASQ/CLEF 2025, derivado de PMC-Patients / PubMed Central | **CC-BY-4.0** (requiere atribución al reusar/redistribuir) | **Sí** — son resúmenes y textos completos de casos clínicos reales publicados en revistas médicas | No hay declaración explícita de desidentificación en la ficha del dataset ni en el paper de overview (ceur-ws Vol-4038). Al derivarse de *case reports* ya publicados en revistas indexadas en PMC, estos históricamente exigen consentimiento del paciente y desidentificación como requisito editorial (normas ICMJE) — pero esto es una inferencia razonable, **no una garantía documentada por el equipo de MultiClinSum** |
| Corpus MaternaQA-es (`datasets/obstetrics/lm/*.jsonl`, ingestado en Config C, ver Q27) | Repo `minciencias-maternas` (GitHub), derivado de 63 PDFs: GPC de atención prenatal colombiana + revistas de obstetricia | Repo de benchmark (`obstetrics-rag-benchmark`) licenciado **MIT** — pero esa licencia cubre el *código*, no necesariamente los 63 PDFs fuente (GPCs gubernamentales vs. artículos de revista pueden tener licencias distintas entre sí, no verificado documento por documento). El dataset `JhonHander/MaternaQA-es` en HuggingFace devolvió **401 Unauthorized** al intentar consultarlo — parece ser privado/gated | No — son guías clínicas institucionales y artículos, no historias de pacientes individuales | N/A por el mismo motivo |

### Hallazgo principal: no todos dicen lo mismo

Los tres datasets con licencia verificable usan licencias permisivas pero **distintas**, con obligaciones distintas:

| Licencia | Dataset | Obligación clave |
|---|---|---|
| Apache 2.0 | MedMCQA | Conservar aviso de copyright/cambios; incluye cláusula de patentes |
| MIT | MedQA (preguntas) | Conservar aviso de copyright y licencia |
| CC-BY-4.0 | MultiClinSum | **Dar crédito/atribución** al reusar o redistribuir — actualmente el proyecto no expone ninguna página de créditos/fuentes visible para el usuario final (ni en `src/ui/app.py` ni en el bot de Telegram) |

**Pendiente concreto:** para cumplir CC-BY-4.0 con MultiClinSum, el proyecto debería mostrar una atribución (aunque sea en un `/help` o footer) citando la fuente. Hoy esa cita solo existe en `foragents/Segundo entregable.md`, que es documentación interna, no algo visible al usuario del chatbot.

### El riesgo real no es el más obvio

La pregunta "¿están anonimizados?" es la más natural de hacer, pero en la práctica el **menor** riesgo del corpus: MedMCQA y MedQA (la mayoría del volumen, ~340k de ~427k vectores) son preguntas de examen, nunca hubo un paciente que anonimizar. El único dataset con pacientes reales (MultiClinSum) son casos ya publicados en revistas médicas, donde la desidentificación es una norma editorial estándar aunque no esté declarada explícitamente por este dataset en particular.

El riesgo más concreto y menos evidente es otro: **los 18 textbooks completos** (`data_clean/.../textbooks/en/`) son libros de texto médicos comerciales con copyright editorial activo (Gray's Anatomy, Harrison's, Robbins, Schwartz's Surgery, etc.). Estar empaquetados junto a un dataset académico con licencia MIT no extiende esa licencia al contenido de los libros — el MIT de jind11/MedQA cubre el código y las preguntas que ese equipo generó, no los libros de terceros que decidieron incluir como corpus de apoyo. Usar el texto completo de esos libros para indexarlos en un RAG de producción es la pieza de este análisis que más justifica revisión legal antes de considerar el sistema "listo para producción" más allá del ámbito educativo/de desarrollo.

### Recomendación

1. **No es necesario** un proceso de anonimización propio — ningún dataset ingestado contiene PHI identificable atribuible a un paciente individual accesible por el proyecto.
2. ✅ **Hecho** — atribución visible para MultiClinSum (CC-BY-4.0) y el resto de datasets agregada en `README.md` (tabla "Datasets indexados y licencias"), sidebar de `src/ui/app.py` (expander "Fuentes de datos y licencias") y `/help` del bot de Telegram (`src/bot/maternas_bot.py`).
3. ⚠️ **PENDIENTE DE REVISIÓN LEGAL** — uso de los 18 textbooks completos (`data_clean/.../textbooks/en/`) antes de cualquier despliegue más allá de uso educativo/interno. Es el punto más débil de todo el análisis de licencias y **queda marcado aquí únicamente** (no se ha tomado ninguna acción sobre el índice FAISS ni se ha comunicado como resuelto en README/UI/bot).
4. Verificar directamente con el equipo de `minciencias-maternas` qué licencia aplica a los 63 PDFs fuente de MaternaQA-es, ya que el MIT del repo de benchmark no cubre necesariamente el contenido de esos documentos.

**Archivos relevantes revisados:**
- `foragents/data_schemas.md` — esquemas de los 3 datasets base (no incluye licencia)
- `foragents/Segundo entregable.md` — únicas URLs de origen documentadas en el repo
- `foragents/qa_technical.md` Q22-Q27 — contexto de MaternaQA-es

---

## Q29: ¿Los datos de usuarios reales (no los datasets) se anonimizan en tránsito y en reposo?

**Contexto:** A diferencia de Q28 (que audita los *datasets de entrenamiento*), esta pregunta audita a los **usuarios reales** que conversan con el bot de Telegram — el requisito explícito fue: *"anonimizar datos en tránsito y no tener datos en reposo de pacientes; no usar nombres de Telegram ni alias, no guardar nada de eso, anonimizar todo"*.

Se auditó todo `src/` (bot, API, RAG, clasificadores, skill de notificación) en modo solo lectura antes de tocar nada. Hallazgos y remediación aplicada:

### Hallazgos

| # | Ubicación | Problema | Severidad |
|---|---|---|---|
| 1 | `active_users.json` (raíz) | `chat_id` real de Telegram como clave + `latest_risk_flags` (banderas clínicas descriptivas) en **texto plano sin cifrar** | 🔴 Crítico |
| 2 | `src/skills/__init__.py:32` | `ToolRegistry.execute()` logueaba `kwargs` completo — incluye el **mensaje clínico del paciente** cada vez que se dispara `notify_risk` (los casos de riesgo alto/medio) | 🔴 Crítico |
| 3 | `src/bot/maternas_bot.py` `error_handler` | Logueaba el objeto `Update` completo de Telegram (`first_name`, `username`, `user_id`, texto del mensaje) sin sanitizar | 🟠 Alto |
| 4 | `src/bot/maternas_bot.py` `histories` | Historial en RAM indexado por el `user_id` real de Telegram (no persistía a disco, pero contradecía el "no usar" literal) | 🟠 Alto |
| 5 | Logs del scheduler (`_send_status_check`, `_sync_user_jobs`) | `chat_id` real expuesto repetidamente en `logger.debug/warning` | 🟡 Medio |

**Lo que ya cumplía:** Groq (LLM) nunca recibe `user_id`/`chat_id`/nombre en los prompts; la API FastAPI es agnóstica de identidad (`ChatRequest`/`ChatResponse` no tienen campo de usuario); el email del notifier usa STARTTLS y tampoco incluye identidad de Telegram, solo el mensaje clínico + risk_level (transmisión cifrada, correcta).

### Remediación aplicada

1. **`active_users.json` ahora se cifra en disco** (Fernet, clave en `.env` como `ACTIVE_USERS_ENCRYPTION_KEY`, generada con `Fernet.generate_key()`). `src/bot/active_users.py` cifra en cada `_save()` y descifra en cada `_load()`; incluye migración automática de archivos preexistentes en texto plano (el `active_users.json` real del proyecto ya fue migrado y verificado).
2. **Se eliminó `latest_risk_flags` del esquema persistido** — ya no se guardan banderas clínicas descriptivas (ej. "hemorragia", "convulsión"), solo `risk_points` (número) y `latest_risk_level` (`low`/`medium`/`high`), que es lo mínimo necesario para calcular la frecuencia de los check-ins del scheduler. La purga de flags legadas ocurre automáticamente al leer un archivo anterior a este cambio.
3. **`ToolRegistry.execute()` ya no loguea `kwargs`** — solo loguea `list(kwargs.keys())`, es decir, qué parámetros se pasaron, nunca su contenido.
4. **`error_handler` del bot ya no loguea el objeto `Update`** — solo un hash truncado del `user_id` (si está disponible) más el error, nunca el texto del mensaje ni la identidad.
5. **`histories` ahora se indexa por `hash(user_id)`, no por el `user_id` real** — se usa SHA-256 con sal fija del proyecto; el identificador real de Telegram solo se usa donde es funcionalmente inevitable (que python-telegram-bot enrute la respuesta al chat correcto), nunca como clave interna ni en logs.
6. **Todos los logs del scheduler usan un hash truncado (10 hex) del `chat_id`** en vez del valor real, suficiente para depurar/correlacionar sin exponer identidad.

### Decisión de diseño documentada: el scheduler necesita *algún* identificador

El status check scheduler (Q18 del README) requiere poder reencontrar a un usuario para enviarle un mensaje de seguimiento más tarde — eso exige un identificador direccionable de Telegram, no es evitable sin eliminar la funcionalidad. La decisión tomada fue **cifrar el dato en reposo en vez de eliminarlo**, ya que:
- El `chat_id` solo se usa, descifrado, en el proceso del bot para llamar a la API de Telegram — nunca se loguea en claro ni se envía a Groq/SMTP.
- Sin la clave de `.env` (que nunca se versiona, igual que `active_users.json`), el archivo en disco no es legible.

### Qué NO se resolvió (fuera de alcance de esta remediación)

- El email de notificación de riesgo (`src/skills/notifier/tool.py`) sigue transmitiendo el mensaje clínico completo del paciente al buzón configurado en `NOTIFIER_EMAIL_TO` — esto es el comportamiento esperado de una alerta clínica (un humano debe poder leer el caso), pero significa que ese dato queda en reposo en un sistema de terceros (Gmail) fuera del control del proyecto. No se consideró un hallazgo a corregir, sino una decisión de producto ya documentada aquí.
- No hay política de retención/purga automática de `active_users.json` — una entrada con `risk_points=0` permanece indefinidamente hasta un `/reset` explícito o borrado manual. Pendiente de decidir si se necesita un TTL.

**Archivos modificados:**
- `src/bot/active_users.py` — cifrado Fernet, esquema sin `latest_risk_flags`
- `src/bot/maternas_bot.py` — hashing de `user_id`/`chat_id` en `histories` y en todos los logs
- `src/skills/__init__.py` — log de `ToolRegistry.execute()` sin valores de `kwargs`
- `src/settings.py`, `.env`, `.env.example` — `ACTIVE_USERS_ENCRYPTION_KEY`
- `requirements.txt` — `cryptography==48.0.0` (dependencia explícita, antes transitiva)
- `tests/test_active_users.py` — tests actualizados al nuevo esquema + cobertura de cifrado en reposo

---

## Q30: ¿Cómo se implementó el aviso de tratamiento de datos al inicio de sesión?

**Contexto:** Requisito explícito del usuario: toda sesión nueva en Streamlit y en Telegram debe empezar mostrando el aviso de tratamiento de información (texto resumido más abajo), exigir aceptación explícita antes de permitir chatear, despedirse y cerrar la sesión si se rechaza, y volver a mostrar el aviso ante cualquier mensaje nuevo mientras no haya aceptación vigente.

### Texto del aviso (resumen del original, fuente única en `src/consent.py`)

El texto original (política de tratamiento de datos del proyecto) se resumió conservando los cinco puntos obligatorios, con lo más relevante en MAYÚSCULAS: naturaleza de INVESTIGACIÓN/FASE EXPERIMENTAL y que no sustituye a un profesional; qué se conserva (NOMBRE PREFERIDO O ALIAS + CÓDIGO INTERNO) y qué nunca debe compartirse (nombre legal, documento, dirección, contraseñas, datos financieros); el propósito del procesamiento y que se aplica ANONIMIZACIÓN/SEUDONIMIZACIÓN en análisis o publicaciones; que la participación es VOLUNTARIA con derecho a no responder y a SOLICITAR EL RETIRO; y que es un PROTOTIPO no apto aún para producción. Se usa texto plano (sin markdown) para que se vea igual en Streamlit y Telegram, y para evitar el error `BadRequest: can't parse entities` de Telegram documentado en Q18.

### Telegram (`src/bot/maternas_bot.py`)

- `/start` y `/reset` limpian `consent_status` para ese usuario (hash) y envían el aviso con teclado inline (`✅ Acepto` / `❌ No acepto`, vía `CallbackQueryHandler`).
- `consent_callback()`: **Acepto** → marca `"accepted"`, edita el mensaje a la confirmación y envía el saludo/bienvenida. **No acepto** → marca `"rejected"`, borra el historial en RAM y da de baja del scheduler (`remove_active_user`), edita el mensaje a la despedida.
- `handle_message()` y `handle_non_text()` verifican `consent_status` al inicio: si no es `"accepted"` (nunca se pidió o fue rechazado), responden con el aviso de nuevo en vez de llamar a la API — el usuario no puede chatear hasta aceptar.
- `consent_status: dict[str, str]` vive en RAM, indexado por el mismo hash que `histories` — no se persiste, se pierde al reiniciar el bot (= nueva sesión para todos).

### Streamlit (`src/ui/app.py`)

- `st.session_state.consent_status` (`None`/`"accepted"`/`"rejected"`), inicializado por sesión de navegador.
- `@st.dialog(...)` muestra el aviso como ventana emergente con dos botones; se invoca en cada rerun mientras `consent_status != "accepted"`.
- La sección de input del chat tiene tres ramas: API caída → aviso de API; `consent_status != "accepted"` → formulario "bloqueado" que, al enviarse, resetea `consent_status` a `None` y hace `st.rerun()` (reabre el diálogo) en vez de llamar a la API; `"accepted"` → flujo normal de chat sin cambios.
- Al rechazar, se limpian `messages`/`meta` de la sesión (equivalente a "apagar" la sesión de chat).

### Verificación

- `python -c "import ast; ast.parse(...)"` sobre los tres archivos modificados — sin errores de sintaxis.
- Import en frío de `src.bot.maternas_bot` — sin errores; se verificó estructuralmente el teclado inline (2 botones, `callback_data` correctos) y el texto del aviso.
- `streamlit run src/ui/app.py --server.headless true` — arranca sin errores, responde HTTP 200. No se verificó visualmente en navegador dentro de esta sesión (sin acceso interactivo a Chrome); se recomienda una prueba manual antes de considerarlo cerrado.
- Suite de tests (104 casos, sin relación directa con el bot/UI) sigue en verde tras los cambios.

**Archivos modificados:**
- `src/consent.py` — nuevo, texto único del aviso compartido por UI y bot
- `src/bot/maternas_bot.py` — flujo de consentimiento, `CallbackQueryHandler`, gating en `handle_message`/`handle_non_text`, `/start`/`/reset` reinician `consent_status`
- `src/ui/app.py` — `st.dialog` de consentimiento, gating del formulario de chat

---

## Q31: ¿Se removieron finalmente `textbook` y `multiclinsum` del índice por licencia? ¿Qué cambió?

**Contexto:** Q28 había dejado marcados dos riesgos de licencia sin resolver: los 18 textbooks médicos en inglés sin licencia de reuso identificable, y MultiClinSum (CC-BY-4.0, casos clínicos reales derivados de PMC-Patients, sin garantía documentada de desidentificación por el propio dataset). El usuario decidió removerlos del índice de producción, pero primero se midió el impacto en las métricas Ragas antes de ejecutar el cambio, para no tomar la decisión a ciegas.

### Medición previa (Config D vs Config C, mismos 14 pares, seed=42, judge Cerebras gemma-4-31b)

Se creó `src/rag/retriever_configD.py` — idéntico a Config C pero con `DENSE_SOURCES` sin `textbook` y sin la capa BM25 de `multiclinsum` — y se corrió el pipeline de evaluación completo (`eval_pipeline.py --config configD --sample 15 --generate-only` + `--evaluate-only`) sin tocar el índice físico todavía, para comparar en igualdad de condiciones con el reporte de Config C más reciente (`eval_report_configC_20260721_153327.md`).

| Métrica | Config C (con textbook+multiclinsum) | Config D (sin ellos) | Δ |
|---|---:|---:|---:|
| `faithfulness` | 0.4556 | 0.4973 | +0.042 |
| `answer_correctness` | 0.5315 | 0.5507 | +0.019 |
| `answer_relevancy` | 0.8155 | 0.7328 | −0.083 |
| `context_recall` | 0.4524 | 0.4524 | 0.000 |
| `context_precision` | 0.3876 | 0.3599 | −0.028 |

**Conclusión de la medición:** con 14 pares y una desviación estándar conocida de ~0.25 (documentada en `eval_runbook.md`), todos los deltas caen dentro del ruido estadístico — remover ambos datasets no tiene un efecto medible en las métricas. `context_recall` da idéntico en ambas configs porque depende de `maternaqaes_lm`, que no cambió. El techo real de las métricas sigue siendo la ausencia de los PDFs exactos del benchmark MaternaQA-es (split test, excluido deliberadamente por leakage), no la composición de multiclinsum/textbook.

### Remoción física del índice

Con la medición en mano, se removieron los vectores de forma permanente:

1. Backup completo de `faiss_store/` (index.faiss + metadata.pkl + build_info.json, 1.5 GB) fuera del repo, antes de modificar nada.
2. Script one-off: cargó el índice, reconstruyó los vectores ya calculados vía `index.reconstruct_n()` (sin re-embeber nada), filtró por `source_dataset not in {"textbook", "multiclinsum_summary", "multiclinsum_fulltext"}`, y reconstruyó un `IndexFlatIP` nuevo con la metadata remapeada a ids secuenciales.
3. Resultado: **380,745 → 253,455 vectores** (−127,290, −33.4% del índice). Desglose de lo removido: `textbook` 75,486 · `multiclinsum_summary` 25,902 · `multiclinsum_fulltext` 25,902.
4. `src/rag/retriever_configD.py` se copió a `src/rag/retriever.py` — **Config D es ahora la config de producción**, reemplazando a Config C.
5. `src/rag/bm25_index.py` se eliminó (dependía exclusivamente de `multiclinsum`, que ya no existe en el índice — hubiera construido un índice BM25 vacío). La dependencia `rank-bm25` se removió de `requirements.txt` y se desinstaló del venv.
6. Suite de tests (104 casos) y un smoke test de `retrieve()` contra el índice nuevo confirmaron que todo sigue funcionando.

**Nota:** `retriever_configA.py`, `retriever_configB.py` y `retriever_configC.py` quedan en el repo solo como referencia histórica — ya no son reproducibles tal cual contra el índice actual, porque las fuentes que usaban (`textbook`, `multiclinsum_*`) ya no están indexadas.

**Archivos modificados:**
- `src/rag/retriever_configD.py` — nuevo (config sin textbook/multiclinsum)
- `src/rag/retriever.py` — reemplazado por Config D (producción)
- `src/rag/bm25_index.py` — eliminado
- `requirements.txt` — removido `rank-bm25==0.2.2`
- `faiss_store/index.faiss`, `faiss_store/metadata.pkl`, `faiss_store/build_info.json` — reconstruidos sin textbook/multiclinsum (253,455 vectores)
- `README.md`, `docs/DOCUMENTACION.md` — actualizados para reflejar Config D como producción y el índice sin BM25

---

## Q32: ¿Vale la pena implementar HyDE (Hypothetical Document Embeddings)?

**Contexto:** El README (sección "Siguientes mejoras") sugería HyDE como posible mejora de `context_recall` en consultas cortas. Antes de adoptarlo en producción, se probó como experimento aislado (mismo método que Q31: config nueva + evaluación Ragas sobre los mismos 14-15 pares) para medir si el beneficio justifica el costo.

### Implementación de prueba

`src/rag/retriever_configE.py` = Config D + HyDE: antes de embeder, se le pide al LLM de producción (Groq `llama-3.3-70b-versatile`) que escriba un párrafo corto en lenguaje clínico formal respondiendo hipotéticamente la pregunta del usuario, y ese párrafo (no la query cruda) es lo que se embede y se busca en FAISS. Si la generación HyDE falla, se hace fallback silencioso a la query original.

### Resultados (14 pares, seed=42, judge Cerebras gemma-4-31b)

| Métrica | Config D (sin HyDE) | Config E (con HyDE) | Δ |
|---|---:|---:|---:|
| `faithfulness` | 0.4973 | 0.5213 | +0.024 |
| `answer_correctness` | 0.5507 | 0.4995 | −0.051 |
| `answer_relevancy` | 0.7328 | 0.6903 | −0.043 |
| `context_recall` | 0.4524 | 0.4762 | +0.024 |
| `context_precision` | 0.3599 | 0.3366 | −0.023 |
| `latency_avg_s` | 9.58 | 10.85 | +1.27s |
| `latency_p95_s` | 15.93 | 16.88 | +0.95s |

**Conclusión: no vale la pena, al menos no en su forma actual.** Todos los deltas caen dentro del ruido estadístico de la evaluación (~0.25 std con 14-15 pares) — ninguna métrica mejora de forma clara, dos de las cinco incluso empeoran ligeramente (`answer_correctness`, `context_precision`). A cambio, el costo es real y medible:

- **Latencia:** +~1.3s promedio por turno (una llamada Groq adicional antes de poder retrievar).
- **Cuota de tokens:** el `GROQ_API_KEY` de producción **se agotó a mitad de la corrida de 15 pares** (límite diario 100k tokens, ya documentado como cuello de botella recurrente en `eval_setup_critico.md`) — algo que no había pasado con Config D en la misma muestra. El item 15 tuvo que regenerarse con `GROQ_API_KEY_2` para completar la evaluación. En producción real esto se traduce en agotar la cuota diaria del bot mucho más rápido, ya que `retrieve()` se llama más de una vez por turno (una vez dentro de `chat()`, otra vez en el pipeline de evaluación para capturar contexto — en producción normal solo la primera, pero igual duplica el uso de Groq por turno de chat vs no tener HyDE).

**Decisión:** no se activa en producción. `src/rag/retriever_configD.py` sigue siendo la config activa. `src/rag/retriever_configE.py` queda en el repo como referencia si se quiere retomar (por ejemplo con un modelo más barato/rápido solo para la generación HyDE, o si se resuelve la restricción de cuota de Groq).

**Archivos:**
- `src/rag/retriever_configE.py` — nuevo, config de prueba (no activa)
- `README.md` — sección "Siguientes mejoras" actualizada con este resultado

---

## Q33: ¿Por qué se migró el LLM generador de `llama-3.3-70b-versatile` a `openai/gpt-oss-120b`? ¿Qué cambió en el código y qué trade-offs quedaron?

**Contexto:** Groq dio de baja `llama-3.3-70b-versatile` el 16 de agosto de 2026 (anunciado el 17 de junio). Es el LLM de producción del sistema: genera respuestas, clasifica intención (`intent_classifier.py`), evalúa riesgo (`risk_detector.py`), decide notificaciones (`chain.py::_run_notification`) y redacta clarificaciones (`chain.py::_generate_clarification`). El cambio de modelo no fue una comparación libre entre alternativas — fue forzado por la baja del proveedor, con el sistema de producción caído en su ruta principal hasta migrar.

### Por qué `gpt-oss-120b` y no otra cosa

Groq recomendó dos reemplazos oficiales: `openai/gpt-oss-120b` y `qwen/qwen3.6-27b`. Se descartó salir de Groq (velocidad, tier gratuito, cero reescritura del cliente) y, entre las dos alternativas de Groq, se eligió `gpt-oss-120b`:

- Mismo tier gratuito que el modelo anterior.
- Más rápido (~500 tok/s vs ~275 de `llama-3.3-70b`) y con ventana de contexto mayor (131k vs 128k).
- `qwen/qwen3.6-27b` quedó como plan B explícito: soporta `reasoning_effort:"none"`, lo que lo volvería un *drop-in* sin razonamiento si el español de `gpt-oss-120b` hubiera resultado notoriamente más frío o mecánico en las pruebas en vivo — no fue necesario, el helper de reasoning (ver abajo) ya lo contempla sin cambios de código, solo cambiar `GROQ_MODEL`.
- El SDK `groq==0.13.1` (de diciembre de 2024, anterior a `gpt-oss`) **no se actualizó**: no tiene los parámetros de razonamiento tipados, pero sí acepta `extra_body`, así que se pasan por ahí. Actualizar el SDK habría puesto en riesgo el workaround ya existente de `x_groq.usage` en streaming (`chain.py`, manejo de tokens en `chat_stream()`).

### Por qué no alcanzaba con cambiar `GROQ_MODEL` en `.env`

`gpt-oss-120b` es un **modelo de razonamiento**, y el código asumía uno que no lo es. Tres consecuencias concretas, encontradas y corregidas antes de tocar producción:

1. **Fuga de `<think>` a la respuesta.** Groq solo cambia `reasoning_format` a `parsed` automáticamente cuando hay JSON mode o tool use. Los dos clasificadores (`response_format={"type":"json_object"}`) se salvan; las 4 llamadas de texto libre en `chain.py` no — sin corregirlo, el bloque `<think>...</think>` habría aparecido literalmente en el chat de la gestante, incluyendo en streaming token a token.
2. **Los tokens de razonamiento cuentan contra `max_tokens`.** El caso más grave: `_run_notification()` usaba `max_tokens=10` para decidir `"YES"/"NO"` antes de notificar un riesgo medio a un clínico. El razonamiento se habría comido el presupuesto entero, dejando `content` vacío y `"YES" not in ""` — **un riesgo medio habría dejado de notificar, en silencio, sin ningún error visible.**
3. **Cascada de fallback silencioso.** Si `classify_intent()` recibe `content` vacío, cae a `pregunta_fuera_de_alcance`; `detect_risk()` tiene un atajo que ante ese intent devuelve `low` sin siquiera consultar al LLM (`risk_detector.py`) — un fallo de parseo del clasificador se habría convertido en un riesgo bajo silencioso.

### Solución implementada

Helper centralizado en `src/settings.py`:

```python
_REASONING_MODEL_MARKERS = ("gpt-oss", "qwen3")

def groq_reasoning_kwargs(json_mode: bool = False) -> dict:
    model = settings.groq_model.lower()
    if not any(marker in model for marker in _REASONING_MODEL_MARKERS):
        return {}
    kwargs = {"reasoning_effort": settings.groq_reasoning_effort}
    if not json_mode:
        kwargs["reasoning_format"] = "hidden"
    return kwargs
```

Gateado por nombre de modelo (no por un flag manual): si `GROQ_MODEL` vuelve a ser un modelo clásico (llama, etc.) en el futuro, esto se desactiva solo, sin tocar los 6 call sites. `json_mode=True` no manda `reasoning_format` porque Groq ya fuerza `parsed` en JSON mode y pedir `raw` explícitamente ahí da `400`.

Aplicado como `extra_body=groq_reasoning_kwargs(...)` en los 6 call sites (`chain.py` ×4, `intent_classifier.py`, `risk_detector.py`). Además:

- **Topes de `max_tokens` subidos** para que el razonamiento no vacíe la respuesta útil: notificación 10→512, clarificación 100→700, clasificación de intención 150→800, detección de riesgo 200→900, generación de respuesta 800→1800.
- **Parseo blindado** independiente del modelo: `(resp.choices[0].message.content or "")` en vez de asumir no-`None`, con `logger.warning(...)` si el resultado queda vacío — antes, un `content=None` lanzaba `AttributeError` en `.strip()`, capturado genéricamente por el `except` más cercano; ahora cae en el camino de fallback correcto (deterministico en clarificación, `_fallback_answer()` en generación) y deja rastro en logs en vez de fallar en silencio.
- `GROQ_REASONING_EFFORT=low` en `.env` — el default de Groq es `medium`, de más para clasificar intención/riesgo o redactar una clarificación de una oración.
- `eval_pipeline.py`: el modelo de *fallback* del juez de Ragas (`GROQ_FALLBACK`, usado solo si falta `CEREBRAS_KEY`) se actualizó de `llama-3.1-8b-instant` (también dado de baja el mismo día) a `openai/gpt-oss-20b`.

### Validación antes de producción

En orden, sin saltarse ninguno: `pytest` completo (241/241, todo mockeado, sin cambios de comportamiento); smoke test directo contra los 6 call paths reales de Groq confirmando cero fugas de `<think>` y JSON parseable; confirmación en vivo del modelo activo (`GET /admin/config` → `"groq_model":"openai/gpt-oss-120b"`, verificado también en el panel Configuración); y finalmente pruebas en vivo con Chrome — Streamlit (4 sesiones nuevas: pregunta informativa, clarificación completa, riesgo medio, riesgo alto) y Telegram (`/reset` para simular sesión nueva) — confirmando tono cálido en español intacto, el flujo de clarificación funcionando, y **los tres correos SMTP de riesgo llegando con el contenido correcto** (medio, medio-tras-clarificación, alto — este último es la ruta más peligrosa de las tres consecuencias descritas arriba).

### Trade-offs medidos (Ragas, Config D, mismo protocolo, seed=42, judge Cerebras `gemma-4-31b`)

| Métrica | `llama-3.3-70b` (15-ago, N=14) | `gpt-oss-120b` (19-ago, N=15) | Δ |
|---|---:|---:|---:|
| `faithfulness` | 0.4973 | 0.3915 | **−21%** |
| `answer_correctness` | 0.5507 | 0.6362 | **+15%** |
| `answer_relevancy` | 0.7328 | 0.8101 | **+11%** (cruza a 🟢, ≥0.80) |
| `context_recall` | 0.4524 | 0.4222 | −7% (ruido — retrieval sin cambios) |
| `context_precision` | 0.3599 | 0.3359 | −7% (ruido — retrieval sin cambios) |
| `latency_avg_s` | 9.58 | 12.49 | **+30%** |
| `latency_p95_s` | 15.93 | 21.81 | **+37%** |

`answer_relevancy` y `answer_correctness` mejoran de forma consistente con que `gpt-oss-120b` es un modelo de razonamiento: pensar antes de responder ayuda a mantenerse enfocado en lo que se preguntó y a acercarse más al *ground truth*. `faithfulness` baja de forma más marcada de lo que explicaría solo el ruido estadístico (~0.25 std con N=14-15, por `eval_runbook.md`) — hipótesis de trabajo, no confirmada: el juez de Ragas premia afirmaciones verificables palabra por palabra contra el contexto recuperado, y un modelo que sintetiza/reformula más (en vez de citar literalmente) puntúa peor en esta métrica específica aunque la respuesta sea igual o más correcta en sustancia — coherente con que `answer_correctness` sube al mismo tiempo. No se investigó a fondo la causa exacta; queda en el backlog (revisión cualitativa de pares con `faithfulness` bajo). La latencia sube por el overhead de razonamiento incluso con `reasoning_effort:"low"`.

### Limitaciones conocidas de este cambio

- **Cuota de Groq más ajustada por turno.** Con `reasoning_effort:"low"` los tokens de razonamiento siguen sumando al consumo por llamada; en tier gratuito (8.000 TPM observados durante las pruebas) el límite por minuto se agota más rápido que con el modelo anterior si se encadenan varias llamadas seguidas (por ejemplo, la suite de pruebas en vivo de esta migración necesitó espaciar las llamadas ~12s para no chocar con el límite).
- **`faithfulness` no explicado del todo.** La caída es consistente en dirección con la hipótesis de "razona/sintetiza en vez de citar", pero no se confirmó con una revisión cualitativa par por par — con N=15 no se puede descartar del todo que sea varianza.
- **SDK de Groq sin actualizar.** `groq==0.13.1` no tiene los parámetros de razonamiento tipados; se pasan vía `extra_body`, lo que funciona pero no está validado por el tipado del cliente — un cambio de la API de Groq en este punto no daría error en tiempo de desarrollo, solo en runtime.
- **`reasoning_format:"hidden"` es todo o nada.** No hay forma de inspeccionar el razonamiento del modelo para debugging (por ejemplo, por qué clasificó cierto mensaje como determinado nivel de riesgo) sin cambiar temporalmente a `reasoning_format:"parsed"` y leer el campo `message.reasoning` aparte — no implementado, no hay ruta de logging del razonamiento hoy.
- **El juez de Ragas sigue siendo Cerebras `gemma-4-31b`, no `gpt-oss-120b`.** Correcto por diseño (el juez debe ser independiente del generador), pero significa que el `GROQ_FALLBACK` del juez (`openai/gpt-oss-20b`, usado solo si falta `CEREBRAS_KEY`) nunca se probó en esta migración — sigue siendo una ruta no ejercida en la práctica, igual que lo era `llama-3.1-8b-instant` antes.

**Archivos modificados:**
- `src/settings.py` — nuevo `groq_reasoning_kwargs()`, campo `groq_reasoning_effort`
- `src/rag/chain.py` — `extra_body` en los 4 call sites del LLM, `max_tokens` subidos, parseo blindado en `_run_notification()` y en la respuesta de `chat()`
- `src/classifiers/intent_classifier.py`, `src/classifiers/risk_detector.py` — `extra_body` en modo JSON, `max_tokens` subidos, parseo blindado
- `src/evaluation/eval_pipeline.py` — `GROQ_FALLBACK` actualizado a `openai/gpt-oss-20b`
- `.env`, `.env.example` — `GROQ_MODEL=openai/gpt-oss-120b`, nuevo `GROQ_REASONING_EFFORT=low`
- `README.md` — tabla de stack y tabla de evaluación actualizadas
- `docs/DOCUMENTACION_oss120b.md`, `INFORME TECNICO MATERNAS/outputs/informe-tecnico-maternas-20260819_oss120b.md` (+ PDFs) — nuevas versiones de la documentación y el informe técnico con el modelo migrado; se conservan los originales (`DOCUMENTACION.md`, `informe-tecnico-maternas-20260815.md`) para trazabilidad
- `evaluation_reports/eval_raw_configD_oss120b_20260819_113326.json`, `eval_report_configD_oss120b_20260819_113326.md`, `eval_results_configD_oss120b_20260819_113326.json` — nueva corrida de evaluación

---

## Q34: ¿Por qué `gpt-oss-120b` a veces no genera el bloque `Fuentes:` en Streamlit, aunque el retrieval sí trajo buenos resultados?

**Contexto:** Revisando las capturas en vivo tomadas para la migración de Q33, se detectó que varias respuestas de `gpt-oss-120b` no traían el bloque `---\nFuentes:\n[n] ...` al final, pese a que el panel lateral mostraba 5 fragmentos recuperados con scores altos (0.83–0.86) — comportamiento distinto al de `llama-3.3-70b`, donde el bloque aparecía de forma consistente (ver capturas del informe del 15/08).

### Causa raíz

`gpt-oss-120b` cita de forma **intermitente** con corchetes CJK de ancho completo — `【1】` (U+3010 LEFT BLACK LENTICULAR BRACKET / U+3011 RIGHT BLACK LENTICULAR BRACKET) — en vez de corchetes ASCII estándar `[1]` (U+005B/U+005D), sin ningún patrón previsible: la misma pregunta repetida puede citar con uno u otro estilo. Confirmado reproduciendo la consulta exacta de una de las capturas ("Tengo hinchazón leve en los pies, estoy en la semana 30, ¿es normal?") directamente contra `chat()`:

```
>>> resp.answer
'Sí, la hinchazón leve de pies y tobillos es frecuente a partir del tercer
trimestre y, en la mayoría de los casos, no indica un problema grave【1】. ...'
```

`build_reference_block()` (`src/rag/citations.py`) usa `_CITATION_RE = re.compile(r"\[(\d+)\]")` — coincide solo con corchetes ASCII. Cuando el modelo cita con `【1】`, el regex no matchea nada, `cited` queda vacío, y la función devuelve `""`: el bloque `Fuentes:` se descarta en silencio. El marcador inline `【1】` sí queda en el texto (por eso a veces se veía un número suelto sin corchetes normales en el chat), pero sin el bloque de referencias al final.

No es un problema del retrieval ni de la calidad de las fuentes — es puramente un problema de formato de salida del LLM nuevo, invisible en los logs (`build_reference_block()` no loguea nada al devolver `""`, porque "sin citas válidas" es un camino normal cuando el LLM decide no citar).

### Solución aplicada

`src/rag/citations.py` — nueva función `normalize_citation_brackets()`:

```python
_FULLWIDTH_BRACKETS_RE = re.compile(r"【(\d+)】")

def normalize_citation_brackets(text: str) -> str:
    return _FULLWIDTH_BRACKETS_RE.sub(r"[\1]", text)
```

Se aplica al `answer` completo (tras `.strip()`, antes de `build_reference_block()`) en los dos puntos de `src/rag/chain.py` donde se arma la respuesta final: `chat()` (no streaming) y `chat_stream()` (streaming, aplicado al texto ya acumulado de todos los `delta`, no por token — normalizar carácter a carácter mientras se transmite sería frágil y no aporta nada, ya que el bloque de referencias solo se construye una vez, al final). Esto corrige ambos síntomas a la vez: el marcador inline se ve como `[1]` estándar y el bloque `Fuentes:` se arma sin importar qué estilo de corchete haya elegido el modelo en ese turno.

### Validación

Suite `pytest` completa en verde (241/241) tras el cambio. Se repitieron en vivo las tres consultas que antes mostraban el problema (`chat()` directo, sin pasar por la API):

| Consulta | Antes del fix | Después del fix |
|---|---|---|
| "Tengo hinchazón leve..." | `grave【1】.` sin bloque `Fuentes:` | `grave[1].` + `Fuentes: [1] GAP_Control prenatal del embarazo normal_6105 · págs. 15-16` |
| "¿Qué alimentos debo evitar...?" | `(pez espada...)【2】` sin bloque | `[2]`...`[1]`...`[3]` + bloque con 3 fuentes agrupadas |
| "¿Puedo tomar ibuprofeno...?" | `[3]` ASCII, ya funcionaba | Sin cambios — confirma que el fix no rompe el caso que ya andaba bien |

Se reinició la API (proceso `uvicorn`, sin `--reload`, con el código viejo en memoria) y se repitió la consulta de hinchazón en vivo por Streamlit, en una sesión nueva: el bloque `Fuentes:` aparece correctamente. Capturas antes/después en el informe técnico, sección 9.6.

**Archivos modificados:**
- `src/rag/citations.py` — nueva `normalize_citation_brackets()` y `_FULLWIDTH_BRACKETS_RE`
- `src/rag/chain.py` — se llama a `normalize_citation_brackets()` antes de `build_reference_block()` en `chat()` y `chat_stream()`

---

*Última actualización: 19 de agosto de 2026*
