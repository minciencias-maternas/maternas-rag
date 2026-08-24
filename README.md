# Maternas — Chatbot RAG de Salud Materna

Chatbot conversacional basado en arquitectura RAG orientado a madres gestantes. Clasifica la intención del usuario, evalúa el riesgo clínico y genera respuestas fundamentadas en literatura médica.

> Proyecto de investigación — Convocatoria 890 Minciencias · Institución Universitaria de Envigado

---

## Stack

| Capa | Tecnología |
|---|---|
| Embedding | `intfloat/multilingual-e5-base` (768 dims, ES/EN/ZH) en CUDA |
| Vector store | FAISS `IndexFlatIP` — 253,455 vectores |
| LLM | `openai/gpt-oss-120b` vía Groq API |
| API | FastAPI + uvicorn |
| UI | Streamlit |
| Bot | Telegram (`python-telegram-bot`) |

## Datasets indexados y licencias

| Dataset | Contenido | Licencia | Atribución |
|---|---|---|---|
| **MedMCQA** | 187,005 preguntas médicas de examen (EN) | [Apache 2.0](https://huggingface.co/datasets/openlifescienceai/medmcqa) | openlifescienceai/medmcqa |
| **MedQA** | Preguntas de licenciatura médica US/Taiwan/Mainland (EN/ZH) | [MIT](https://github.com/jind11/MedQA) | Jin et al., 2020 — *What Disease does this Patient Have?* (arXiv:2009.13081) |
| **MaternaQA-es** (corpus obstétrico) | Chunks de GPCs y revistas de obstetricia colombianas | MIT (repo de benchmark) | `minciencias-maternas/obstetrics-rag-benchmark` |

> **MultiClinSum** (casos clínicos reales, CC-BY-4.0) y los **18 textbooks médicos en inglés** de `data_clean/.../textbooks/en/` (Gray's Anatomy, Harrison's, Robbins, etc. — sin licencia de reuso identificada) fueron **removidos del índice FAISS** por riesgo de licencia/datos de pacientes — ver `foragents/qa_technical.md` (Q28, Q31). El impacto en las métricas Ragas fue nulo (dentro del margen de ruido de la evaluación).

## Estructura

```
src/
├── ingestion/      # formatters, chunkers, embedder, FAISS store, scripts de ingestión
├── classifiers/    # intent_classifier.py, risk_detector.py
├── rag/            # retriever.py, chain.py
├── api/            # main.py (FastAPI), schemas.py
├── ui/             # app.py (Streamlit)
└── settings.py
foragents/          # plan técnico y Q&A del proyecto
```

## Inicio rápido

```bash
# 1. Entorno
python -m venv venv
.\venv\Scripts\activate       # Windows
pip install -r requirements.txt
# Si tienes GPU NVIDIA (recomendado), reemplaza el torch CPU-only por la build CUDA:
pip install torch==2.12.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 2. Configuración
cp .env.example .env          # completar GROQ_API_KEY y rutas de datasets

# 3. Ingestión (una sola vez, ~5h en GPU)
python src/ingestion/run_ingestion.py

# 4. Arrancar
python -m uvicorn src.api.main:app --port 8080   # Terminal 1
streamlit run src/ui/app.py                       # Terminal 2
python src/bot/maternas_bot.py                    # Terminal 3 (opcional, Telegram bot)
```

UI disponible en `http://localhost:8501` · API docs en `http://localhost:8080/docs` · Bot Telegram: `python src/bot/maternas_bot.py`

## Interfaz Streamlit

Cada sesión nueva (pestaña/navegador nuevo) abre una ventana emergente (`st.dialog`) con el aviso de tratamiento de datos (`src/consent.py`) y botones **✅ Acepto** / **❌ No acepto**, antes de habilitar el chat:

- **Acepta:** el diálogo se cierra y el chat queda habilitado normalmente.
- **No acepta:** se muestra un mensaje de despedida y el envío de mensajes queda bloqueado.
- **Sin decisión o tras rechazar:** cualquier intento de enviar un mensaje vuelve a abrir el diálogo en lugar de consultar la API — no se puede chatear hasta aceptar.

El estado (`consent_status`) vive en `st.session_state`, por lo que es por sesión de navegador y se pierde al recargar la página o abrir una nueva pestaña.

## Bot Telegram

El bot permite chatear con Maternas directamente desde Telegram usando polling.

```bash
python src/bot/maternas_bot.py   # Terminal 3 (requiere API ya corriendo)
```

Comandos: `/start` — muestra el aviso de datos (nueva sesión) · `/help` — instrucciones · `/reset` — reinicia historial, da de baja del scheduler y vuelve a exigir el aviso · `/stats` — estadísticas del bot.

Historial conversacional en RAM por usuario, indexado por un hash del `chat_id` de Telegram (no por el identificador real), persistido siempre en memoria (incluso durante una pregunta de clarificación) para no perder el contexto de síntomas entre turnos, y descartado al reiniciar el bot — nunca se escribe a disco. La respuesta de Maternas se envía como texto plano.

El token se configura en `.env` como `TELEGRAM_BOT_TOKEN`.

### Aviso de tratamiento de datos

Toda sesión nueva (bot recién iniciado, `/start` o `/reset`) muestra primero el aviso de tratamiento de información (`src/consent.py`) con botones **✅ Acepto** / **❌ No acepto**, antes de procesar cualquier mensaje:

- **Acepta:** se envía la confirmación y el mensaje de bienvenida; el chat queda habilitado.
- **No acepta:** se envía un mensaje de despedida, se borra su historial y se da de baja del scheduler — la sesión queda cerrada.
- **Sin decisión o tras rechazar:** cualquier mensaje nuevo del usuario vuelve a mostrarle el aviso en lugar de procesarse — no puede chatear hasta aceptar.

El estado de aceptación vive solo en RAM (`consent_status`, indexado por el mismo hash que `histories`) y se pierde al reiniciar el bot.

### Privacidad

- **Sin datos en reposo identificables:** el historial conversacional vive solo en RAM y se indexa por un hash SHA-256 del `chat_id`, nunca por el identificador real. El único dato persistido a disco (`active_users.json`, ver abajo) está cifrado.
- **Sin nombres ni alias de Telegram:** el `first_name`/`username` de Telegram nunca se loguea ni se guarda — se usa una única vez para el saludo posterior a la aceptación del aviso y no vuelve a aparecer en ningún registro interno.
- **Logs sin contenido clínico:** los mensajes del usuario (que pueden incluir síntomas) nunca se escriben a un log — ni en errores del bot, ni al ejecutar la notificación de riesgo alto.
- Detalle completo de la auditoría y las decisiones tomadas en `foragents/qa_technical.md` (Q29, Q30).

### Status Check Scheduler

El mismo proceso del bot envía mensajes automáticos de seguimiento a cada
usuario que le haya escrito ("¿cómo te encuentras hoy?"), con una frecuencia
que depende de su nivel de riesgo acumulado — usa la `JobQueue` nativa de
`python-telegram-bot` (basada en APScheduler), sin proceso ni terminal aparte.

- `src/bot/active_users.py` — registro persistido **cifrado** (Fernet,
  `ACTIVE_USERS_ENCRYPTION_KEY` en `.env`) en `active_users.json`
  (raíz del proyecto, **no versionado**). Solo guarda `chat_id` + nivel de
  riesgo agregado (`low`/`medium`/`high`) — no persiste banderas clínicas
  descriptivas. Cada usuario acumula `risk_points` (`low=0, medium=+3,
  high=+10`, tope 50, decae 1 punto por hora de inactividad).
- Cada 15s el bot sincroniza un job de `JobQueue` por usuario según su
  `risk_points` actual: `LOW` → `STATUS_CHECK_INTERVAL_LOW_SECONDS` (60s dev),
  `MEDIUM` → `..._MEDIUM_SECONDS` (45s), `HIGH` → `..._HIGH_SECONDS` (30s).
  En producción se sugiere subir a minutos/horas (ver `.env.example`).
- Los checks solo se envían si la API respondió OK en el último turno del
  bot (congruencia — nunca manda mensajes automáticos si el sistema está caído).
- `/reset` da de baja al usuario del registro, deteniendo sus check-ins.

## Flujo por turno

```
query → classify_intent() → detect_risk()
              │
              ├─ ¿query vaga? → pregunta de clarificación al usuario
              │
              └─ Búsqueda densa FAISS
                   (medmcqa + medqa_* + maternaqaes_lm)
                       │
                       └─ Groq LLM → respuesta con citas [n]
```

- **Riesgo HIGH** → alerta inmediata + notificación email + respuesta de urgencia
- **Riesgo MEDIUM** → respuesta con recomendación médica (LLM decide si notificar)
- **Riesgo LOW** → respuesta educativa con citas a la fuente

## Streaming y citas por documento

`POST /chat/stream` devuelve la respuesta como NDJSON (una línea JSON por evento: `status` → `meta` → `delta`* → `done`, o `error`), y la UI de Streamlit la renderiza token a token en vez de esperar el turno completo. La detección/notificación de riesgo corre en paralelo en un hilo de fondo, sin bloquear la generación del texto.

Las citas dentro de la respuesta usan marcadores numerados `[n]` en el texto, con una lista agrupada **"Fuentes:"** al final que muestra el **nombre del documento o dataset de origen** de cada fragmento — no `fragmento [n]` genérico. `src/rag/citations.py` resuelve ese nombre (`document_name()`): usa el archivo fuente cuando existe, y para el ~73% del índice sin archivo asociado (MedMCQA/MedQA) degrada a un nombre descriptivo tipo `"MedMCQA · {subject} — {topic}"`.

`POST /chat` (sin streaming) sigue funcionando igual que antes — lo sigue usando el bot de Telegram y el pipeline de evaluación — y ahora también expone `notified`/`needs_clarification`/`clarification_question` en la respuesta.

## Retrieval

El índice tiene 253,455 vectores de tres fuentes (`medmcqa`, `medqa_us`/`medqa_taiwan`/`medqa_mainland`, `maternaqaes_lm`), todas recuperadas por búsqueda densa FAISS (similitud coseno). `textbook` y `multiclinsum_*` fueron removidos del índice por riesgo de licencia — ver `foragents/qa_technical.md` (Q31).

## Preguntas de clarificación

Cuando la query es corta y le falta contexto clínico (semana de gestación, síntoma específico, si está en lactancia), el sistema pide esa información antes de recuperar fragmentos:

- `"me duele la cabeza"` → *"¿Cuántas semanas de embarazo estás actualmente?"*
- `"puedo tomar algo"` → *"¿Para qué síntoma y en qué semana de gestación estás?"*
- `"me siento triste"` → *"¿Cuánto tiempo llevas así y estás embarazada o en postparto?"*

**Nunca** se pide clarificación para `signos_de_alarma` ni cuando el riesgo es medium/high — en esos casos siempre se responde de inmediato.

## Skill System

Arquitectura extensible de herramientas (`tools`) agrupadas en habilidades (`skills`). Cada skill vive en `src/skills/<nombre>/` y expone tools registrables con nombre, descripción, esquema de parámetros y función asociada.

### Notifier (notificaciones por email)

Envío de alertas SMTP cuando se detecta riesgo clínico alto:
- **Risk HIGH** → notificación automática siempre
- **Risk MEDIUM** → una llamada adicional al LLM decide si amerita notificar
- **Risk LOW** → no notifica

Configuración en `.env` con prefijo `NOTIFIER_*`. Por defecto remitente y destinatario son la misma cuenta Google (`maternasrag@gmail.com`).

### Crear una skill nueva

```
src/skills/mi_skill/
├── __init__.py
├── skill.py      # Skill(name, desc, tools=[ToolSpec(...)])
└── tool.py       # Implementación de la función
```

Registrar en `ToolRegistry` y ejecutar desde `chain.py` vía `ToolRegistry.execute("tool_name", ...)`.

## Panel de administración

El panel (`/documents*`, `/admin*`, protegido por `X-Admin-Token`, ver `.env` → `ADMIN_API_TOKEN`) tiene, además de gestión de documentos y evaluaciones, dos secciones para operar el sistema sin tocar `.env` a mano ni la terminal:

### Configuración editable

10 variables se pueden editar desde la página **Configuración** de Streamlit:

| Variable | Aplica |
|---|---|
| `GROQ_MODEL`, `GROQ_API_KEY` | En caliente, sin reiniciar nada |
| `NOTIFIER_ENABLED`, `NOTIFIER_EMAIL_TO`, `NOTIFIER_SMTP_USER`, `NOTIFIER_SMTP_PASSWORD` | En caliente, sin reiniciar nada |
| `TELEGRAM_BOT_TOKEN`, `STATUS_CHECK_INTERVAL_LOW/MEDIUM/HIGH_SECONDS` | Requiere reiniciar el bot (botón en la misma página) |

Los primeros 6 campos los lee el propio proceso de la API en cada llamada, así que `PATCH /admin/config` los aplica mutando el singleton `settings` en memoria — visible al instante desde cualquier módulo — y los persiste en `.env` con `dotenv.set_key()` para que sobrevivan un reinicio. `GROQ_API_KEY` además fuerza reconstruir los tres clientes Groq cacheados (`chain.py`, `intent_classifier.py`, `risk_detector.py`) vía `reset_client()`. Los últimos 4 solo los lee el proceso del bot de Telegram al arrancar (es otro intérprete de Python) — por eso la respuesta indica `requires_bot_restart` y la UI ofrece reiniciarlo ahí mismo.

**Endurecido contra inyección** (revisado explícitamente para esta función, ya que escribe en `.env`): `PATCH /admin/config` no acepta un nombre de variable libre — expone un campo Pydantic fijo por variable con `extra="forbid"`, así que no hay forma de pedirle que toque `ADMIN_API_TOKEN` ni nada fuera de esa lista (un campo desconocido es 422, no se ignora en silencio). Cualquier valor con `\n`/`\r` se rechaza antes de escribir, como segunda barrera contra inyectar una línea nueva en el archivo. Los secretos (`groq_api_key`, `notifier_smtp_password`, `telegram_bot_token`) nunca se devuelven en ninguna respuesta — ni en el `GET`, ni en el `PATCH` — solo viaja un booleano "configurado", igual que `secrets_configured` ya hacía.

### Bot de Telegram como subproceso administrado

El bot se puede iniciar, detener y reiniciar desde la página **Consola** de Streamlit (`src/api/bot_supervisor.py`): la API lo lanza como subproceso hijo (`subprocess.Popen`), con PID, hora de inicio, uptime y un buffer de log en memoria. Al apagar la API se detiene también el subproceso del bot (`lifespan()` en `main.py`), para no dejarlo huérfano.

`status()` distingue "detenido a propósito" (por el botón Detener/Reiniciar) de "se cerró solo" (`crashed=true`, p. ej. un token inválido) — en Windows `Popen.terminate()` no manda una señal real, así que sin esta distinción ambos casos lucían idénticos.

### Consola

Página nueva con dos paneles de solo estado, refresco manual (sin auto-poll, igual que el resto del panel):

- **API**: uptime del proceso y últimas líneas de log (`GET /admin/logs`, buffer en memoria de hasta 1000 líneas colgado del logger raíz). No tiene control de reinicio — auto-reiniciar el propio proceso que atiende la request es frágil y de bajo valor cuando quien arrancó `uvicorn` ya lo controla por terminal.
- **Bot de Telegram**: badge de estado (corriendo / detenido / se cerró solo), PID, uptime, botones Iniciar/Detener/Reiniciar y sus últimas líneas de log.

## Estructura

```
src/
├── ingestion/      # formatters, chunkers, embedder, FAISS store, scripts de ingestión
├── classifiers/    # intent_classifier.py, risk_detector.py
├── rag/            # retriever.py, chain.py (+ chat_stream), citations.py
├── api/            # main.py (FastAPI, incl. /chat/stream), schemas.py,
│                   # routes_documents.py, routes_admin.py (config editable + logs),
│                   # routes_bot.py, bot_supervisor.py (subproceso del bot)
├── ui/             # app.py (Streamlit), client.py, views/ (chat, config, consola, ...)
├── bot/            # maternas_bot.py (Telegram)
├── skills/         # notifier/ (email SMTP), base ToolRegistry
└── settings.py
foragents/          # plan técnico y Q&A del proyecto (Q1–Q21)
```

## Evaluación automática

El sistema se evalúa con el framework **Ragas** sobre el benchmark **MaternaQA-es** (split test, 328 pares QA de obstetricia en español). La evaluación opera en dos fases con modelos independientes:

| Fase | Modelo | Rol |
|---|---|---|
| 1 — Generación | `openai/gpt-oss-120b` (Groq) | El chatbot RAG genera respuestas reales |
| 2 — Evaluación | `gemma-4-31b` (Cerebras) | Juez externo independiente vía Ragas |

```bash
# Fase 1: generar respuestas
python src/evaluation/eval_pipeline.py --config configD --sample 15 --generate-only

# Fase 2: evaluar con Ragas
python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_configD_<TIMESTAMP>.json
```

Ver guía completa en `foragents/eval_runbook.md`.

### Mejores resultados obtenidos — Config D (producción actual)

Configuración: FAISS densa pura sobre `medmcqa` + `medqa_*` + `maternaqaes_lm` (sin `textbook` ni `multiclinsum`, removidos por licencia), evaluado sobre 14 pares sin preguntas de clarificación.

| Métrica | Config D | Config C (con textbook+multiclinsum) | Baseline MaternaQA-es |
|---|:---:|:---:|:---:|
| `faithfulness` | **0.497** | 0.456 | 0.713 |
| `answer_relevancy` | **0.733** | 0.816 | 0.558 |
| `answer_correctness` | **0.551** | 0.532 | — |
| `context_recall` | **0.452** | 0.452 | — |
| `context_precision` | **0.360** | 0.388 | — |
| `latency_avg_s` | ~9.6 s | ~10.2 s | — |

Remover `textbook` y `multiclinsum` no cambió las métricas de forma significativa (deltas dentro del margen de ruido, std~0.25 con 14-15 pares) — la brecha en `faithfulness` frente al baseline se debe principalmente a que los PDFs exactos que generaron el benchmark MaternaQA-es (split test) no están indexados, para evitar data leakage.

## Siguientes mejoras

- **Reranker cross-encoder local** (`BAAI/bge-reranker-v2-m3`) — recuperar k=20 candidatos y reranquear a top-5 antes del LLM; mejora `context_precision` sin costo de API ni latencia significativa
- **System prompt restrictivo** — instruir al LLM a responder solo con información de los fragmentos recuperados y declarar explícitamente cuando no tiene suficiente contexto; sube `faithfulness` en pares donde el retrieval ya es correcto

~~**HyDE (Hypothetical Document Embeddings)**~~ — probado en `src/rag/retriever_configE.py` y evaluado contra Config D (14 pares, ver `foragents/qa_technical.md` Q32): todos los deltas de métricas caen dentro del margen de ruido de la evaluación (ninguna mejora de forma clara, dos empeoran ligeramente), mientras que el costo es real — +~1.3s de latencia por turno y agotó la cuota diaria de Groq a mitad de una corrida de solo 15 pares. **No se adoptó en producción.**

---

## Advertencia de uso y puesta en producción

Este software corresponde a un prototipo desarrollado con fines de investigación. Su disponibilidad en un repositorio, ambiente de demostración o prueba piloto no implica que se encuentre autorizado para operar como servicio productivo, comercial, clínico, psicológico, educativo o institucional.

Antes de su puesta en producción, la organización responsable deberá realizar una revisión jurídica, ética, técnica y de seguridad que incluya, como mínimo:

* Política de tratamiento de datos personales.
* Aviso de privacidad y mecanismos de autorización.
* Tratamiento especial de datos sensibles.
* Evaluación de impacto en privacidad.
* Plazos de conservación y eliminación.
* Gestión de derechos de los titulares.
* Revisión de transferencias y transmisiones nacionales o internacionales.
* Contratos con proveedores de inteligencia artificial, alojamiento, mensajería y analítica.
* Controles de acceso, cifrado, registros y respuesta a incidentes.
* Revisión y aprobación por las instancias jurídicas y éticas que correspondan.

Los responsables de una implementación productiva deberán verificar el cumplimiento de la legislación vigente en la jurisdicción en la que se utilice la herramienta.
