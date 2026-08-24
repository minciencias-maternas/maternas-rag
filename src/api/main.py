"""
main.py — API FastAPI del chatbot Maternas.

Endpoints:
    GET  /health        — estado del servicio, vectores cargados
    POST /chat          — turno completo del chatbot (intent + risk + RAG + LLM)
    POST /classify      — solo clasificación (intent + risk, sin generar respuesta)

Arrancar:
    uvicorn src.api.main:app --reload --port 8000

Docs interactivas:
    http://localhost:8000/docs

Panel de administración (/documents*, /admin*, /admin/bot*): requiere
ADMIN_API_TOKEN en .env (ver src/api/auth.py) y un único worker de
uvicorn — el FAISSStore singleton y sus locks son por-proceso, así que
con --workers > 1 cada proceso mantiene una copia divergente del índice
(y bot_supervisor.py mantendría un subproceso del bot por worker).

/admin/config acepta además un PATCH para un conjunto fijo de variables
(ver UpdateConfigRequest en schemas.py) que se aplican en caliente y se
persisten en .env. /admin/bot/* arranca/detiene el proceso del bot de
Telegram como subproceso hijo de esta API (ver bot_supervisor.py).
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    SourceDoc,
)
from src.api import bot_supervisor, usage_sessions
from src.api.routes_admin import router as admin_router
from src.api.routes_bot import router as bot_router
from src.api.routes_documents import router as documents_router
from src.classifiers.intent_classifier import classify_intent
from src.classifiers.risk_detector import detect_risk
from src.rag.chain import chat as rag_chat, chat_stream as rag_chat_stream
from src.rag.citations import document_name
from src.rag.retriever import _get_store, source_path
from src.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Buffer de logs en memoria — alimenta GET /admin/logs (consola del panel
# admin). Cuelga del logger raíz para capturar todo lo que ya se loguea
# vía logging.basicConfig arriba, sin duplicar formato ni handlers.
# ---------------------------------------------------------------------------

_api_started_at = datetime.now(timezone.utc)
_log_buffer: deque[str] = deque(maxlen=1000)


class _DequeHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _log_buffer.append(self.format(record))


_deque_handler = _DequeHandler()
_deque_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
logging.getLogger().addHandler(_deque_handler)


# ---------------------------------------------------------------------------
# Startup: cargar FAISS al arrancar (no en el primer request)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando índice FAISS al arrancar...")
    try:
        store = _get_store()
        logger.info(f"FAISS listo: {store.total:,} vectores")
    except Exception as e:
        logger.error(f"Error cargando FAISS: {e}")
    yield
    bot_supervisor.shutdown()   # no dejar el subproceso del bot huérfano
    logger.info("Apagando servidor.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Maternas API",
    description="Chatbot RAG de salud materna — clasificación de intención, detección de riesgo y respuestas basadas en evidencia.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en producción restringir a la URL de Streamlit
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(admin_router)
app.include_router(bot_router)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["sistema"])
def health() -> HealthResponse:
    """Estado del servicio y métricas básicas."""
    try:
        store = _get_store()
        return HealthResponse(
            status="ok",
            model=settings.embedding_model,
            total_vectors=store.total,
            faiss_loaded=True,
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {str(e)[:80]}",
            model=settings.embedding_model,
            total_vectors=0,
            faiss_loaded=False,
        )


def _source_doc(s: dict) -> SourceDoc:
    return SourceDoc(
        score=s.get("score", 0.0),
        source_dataset=s.get("source_dataset", ""),
        language=s.get("language", ""),
        doc_id=s.get("doc_id"),
        chunk_id=s.get("chunk_id"),
        source_path=source_path(s),
        document_name=document_name(s),
        pages=s.get("pages") or [],
    )


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, tags=["chatbot"])
def chat(request: ChatRequest) -> ChatResponse:
    """
    Turno completo del chatbot.

    Recibe el mensaje del usuario y el historial de la conversación.
    Retorna la respuesta generada junto con metadatos de clasificación y fuentes.

    El caller es responsable de mantener y pasar el historial entre turnos.
    """
    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        result = rag_chat(
            query=request.message,
            history=history,
            k=request.k,
        )
    except Exception as e:
        logger.error(f"[/chat] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)[:120]}")

    sources = [_source_doc(s) for s in result.sources]

    usage_sessions.touch(request.session_id, request.platform, result.tokens_used)

    return ChatResponse(
        answer=result.answer,
        intent=result.intent,
        risk_level=result.risk_level,
        action=result.action,
        risk_flags=result.risk_flags,
        sources=sources,
        reasoning=result.reasoning,
        tokens_used=result.tokens_used,
        notified=result.notified,
        needs_clarification=result.needs_clarification,
        clarification_question=result.clarification_question,
    )


# ---------------------------------------------------------------------------
# POST /chat/stream
# ---------------------------------------------------------------------------

@app.post("/chat/stream", tags=["chatbot"])
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Igual que POST /chat, pero como NDJSON: una línea JSON por evento
    (status → meta → delta* → done, o error), para que la UI pueda
    mostrar la respuesta a medida que se genera en vez de esperar el
    turno completo.

    Declarado `def` (no `async def`) a propósito: FastAPI corre un
    endpoint sync en el threadpool, así que el cliente Groq síncrono de
    src.rag.chain no bloquea el event loop mientras transmite.

    El bot de Telegram y el pipeline de evaluación siguen usando POST
    /chat sin cambios — este endpoint es aditivo.
    """
    history = [{"role": m.role, "content": m.content} for m in request.history]

    def _events():
        try:
            for event in rag_chat_stream(
                query=request.message,
                history=history,
                k=request.k,
            ):
                if event.get("type") == "meta":
                    event = {**event, "sources": [
                        _source_doc(s).model_dump() for s in event.get("sources", [])
                    ]}
                elif event.get("type") == "done":
                    usage_sessions.touch(
                        request.session_id, request.platform, event.get("tokens_used", 0)
                    )
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(f"[/chat/stream] Error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "detail": str(e)[:200]}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        _events(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------

@app.post("/classify", response_model=ClassifyResponse, tags=["clasificadores"])
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """
    Solo clasificación: intención + riesgo clínico sin generar respuesta.

    Útil para pruebas rápidas de los clasificadores o para pipelines
    donde la generación se hace por separado.
    """
    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        intent_result = classify_intent(request.message, conversation_history=history)
        risk_result   = detect_risk(request.message, intent=intent_result.intent, history=history)
    except Exception as e:
        logger.error(f"[/classify] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:120])

    return ClassifyResponse(
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        risk_level=risk_result.level,
        risk_action=risk_result.action,
        risk_flags=risk_result.flags,
        risk_reasoning=risk_result.reasoning,
        used_heuristic=risk_result.used_heuristic,
    )
