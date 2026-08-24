"""
test_chat_stream.py — Cobertura del endpoint POST /chat/stream (NDJSON).

TestClient(app) se usa SIN el bloque `with`, igual que
tests/test_api_documents.py, para saltear el lifespan de FastAPI (que
intentaría cargar el índice FAISS real). src.rag.chain.chat_stream se
parchea directo en src.api.main (donde el endpoint lo importó como
rag_chat_stream) para no depender de Groq/FAISS reales.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)   # sin "with": no dispara el lifespan


def _events(response) -> list[dict]:
    return [json.loads(line) for line in response.iter_lines() if line.strip()]


def _happy_path_stream(query, history, k=None, session_id=None):
    yield {"type": "status", "stage": "classifying"}
    yield {"type": "status", "stage": "retrieving"}
    yield {
        "type": "meta",
        "intent": "sintomas_embarazo",
        "risk_level": "low",
        "action": "educational_answer",
        "risk_flags": [],
        "sources": [{"source_dataset": "medmcqa", "language": "en", "score": 0.8,
                      "doc_id": "d1", "chunk_id": "c1"}],
        "needs_clarification": False,
    }
    yield {"type": "status", "stage": "generating"}
    for tok in ["Es ", "normal ", "sentir ", "nauseas."]:
        yield {"type": "delta", "text": tok}
    yield {"type": "done", "answer": "Es normal sentir nauseas.", "tokens_used": 42, "notified": False}


def test_stream_events_are_valid_ndjson(client, monkeypatch):
    monkeypatch.setattr("src.api.main.rag_chat_stream", _happy_path_stream)
    r = client.post("/chat/stream", json={"message": "hola", "history": []})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = _events(r)
    assert len(events) == 9


def test_stream_event_order_is_status_meta_delta_done(client, monkeypatch):
    monkeypatch.setattr("src.api.main.rag_chat_stream", _happy_path_stream)
    r = client.post("/chat/stream", json={"message": "hola", "history": []})
    types = [e["type"] for e in _events(r)]
    assert types == [
        "status", "status", "meta", "status",
        "delta", "delta", "delta", "delta", "done",
    ]


def test_stream_done_is_last_event_and_matches_deltas(client, monkeypatch):
    monkeypatch.setattr("src.api.main.rag_chat_stream", _happy_path_stream)
    r = client.post("/chat/stream", json={"message": "hola", "history": []})
    events = _events(r)
    assert events[-1]["type"] == "done"
    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert events[-1]["answer"].startswith(deltas.strip()[:10])  # arranca igual que lo transmitido


def test_stream_meta_sources_are_enriched_with_document_name(client, monkeypatch):
    monkeypatch.setattr("src.api.main.rag_chat_stream", _happy_path_stream)
    r = client.post("/chat/stream", json={"message": "hola", "history": []})
    events = _events(r)
    meta = next(e for e in events if e["type"] == "meta")
    assert meta["sources"][0]["document_name"] == "MedMCQA"


def test_stream_llm_failure_emits_error_event(client, monkeypatch):
    def _failing_stream(query, history, k=None, session_id=None):
        yield {"type": "status", "stage": "classifying"}
        yield {"type": "status", "stage": "retrieving"}
        yield {"type": "meta", "intent": "x", "risk_level": "low", "action": "a",
               "risk_flags": [], "sources": [], "needs_clarification": False}
        yield {"type": "status", "stage": "generating"}
        yield {"type": "delta", "text": "Empez"}
        raise RuntimeError("groq se cayó a mitad de camino")

    monkeypatch.setattr("src.api.main.rag_chat_stream", _failing_stream)
    r = client.post("/chat/stream", json={"message": "hola", "history": []})
    assert r.status_code == 200   # los headers ya se enviaron; el error va en el body NDJSON
    events = _events(r)
    assert events[-1]["type"] == "error"
    assert "detail" in events[-1]


def test_stream_clarification_short_circuits_to_done(client, monkeypatch):
    def _clarify_stream(query, history, k=None, session_id=None):
        yield {"type": "status", "stage": "classifying"}
        yield {"type": "meta", "intent": "medicamentos", "risk_level": "low",
               "action": "educational_answer", "risk_flags": [], "sources": [],
               "needs_clarification": True}
        yield {"type": "delta", "text": "¿En qué semana estás?"}
        yield {"type": "done", "answer": "¿En qué semana estás?", "tokens_used": 0, "notified": False}

    monkeypatch.setattr("src.api.main.rag_chat_stream", _clarify_stream)
    r = client.post("/chat/stream", json={"message": "me duele", "history": []})
    events = _events(r)
    meta = next(e for e in events if e["type"] == "meta")
    assert meta["needs_clarification"] is True
    assert events[-1]["type"] == "done"


def test_chat_post_still_works_unchanged(client, monkeypatch):
    """POST /chat (no streaming) no cambia de shape para bot/eval_pipeline,
    salvo que ahora sí propaga needs_clarification/notified (bug adyacente
    arreglado junto con el streaming)."""
    from src.rag.chain import ChatResponse

    def _fake_chat(query, history, k=None, session_id=None):
        return ChatResponse(
            answer="Es normal.",
            intent="sintomas_embarazo",
            risk_level="low",
            action="educational_answer",
            sources=[{"source_dataset": "medmcqa", "language": "en", "score": 0.5}],
            notified=False,
            needs_clarification=False,
        )

    monkeypatch.setattr("src.api.main.rag_chat", _fake_chat)
    r = client.post("/chat", json={"message": "hola", "history": []})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Es normal."
    assert body["sources"][0]["document_name"] == "MedMCQA"
    assert body["needs_clarification"] is False
    assert body["notified"] is False
