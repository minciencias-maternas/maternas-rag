"""
client.py — Cliente HTTP para la API del backend Maternas.

Sin dependencias de Streamlit. Las excepciones de httpx se propagan
hacia la capa de presentación (vistas), que decide cómo mostrarlas.
"""

import json
from typing import Iterator
from urllib.parse import quote

import httpx

from src.settings import settings

API_URL     = settings.api_url
API_TIMEOUT = 60


def _admin_headers(admin_token: str) -> dict:
    """Header de autenticación para /documents* y /admin*.

    No se envía en /health ni /chat, que son públicos. `admin_token` lo
    pasa cada vista, leído de st.session_state — NUNCA de
    settings.admin_api_token acá: este módulo no sabe qué sesión de
    navegador está llamando, y en modo servidor Streamlit corre un solo
    proceso para todas las sesiones concurrentes. Si el token viviera en
    una variable de módulo, la sesión de un admin autenticado quedaría
    expuesta a cualquier otra sesión (ver src/ui/admin_gate.py).
    """
    return {"X-Admin-Token": admin_token} if admin_token else {}


# ---------------------------------------------------------------------------
# Chat (público)
# ---------------------------------------------------------------------------

def check_health() -> dict:
    r = httpx.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def call_chat(message: str, history: list, session_id: str = "") -> dict:
    payload = {"message": message, "history": history, "k": 5, "platform": "streamlit"}
    if session_id:
        payload["session_id"] = session_id
    r = httpx.post(f"{API_URL}/chat", json=payload, timeout=API_TIMEOUT)
    r.raise_for_status()
    return r.json()


def stream_chat(message: str, history: list, k: int = 5, session_id: str = "") -> Iterator[dict]:
    """Turno del chat como secuencia de eventos NDJSON (ver POST /chat/stream).

    Igual que call_chat(), sin captura de excepciones: httpx.HTTPError
    sube tal cual a la vista, que decide cómo mostrarlo.
    """
    payload = {"message": message, "history": history, "k": k, "platform": "streamlit"}
    if session_id:
        payload["session_id"] = session_id
    timeout = httpx.Timeout(connect=5.0, read=API_TIMEOUT, write=10.0, pool=5.0)
    with httpx.stream("POST", f"{API_URL}/chat/stream", json=payload, timeout=timeout) as r:
        if r.status_code >= 400:
            r.read()   # sin esto, e.response.text lanza ResponseNotRead
        r.raise_for_status()
        for line in r.iter_lines():
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Documentos (requieren X-Admin-Token)
# ---------------------------------------------------------------------------

def list_documents(admin_token: str, search: str = "", page: int = 1, per_page: int = 20) -> dict:
    r = httpx.get(
        f"{API_URL}/documents",
        params={"search": search, "page": page, "per_page": per_page},
        headers=_admin_headers(admin_token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_document_stats(admin_token: str) -> dict:
    r = httpx.get(f"{API_URL}/documents/stats", headers=_admin_headers(admin_token), timeout=10)
    r.raise_for_status()
    return r.json()


def get_document_detail(admin_token: str, doc_id: str, page: int = 1, per_page: int = 20) -> dict:
    r = httpx.get(
        f"{API_URL}/documents/{quote(doc_id, safe='')}",
        params={"page": page, "per_page": per_page},
        headers=_admin_headers(admin_token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def toggle_document_status(admin_token: str, doc_id: str, active: bool) -> dict:
    r = httpx.patch(
        f"{API_URL}/documents/{quote(doc_id, safe='')}",
        json={"active": active},
        headers=_admin_headers(admin_token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def upload_document(admin_token: str, filename: str, content: bytes) -> dict:
    r = httpx.post(
        f"{API_URL}/documents/upload",
        files={"file": (filename, content)},
        headers=_admin_headers(admin_token),
        # El embedding corre síncrono en el backend (ver
        # src/api/routes_documents.py, MAX_UPLOAD_CHUNKS); 300s da margen
        # incluso para una subida cerca del límite corriendo en CPU.
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Administración (requieren X-Admin-Token)
# ---------------------------------------------------------------------------

def list_evaluations(admin_token: str) -> dict:
    r = httpx.get(f"{API_URL}/admin/evaluations", headers=_admin_headers(admin_token), timeout=10)
    r.raise_for_status()
    return r.json()


def get_evaluation_detail(admin_token: str, run_id: str) -> dict:
    r = httpx.get(
        f"{API_URL}/admin/evaluations/{quote(run_id, safe='')}",
        headers=_admin_headers(admin_token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_admin_config(admin_token: str) -> dict:
    r = httpx.get(f"{API_URL}/admin/config", headers=_admin_headers(admin_token), timeout=10)
    r.raise_for_status()
    return r.json()


def update_admin_config(admin_token: str, **fields) -> dict:
    """Solo manda los campos presentes en `fields` — el caller decide
    qué cambió (ver config_view.py, que diffea contra el valor cargado
    antes de armar el payload)."""
    r = httpx.patch(
        f"{API_URL}/admin/config",
        json=fields,
        headers=_admin_headers(admin_token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_usage_sessions(admin_token: str) -> dict:
    r = httpx.get(f"{API_URL}/admin/usage_sessions", headers=_admin_headers(admin_token), timeout=10)
    r.raise_for_status()
    return r.json()


def get_admin_logs(admin_token: str, limit: int = 200) -> dict:
    r = httpx.get(
        f"{API_URL}/admin/logs",
        params={"limit": limit},
        headers=_admin_headers(admin_token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Bot de Telegram (requieren X-Admin-Token)
# ---------------------------------------------------------------------------

def get_bot_status(admin_token: str) -> dict:
    r = httpx.get(f"{API_URL}/admin/bot/status", headers=_admin_headers(admin_token), timeout=10)
    r.raise_for_status()
    return r.json()


def start_bot(admin_token: str) -> dict:
    r = httpx.post(f"{API_URL}/admin/bot/start", headers=_admin_headers(admin_token), timeout=15)
    r.raise_for_status()
    return r.json()


def stop_bot(admin_token: str) -> dict:
    r = httpx.post(f"{API_URL}/admin/bot/stop", headers=_admin_headers(admin_token), timeout=15)
    r.raise_for_status()
    return r.json()


def restart_bot(admin_token: str) -> dict:
    r = httpx.post(f"{API_URL}/admin/bot/restart", headers=_admin_headers(admin_token), timeout=20)
    r.raise_for_status()
    return r.json()


def get_bot_logs(admin_token: str, limit: int = 200) -> dict:
    r = httpx.get(
        f"{API_URL}/admin/bot/logs",
        params={"limit": limit},
        headers=_admin_headers(admin_token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()
