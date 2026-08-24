"""
usage_sessions.py — Registro en memoria de sesiones activas, para la
subsección "Uso en tiempo real" del panel de Métricas.

El session_id lo genera el CLIENTE (uuid4 aleatorio — nunca derivado de un
chat_id de Telegram ni de ningún identificador real) y viaja en cada
POST /chat. Este módulo solo lo usa como clave interna para acumular
duración y tokens; GET /admin/usage_sessions (routes_admin.py) nunca lo
expone ni expone nada derivado de él — así ninguna sesión es distinguible
de otra entre dos lecturas del panel.

En memoria únicamente, con lock — mismo criterio MVP que 'histories' en
src/bot/maternas_bot.py ("se pierde al reiniciar, suficiente por ahora").
No hace falta que sobreviva un reinicio: es una vista de monitoreo en vivo,
no una métrica histórica.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

# Una sesión sin turnos nuevos en este lapso deja de contar como activa.
IDLE_TIMEOUT_SECONDS = 15 * 60

_lock = threading.Lock()
_sessions: dict[str, dict] = {}   # session_id -> {platform, started_at, last_activity, tokens_total}


def touch(session_id: str | None, platform: str, tokens_used: int = 0) -> None:
    """Registra un turno de chat contra la sesión. Crea la entrada si es
    la primera vez que se ve este session_id."""
    if not session_id:
        return
    now = datetime.now(timezone.utc)
    with _lock:
        entry = _sessions.get(session_id)
        if entry is None:
            entry = {
                "platform":      platform,
                "started_at":    now,
                "last_activity": now,
                "tokens_total":  0,
            }
            _sessions[session_id] = entry
        entry["last_activity"] = now
        entry["tokens_total"] += max(0, tokens_used)


def active_by_platform() -> dict[str, list[dict]]:
    """Snapshot de sesiones activas (last_activity dentro de
    IDLE_TIMEOUT_SECONDS), agrupadas por plataforma y ordenadas por
    duración descendente. Cada fila trae solo active_seconds y
    tokens_total — nunca el session_id."""
    now = datetime.now(timezone.utc)
    result: dict[str, list[dict]] = {"streamlit": [], "telegram": []}

    with _lock:
        expired = [
            sid for sid, entry in _sessions.items()
            if (now - entry["last_activity"]).total_seconds() > IDLE_TIMEOUT_SECONDS
        ]
        for sid in expired:
            _sessions.pop(sid, None)

        for entry in _sessions.values():
            bucket = result.get(entry["platform"])
            if bucket is None:
                continue
            bucket.append({
                "active_seconds": (now - entry["started_at"]).total_seconds(),
                "tokens_total":   entry["tokens_total"],
            })

    for rows in result.values():
        rows.sort(key=lambda r: r["active_seconds"], reverse=True)
    return result
