"""
risk_episodes.py — Deduplicación de notificaciones de riesgo clínico por
"episodio", en memoria, por sesión.

Problema que resuelve: sin este registro, cada turno con riesgo medium/high
dispara notify_risk() de cero — incluyendo turnos que ya no describen el
mismo cuadro (ver el fix de contaminación en risk_detector.py, que ataca la
otra mitad del problema: dejar de heredar keywords de turnos viejos). Este
módulo guarda únicamente la ÚLTIMA señal EFECTIVAMENTE notificada por sesión
(nivel + categorías de flags — nunca texto del mensaje) para decidir si un
turno nuevo es la MISMA alerta ya enviada o un riesgo distinto/mayor que sí
amerita un correo nuevo.

API de dos fases a propósito (peek + commit), porque "medium" pasa por una
decisión adicional del LLM antes de saber si el correo se envía de verdad:
  - is_new_signal(): solo LEE — decide si vale la pena seguir evaluando
    (y, para medium, si vale la pena gastar la llamada al LLM). No marca
    nada como notificado.
  - commit(): se llama SOLO en el momento en que el correo efectivamente
    se envió — ese es el que actualiza "última señal notificada". Si se
    llamara en is_new_signal(), un medium que el LLM termina descartando
    (NO amerita notificar) quedaría registrado como si sí se hubiera
    avisado, y una repetición real de ese mismo riesgo después se
    suprimiría sin que nunca se haya mandado el primer correo.
  - register_low(): un turno "low" cierra el episodio de inmediato — la
    próxima alerta medium/high se trata como nueva sin importar qué flags
    comparta con la anterior.

Reglas de is_new_signal():
  - medium/high sin episodio previo → nueva señal.
  - medium/high con episodio previo → nueva señal solo si el nivel escala
    (medium→high) o aparece una flag que no estaba en el último aviso;
    si es exactamente la misma señal (mismo nivel, flags ⊆ las ya avisadas),
    NO es nueva.
  - Sin session_id (llamadas internas/eval sin sesión de usuario real) →
    no hay forma de deduplicar, se preserva el comportamiento anterior
    (notificar siempre que el nivel lo amerite).

En memoria únicamente, con lock — mismo criterio MVP que src/api/usage_sessions.py
y 'histories' en el bot de Telegram: se pierde al reiniciar el proceso de la
API. IDLE_TIMEOUT_SECONDS coincide a propósito con el de usage_sessions
(misma noción de "sesión inactiva"), pero se define aparte: src/rag/ no debe
importar de src/api/ (la dependencia entre capas va siempre en el otro
sentido — src/api/main.py importa de src.rag.chain, nunca al revés).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

IDLE_TIMEOUT_SECONDS = 15 * 60

_LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}

_lock = threading.Lock()
_episodes: dict[str, dict] = {}   # session_id -> {"level": str, "flags": frozenset[str], "last_at": datetime}


def _prune_locked(now: datetime) -> None:
    expired = [
        sid for sid, ep in _episodes.items()
        if (now - ep["last_at"]).total_seconds() > IDLE_TIMEOUT_SECONDS
    ]
    for sid in expired:
        _episodes.pop(sid, None)


def register_low(session_id: str | None) -> None:
    """Turno 'low': cierra el episodio de inmediato, sin importar si había
    uno activo. Llamar en CADA turno low, no solo cuando había riesgo antes
    — es barato (un pop) y evita depender de que el caller sepa si había
    episodio previo."""
    if not session_id:
        return
    with _lock:
        _episodes.pop(session_id, None)


def is_new_signal(session_id: str | None, level: str, flags: list[str] | None) -> bool:
    """Solo lectura (aparte de la poda por inactividad): ¿este nivel/flags
    representa una señal distinta de la última EFECTIVAMENTE notificada?"""
    if not session_id:
        return True   # sin sesión no hay forma de deduplicar

    now = datetime.now(timezone.utc)
    flag_set = frozenset(flags or [])

    with _lock:
        _prune_locked(now)
        existing = _episodes.get(session_id)

    if existing is None:
        return True

    escalated = _LEVEL_RANK.get(level, 1) > _LEVEL_RANK.get(existing["level"], 1)
    # Si ambos lados no tienen flags, no hay forma de detectar una señal
    # "nueva" por esa vía — solo la escalada de nivel cuenta.
    new_flag = bool(flag_set or existing["flags"]) and not flag_set.issubset(existing["flags"])
    return escalated or new_flag


def commit(session_id: str | None, level: str, flags: list[str] | None) -> None:
    """Registra que el correo para este nivel/flags SE ENVIÓ. Llamar
    únicamente en el punto donde ya se confirmó el envío (nunca antes)."""
    if not session_id:
        return
    now = datetime.now(timezone.utc)
    with _lock:
        _episodes[session_id] = {"level": level, "flags": frozenset(flags or []), "last_at": now}
