from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any

from src.settings import settings

logger = logging.getLogger(__name__)


def notify_risk(
    query: str,
    risk_level: str,
    intent: str,
    reasoning: str,
    flags: list[str] | None = None,
    conversation: list[dict] | None = None,
) -> dict[str, Any]:
    """Envía alerta por email cuando se detecta riesgo clínico alto o medio.

    Args:
        query:        Mensaje del turno actual (el que disparó la alerta).
        risk_level:   "high" | "medium"
        intent:       Intención clasificada por el intent classifier.
        reasoning:    Explicación del detector de riesgo.
        flags:        Lista de banderas de riesgo detectadas.
        conversation: Secuencia de turnos [{"role","content"}] que llevó a
                      esta alerta (historial + mensaje actual al final). Si
                      no viene, el email cae de vuelta a mostrar solo `query`
                      (compatibilidad con callers que no la arman).

    Returns:
        dict con {"success": bool, "message": str}
    """
    if not settings.notifier_enabled:
        logger.info("[Notifier] Notificaciones deshabilitadas (NOTIFIER_ENABLED=false)")
        return {"success": False, "message": "Notifier disabled"}

    if not settings.notifier_email_to:
        logger.warning("[Notifier] NOTIFIER_EMAIL_TO vacío — no se envía correo")
        return {"success": False, "message": "No recipient configured"}

    try:
        subject = f"[MATERNAS - ALERTA] Riesgo {risk_level.upper()} - {intent}"
        body = _build_email_body(query, risk_level, intent, reasoning, flags or [], conversation or [])

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = settings.notifier_smtp_user
        msg["To"] = settings.notifier_email_to

        with smtplib.SMTP(settings.notifier_smtp_host, settings.notifier_smtp_port) as server:
            server.starttls()
            server.login(settings.notifier_smtp_user, settings.notifier_smtp_password)
            server.send_message(msg)

        logger.info(f"[Notifier] Email enviado a {settings.notifier_email_to} — {subject}")
        return {"success": True, "message": f"Email sent to {settings.notifier_email_to}"}

    except Exception as e:
        logger.error(f"[Notifier] Error enviando email: {e}")
        return {"success": False, "message": str(e)}


_ROLE_LABEL = {"user": "Paciente", "assistant": "Asistente"}


def _build_email_body(
    query: str,
    risk_level: str,
    intent: str,
    reasoning: str,
    flags: list[str],
    conversation: list[dict],
) -> str:
    lines = [
        "=" * 60,
        "  MATERNAS - ALERTA DE RIESGO CLINICO",
        "=" * 60,
        "",
        f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Riesgo:      {risk_level.upper()}",
        f"Intencion:   {intent}",
        f"Razonamiento: {reasoning}",
        "",
    ]
    if flags:
        lines.append(f"Banderas:    {', '.join(flags)}")
        lines.append("")

    lines += ["-" * 60, "SECUENCIA DE MENSAJES QUE LLEVARON A ESTA ALERTA:", "-" * 60, ""]

    if conversation:
        # El último turno (el mensaje actual, el que disparó la alerta) se
        # marca aparte para que quede claro cuál de todos gatilló el correo.
        last_idx = len(conversation) - 1
        for i, turn in enumerate(conversation):
            role = _ROLE_LABEL.get(turn.get("role", ""), turn.get("role", "?"))
            content = turn.get("content", "")
            marker = "  <<< MENSAJE QUE DISPARO LA ALERTA" if i == last_idx else ""
            lines.append(f"[{role}]{marker}")
            lines.append(content)
            lines.append("")
    else:
        # Compatibilidad: caller sin historial armado, solo el mensaje actual.
        lines.append("[Paciente]  <<< MENSAJE QUE DISPARO LA ALERTA")
        lines.append(query)
        lines.append("")

    lines += [
        "=" * 60,
        "  Este es un mensaje automatico del sistema Maternas.",
        "  No responder directamente a este correo.",
        "=" * 60,
    ]
    return "\n".join(lines)
