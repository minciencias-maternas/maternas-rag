"""
risk_detector.py — Detector de riesgo clínico para el chatbot Maternas.

Evalúa el nivel de riesgo clínico de un mensaje y detecta señales de alarma
específicas. Combina dos capas:

  1. Capa rápida (heurística): keywords de alarma → detección instantánea
     de emergencias sin llamada a la API.
  2. Capa LLM (Groq): evaluación contextual del riesgo con razonamiento.

La capa heurística tiene prioridad: si detecta HIGH, no se llama al LLM,
lo que reduce latencia en los casos más urgentes.

Niveles de riesgo:
    low    → respuesta educativa, no requiere acción inmediata
    medium → recomendar consulta médica en los próximos días
    high   → requiere atención médica urgente / ir a urgencias

Retorna:
    RiskResult(level, flags, action, reasoning, used_heuristic)

Uso:
    from src.classifiers.risk_detector import detect_risk
    result = detect_risk("Estoy sangrando mucho y tengo dolor fuerte")
    print(result.level)   # "high"
    print(result.flags)   # ["hemorragia", "dolor_intenso"]
    print(result.action)  # "urgent_care"
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from groq import Groq
from src.settings import settings, groq_reasoning_kwargs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos de retorno
# ---------------------------------------------------------------------------

VALID_LEVELS  = frozenset(["low", "medium", "high"])
VALID_ACTIONS = frozenset(["educational_answer", "medical_consultation", "urgent_care"])


@dataclass
class RiskResult:
    level:           str               # "low" | "medium" | "high"
    flags:           list[str]         # señales de alarma detectadas
    action:          str               # "educational_answer" | "medical_consultation" | "urgent_care"
    reasoning:       str               # explicación
    used_heuristic:  bool = False      # True si fue la capa rápida
    raw:             Optional[str] = None


# ---------------------------------------------------------------------------
# Capa 1: Heurística de keywords de alarma
# ---------------------------------------------------------------------------
# Organizado por categoría clínica. Si se detecta cualquier keyword → HIGH.

HIGH_RISK_KEYWORDS: dict[str, list[str]] = {
    "hemorragia": [
        "sangrando mucho", "hemorragia", "sangrado abundante",
        "sangre en cantidad", "empapé", "coágulos",
    ],
    "preeclampsia": [
        "visión borrosa", "ver borroso", "manchas en la vista",
        "zumbido en los oídos", "dolor de cabeza intenso con visión",
        "hinchazón súbita de cara", "hinchazón de manos y cara",
    ],
    "eclampsia_convulsion": [
        "convulsión", "convulsiones", "me convulsioné",
        "espasmos", "pérdida de conocimiento", "desmayo",
    ],
    "trabajo_parto_prematuro": [
        "contracciones antes de las 37", "parto prematuro",
        "rompí fuente antes de tiempo", "líquido antes de las 37",
        "presión en la pelvis con contracciones regulares",
    ],
    "ruptura_membranas": [
        "rompí fuente", "se rompió la bolsa", "chorro de líquido",
        "líquido amniótico", "pérdida de líquido continua",
    ],
    "movimiento_fetal_ausente": [
        "no se mueve", "dejó de moverse", "no siento movimientos",
        "no patea", "sin movimiento fetal",
    ],
    "dolor_intenso": [
        "dolor insoportable", "dolor muy fuerte en el abdomen",
        "dolor abdominal agudo", "dolor que no cede",
        "dolor en el pecho fuerte",
    ],
    "signos_sepsis": [
        "fiebre muy alta", "fiebre de 39", "fiebre de 40",
        "escalofríos con fiebre", "mal olor vaginal con fiebre",
    ],
    "depresion_grave": [
        "quiero hacerme daño", "quiero lastimarme",
        "pienso en hacerle daño al bebé", "no quiero vivir",
        "ideas de suicidio", "pensamientos de muerte",
    ],
}

MEDIUM_RISK_KEYWORDS: dict[str, list[str]] = {
    "sangrado_leve": [
        "manchado", "pequeño sangrado", "spotting", "sangre rosada",
    ],
    "presion_alta_leve": [
        "presión alta", "hipertensión", "tensión elevada",
    ],
    "edema_moderado": [
        "pies muy hinchados", "tobillos muy hinchados", "no puedo ponerme los zapatos",
    ],
    "dolor_moderado": [
        "dolor de cabeza que no pasa", "migraña en el embarazo",
        "dolor en el costado derecho",
    ],
    "fiebre_moderada": [
        "fiebre", "temperatura alta", "febrícula persistente",
    ],
    "reduccion_movimiento": [
        "se mueve menos", "patea menos de lo normal",
        "noto menos movimiento",
    ],
    "sintomas_infeccion": [
        "ardor al orinar", "orina con mal olor", "flujo con mal olor",
        "picazón vaginal intensa",
    ],
}


def _check_heuristic(message: str) -> Optional[RiskResult]:
    """
    Revisa keywords en el mensaje. Retorna RiskResult si detecta HIGH o MEDIUM,
    None si no encuentra nada relevante (continuar con LLM).
    """
    msg_lower = message.lower()
    found_high: list[str] = []
    found_medium: list[str] = []

    for flag, keywords in HIGH_RISK_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            found_high.append(flag)

    if found_high:
        return RiskResult(
            level="high",
            flags=found_high,
            action="urgent_care",
            reasoning=f"Señales de alarma detectadas: {', '.join(found_high)}. Requiere atención médica urgente.",
            used_heuristic=True,
        )

    for flag, keywords in MEDIUM_RISK_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            found_medium.append(flag)

    if found_medium:
        return RiskResult(
            level="medium",
            flags=found_medium,
            action="medical_consultation",
            reasoning=f"Síntomas que requieren evaluación médica: {', '.join(found_medium)}.",
            used_heuristic=True,
        )

    return None  # No detectado por heurística → usar LLM


# ---------------------------------------------------------------------------
# Capa 2: LLM (Groq)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Eres un evaluador de riesgo clínico para un chatbot de salud materna.
Tu tarea es evaluar el nivel de riesgo del mensaje de una gestante o puérpera.

Niveles de riesgo:
- low: pregunta informativa, síntoma leve normal del embarazo, sin urgencia
- medium: síntoma que merece evaluación médica en los próximos días (no urgente)
- high: síntoma o situación que requiere atención médica INMEDIATA o ir a urgencias

Acciones según nivel:
- low    → "educational_answer"
- medium → "medical_consultation"
- high   → "urgent_care"

Señales que SIEMPRE son HIGH:
hemorragia activa, convulsiones, ausencia de movimiento fetal, visión borrosa + cefalea + edema,
dolor abdominal agudo, fiebre alta con signos de infección, ideas de autolesión.

Si el mensaje incluye síntomas mencionados en turnos previos de la misma conversación,
evalúa el CONJUNTO de síntomas como un cuadro clínico único, no cada uno por separado.
Varios síntomas leves combinados pueden justificar un nivel de riesgo mayor al de
cualquiera de ellos visto de forma aislada.

Responde ÚNICAMENTE con un JSON válido:
{
  "level": "low"|"medium"|"high",
  "flags": ["señal1", "señal2"],
  "action": "educational_answer"|"medical_consultation"|"urgent_care",
  "reasoning": "<explicación en 1-2 oraciones>"
}

No agregues texto antes ni después del JSON.\
"""

# Cuántos turnos previos de la usuaria se incluyen al evaluar riesgo.
# Acota el crecimiento del prompt en conversaciones largas.
RISK_HISTORY_USER_TURNS = 5


def _extract_recent_user_text(history: Optional[list[dict]], limit: int = RISK_HISTORY_USER_TURNS) -> str:
    """Concatena los últimos mensajes de la usuaria (rol 'user') en el historial."""
    if not history:
        return ""
    user_msgs = [str(h.get("content", "")).strip() for h in history if h.get("role") == "user"]
    user_msgs = [m for m in user_msgs if m]
    return " ".join(user_msgs[-limit:])

_groq_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def reset_client() -> None:
    """Fuerza reconstruir el cliente Groq en la próxima llamada — usado
    cuando GROQ_API_KEY cambia en caliente desde el panel admin."""
    global _groq_client
    _groq_client = None


def _llm_risk(message: str, history: Optional[list[dict]] = None) -> RiskResult:
    """Evalúa riesgo usando el LLM de Groq, considerando síntomas de turnos previos."""
    client = _get_client()
    user_content = message.strip()
    prior_user_text = _extract_recent_user_text(history)
    if prior_user_text:
        user_content = (
            f"Mensajes previos de la paciente en esta conversación: {prior_user_text}\n\n"
            f"Mensaje actual: {message.strip()}\n\n"
            "Evalúa el riesgo considerando el conjunto de síntomas mencionados en toda "
            "la conversación, no solo el mensaje actual."
        )
    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.0,
            max_tokens=900,
            response_format={"type": "json_object"},
            extra_body=groq_reasoning_kwargs(json_mode=True),
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_llm_response(raw)

    except Exception as e:
        logger.error(f"[RiskDetector] Error llamando a Groq: {e}")
        return RiskResult(
            level="medium",
            flags=[],
            action="medical_consultation",
            reasoning=f"Error en evaluación de riesgo — por precaución se recomienda consulta. ({str(e)[:80]})",
        )


def _parse_llm_response(raw: str) -> RiskResult:
    try:
        data     = json.loads(raw)
        level    = str(data.get("level", "medium")).strip().lower()
        flags    = [str(f) for f in data.get("flags", [])]
        action   = str(data.get("action", "medical_consultation")).strip()
        reasoning = str(data.get("reasoning", "")).strip()

        if level not in VALID_LEVELS:
            level = "medium"
        if action not in VALID_ACTIONS:
            action = _action_for_level(level)

        return RiskResult(
            level=level, flags=flags, action=action,
            reasoning=reasoning, raw=raw,
        )
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"[RiskDetector] No se pudo parsear JSON: {raw[:100]}")
        return RiskResult(
            level="medium", flags=[],
            action="medical_consultation",
            reasoning="No se pudo evaluar el riesgo correctamente.",
            raw=raw,
        )


def _action_for_level(level: str) -> str:
    return {"low": "educational_answer",
            "medium": "medical_consultation",
            "high": "urgent_care"}.get(level, "medical_consultation")


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def detect_risk(
    message: str,
    intent: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> RiskResult:
    """
    Detecta el nivel de riesgo clínico del mensaje.

    Primero aplica la capa heurística (instantánea). Si no detecta nada,
    escala al LLM para evaluación contextual.

    Cuando se provee `history`, los síntomas mencionados en turnos anteriores
    de la misma conversación (p. ej. durante un intercambio de clarificación)
    se combinan con el mensaje actual, para no evaluar cada síntoma aislado
    del resto del cuadro clínico.

    Args:
        message: Texto del usuario.
        intent:  Intención ya clasificada (opcional). Si es
                 'pregunta_fuera_de_alcance', retorna LOW directamente.
        history: Turnos previos [{"role": "user"|"assistant", "content": "..."}].

    Returns:
        RiskResult con level, flags, action y reasoning.
    """
    if not message or not message.strip():
        return RiskResult(
            level="low", flags=[], action="educational_answer",
            reasoning="Mensaje vacío.",
        )

    # Shortcut: off-topic → siempre low
    if intent == "pregunta_fuera_de_alcance":
        return RiskResult(
            level="low", flags=[], action="educational_answer",
            reasoning="Pregunta fuera del dominio de salud materna.",
        )

    # Capa 1: heurística rápida — combina síntomas de turnos previos + mensaje actual
    prior_user_text = _extract_recent_user_text(history)
    combined_message = f"{prior_user_text} {message.strip()}" if prior_user_text else message
    heuristic_result = _check_heuristic(combined_message)
    if heuristic_result is not None:
        return heuristic_result

    # Capa 2: LLM para evaluación contextual
    return _llm_risk(message, history=history)
