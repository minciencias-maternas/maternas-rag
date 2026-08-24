"""
intent_classifier.py — Clasificador de intención para el chatbot Maternas.

Clasifica cada mensaje del usuario en una de 12 categorías usando el LLM de Groq
con un prompt zero-shot estructurado. No requiere fine-tuning ni modelo local adicional.

Categorías:
    control_prenatal       — controles, ecografías, citas, semanas de gestación
    signos_de_alarma       — síntomas que requieren atención urgente
    sintomas_embarazo      — náuseas, hinchazón, dolores, cambios físicos normales
    postparto              — recuperación, loquios, cicatrización, puerperio
    lactancia              — leche materna, agarre, mastitis, destete
    salud_mental_perinatal — depresión, ansiedad, miedo, estrés perinatal
    medicamentos           — fármacos, suplementos, dosis, contraindicaciones
    nutricion              — dieta, alimentos, vitaminas, peso gestacional
    actividad_fisica       — ejercicio, reposo, restricciones físicas
    planificacion_familiar — anticonceptivos postparto, próximo embarazo
    consulta_administrativa — turnos, hospitales, documentos, sistemas de salud
    pregunta_fuera_de_alcance — fuera del dominio materno/salud

Retorna:
    IntentResult(intent, confidence, reasoning)

Uso:
    from src.classifiers.intent_classifier import classify_intent
    result = classify_intent("Tengo hinchazón en los pies, ¿es normal?")
    print(result.intent)  # "sintomas_embarazo"
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from groq import Groq
from src.settings import settings, groq_reasoning_kwargs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos de retorno
# ---------------------------------------------------------------------------

VALID_INTENTS = frozenset([
    "control_prenatal",
    "signos_de_alarma",
    "sintomas_embarazo",
    "postparto",
    "lactancia",
    "salud_mental_perinatal",
    "medicamentos",
    "nutricion",
    "actividad_fisica",
    "planificacion_familiar",
    "consulta_administrativa",
    "pregunta_fuera_de_alcance",
])


@dataclass
class IntentResult:
    intent:     str
    confidence: float        # 0.0 – 1.0 (autoreportado por el LLM)
    reasoning:  str          # explicación breve del LLM
    raw:        Optional[str] = None  # respuesta cruda para debug


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Eres un clasificador de intención para un chatbot de salud materna.
Tu única tarea es clasificar el mensaje del usuario en UNA de las siguientes categorías:

- control_prenatal: controles médicos, ecografías, semanas de gestación, citas prenatales
- signos_de_alarma: síntomas que requieren atención urgente (hemorragia, dolor intenso, falta de movimiento fetal, convulsiones, visión borrosa)
- sintomas_embarazo: síntomas comunes del embarazo (náuseas, vómitos, hinchazón leve, cansancio, acidez)
- postparto: recuperación postparto, loquios, cicatrización, puerperio, cambios físicos tras el parto
- lactancia: lactancia materna, leche, agarre, mastitis, biberón, destete
- salud_mental_perinatal: depresión postparto, ansiedad, miedos, estrés, bienestar emocional perinatal
- medicamentos: fármacos, suplementos, vitaminas, dosis, seguridad de medicamentos en embarazo
- nutricion: dieta, alimentación, alimentos permitidos/prohibidos, peso gestacional, hidratación
- actividad_fisica: ejercicio durante el embarazo, reposo médico, restricciones físicas
- planificacion_familiar: anticonceptivos postparto, intervalo intergenésico, próximo embarazo
- consulta_administrativa: turnos, hospitales, documentos, sistemas de salud pública, trámites
- pregunta_fuera_de_alcance: cualquier tema que no sea salud materna o del recién nacido

Responde ÚNICAMENTE con un JSON válido con este formato exacto:
{
  "intent": "<categoría>",
  "confidence": <número entre 0.0 y 1.0>,
  "reasoning": "<explicación en 1 oración>"
}

No agregues texto antes ni después del JSON.\
"""


# ---------------------------------------------------------------------------
# Cliente Groq (singleton)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Clasificador principal
# ---------------------------------------------------------------------------

def classify_intent(
    message: str,
    conversation_history: Optional[list[dict]] = None,
) -> IntentResult:
    """
    Clasifica la intención del mensaje del usuario.

    Args:
        message: Texto del usuario a clasificar.
        conversation_history: Turns anteriores en formato
            [{"role": "user"|"assistant", "content": "..."}].
            Si se provee, se incluye contexto del último turno para
            desambiguar intenciones dependientes del contexto.

    Returns:
        IntentResult con intent, confidence y reasoning.
    """
    if not message or not message.strip():
        return IntentResult(
            intent="pregunta_fuera_de_alcance",
            confidence=1.0,
            reasoning="Mensaje vacío.",
        )

    # Construir el mensaje de usuario con contexto opcional
    user_content = message.strip()
    if conversation_history:
        # Incluir solo el último turno para contexto mínimo
        last = conversation_history[-1]
        if last.get("role") == "assistant":
            user_content = (
                f"[Contexto previo del asistente: {last['content'][:200]}]\n\n"
                f"Nuevo mensaje del usuario: {message.strip()}"
            )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.0,          # clasificación determinista
            max_tokens=800,
            response_format={"type": "json_object"},
            extra_body=groq_reasoning_kwargs(json_mode=True),
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_response(raw, message)

    except Exception as e:
        logger.error(f"[IntentClassifier] Error llamando a Groq: {e}")
        return IntentResult(
            intent="pregunta_fuera_de_alcance",
            confidence=0.0,
            reasoning=f"Error en clasificación: {str(e)[:100]}",
        )


# ---------------------------------------------------------------------------
# Parser de respuesta
# ---------------------------------------------------------------------------

def _parse_response(raw: str, original_message: str) -> IntentResult:
    """Parsea el JSON del LLM con fallback robusto."""
    try:
        data = json.loads(raw)
        intent = str(data.get("intent", "")).strip().lower()
        confidence = float(data.get("confidence", 0.5))
        reasoning  = str(data.get("reasoning", "")).strip()

        # Validar intent
        if intent not in VALID_INTENTS:
            logger.warning(f"[IntentClassifier] Intent desconocido: '{intent}' — usando fallback")
            intent = _fallback_intent(original_message)
            confidence = 0.3

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        return IntentResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            raw=raw,
        )

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"[IntentClassifier] No se pudo parsear JSON: {e} | raw: {raw[:100]}")
        # Intentar extraer intent con regex como último recurso
        match = re.search(r'"intent"\s*:\s*"([^"]+)"', raw)
        if match and match.group(1) in VALID_INTENTS:
            return IntentResult(
                intent=match.group(1),
                confidence=0.4,
                reasoning="Parseado con fallback regex.",
                raw=raw,
            )
        return IntentResult(
            intent=_fallback_intent(original_message),
            confidence=0.2,
            reasoning="No se pudo parsear la respuesta del clasificador.",
            raw=raw,
        )


def _fallback_intent(message: str) -> str:
    """Heurística simple de palabras clave como último recurso."""
    msg = message.lower()
    if any(w in msg for w in ["hemorragia", "sangrado", "convulsión", "desmayo", "no se mueve"]):
        return "signos_de_alarma"
    if any(w in msg for w in ["náusea", "vómito", "hinchazón", "cansancio"]):
        return "sintomas_embarazo"
    if any(w in msg for w in ["leche", "pecho", "amamantar", "lactar"]):
        return "lactancia"
    if any(w in msg for w in ["medicamento", "pastilla", "dosis", "suplemento"]):
        return "medicamentos"
    if any(w in msg for w in ["comer", "dieta", "alimento", "vitamina"]):
        return "nutricion"
    return "pregunta_fuera_de_alcance"
