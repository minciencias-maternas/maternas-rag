"""
chain.py — Cadena RAG completa del chatbot Maternas.

Orquesta el flujo completo por turno:
  1. classify_intent()   → qué quiere el usuario
  2. detect_risk()       → nivel de riesgo clínico
  3. retrieve()          → fragmentos relevantes del FAISS
  4. build_prompt()      → prompt con contexto + historial
  5. Groq LLM            → respuesta generada
  6. ChatResponse        → objeto de retorno estructurado

Historial conversacional:
  Se pasa como lista de dicts [{"role": "user"|"assistant", "content": "..."}].
  El caller es responsable de mantenerlo entre turnos.

Uso básico:
    from src.rag.chain import chat
    history = []
    response = chat("¿Qué alimentos debo evitar en el embarazo?", history)
    history.append({"role": "user",      "content": "¿Qué alimentos debo evitar?"})
    history.append({"role": "assistant", "content": response.answer})
    print(response.answer)
    print(response.intent, response.risk_level)
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from groq import Groq

from src.classifiers.intent_classifier import classify_intent, IntentResult
from src.classifiers.risk_detector import detect_risk, RiskResult
from src.rag.citations import build_reference_block, format_context, normalize_citation_brackets
from src.rag.retriever import retrieve
from src.settings import settings, groq_reasoning_kwargs
from src.skills import ToolRegistry
import src.skills.notifier  # noqa: F401 — registra tools del notifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipo de retorno
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    answer:                  str                    # Respuesta generada al usuario
    intent:                  str                    # Intención clasificada
    risk_level:              str                    # "low" | "medium" | "high"
    action:                  str                    # "educational_answer" | "medical_consultation" | "urgent_care"
    risk_flags:              list[str] = field(default_factory=list)
    sources:                 list[dict] = field(default_factory=list)
    reasoning:               str = ""
    tokens_used:             int = 0
    notified:                bool = False
    needs_clarification:     bool = False           # True → el sistema pide más info antes de responder
    clarification_question:  str = ""               # Pregunta empática para el usuario


# ---------------------------------------------------------------------------
# System prompt base
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = (
    "Eres Maternas, una asistente de salud dedicada a acompanar con calidez a madres "
    "gestantes y en puerperio. Tu objetivo es que cada mujer se sienta escuchada, "
    "apoyada e informada.\n\n"

    "COMO RESPONDER:\n"
    "- Usa un tono calido, cercano y empatico. Nunca frio ni clinico.\n"
    "- Responde de forma CONCISA y DIRECTA. No repitas la pregunta, no hagas "
    "introducciones largas, no agregues despedidas innecesarias.\n"
    "- Usa lenguaje sencillo. Evita tecnicismos salvo que sean imprescindibles "
    "(y si los usas, explicalos brevemente).\n"
    "- Responde siempre en espanol.\n\n"

    "SOBRE LAS FUENTES:\n"
    "- Cada fuente del contexto viene encabezada por [n] y el nombre del "
    "documento. Si una fuente respalda lo que afirmas, cita [n] al final "
    "de esa oracion especifica. Solo cita si la fuente realmente dice lo "
    "que afirmas — nunca cites para aparentar respaldo.\n"
    "- Nunca escribas la palabra 'Fragmento' ni menciones numeros de "
    "fragmento — [n] es la unica referencia que debes usar. Nunca "
    "repitas el nombre del documento dentro del texto de tu respuesta.\n"
    "- Si las fuentes no contienen la informacion exacta pero si informacion "
    "relacionada, usala como apoyo y complementa con conocimiento medico general "
    "bien establecido. En ese caso no es necesario aclarar 'no tengo fuentes' — "
    "simplemente responde con naturalidad.\n"
    "- Si las fuentes no tienen absolutamente nada relevante, responde desde "
    "conocimiento general sin citar [n].\n\n"

    "LIMITES:\n"
    "- No eres medico y no reemplazas una consulta. Cuando corresponda, "
    "orienta a consultar con su medico o matrona.\n"
    "- Nunca inventes datos clinicos, dosis ni procedimientos."
)

URGENT_SUFFIX = (
    "\n\nALERTA: Este mensaje contiene senales de alarma clinica. "
    "Comienza tu respuesta indicando de forma clara y directa que debe "
    "buscar atencion medica INMEDIATA. Se breve, urgente y empatica. "
    "No des informacion que pueda hacerla postergar ir a urgencias."
)

MEDIUM_SUFFIX = (
    "\n\nNOTA: El mensaje sugiere un sintoma que merece evaluacion medica. "
    "Incluye al final una recomendacion breve de consultar con su medico o matrona."
)


# ---------------------------------------------------------------------------
# Clarificación — reglas por intent y lógica de detección
# ---------------------------------------------------------------------------

# Intents donde nunca se pide clarificación (actuar de inmediato)
NEVER_CLARIFY = {"signos_de_alarma"}

# Intent → información mínima esperada para responder bien
# Si la query es corta Y no menciona ninguna de esas keywords, se activa la regla
CLARIFICATION_RULES: dict[str, dict] = {
    "medicamentos": {
        "min_tokens": 6,
        "keywords": ["para", "semana", "trimestre", "embarazo", "lactancia",
                     "dolor", "nausea", "fiebre", "infeccion", "gripa"],
        "missing_info": ["el síntoma o motivo", "en qué semana de gestación estás"],
    },
    "sintomas_embarazo": {
        "min_tokens": 5,
        "keywords": ["semana", "trimestre", "mes", "hace", "dias", "horas",
                     "primer", "segundo", "tercer", "meses"],
        "missing_info": ["cuántas semanas de gestación tienes"],
    },
    "control_prenatal": {
        "min_tokens": 6,
        "keywords": ["semana", "mes", "primera", "segunda", "vez", "trimestre",
                     "cuantas", "cuando", "proxima"],
        "missing_info": ["en qué semana o mes de embarazo estás"],
    },
    "nutricion": {
        "min_tokens": 5,
        "keywords": ["embarazo", "lactancia", "semana", "trimestre", "puedo",
                     "debo", "evitar", "comer", "tomar"],
        "missing_info": ["si estás embarazada o en periodo de lactancia"],
    },
    "actividad_fisica": {
        "min_tokens": 5,
        "keywords": ["semana", "trimestre", "mes", "cuantas", "embarazo",
                     "puedo", "seguro", "riesgo"],
        "missing_info": ["en qué trimestre de embarazo estás"],
    },
    "salud_mental_perinatal": {
        "min_tokens": 5,
        "keywords": ["semana", "parto", "embarazo", "bebe", "hace", "dias",
                     "meses", "postparto", "desde"],
        "missing_info": ["si estás embarazada o en el postparto, y hace cuánto tiempo sientes esto"],
    },
}


def _should_clarify(
    query: str,
    intent: str,
    risk_level: str,
    history: list[dict] | None = None,
) -> bool:
    """
    Determina si se debe pedir clarificación antes de responder.

    Reglas:
    - Nunca clarificar si risk != low (urgente o medio → responder siempre)
    - Nunca clarificar para intents en NEVER_CLARIFY
    - Nunca clarificar si ya hay historial en esta conversación — ya se pidió
      contexto antes (o la usuaria ya viene contando algo); preguntar de nuevo
      en cada turno corto hace que el sistema pierda de vista los síntomas
      previos en vez de acumularlos.
    - Nunca clarificar si la query ya es larga (≥ 20 tokens) — tiene suficiente contexto
    - Clarificar si el intent está en CLARIFICATION_RULES Y la query es corta
      Y no menciona ninguna keyword de contexto esperada
    """
    if risk_level != "low":
        return False
    if intent in NEVER_CLARIFY:
        return False
    if history:
        return False

    rule = CLARIFICATION_RULES.get(intent)
    if not rule:
        return False

    tokens = query.lower().split()
    if len(tokens) >= 20:
        return False

    # Si la query tiene pocos tokens Y ninguna keyword de contexto → clarificar
    has_context = any(kw in query.lower() for kw in rule["keywords"])
    if len(tokens) < rule["min_tokens"] or not has_context:
        return True

    return False


def _generate_clarification(
    query: str,
    intent: str,
    risk_level: str,
) -> str:
    """
    Genera una pregunta de clarificación empática y específica usando el LLM.
    Se llama solo cuando _should_clarify() devuelve True.
    """
    rule = CLARIFICATION_RULES.get(intent, {})
    missing = rule.get("missing_info", ["más información"])
    missing_str = " y ".join(missing)

    prompt = (
        "Eres Maternas, una asistente de salud calida y empatica para madres gestantes.\n\n"
        f"Una usuaria escribio: '{query}'\n\n"
        f"Para poder ayudarla bien, necesitas saber: {missing_str}.\n\n"
        "Escribe UNA sola pregunta de clarificacion, en espanol, con tono calido y cercano. "
        "Maximo 2 oraciones. No repitas la pregunta del usuario. "
        "No digas que eres una IA. Solo haz la pregunta de forma natural y amigable."
    )

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=700,
            extra_body=groq_reasoning_kwargs(),
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            logger.warning("[Chain] Clarificacion vacia del LLM (razonamiento sin respuesta final?) — usando fallback")
            return f"Con gusto te ayudo. Para darte la mejor respuesta, ¿podrías contarme {missing_str}?"
        return text
    except Exception as e:
        logger.warning(f"[Chain] Error generando clarificacion: {e}")
        # Fallback determinista si el LLM falla
        return f"Con gusto te ayudo. Para darte la mejor respuesta, ¿podrías contarme {missing_str}?"


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
# Construcción del prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(risk: RiskResult) -> str:
    prompt = BASE_SYSTEM_PROMPT
    if risk.level == "high":
        prompt += URGENT_SUFFIX
    elif risk.level == "medium":
        prompt += MEDIUM_SUFFIX
    return prompt


def _build_messages(
    query: str,
    context: str,
    history: list[dict],
    system_prompt: str,
) -> list[dict]:
    """
    Construye la lista de mensajes para el LLM.
    Incluye hasta los últimos 6 turnos del historial para no exceder
    el context window de Groq con historial largo.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Historial reciente (máx 6 turns = 3 pares user/assistant)
    recent_history = history[-6:] if len(history) > 6 else history
    messages.extend(recent_history)

    # Mensaje actual con contexto inyectado
    user_message = (
        f"CONTEXTO DE LA BASE DE CONOCIMIENTO MÉDICO:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"PREGUNTA DEL USUARIO:\n{query}"
    )
    messages.append({"role": "user", "content": user_message})

    return messages


# ---------------------------------------------------------------------------
# Clasificación del turno (intent + riesgo + clarificación) — compartida
# entre chat() y chat_stream() para que no puedan divergir.
# ---------------------------------------------------------------------------

@dataclass
class TurnClassification:
    intent_result:  IntentResult
    risk_result:    RiskResult
    needs_clarification: bool = False
    clarification_question: str = ""


def _classify_turn(query: str, history: list[dict]) -> TurnClassification:
    # 1. Clasificar intención
    intent_result: IntentResult = classify_intent(query, conversation_history=history)
    logger.info(f"[Chain] intent={intent_result.intent} conf={intent_result.confidence:.2f}")

    # 2. Detectar riesgo (combina síntomas de turnos previos con el mensaje actual)
    risk_result: RiskResult = detect_risk(query, intent=intent_result.intent, history=history)
    logger.info(f"[Chain] risk={risk_result.level} action={risk_result.action}")

    # 2b. Clarificación — pedir más contexto si la query es vaga
    if _should_clarify(query, intent_result.intent, risk_result.level, history):
        clarification_q = _generate_clarification(query, intent_result.intent, risk_result.level)
        logger.info(f"[Chain] Clarificacion activada para intent={intent_result.intent}")
        return TurnClassification(
            intent_result=intent_result,
            risk_result=risk_result,
            needs_clarification=True,
            clarification_question=clarification_q,
        )

    return TurnClassification(intent_result=intent_result, risk_result=risk_result)


# ---------------------------------------------------------------------------
# Notificación por riesgo clínico
# ---------------------------------------------------------------------------

def _run_notification(query: str, intent: str, risk_result: RiskResult) -> bool:
    """Decide si notificar a un clínico y, si corresponde, dispara el email.

    high → notifica siempre. medium → una llamada barata al LLM decide.
    Devuelve si se notificó, para popular ChatResponse.notified.
    """
    if risk_result.level == "high":
        ToolRegistry.execute("notify_risk", query=query, risk_level=risk_result.level,
                             intent=intent, reasoning=risk_result.reasoning,
                             flags=risk_result.flags)
        return True

    if risk_result.level != "medium":
        return False

    notify_prompt = (
        "Eres un clasificador medico. Decide si este mensaje de una paciente "
        "amerita notificar a un clinico para revision.\n\n"
        "Contexto:\n"
        f"- Nivel de riesgo: {risk_result.level}\n"
        f"- Intencion: {intent}\n"
        f"- Razonamiento: {risk_result.reasoning}\n"
        f"- Banderas: {risk_result.flags}\n\n"
        f"Mensaje de la paciente:\n{query}\n\n"
        "Responde SOLO con 'YES' si un medico debe revisar este caso, "
        "o 'NO' si no es necesario."
    )
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": notify_prompt}],
            temperature=0,
            max_tokens=512,
            extra_body=groq_reasoning_kwargs(),
        )
        decision = (resp.choices[0].message.content or "").strip().upper()
        if not decision:
            # No debe pasar inadvertido: un vacio aca hoy significa "no
            # notificar" sin dejar rastro, y es la ruta que decide si un
            # riesgo medio llega o no a un clinico.
            logger.warning("[Chain] Decision de notificacion vacia del LLM — tratando como NO")
        if "YES" in decision:
            ToolRegistry.execute("notify_risk", query=query, risk_level=risk_result.level,
                                 intent=intent, reasoning=risk_result.reasoning,
                                 flags=risk_result.flags)
            return True
    except Exception as e:
        logger.warning(f"[Chain] Error en decision de notificacion medium: {e}")
    return False


# ---------------------------------------------------------------------------
# Retrieval + construcción del prompt — compartido entre chat() y
# chat_stream().
# ---------------------------------------------------------------------------

def _retrieve_and_build(
    query: str,
    history: list[dict],
    risk_result: RiskResult,
    k: int | None,
) -> tuple[list[dict], list[dict]]:
    # Para preguntas fuera de alcance recuperamos igualmente por si acaso
    docs = retrieve(query, k=k)
    context = format_context(docs)
    logger.info(f"[Chain] {len(docs)} fragmentos recuperados")

    system_prompt = _build_system_prompt(risk_result)
    messages      = _build_messages(query, context, history, system_prompt)
    return docs, messages


def _docs_to_sources(docs: list[dict]) -> list[dict]:
    """Fuentes sin el texto completo, para no saturar el objeto de retorno."""
    return [{k: v for k, v in doc.items() if k != "text"} for doc in docs]


# ---------------------------------------------------------------------------
# Función pública principal (no streaming)
# ---------------------------------------------------------------------------

def chat(
    query: str,
    history: list[dict] | None = None,
    k: int | None = None,
) -> ChatResponse:
    """
    Procesa un turno completo del chatbot.

    Args:
        query:   Mensaje del usuario.
        history: Historial de la conversación (lista de dicts role/content).
                 Se modifica externamente por el caller.
        k:       Número de fragmentos a recuperar (default: settings.rag_top_k).

    Returns:
        ChatResponse con respuesta, intent, risk_level, sources, etc.
    """
    if history is None:
        history = []

    if not query or not query.strip():
        return ChatResponse(
            answer="No recibí ningún mensaje. ¿En qué puedo ayudarte?",
            intent="pregunta_fuera_de_alcance",
            risk_level="low",
            action="educational_answer",
        )

    turn = _classify_turn(query, history)
    intent_result, risk_result = turn.intent_result, turn.risk_result

    if turn.needs_clarification:
        return ChatResponse(
            answer=turn.clarification_question,
            intent=intent_result.intent,
            risk_level=risk_result.level,
            action=risk_result.action,
            risk_flags=risk_result.flags,
            reasoning=risk_result.reasoning,
            needs_clarification=True,
            clarification_question=turn.clarification_question,
        )

    notified = _run_notification(query, intent_result.intent, risk_result)

    docs, messages = _retrieve_and_build(query, history, risk_result, k)

    client = _get_client()
    tokens_used = 0

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=1800,
            extra_body=groq_reasoning_kwargs(),
        )
        answer      = (response.choices[0].message.content or "").strip()
        tokens_used = response.usage.total_tokens if response.usage else 0

        if not answer:
            logger.error("[Chain] Respuesta vacia del LLM (max_tokens agotado en razonamiento?)")
            answer = _fallback_answer(risk_result)
        else:
            answer = normalize_citation_brackets(answer)
            ref_block = build_reference_block(answer, docs)
            if ref_block:
                answer += "\n\n" + ref_block

    except Exception as e:
        logger.error(f"[Chain] Error generando respuesta: {e}")
        answer = _fallback_answer(risk_result)

    return ChatResponse(
        answer=answer,
        intent=intent_result.intent,
        risk_level=risk_result.level,
        action=risk_result.action,
        risk_flags=risk_result.flags,
        sources=_docs_to_sources(docs),
        reasoning=risk_result.reasoning,
        tokens_used=tokens_used,
        notified=notified,
    )


# ---------------------------------------------------------------------------
# Función pública — streaming
# ---------------------------------------------------------------------------

def chat_stream(
    query: str,
    history: list[dict] | None = None,
    k: int | None = None,
) -> Iterator[dict]:
    """
    Igual que chat(), pero emite el turno como una secuencia de eventos
    (dicts) en vez de devolver un único ChatResponse al final:

        {"type": "status", "stage": "classifying" | "retrieving" | "generating"}
        {"type": "meta", "intent", "risk_level", "action", "risk_flags",
                  "sources", "needs_clarification"}
        {"type": "delta", "text": "..."}                 # uno por token
        {"type": "done", "answer", "tokens_used", "notified"}
        {"type": "error", "detail": "..."}

    El caller (src/api/main.py) serializa cada evento a una línea NDJSON.
    La notificación de riesgo clínico corre en un hilo aparte, en paralelo
    con la generación, en vez de bloquear antes de ella.
    """
    if history is None:
        history = []

    if not query or not query.strip():
        yield {"type": "meta", "intent": "pregunta_fuera_de_alcance", "risk_level": "low",
               "action": "educational_answer", "risk_flags": [], "sources": [],
               "needs_clarification": False}
        yield {"type": "delta", "text": "No recibí ningún mensaje. ¿En qué puedo ayudarte?"}
        yield {"type": "done", "answer": "No recibí ningún mensaje. ¿En qué puedo ayudarte?",
               "tokens_used": 0, "notified": False}
        return

    yield {"type": "status", "stage": "classifying"}
    turn = _classify_turn(query, history)
    intent_result, risk_result = turn.intent_result, turn.risk_result

    if turn.needs_clarification:
        yield {"type": "meta", "intent": intent_result.intent, "risk_level": risk_result.level,
               "action": risk_result.action, "risk_flags": risk_result.flags, "sources": [],
               "needs_clarification": True}
        yield {"type": "delta", "text": turn.clarification_question}
        yield {"type": "done", "answer": turn.clarification_question,
               "tokens_used": 0, "notified": False}
        return

    # Notificación en paralelo: arranca ahora, se recoge recién al final.
    notify_result: dict[str, bool] = {}

    def _notify_worker() -> None:
        notify_result["notified"] = _run_notification(query, intent_result.intent, risk_result)

    notify_thread = threading.Thread(target=_notify_worker, daemon=True)
    notify_thread.start()

    yield {"type": "status", "stage": "retrieving"}
    docs, messages = _retrieve_and_build(query, history, risk_result, k)

    sources = _docs_to_sources(docs)
    yield {
        "type": "meta",
        "intent": intent_result.intent,
        "risk_level": risk_result.level,
        "action": risk_result.action,
        "risk_flags": risk_result.flags,
        "sources": sources,
        "needs_clarification": False,
    }

    yield {"type": "status", "stage": "generating"}
    client = _get_client()
    answer_parts: list[str] = []
    tokens_used = 0

    try:
        stream = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=1800,
            stream=True,
            extra_body=groq_reasoning_kwargs(),
        )
        for chunk in stream:
            # groq==0.13.1 no acepta stream_options={"include_usage": True}
            # (SDK viejo): el uso de tokens en modo streaming viaja en el
            # chunk final bajo x_groq.usage, no bajo chunk.usage.
            usage = getattr(chunk, "usage", None) or getattr(
                getattr(chunk, "x_groq", None), "usage", None
            )
            if usage is not None:
                tokens_used = getattr(usage, "total_tokens", 0) or 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                answer_parts.append(text)
                yield {"type": "delta", "text": text}

    except Exception as e:
        logger.error(f"[Chain] Error en streaming: {e}")
        if not answer_parts:
            # Nada transmitido todavía: degradar a la respuesta de emergencia
            # en vez de dejar a la usuaria sin nada.
            fallback = _fallback_answer(risk_result)
            yield {"type": "delta", "text": fallback}
            notify_thread.join(timeout=10)
            yield {"type": "done", "answer": fallback, "tokens_used": 0,
                   "notified": notify_result.get("notified", False)}
            return
        notify_thread.join(timeout=10)
        yield {"type": "error", "detail": str(e)[:200]}
        return

    answer = normalize_citation_brackets("".join(answer_parts).strip())
    ref_block = build_reference_block(answer, docs)
    if ref_block:
        answer += "\n\n" + ref_block

    notify_thread.join(timeout=10)
    yield {
        "type": "done",
        "answer": answer,
        "tokens_used": tokens_used,
        "notified": notify_result.get("notified", False),
    }


def _fallback_answer(risk: RiskResult) -> str:
    """Respuesta de emergencia si el LLM falla."""
    if risk.level == "high":
        return (
            "⚠️ He detectado posibles señales de alarma en tu mensaje. "
            "Por favor busca atención médica urgente de inmediato. "
            "No puedo brindarte más información ahora debido a un error técnico."
        )
    return (
        "Lo siento, tuve un problema técnico al generar la respuesta. "
        "Por favor intenta de nuevo. Si tienes alguna urgencia médica, "
        "contacta a tu médico o ve a urgencias."
    )
