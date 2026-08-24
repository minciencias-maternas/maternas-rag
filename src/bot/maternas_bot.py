"""
maternas_bot.py — Bot de Telegram para Maternas.

Conecta el chatbot RAG con Telegram usando polling.
Requiere la API FastAPI corriendo en localhost:8080.

Incluye el Status Check Scheduler: mensajes automaticos de seguimiento
a usuarios activos, con frecuencia segun su nivel de riesgo acumulado
(ver src/bot/active_users.py). Un solo proceso — no requiere terminal
ni servicio aparte.

Arrancar:
    python src/bot/maternas_bot.py

Comandos:
    /start  — muestra el aviso de tratamiento de datos (nueva sesion)
    /help   — instrucciones de uso
    /reset  — reinicia la conversacion y vuelve a exigir ambas capas
    /stats  — info del sistema (vectores indexados)

Toda sesion nueva (bot recien iniciado, /start o /reset) exige aceptar DOS
capas SECUENCIALES (src/consent.py) antes de procesar cualquier mensaje:
1. aviso de tratamiento de datos (CONSENT_TEXT), y
2. Terminos y Condiciones de uso (TERMS_TEXT) — recien despues de aceptar
   la capa 1.
Si el usuario rechaza cualquiera de las dos, la sesion termina: cualquier
mensaje nuevo vuelve a mostrar la capa 1 desde cero.
"""

from __future__ import annotations

import asyncio
import hashlib
import httpx
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, constants
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    Job,
    MessageHandler,
    filters,
)
from src.settings import settings
from src.consent import CONSENT_TEXT, FAREWELL_TEXT, TERMS_ACCEPTED_TEXT, TERMS_TEXT

from src.bot.active_users import (
    register as register_active_user,
    get_all as get_active_users,
    update_check_sent,
    remove as remove_active_user,
)

# ---------------------------------------------------------------------------
# Anonimización de identificadores de Telegram
#
# El chat_id/user_id real nunca se usa como clave de indexación interna ni
# aparece en logs — se deriva un hash SHA-256 (con sal fija del proyecto).
# El identificador real solo se usa en el punto mínimo necesario: la llamada
# a la API de Telegram para enrutar el mensaje (context.bot.send_message /
# update.message.reply_text), que python-telegram-bot ya resuelve a partir
# del objeto Update entrante, no de nuestras estructuras internas.
# ---------------------------------------------------------------------------

_HASH_SALT = "maternas-bot:"


def _hash_id(user_id: int) -> str:
    """Hash completo (64 hex) — usado como clave interna, sin colisiones prácticas."""
    return hashlib.sha256(f"{_HASH_SALT}{user_id}".encode()).hexdigest()


def _short_hash(user_id: int) -> str:
    """Hash truncado (10 hex) — solo para trazabilidad legible en logs."""
    return _hash_id(user_id)[:10]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL      = "http://localhost:8080"
API_TIMEOUT  = 60
TOKEN        = settings.telegram_bot_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Historial conversacional en memoria: { hash(user_id): [{"role": ..., "content": ...}] }
# Indexado por hash, no por el user_id real de Telegram. Se pierde al
# reiniciar el bot — suficiente para MVP.
# ---------------------------------------------------------------------------

histories: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# session_id de "uso en tiempo real": { hash(user_id): uuid4 } — aleatorio,
# nunca derivado del chat_id real. Se regenera en /start y /reset (cierran
# la sesión anterior para las métricas: ver src/api/usage_sessions.py, la
# entrada vieja simplemente deja de recibir turnos y expira sola por
# inactividad). En RAM únicamente, igual que histories.
# ---------------------------------------------------------------------------

usage_session_ids: dict[str, str] = {}


def _usage_session_id(user_id: int) -> str:
    key = _hash_id(user_id)
    if key not in usage_session_ids:
        usage_session_ids[key] = str(uuid.uuid4())
    return usage_session_ids[key]


def _new_usage_session(user_id: int) -> None:
    usage_session_ids[_hash_id(user_id)] = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Consentimiento de tratamiento de datos + Términos y Condiciones:
# { hash(user_id): "accepted" | "rejected" } — dos dicts, dos capas
# SECUENCIALES (primero consent_status, recién después terms_status).
# Toda nueva sesión (bot recién iniciado, /start o /reset) empieza sin
# entrada en ninguno de los dos — se exige aceptación explícita de ambos
# antes de procesar cualquier mensaje. En RAM únicamente, no se persiste.
# ---------------------------------------------------------------------------

consent_status: dict[str, str] = {}
terms_status: dict[str, str] = {}


def _is_fully_accepted(key: str) -> bool:
    return consent_status.get(key) == "accepted" and terms_status.get(key) == "accepted"


def _consent_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Acepto", callback_data="consent_accept"),
        InlineKeyboardButton("❌ No acepto", callback_data="consent_reject"),
    ]])


def _terms_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Acepto", callback_data="terms_accept"),
        InlineKeyboardButton("❌ No acepto", callback_data="terms_reject"),
    ]])


async def _send_consent_prompt(update: Update) -> None:
    await update.message.reply_text(CONSENT_TEXT, reply_markup=_consent_keyboard())


async def _send_terms_prompt(update: Update) -> None:
    await update.message.reply_text(TERMS_TEXT, reply_markup=_terms_keyboard())


async def _send_next_prompt(update: Update, key: str) -> None:
    """Muestra la capa que falta aceptar — datos primero, T&C después."""
    if consent_status.get(key) != "accepted":
        await _send_consent_prompt(update)
    else:
        await _send_terms_prompt(update)


def _end_session(key: str, user_id: int) -> None:
    """Limpieza compartida al rechazar cualquiera de las dos capas, o al
    empezar una sesión nueva (/start, /reset): ambos estados de
    aceptación, historial, scheduler de riesgo y sesión de métricas."""
    consent_status.pop(key, None)
    terms_status.pop(key, None)
    histories.pop(key, None)
    remove_active_user(user_id)
    usage_session_ids.pop(key, None)

# ---------------------------------------------------------------------------
# Flag de salud de la API (para el scheduler)
# ---------------------------------------------------------------------------
# Se actualiza en cada handle_message(). Si la API no responde, el scheduler
# no enviará status checks (congruencia: no mandar check si el bot está caído).
# ---------------------------------------------------------------------------

_api_healthy: bool = False

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def call_chat(message: str, user_id: int) -> dict | None:
    payload = {
        "message": message,
        "history": histories.get(_hash_id(user_id), []),
        "k": 5,
        "session_id": _usage_session_id(user_id),
        "platform": "telegram",
    }
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            r = await client.post(f"{API_URL}/chat", json=payload)
            if r.status_code == 200:
                return r.json()
            logger.error(f"API error {r.status_code}: {r.text[:200]}")
    except httpx.ConnectError:
        logger.error("No se pudo conectar con la API en localhost:8080")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
    return None

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

WELCOME_TEXT = (
    "Soy Maternas, un asistente de salud para madres gestantes.\n\n"
    "Puedes preguntarme sobre:\n"
    "• Síntomas del embarazo\n"
    "• Nutrición y ejercicios\n"
    "• Medicamentos y suplementos\n"
    "• Lactancia y postparto\n"
    "• Salud mental perinatal\n\n"
    "Comandos:\n"
    "/help  — más información\n"
    "/reset — reiniciar conversación\n"
    "/stats — estado del sistema\n\n"
    "⚠️ No reemplazo a un médico — si tienes una emergencia, busca atención profesional."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /start siempre marca el inicio de una nueva sesión — se exige
    # aceptar de nuevo AMBAS capas (datos, luego T&C), aunque ya se
    # hubieran aceptado antes.
    key = _hash_id(update.effective_user.id)
    consent_status.pop(key, None)
    terms_status.pop(key, None)
    _new_usage_session(update.effective_user.id)
    await _send_consent_prompt(update)


async def consent_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user = query.from_user
    key = _hash_id(user.id)

    if query.data == "consent_accept":
        consent_status[key] = "accepted"
        # Capa 1 aceptada -> mostrar la capa 2 (T&C) a continuación, en el
        # mismo mensaje editado; el chat todavía no se habilita.
        await query.edit_message_text(TERMS_TEXT, reply_markup=_terms_keyboard())
    else:
        _end_session(key, user.id)
        await query.edit_message_text(FAREWELL_TEXT, reply_markup=None)


async def terms_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user = query.from_user
    key = _hash_id(user.id)

    if query.data == "terms_accept":
        terms_status[key] = "accepted"
        await query.edit_message_text(TERMS_ACCEPTED_TEXT, reply_markup=None)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"¡Hola {user.first_name}! 🤰\n\n{WELCOME_TEXT}",
        )
    else:
        _end_session(key, user.id)
        await query.edit_message_text(FAREWELL_TEXT, reply_markup=None)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🤰 **Maternas** — asistente de salud materna\n\n"
        "Escribe tu pregunta en lenguaje natural. El sistema:\n"
        "1. Clasifica tu intención (síntomas, nutrición, alarma, etc.)\n"
        "2. Evalúa el riesgo clínico\n"
        "3. Busca información médica relevante\n"
        "4. Genera una respuesta fundamentada\n\n"
        "Ejemplos:\n"
        "• \"¿Es normal tener náuseas a las 10 semanas?\"\n"
        "• \"¿Qué alimentos debo evitar?\"\n"
        "• \"Tengo dolor de cabeza intenso y veo manchas\"\n\n"
        "Si tienas una urgencia, por favor contacta a tu médico de inmediato.\n\n"
        "📖 Fuentes de datos: MedMCQA (Apache 2.0), MedQA (MIT), MultiClinSum "
        "(CC-BY 4.0, BioASQ/CLEF 2025), MaternaQA-es (MIT). Detalle en el README del proyecto.",
        parse_mode=constants.ParseMode.MARKDOWN,
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    key = _hash_id(user_id)
    if key in histories:
        del histories[key]
    # Dar de baja del scheduler — /reset también detiene los check-ins automáticos
    remove_active_user(user_id)
    # /reset también cuenta como nueva sesión — se vuelven a exigir AMBAS capas
    consent_status.pop(key, None)
    terms_status.pop(key, None)
    _new_usage_session(user_id)
    await _send_consent_prompt(update)

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{API_URL}/health")
            if r.status_code == 200:
                data = r.json()
                await update.message.reply_text(
                    f"📊 **Estado del sistema**\n\n"
                    f"• Fragmentos médicos: {data.get('total_vectors', 0):,}\n"
                    f"• Modelo embedding: {data.get('model', '?').split('/')[-1]}\n"
                    f"• FAISS cargado: {'✅' if data.get('faiss_loaded') else '❌'}\n"
                    f"• Usuarios activos en este turno: {len(histories)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
            else:
                await update.message.reply_text("❌ No se pudo conectar con la API.")
    except Exception:
        await update.message.reply_text("❌ API no disponible. Asegúrate de que el servidor esté corriendo en localhost:8080.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global _api_healthy

    user_id = update.effective_user.id
    text    = update.message.text.strip()

    if not text:
        return

    # Sin aceptación vigente de AMBAS capas (nunca se pidió, o el usuario
    # rechazó alguna antes): cualquier mensaje nuevo vuelve a mostrar la
    # capa que falte, en vez de procesarse.
    key_gate = _hash_id(user_id)
    if not _is_fully_accepted(key_gate):
        await _send_next_prompt(update, key_gate)
        return

    # Indicador de escritura
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    result = await call_chat(text, user_id)

    if result is None:
        _api_healthy = False
        await update.message.reply_text(
            "❌ No pude conectarme con el sistema. Verifica que la API esté funcionando."
        )
        return

    _api_healthy = True

    answer               = result.get("answer", "No se pudo generar una respuesta.")
    needs_clarification  = result.get("needs_clarification", False)

    # Siempre guardamos en el historial, incluso durante clarificación — de lo
    # contrario el sistema pierde de vista los síntomas mencionados antes de
    # que la usuaria responda la pregunta de clarificación.
    key = _hash_id(user_id)
    if key not in histories:
        histories[key] = []
    histories[key].append({"role": "user",      "content": text})
    histories[key].append({"role": "assistant", "content": answer})

    # Registro para el scheduler de status checks — se hace aunque la
    # respuesta sea una pregunta de clarificación, porque el riesgo ya
    # se evaluó sobre el mensaje. No se persisten las banderas clínicas
    # descriptivas (risk_flags), solo el nivel agregado.
    register_active_user(
        user_id,
        risk_level=result.get("risk_level", "low"),
    )

    if needs_clarification:
        await update.message.reply_text(f"💬 {answer}")
        return

    await update.message.reply_text(answer[:4000])

async def handle_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    key = _hash_id(update.effective_user.id)
    if not _is_fully_accepted(key):
        await _send_next_prompt(update, key)
        return
    await update.message.reply_text(
        "Solo entiendo mensajes de texto. Por favor escribe tu pregunta."
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # No se loguea el objeto Update completo — contiene first_name, username
    # y el texto del mensaje del usuario (posiblemente dato clínico).
    ref = "desconocido"
    if isinstance(update, Update) and update.effective_user:
        ref = _short_hash(update.effective_user.id)
    logger.error(f"Error procesando update de usuario {ref}: {context.error}")

# ---------------------------------------------------------------------------
# Status Check Scheduler (integrado — usa la JobQueue nativa de PTB)
# ---------------------------------------------------------------------------
# Cada usuario activo recibe un mensaje periódico de "check de estado".
# La frecuencia depende del nivel de riesgo:
#   LOW    → cada STATUS_CHECK_INTERVAL_LOW_SECONDS    (default: 60s)
#   MEDIUM → cada STATUS_CHECK_INTERVAL_MEDIUM_SECONDS (default: 45s)
#   HIGH   → cada STATUS_CHECK_INTERVAL_HIGH_SECONDS   (default: 30s)
#
# El scheduler solo envía checks si _api_healthy == True. Esto asegura
# congruencia: si el bot no puede hablar con la API, no se mandan
# mensajes como si todo funcionara.
# ---------------------------------------------------------------------------

_user_status_jobs: dict[str, Job] = {}
_user_risk_levels: dict[str, str] = {}


def _check_interval(risk_level: str) -> float:
    return {
        "low":    settings.status_check_interval_low_seconds,
        "medium": settings.status_check_interval_medium_seconds,
        "high":   settings.status_check_interval_high_seconds,
    }.get(risk_level, settings.status_check_interval_low_seconds)


async def _send_status_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envía el mensaje de check de estado a un usuario (solo si API responde)."""
    if not _api_healthy:
        logger.debug("API no disponible — status check omitido")
        return

    chat_id = context.job.data
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=settings.status_check_message,
            parse_mode=constants.ParseMode.MARKDOWN,
        )
        update_check_sent(chat_id)
        logger.debug("Status check enviado a %s", _short_hash(chat_id))
    except Exception as e:
        logger.warning("Error enviando status check a %s: %s", _short_hash(chat_id), e)


async def _sync_user_jobs(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sincroniza los jobs de status check con active_users.json."""
    users = get_active_users()
    active = set(users.keys())
    existing = set(_user_status_jobs.keys())

    # Eliminar jobs de usuarios que ya no existen (incluye los dados de baja con /reset)
    for chat_id in existing - active:
        _user_status_jobs.pop(chat_id, None).schedule_removal()
        _user_risk_levels.pop(chat_id, None)
        logger.debug("Job eliminado para chat %s", _short_hash(chat_id))

    # Crear o actualizar jobs según riesgo
    for chat_id, user_data in users.items():
        risk_level = user_data.get("latest_risk_level", "low")
        interval = _check_interval(risk_level)

        if chat_id not in existing:
            job = context.job_queue.run_repeating(
                _send_status_check,
                interval=interval,
                first=interval,
                data=int(chat_id),
                name=f"status_check_{chat_id}",
            )
            _user_status_jobs[chat_id] = job
            _user_risk_levels[chat_id] = risk_level
            logger.debug(
                "Job creado para %s — riesgo=%s, cada %.0fs",
                _short_hash(chat_id), risk_level, interval,
            )
        else:
            prev = _user_risk_levels.get(chat_id)
            if prev != risk_level:
                _user_status_jobs[chat_id].schedule_removal()
                job = context.job_queue.run_repeating(
                    _send_status_check,
                    interval=interval,
                    first=interval,
                    data=int(chat_id),
                    name=f"status_check_{chat_id}",
                )
                _user_status_jobs[chat_id] = job
                _user_risk_levels[chat_id] = risk_level
                logger.debug(
                    "Job re-programado para %s — riesgo %s→%s, cada %.0fs",
                    _short_hash(chat_id), prev, risk_level, interval,
                )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN no configurado en .env")
        sys.exit(1)

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CallbackQueryHandler(consent_callback, pattern="^consent_"))
    app.add_handler(CallbackQueryHandler(terms_callback, pattern="^terms_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(~filters.TEXT, handle_non_text))
    app.add_error_handler(error_handler)

    # ── Status Check Scheduler (integrado) ──
    app.job_queue.run_repeating(
        _sync_user_jobs,
        interval=15,
        first=5,
        name="sync_user_jobs",
    )
    logger.info("Scheduler de status check integrado — sync cada 15s")

    logger.info("Bot Maternas iniciado. Presiona Ctrl+C para detener.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
