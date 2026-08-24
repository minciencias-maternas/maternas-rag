"""
routes_admin.py — Métricas de evaluación, configuración editable y logs
de la API, para el panel de administración.

Las evaluaciones son de solo lectura (lee los reportes que
src/evaluation/eval_pipeline.py ya generó en evaluation_reports/, sin
recomputar nada). La configuración SÍ es editable — ver PATCH /config —
para un conjunto fijo y acotado de variables (ver UpdateConfigRequest en
schemas.py): nunca se acepta un nombre de variable libre, así que no hay
forma de pedirle a este endpoint que toque algo fuera de esa lista (p.
ej. ADMIN_API_TOKEN).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import dotenv
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api import usage_sessions
from src.api.auth import require_admin_token
from src.api.schemas import (
    AdminConfigResponse,
    ApiLogsResponse,
    EditableConfigView,
    EvaluationDetailResponse,
    EvaluationListResponse,
    EvaluationSummary,
    UpdateConfigRequest,
    UpdateConfigResponse,
    UsageSessionRow,
    UsageSessionsResponse,
)
from src.classifiers import intent_classifier, risk_detector
from src.rag import chain
from src.rag.retriever import DENSE_SOURCES, _get_store
from src.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["administracion"], dependencies=[Depends(require_admin_token)])

REPORTS_DIR = Path("evaluation_reports")   # igual que src/evaluation/eval_pipeline.py

# Ruta real de .env — mismo patrón parents[2] que el bootstrap de sys.path
# en src/api/main.py (este archivo está en src/api/, dos niveles bajo raíz).
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

# field de UpdateConfigRequest -> nombre de la variable en .env
ENV_KEY_MAP = {
    "groq_model":                            "GROQ_MODEL",
    "groq_api_key":                          "GROQ_API_KEY",
    "notifier_enabled":                      "NOTIFIER_ENABLED",
    "notifier_email_to":                     "NOTIFIER_EMAIL_TO",
    "notifier_smtp_user":                    "NOTIFIER_SMTP_USER",
    "notifier_smtp_password":                "NOTIFIER_SMTP_PASSWORD",
    "telegram_bot_token":                    "TELEGRAM_BOT_TOKEN",
    "status_check_interval_low_seconds":     "STATUS_CHECK_INTERVAL_LOW_SECONDS",
    "status_check_interval_medium_seconds":  "STATUS_CHECK_INTERVAL_MEDIUM_SECONDS",
    "status_check_interval_high_seconds":    "STATUS_CHECK_INTERVAL_HIGH_SECONDS",
}

# Variables que solo lee el proceso del bot de Telegram (settings.py se
# carga una sola vez, al importar) — un cambio acá no le llega al bot
# corriendo, hace falta reiniciarlo (ver src/api/bot_supervisor.py).
BOT_SCOPED_FIELDS = {
    "telegram_bot_token",
    "status_check_interval_low_seconds",
    "status_check_interval_medium_seconds",
    "status_check_interval_high_seconds",
}

_config_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_run_path(run_id: str) -> Path:
    """Resuelve run_id -> evaluation_reports/eval_results_<run_id>.json,
    verificando que el resultado siga dentro de REPORTS_DIR (sin esto,
    un run_id tipo '../../secrets' podría leer fuera del directorio)."""
    reports_dir = REPORTS_DIR.resolve()
    candidate = (reports_dir / f"eval_results_{run_id}.json").resolve()
    if reports_dir not in candidate.parents:
        raise HTTPException(status_code=400, detail="run_id inválido")
    return candidate


def _list_run_files() -> list[Path]:
    if not REPORTS_DIR.exists():
        return []
    return sorted(REPORTS_DIR.glob("eval_results_*.json"))


# ---------------------------------------------------------------------------
# GET /admin/evaluations
# ---------------------------------------------------------------------------

@router.get("/evaluations", response_model=EvaluationListResponse)
def list_evaluations() -> EvaluationListResponse:
    """Lista las corridas de evaluación ya generadas, más recientes primero."""
    runs = []
    for path in _list_run_files():
        run_id = path.stem.removeprefix("eval_results_")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("No se pudo leer el reporte de evaluación '%s'", path)
            continue

        runs.append(EvaluationSummary(
            run_id=run_id,
            config=data.get("config", ""),
            timestamp=data.get("timestamp", ""),
            dataset=data.get("dataset", ""),
            n_evaluated=data.get("n_evaluated", 0),
            n_failed=data.get("n_failed", 0),
            metrics_global=data.get("metrics_global", {}),
        ))

    runs.sort(key=lambda r: r.timestamp, reverse=True)
    return EvaluationListResponse(runs=runs)


# ---------------------------------------------------------------------------
# GET /admin/evaluations/{run_id}
# ---------------------------------------------------------------------------

@router.get("/evaluations/{run_id}", response_model=EvaluationDetailResponse)
def evaluation_detail(run_id: str, include_rows: bool = Query(False)) -> EvaluationDetailResponse:
    """Detalle completo de una corrida. 'rows' (por-pregunta) se omite por
    defecto: es el grueso del archivo y solo hace falta para inspección fina."""
    path = _resolve_run_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Corrida '{run_id}' no encontrada")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Error al leer el reporte de evaluación '%s'", path)
        raise HTTPException(status_code=500, detail="No se pudo leer el reporte de evaluación")

    return EvaluationDetailResponse(
        run_id=run_id,
        config=data.get("config", ""),
        timestamp=data.get("timestamp", ""),
        dataset=data.get("dataset", ""),
        n_sample=data.get("n_sample", 0),
        n_evaluated=data.get("n_evaluated", 0),
        n_failed=data.get("n_failed", 0),
        metrics_global=data.get("metrics_global", {}),
        metrics_by_tipo=data.get("metrics_by_tipo", {}),
        metrics_by_dificultad=data.get("metrics_by_dificultad", {}),
        rows=data.get("rows") if include_rows else None,
    )


# ---------------------------------------------------------------------------
# GET /admin/config
# ---------------------------------------------------------------------------

@router.get("/config", response_model=AdminConfigResponse)
def admin_config() -> AdminConfigResponse:
    """Configuración efectiva del proceso, con secretos redactados a un
    booleano 'configurado'. Nunca se expone el valor de un secreto."""
    try:
        store = _get_store()
        build_info = store.build_info()
    except Exception:
        build_info = {}

    return AdminConfigResponse(
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        rag_top_k=settings.rag_top_k,
        dense_sources=sorted(DENSE_SOURCES),
        groq_model=settings.groq_model,
        index_build_info=build_info,
        secrets_configured={
            "groq_api_key":              bool(settings.groq_api_key),
            "groq_api_key_2":            bool(settings.groq_api_key_2),
            "telegram_bot_token":        bool(settings.telegram_bot_token),
            "admin_api_token":           bool(settings.admin_api_token),
            "openrouter_key":            bool(settings.openrouter_key),
            "cerebras_key":              bool(settings.cerebras_key),
            "active_users_encryption_key": bool(settings.active_users_encryption_key),
            "notifier_smtp_password":    bool(settings.notifier_smtp_password),
        },
        editable=EditableConfigView(
            groq_model=settings.groq_model,
            notifier_enabled=settings.notifier_enabled,
            notifier_email_to=settings.notifier_email_to,
            notifier_smtp_user=settings.notifier_smtp_user,
            status_check_interval_low_seconds=settings.status_check_interval_low_seconds,
            status_check_interval_medium_seconds=settings.status_check_interval_medium_seconds,
            status_check_interval_high_seconds=settings.status_check_interval_high_seconds,
        ),
    )


# ---------------------------------------------------------------------------
# PATCH /admin/config
# ---------------------------------------------------------------------------

def _env_string(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _validate_env_value(env_key: str, value: str) -> None:
    # Segunda barrera además de max_length en el schema: ningún valor
    # puede inyectar una línea nueva en .env (p. ej. "x\nADMIN_API_TOKEN=y"),
    # que en la próxima lectura del archivo definiría una variable aparte.
    if "\n" in value or "\r" in value:
        raise HTTPException(
            status_code=400,
            detail=f"{env_key}: el valor no puede contener saltos de línea",
        )


@router.patch("/config", response_model=UpdateConfigResponse)
def update_config(body: UpdateConfigRequest) -> UpdateConfigResponse:
    """Actualiza settings en memoria Y persiste en .env — ambos, para que
    el cambio no se pierda en el próximo reinicio del proceso.

    Solo acepta los campos declarados en UpdateConfigRequest (extra
    "forbid"): no hay forma de pedirle que toque una variable fuera de
    esa lista. Los 4 campos de BOT_SCOPED_FIELDS los usa únicamente el
    proceso del bot de Telegram, que ya cargó su propio settings al
    arrancar — por eso 'requires_bot_restart' avisa que ese cambio no
    aplica hasta reiniciarlo (ver /admin/bot/restart).
    """
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No se envió ningún cambio")

    with _config_write_lock:
        for field, value in updates.items():
            env_key = ENV_KEY_MAP[field]
            str_value = _env_string(value)
            _validate_env_value(env_key, str_value)
            dotenv.set_key(str(ENV_PATH), env_key, str_value)
            setattr(settings, field, value)

        if "groq_api_key" in updates:
            chain.reset_client()
            intent_classifier.reset_client()
            risk_detector.reset_client()

    changed = sorted(updates.keys())
    logger.info("[AdminConfig] Variables actualizadas: %s", changed)   # nombres, nunca valores

    return UpdateConfigResponse(
        updated=changed,
        requires_bot_restart=bool(BOT_SCOPED_FIELDS & updates.keys()),
        config=admin_config(),
    )


# ---------------------------------------------------------------------------
# GET /admin/logs — logs recientes del propio proceso de la API
# ---------------------------------------------------------------------------

@router.get("/logs", response_model=ApiLogsResponse)
def api_logs(limit: int = Query(200, ge=1, le=1000)) -> ApiLogsResponse:
    from src.api.main import _api_started_at, _log_buffer

    uptime = (datetime.now(timezone.utc) - _api_started_at).total_seconds()
    return ApiLogsResponse(
        started_at=_api_started_at.isoformat(),
        uptime_seconds=uptime,
        lines=list(_log_buffer)[-limit:],
    )


# ---------------------------------------------------------------------------
# GET /admin/usage_sessions — sesiones activas en tiempo real, anonimizado
# ---------------------------------------------------------------------------

@router.get("/usage_sessions", response_model=UsageSessionsResponse)
def usage_sessions_endpoint() -> UsageSessionsResponse:
    """Sesiones activas de Streamlit y Telegram (últimos
    usage_sessions.IDLE_TIMEOUT_SECONDS de actividad), con duración y
    tokens por sesión. A propósito no lleva session_id ni nada derivado
    de él — ver usage_sessions.py y UsageSessionsResponse."""
    data = usage_sessions.active_by_platform()
    return UsageSessionsResponse(
        streamlit=[UsageSessionRow(**row) for row in data["streamlit"]],
        telegram=[UsageSessionRow(**row) for row in data["telegram"]],
    )
