"""
metrics_view.py — Vista de Métricas de evaluación RAG.

Lee las corridas que src/evaluation/eval_pipeline.py ya generó en
evaluation_reports/ (vía GET /admin/evaluations*). No recomputa nada.
"""

import httpx
import pandas as pd
import streamlit as st

from src.ui.client import get_evaluation_detail, get_usage_sessions, list_evaluations

# Mismo semáforo que usa el reporte markdown del pipeline de evaluación
# (foragents/eval_runbook.md): 🟢 ≥0.80 · 🟡 0.60–0.80 · 🔴 <0.60
SCORE_METRICS = [
    ("faithfulness", "Faithfulness"),
    ("answer_correctness", "Corrección de respuesta"),
    ("answer_relevancy", "Relevancia de respuesta"),
    ("context_recall", "Recall de contexto"),
    ("context_precision", "Precisión de contexto"),
]


def _semaforo(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if v >= 0.80:
        return "🟢"
    if v >= 0.60:
        return "🟡"
    return "🔴"


def _format_duration(seconds) -> str:
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return "—"
    m, s = divmod(max(0, s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _render_usage_platform(label: str, icon: str, sessions: list[dict]) -> None:
    with st.container(border=True):
        st.caption(f"{icon} {label}")
        st.write(f"{len(sessions)} sesión(es) activa(s)")
        if sessions:
            rows = [
                {
                    "Duración activa": _format_duration(s.get("active_seconds")),
                    "Tokens usados":   f"{s.get('tokens_total', 0):,}",
                }
                for s in sessions
            ]
            # Sin columna de identificador ni index estable entre refrescos —
            # el orden ya viene por duración descendente desde el backend
            # (usage_sessions.active_by_platform), no hay forma de mapear
            # una fila a "la misma persona" de un refresh al siguiente.
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_usage_section(admin_token: str) -> None:
    st.subheader("Uso en tiempo real")
    st.caption(
        "Sesiones activas (sin actividad hace más de 15 min se consideran cerradas). "
        "No se muestra ningún identificador de sesión ni de usuario — solo duración y "
        "tokens, para que no sea posible diferenciar una sesión de otra."
    )
    try:
        usage = get_usage_sessions(admin_token)
    except httpx.HTTPError:
        st.warning("No se pudo cargar el uso en tiempo real.")
        return

    col1, col2 = st.columns(2)
    with col1:
        _render_usage_platform("Streamlit", "💻", usage.get("streamlit", []))
    with col2:
        _render_usage_platform("Telegram", "✈️", usage.get("telegram", []))

    st.write("")


def _render_breakdown(title: str, data: dict) -> None:
    if not data:
        return
    st.subheader(title)
    rows = []
    for group, gmetrics in data.items():
        row = {"Grupo": group, "N": gmetrics.get("n", "—")}
        for key, label in SCORE_METRICS:
            v = gmetrics.get(key)
            row[label] = round(v, 4) if isinstance(v, (int, float)) else "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_metrics() -> None:
    if not st.session_state.get("is_admin"):
        st.error("Acceso restringido a administradores.")
        st.stop()

    st.title("📊 Métricas")
    st.caption("Resultados de las corridas de evaluación RAG (src/evaluation/eval_pipeline.py).")
    st.divider()

    if not st.session_state.get("api_ok", False):
        st.warning("La API no está disponible. Inicia el servidor para ver las métricas.")
        return

    admin_token = st.session_state.admin_token

    _render_usage_section(admin_token)
    st.divider()

    try:
        runs = list_evaluations(admin_token).get("runs", [])
    except httpx.HTTPError:
        st.error("No se pudieron cargar las corridas de evaluación.")
        return

    if not runs:
        st.info("Todavía no hay corridas de evaluación generadas.")
        return

    # ------------------------------------------------------------------
    # Selector de corrida
    # ------------------------------------------------------------------
    options = {f"{r['config']} — {r['timestamp']}": r["run_id"] for r in runs}
    selected_label = st.selectbox("Corrida", options=list(options.keys()))
    run_id = options[selected_label]

    try:
        detail = get_evaluation_detail(admin_token, run_id)
    except httpx.HTTPError:
        st.error("No se pudo cargar el detalle de la corrida seleccionada.")
        return

    st.caption(
        f"Dataset: {detail.get('dataset','—')} · "
        f"{detail.get('n_evaluated',0)} pares evaluados · "
        f"{detail.get('n_failed',0)} fallidos"
    )

    # ------------------------------------------------------------------
    # KPIs con semáforo
    # ------------------------------------------------------------------
    metrics_global = detail.get("metrics_global", {})
    cols = st.columns(len(SCORE_METRICS))
    for col, (key, label) in zip(cols, SCORE_METRICS):
        value = metrics_global.get(key)
        with col:
            with st.container(border=True):
                st.caption(label)
                st.write("—" if value is None else f"{_semaforo(value)} {value:.4f}")

    latency_avg = metrics_global.get("latency_avg_s")
    latency_p95 = metrics_global.get("latency_p95_s")
    if latency_avg is not None:
        txt = f"Latencia promedio: {latency_avg:.2f}s"
        if latency_p95 is not None:
            txt += f" · p95: {latency_p95:.2f}s"
        st.caption(txt)

    st.write("")

    _render_breakdown("Por tipo de pregunta", detail.get("metrics_by_tipo", {}))
    st.write("")
    _render_breakdown("Por dificultad", detail.get("metrics_by_dificultad", {}))

    st.write("")

    # ------------------------------------------------------------------
    # Comparativa entre corridas
    # ------------------------------------------------------------------
    st.subheader("Comparativa entre corridas")
    comp_rows = [
        {
            "corrida": f"{r['config']} ({r['timestamp']})",
            "faithfulness": r.get("metrics_global", {}).get("faithfulness"),
            "answer_relevancy": r.get("metrics_global", {}).get("answer_relevancy"),
        }
        for r in runs
    ]
    comp_df = pd.DataFrame(comp_rows).dropna(how="all", subset=["faithfulness", "answer_relevancy"])
    if not comp_df.empty:
        st.line_chart(comp_df.set_index("corrida")[["faithfulness", "answer_relevancy"]])
    else:
        st.caption("Sin datos suficientes para graficar.")
