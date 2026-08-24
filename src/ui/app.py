"""
app.py — Interfaz Streamlit del chatbot Maternas y su panel de administración.

Conecta al backend FastAPI en settings.api_url (default http://localhost:8080).
Arranca con: streamlit run src/ui/app.py

Cada sesión nueva (nueva pestaña/navegador — session_state fresco) exige
aceptar el aviso de tratamiento de datos (src/consent.py, vía
src/ui/consent_gate.py) antes de poder usar CUALQUIER página del panel,
no solo el chat: el gate corre acá, antes de st.navigation(), y detiene
el script con st.stop() si no hay aceptación vigente. Si rechaza, la
sesión queda cerrada; cualquier intento de enviar un mensaje vuelve a
mostrar el aviso.
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.ui.admin_gate import is_admin, render_admin_gate
from src.ui.client import check_health
from src.ui.consent_gate import enforce_consent
from src.ui.styles import STYLES
from src.ui.views.chat_view import render_chat
from src.ui.views.config_view import render_config
from src.ui.views.console_view import render_console
from src.ui.views.dashboard_view import render_dashboard
from src.ui.views.documents_view import render_documents
from src.ui.views.metrics_view import render_metrics

# ---------------------------------------------------------------------------
# Config — ÚNICA llamada a set_page_config de todo el proyecto. Una
# segunda llamada, aunque sea desde una vista, lanza
# StreamlitSetPageConfigMustBeFirstCommandError.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Maternas — Asistente de Salud",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos globales, antes del dispatch: aplican a todas las páginas, no
# solo a la que los definía originalmente.
st.markdown(STYLES, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []        # [{role, content}]
if "meta" not in st.session_state:
    st.session_state.meta = []            # metadata del último turno
if "api_ok" not in st.session_state:
    st.session_state.api_ok = None
if "consent_status" not in st.session_state:
    st.session_state.consent_status = None   # None | "accepted" | "rejected"
if "terms_status" not in st.session_state:
    st.session_state.terms_status = None     # None | "accepted" | "rejected" — capa 2, tras consent_status
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "admin_token" not in st.session_state:
    st.session_state.admin_token = ""
if "session_id" not in st.session_state:
    # Aleatorio, nunca derivado de nada identificable — solo alimenta el
    # conteo de "uso en tiempo real" de Métricas (ver src/api/usage_sessions.py).
    # Un refresh duro del navegador crea una sesión de Streamlit nueva de
    # cero, así que también genera un session_id nuevo: limitación conocida,
    # no hay forma de persistirlo sin cookies/localStorage.
    st.session_state.session_id = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Gate de consentimiento — antes de la sonda de salud y de la navegación,
# para que ninguna página quede alcanzable sin aceptación vigente.
# ---------------------------------------------------------------------------

enforce_consent()

# ---------------------------------------------------------------------------
# Sidebar — sonda de salud y licencias, visibles en todas las páginas.
# Corre una sola vez por rerun, antes del dispatch: las páginas que leen
# st.session_state.api_ok (el chat, por ejemplo) siempre lo encuentran
# ya actualizado en este mismo rerun.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤰 Maternas")
    st.caption("Asistente de salud para madres gestantes")
    st.divider()

    try:
        health = check_health()
        api_ok = bool(health.get("faiss_loaded"))
    except Exception:
        # Amplio a propósito (no solo httpx.HTTPError): un 200 con cuerpo
        # no-JSON, por ejemplo, no debe tirar la app entera.
        health = {}
        api_ok = False

    st.session_state.health = health
    st.session_state.api_ok = api_ok

    if api_ok:
        st.success("API conectada")
        st.caption(f"📚 {health.get('total_vectors', 0):,} fragmentos médicos indexados")
        st.caption(f"🧠 {health.get('model', '').split('/')[-1]}")
    else:
        st.error("API no disponible")
        st.caption("Arranca el servidor con:\n```\npython -m uvicorn src.api.main:app --port 8080\n```")

    render_admin_gate()

    st.divider()

    with st.expander("📖 Fuentes de datos y licencias"):
        st.markdown(
            "- **MedMCQA** — [Apache 2.0](https://huggingface.co/datasets/openlifescienceai/medmcqa)\n"
            "- **MedQA** — [MIT](https://github.com/jind11/MedQA) (Jin et al., 2020)\n"
            "- **MultiClinSum** — [CC-BY 4.0](https://zenodo.org/records/15517617) (BioASQ / CLEF 2025)\n"
            "- **MaternaQA-es** — MIT (`minciencias-maternas/obstetrics-rag-benchmark`)\n"
            "- **Documentos cargados** — bajo responsabilidad de quien administra el panel\n\n"
            "Detalle completo en `foragents/qa_technical.md` (Q28)."
        )

# ---------------------------------------------------------------------------
# Navegación — el resto de las páginas ni siquiera se declara acá si la
# sesión no está autenticada como admin: st.navigation solo puede
# resolver páginas que están en esta lista, así que quedan inalcanzables
# incluso por URL directa, no solo ocultas en el menú.
# ---------------------------------------------------------------------------

pages = [st.Page(render_chat, title="Chat", icon="💬", default=True)]
if is_admin():
    pages += [
        st.Page(render_dashboard, title="Dashboard", icon="🏠"),
        st.Page(render_documents, title="Documentos", icon="📁"),
        st.Page(render_metrics, title="Métricas", icon="📊"),
        st.Page(render_config, title="Configuración", icon="⚙️"),
        st.Page(render_console, title="Consola", icon="🖥️"),
    ]

pg = st.navigation(pages)
pg.run()
