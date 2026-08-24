"""
consent_gate.py — Aviso de tratamiento de datos + Términos y Condiciones,
obligatorios (en ese orden, dos pantallas secuenciales) antes de usar
cualquier página del panel.

A diferencia de la versión anterior de app.py (que abría el diálogo pero
seguía renderizando el chat detrás, bloqueando solo el formulario de
envío), acá el gate detiene el script con st.stop(): en un shell
multipágina, seguir adelante dejaría Documentos, Métricas y
Configuración accesibles detrás del modal.

Rechazar cualquiera de las dos capas termina la sesión — el próximo intento
de escribir vuelve a mostrar la capa 1 (tratamiento de datos) desde cero,
nunca reanuda a mitad de camino.
"""

import streamlit as st

from src.consent import CONSENT_TEXT, FAREWELL_TEXT, TERMS_TEXT


def _reset_session_on_reject() -> None:
    st.session_state.consent_status = "rejected"
    st.session_state.terms_status = None
    st.session_state.messages = []
    st.session_state.meta = []


@st.dialog("Aviso sobre tratamiento de información")
def _consent_dialog() -> None:
    st.text(CONSENT_TEXT)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Acepto", use_container_width=True):
            st.session_state.consent_status = "accepted"
            st.rerun()
    with col_b:
        if st.button("❌ No acepto", use_container_width=True):
            _reset_session_on_reject()
            st.rerun()


@st.dialog("Términos y Condiciones de uso")
def _terms_dialog() -> None:
    st.text(TERMS_TEXT)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Acepto", use_container_width=True, key="terms_accept"):
            st.session_state.terms_status = "accepted"
            st.rerun()
    with col_b:
        if st.button("❌ No acepto", use_container_width=True, key="terms_reject"):
            _reset_session_on_reject()
            st.rerun()


def enforce_consent() -> None:
    """Bloquea toda la aplicación hasta aceptar AMBAS capas, en orden:
    tratamiento de datos primero, Términos y Condiciones después."""
    if st.session_state.consent_status == "accepted" and st.session_state.terms_status == "accepted":
        return

    if st.session_state.consent_status != "accepted":
        _consent_dialog()
        placeholder = "Acepta el aviso de datos para poder escribirme"
    else:
        _terms_dialog()
        placeholder = "Acepta los Términos y Condiciones para poder escribirme"

    if st.session_state.consent_status == "rejected" or st.session_state.terms_status == "rejected":
        st.warning(FAREWELL_TEXT)
    elif st.session_state.consent_status == "accepted":
        st.info("Debes aceptar los Términos y Condiciones para poder chatear.")
    else:
        st.info("Debes aceptar el aviso de tratamiento de información para poder chatear.")

    with st.form("chat_form_locked", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            locked_input = st.text_input(
                "Tu mensaje",
                placeholder=placeholder,
                label_visibility="collapsed",
            )
        with col2:
            locked_submitted = st.form_submit_button("Enviar", use_container_width=True)

    if locked_submitted and locked_input.strip():
        # Cualquier intento de enviar un mensaje sin aceptación vigente de
        # AMBAS capas reinicia el flujo desde la capa 1, en vez de procesar
        # el mensaje.
        st.session_state.consent_status = None
        st.session_state.terms_status = None
        st.rerun()

    st.stop()
