"""
chat_view.py — Vista del chat Maternas.

Solo lógica de renderizado; la comunicación HTTP delega en src.ui.client.
El aviso de tratamiento de datos se resuelve en src.ui.consent_gate,
antes de que el shell (app.py) llegue siquiera a la navegación — acá no
hace falta volver a chequearlo ni hay una rama "bloqueada".

Derivado del app.py de MASTER (no del de la rama original, que fue
escrito antes del flujo de consentimiento y de needs_clarification/
source_path): conserva la burbuja de clarificación, el source_path en
la píldora de fuentes y que el historial enviado a la API lleve
únicamente role/content.

La respuesta se transmite token a token (POST /chat/stream, vía
client.stream_chat) para que se sienta más rápida aunque el turno
completo tarde lo mismo: el pipeline (clasificar → recuperar → generar)
se muestra en un st.status y el texto se pinta a medida que llega, en
vez de esperar el turno entero detrás de un spinner.
"""

import httpx
import streamlit as st

from src.ui.client import call_chat, stream_chat
from src.ui.helpers import intent_label, risk_badge, source_dataset_label

STAGE_LABELS = {
    "classifying": "Analizando tu mensaje…",
    "retrieving":  "Buscando en la base de conocimiento médico…",
    "generating":  "Redactando la respuesta…",
}


def _render_message(msg: dict) -> None:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    elif msg.get("clarification"):
        st.markdown(f'<div class="msg-clarification">🤰 💬 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-assistant">🤰 {msg["content"]}</div>', unsafe_allow_html=True)


def _render_sidebar_turn() -> None:
    if not st.session_state.messages:
        return
    if not st.session_state.meta:
        return

    st.divider()

    m = st.session_state.meta[-1]
    st.subheader("Último turno")

    st.markdown(f"**Intención:** {intent_label(m.get('intent',''))}", unsafe_allow_html=True)
    st.markdown(f"**Riesgo:** {risk_badge(m.get('risk_level','low'))}", unsafe_allow_html=True)
    st.markdown(f"**Acción:** `{m.get('action','')}`")

    if m.get("risk_flags"):
        st.markdown("**Señales:**")
        for flag in m["risk_flags"]:
            st.markdown(f"- `{flag}`")

    if m.get("sources"):
        st.markdown("**Fuentes recuperadas:**")
        for s in m["sources"]:
            dataset_label = source_dataset_label(s.get("source_dataset", ""))
            doc_name = s.get("document_name") or dataset_label
            score = s.get("score", 0)
            pages = s.get("pages") or []
            locator = f"pág. {pages[0]}" if len(pages) == 1 else (
                f"págs. {pages[0]}-{pages[-1]}" if len(pages) > 1 else ""
            )
            pill_text = f"📄 {doc_name}"
            if locator:
                pill_text += f" · {locator}"
            pill_text += f" · {score:.3f}"
            st.markdown(
                f'<span class="source-pill">{pill_text}</span>',
                unsafe_allow_html=True,
            )

    if m.get("tokens_used"):
        st.caption(f"Tokens usados: {m['tokens_used']:,}")

    st.divider()

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.meta = []
        st.rerun()


def _turn_result(meta_event: dict | None, done_event: dict, acc: str) -> dict:
    """Combina el evento meta + el evento done en un dict con la misma
    forma que el ChatResponse de call_chat(), para que el sidebar (que
    lee de st.session_state.meta) no necesite dos caminos distintos."""
    meta_event = meta_event or {}
    return {
        "intent":               meta_event.get("intent", ""),
        "risk_level":           meta_event.get("risk_level", "low"),
        "action":               meta_event.get("action", ""),
        "risk_flags":           meta_event.get("risk_flags", []),
        "sources":              meta_event.get("sources", []),
        "needs_clarification":  meta_event.get("needs_clarification", False),
        "tokens_used":          done_event.get("tokens_used", 0),
        "notified":             done_event.get("notified", False),
        "answer":               done_event.get("answer", acc),
    }


def render_chat() -> None:
    with st.sidebar:
        _render_sidebar_turn()

    st.title("Maternas — Asistente de Salud Materna")
    st.caption("Respondo preguntas sobre embarazo, parto, postparto y lactancia basándome en literatura médica.")

    for msg in st.session_state.messages:
        _render_message(msg)

    if not st.session_state.api_ok:
        st.warning("La API no está disponible. Inicia el servidor para continuar.")
        return

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            user_input = st.text_input(
                "Tu mensaje",
                placeholder="Ej: ¿Es normal tener náuseas a las 10 semanas?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Enviar", use_container_width=True)

    if not (submitted and user_input.strip()):
        return

    user_text = user_input.strip()

    # Solo role/content: 'clarification' es un detalle de presentación de
    # esta vista y la API no lo necesita ni debe recibirlo. Se calcula
    # ANTES de appendear el mensaje actual — el turno no debe verse a
    # sí mismo en su propio historial.
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": user_text})
    # Se pinta de inmediato en este mismo run — sin esto, la burbuja de la
    # usuaria no aparecería hasta el st.rerun() del final del turno.
    _render_message(st.session_state.messages[-1])

    status = st.status(STAGE_LABELS["classifying"], expanded=True)

    acc = ""
    meta_event: dict | None = None
    # Se crea recién con el primer delta (no antes) para que un st.error()
    # de riesgo alto disparado por el evento 'meta' quede POR ENCIMA de la
    # burbuja de respuesta en el DOM — si el placeholder ya existiera,
    # Streamlit insertaría el error después de él, es decir debajo del
    # texto que se está escribiendo.
    placeholder = None   # st.empty(), creado recién con el primer delta
    got_delta = False
    fell_back = False

    try:
        for event in stream_chat(user_text, history_payload, session_id=st.session_state.session_id):
            etype = event.get("type")

            if etype == "status":
                status.update(label=STAGE_LABELS.get(event.get("stage"), "Procesando…"))

            elif etype == "meta":
                meta_event = event
                if event.get("risk_level") == "high":
                    st.error("⚠️ Se detectaron señales de alarma. Busca atención médica de inmediato.")

            elif etype == "delta":
                if not got_delta:
                    status.update(label=STAGE_LABELS["generating"], state="complete", expanded=False)
                    got_delta = True
                    placeholder = st.empty()
                acc += event.get("text", "")
                placeholder.markdown(
                    f'<div class="msg-assistant">🤰 {acc}<span class="typing-caret">▌</span></div>',
                    unsafe_allow_html=True,
                )

            elif etype == "done":
                if placeholder:
                    placeholder.empty()
                result = _turn_result(meta_event, event, acc)
                answer = result.pop("answer")
                if result["needs_clarification"]:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "clarification": True}
                    )
                else:
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.meta.append(result)
                status.update(state="complete", expanded=False)
                st.rerun()

            elif etype == "error":
                if not got_delta:
                    # No se transmitió nada todavía: reintentar sin
                    # streaming en vez de dejar el turno a medias.
                    fell_back = True
                    break
                placeholder.empty()
                st.session_state.messages.append({"role": "assistant", "content": acc})
                st.session_state.meta.append(_turn_result(meta_event, {}, acc))
                status.update(label="La respuesta se interrumpió", state="error", expanded=False)
                st.error(event.get("detail", "Error desconocido durante la generación."))
                st.rerun()

    except httpx.ConnectError:
        status.update(state="error", expanded=False)
        st.error("No se puede conectar con la API. ¿Está corriendo en el puerto 8080?")
        return
    except httpx.TimeoutException:
        status.update(state="error", expanded=False)
        st.error("La API tardó demasiado. Intenta de nuevo.")
        return
    except httpx.HTTPStatusError as e:
        status.update(state="error", expanded=False)
        st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
        return
    except Exception as e:
        status.update(state="error", expanded=False)
        st.error(f"Error inesperado: {e}")
        return

    if not fell_back:
        return

    status.update(label="Reintentando sin streaming…")
    try:
        result = call_chat(user_text, history_payload, session_id=st.session_state.session_id)
    except httpx.ConnectError:
        status.update(state="error", expanded=False)
        st.error("No se puede conectar con la API. ¿Está corriendo en el puerto 8080?")
        return
    except httpx.TimeoutException:
        status.update(state="error", expanded=False)
        st.error("La API tardó demasiado. Intenta de nuevo.")
        return
    except httpx.HTTPStatusError as e:
        status.update(state="error", expanded=False)
        st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
        return
    except Exception as e:
        status.update(state="error", expanded=False)
        st.error(f"Error inesperado: {e}")
        return

    if placeholder:
        placeholder.empty()
    status.update(state="complete", expanded=False)

    answer = result.get("answer", "Sin respuesta")
    if result.get("risk_level") == "high":
        st.error("⚠️ Se detectaron señales de alarma. Busca atención médica de inmediato.")

    if result.get("needs_clarification"):
        st.session_state.messages.append({"role": "assistant", "content": answer, "clarification": True})
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.session_state.meta.append(result)
    st.rerun()
