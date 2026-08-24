"""
test_notifier.py — Cuerpo del email de alerta (src/skills/notifier/tool.py).

Solo cubre _build_email_body(), función pura sin I/O. notify_risk() en sí
(SMTP real) no se cubre acá — el envío se prueba manualmente contra un
servidor real (ver foragents/qa_technical.md).
"""

from __future__ import annotations

from src.skills.notifier.tool import _build_email_body


class TestEmailConversationSequence:
    def test_includes_full_conversation_in_order(self):
        conversation = [
            {"role": "user", "content": "tengo hinchazon"},
            {"role": "assistant", "content": "cuentame mas"},
            {"role": "user", "content": "y me duele la cabeza fuerte"},
        ]
        body = _build_email_body(
            "y me duele la cabeza fuerte", "high", "sintomas_embarazo",
            "posible preeclampsia", ["preeclampsia"], conversation,
        )
        assert "tengo hinchazon" in body
        assert "cuentame mas" in body
        assert "y me duele la cabeza fuerte" in body
        assert body.index("tengo hinchazon") < body.index("cuentame mas") < body.index("y me duele la cabeza fuerte")

    def test_marks_last_message_as_the_trigger(self):
        conversation = [
            {"role": "user", "content": "primer mensaje"},
            {"role": "user", "content": "mensaje que dispara"},
        ]
        body = _build_email_body("mensaje que dispara", "high", "x", "r", [], conversation)
        lines = body.splitlines()
        marker_idx = next(i for i, l in enumerate(lines) if "MENSAJE QUE DISPARO LA ALERTA" in l)
        # La línea inmediatamente después del marcador es el contenido del
        # ÚLTIMO turno (el que disparó la alerta), no del primero.
        assert lines[marker_idx + 1] == "mensaje que dispara"
        assert body.index("primer mensaje") < body.index("mensaje que dispara")

    def test_role_labels_are_spanish(self):
        conversation = [
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "hola, como estas"},
        ]
        body = _build_email_body("hola", "medium", "x", "r", [], conversation)
        assert "[Paciente]" in body
        assert "[Asistente]" in body

    def test_falls_back_to_query_when_no_conversation(self):
        body = _build_email_body("dolor fuerte", "high", "x", "r", [], [])
        assert "dolor fuerte" in body
        assert "MENSAJE QUE DISPARO LA ALERTA" in body

    def test_flags_are_listed_when_present(self):
        body = _build_email_body("q", "high", "x", "r", ["hemorragia", "dolor_intenso"], [])
        assert "hemorragia" in body
        assert "dolor_intenso" in body

    def test_no_flags_section_when_empty(self):
        body = _build_email_body("q", "low", "x", "r", [], [])
        assert "Banderas:" not in body
