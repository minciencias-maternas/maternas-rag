from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.classifiers.risk_detector import (
    VALID_ACTIONS,
    VALID_LEVELS,
    RiskResult,
    _check_heuristic,
    _parse_llm_response,
    detect_risk,
)


class TestCheckHeuristic:
    def test_high_hemorragia(self):
        result = _check_heuristic("Estoy sangrando mucho")
        assert result is not None
        assert result.level == "high"
        assert "hemorragia" in result.flags
        assert result.action == "urgent_care"
        assert result.used_heuristic is True

    def test_high_convulsion(self):
        result = _check_heuristic("tuve una convulsión")
        assert result is not None
        assert result.level == "high"
        assert "eclampsia_convulsion" in result.flags

    def test_high_movimiento_ausente(self):
        result = _check_heuristic("el bebé no se mueve")
        assert result is not None
        assert result.level == "high"
        assert "movimiento_fetal_ausente" in result.flags

    def test_high_dolor_intenso(self):
        result = _check_heuristic("tengo un dolor insoportable en el abdomen")
        assert result is not None
        assert result.level == "high"
        assert "dolor_intenso" in result.flags

    def test_high_preeclampsia(self):
        result = _check_heuristic("tengo visión borrosa y dolor de cabeza intenso")
        assert result is not None
        assert result.level == "high"
        assert "preeclampsia" in result.flags

    def test_high_ruptura_membranas(self):
        result = _check_heuristic("se rompió la bolsa")
        assert result is not None
        assert result.level == "high"
        assert "ruptura_membranas" in result.flags

    def test_high_signos_sepsis(self):
        result = _check_heuristic("tengo fiebre de 40 con escalofríos")
        assert result is not None
        assert result.level == "high"
        assert "signos_sepsis" in result.flags

    def test_high_depresion_grave(self):
        result = _check_heuristic("quiero hacerme daño")
        assert result is not None
        assert result.level == "high"
        assert "depresion_grave" in result.flags

    def test_high_trabajo_parto_prematuro(self):
        result = _check_heuristic("tengo contracciones antes de las 37 semanas")
        assert result is not None
        assert result.level == "high"
        assert "trabajo_parto_prematuro" in result.flags

    def test_medium_sangrado_leve(self):
        result = _check_heuristic("tengo un pequeño manchado rosado")
        assert result is not None
        assert result.level == "medium"
        assert "sangrado_leve" in result.flags
        assert result.action == "medical_consultation"

    def test_medium_presion_alta(self):
        result = _check_heuristic("tengo la presión alta")
        assert result is not None
        assert result.level == "medium"
        assert "presion_alta_leve" in result.flags

    def test_medium_edema(self):
        result = _check_heuristic("tengo los pies muy hinchados")
        assert result is not None
        assert result.level == "medium"
        assert "edema_moderado" in result.flags

    def test_medium_fiebre_moderada(self):
        result = _check_heuristic("tengo fiebre desde ayer")
        assert result is not None
        assert result.level == "medium"
        assert "fiebre_moderada" in result.flags

    def test_medium_reduccion_movimiento(self):
        result = _check_heuristic("siento que el bebé se mueve menos")
        assert result is not None
        assert result.level == "medium"
        assert "reduccion_movimiento" in result.flags

    def test_medium_sintomas_infeccion(self):
        result = _check_heuristic("tengo ardor al orinar")
        assert result is not None
        assert result.level == "medium"
        assert "sintomas_infeccion" in result.flags

    def test_no_match_returns_none(self):
        result = _check_heuristic("¿cuántas semanas de embarazo son normales?")
        assert result is None

    def test_case_insensitive(self):
        result = _check_heuristic("Sangrando Mucho")
        assert result is not None
        assert result.level == "high"

    def test_high_trumps_medium(self):
        result = _check_heuristic("tengo un manchado y también estoy sangrando mucho")
        assert result is not None
        assert result.level == "high"
        assert "hemorragia" in result.flags

    def test_special_characters(self):
        result = _check_heuristic("¡no siento movimientos! ¿qué hago?")
        assert result is not None
        assert result.level == "high"

    @pytest.mark.parametrize("msg", [
        "no se mueve el bebé",
        "dejó de moverse",
        "no siento movimientos",
        "no patea",
        "sin movimiento fetal",
    ])
    def test_parametric_movement_absent(self, msg):
        result = _check_heuristic(msg)
        assert result is not None
        assert result.level == "high"

    @pytest.mark.parametrize("msg", [
        "tengo fiebre muy alta",
        "fiebre de 39",
        "fiebre de 40",
        "escalofríos con fiebre",
    ])
    def test_parametric_sepsis_signs(self, msg):
        result = _check_heuristic(msg)
        assert result is not None
        assert result.level == "high"

    @pytest.mark.parametrize("msg,expected_flag", [
        ("quiero hacerme daño", "depresion_grave"),
        ("quiero lastimarme", "depresion_grave"),
        ("no quiero vivir", "depresion_grave"),
        ("pensamientos de muerte", "depresion_grave"),
    ])
    def test_parametric_severe_depression(self, msg, expected_flag):
        result = _check_heuristic(msg)
        assert result is not None
        assert result.level == "high"
        assert expected_flag in result.flags


class TestParseLlmResponse:
    def test_valid_json(self):
        raw = '{"level": "high", "flags": ["hemorragia"], "action": "urgent_care", "reasoning": "Paciente con sangrado activo."}'
        result = _parse_llm_response(raw)
        assert result.level == "high"
        assert result.flags == ["hemorragia"]
        assert result.action == "urgent_care"
        assert result.raw == raw

    def test_valid_json_low(self):
        raw = '{"level": "low", "flags": [], "action": "educational_answer", "reasoning": "Consulta informativa."}'
        result = _parse_llm_response(raw)
        assert result.level == "low"
        assert result.action == "educational_answer"

    def test_invalid_level_fallback_to_medium(self):
        raw = '{"level": "critical", "flags": [], "action": "urgent_care", "reasoning": "test"}'
        result = _parse_llm_response(raw)
        assert result.level == "medium"

    def test_invalid_action_fallback_by_level(self):
        raw = '{"level": "low", "flags": [], "action": "evacuate", "reasoning": "test"}'
        result = _parse_llm_response(raw)
        assert result.action == "educational_answer"

    def test_missing_fields(self):
        raw = '{"level": "medium"}'
        result = _parse_llm_response(raw)
        assert result.level == "medium"
        assert result.flags == []
        assert result.action == "medical_consultation"

    def test_invalid_json(self):
        raw = "not json at all"
        result = _parse_llm_response(raw)
        assert result.level == "medium"
        assert result.action == "medical_consultation"
        assert result.raw == raw

    def test_empty_string(self):
        result = _parse_llm_response("")
        assert result.level == "medium"
        assert result.action == "medical_consultation"


class TestDetectRisk:
    def test_empty_message_returns_low(self):
        result = detect_risk("")
        assert result.level == "low"
        assert result.action == "educational_answer"

    def test_whitespace_message_returns_low(self):
        result = detect_risk("   ")
        assert result.level == "low"
        assert result.action == "educational_answer"

    def test_none_message_returns_low(self):
        result = detect_risk(None)
        assert result.level == "low"
        assert result.action == "educational_answer"

    def test_off_topic_shortcut_returns_low(self):
        result = detect_risk("¿cuál es la capital de Francia?", intent="pregunta_fuera_de_alcance")
        assert result.level == "low"
        assert result.action == "educational_answer"
        assert "fuera del dominio" in result.reasoning

    def test_heuristic_high_triggers_before_llm(self):
        with patch("src.classifiers.risk_detector._llm_risk") as mock_llm:
            result = detect_risk("estoy sangrando mucho")
            mock_llm.assert_not_called()
            assert result.level == "high"
            assert result.used_heuristic is True

    def test_heuristic_medium_triggers_before_llm(self):
        with patch("src.classifiers.risk_detector._llm_risk") as mock_llm:
            result = detect_risk("tengo la presión alta")
            mock_llm.assert_not_called()
            assert result.level == "medium"
            assert result.used_heuristic is True

    def test_no_heuristic_calls_llm(self, mock_groq_client):
        llm_response = '{"level": "low", "flags": [], "action": "educational_answer", "reasoning": "Pregunta informativa normal."}'
        mock_groq_client.chat.completions.create.return_value.choices[0].message.content = llm_response
        result = detect_risk("¿es normal tener náuseas en el primer trimestre?")
        assert result.level == "low"
        mock_groq_client.chat.completions.create.assert_called_once()

    def test_llm_error_returns_medium_safe_fallback(self, mock_groq_client):
        mock_groq_client.chat.completions.create.side_effect = Exception("API error")
        result = detect_risk("tengo un dolor de cabeza")
        assert result.level == "medium"
        assert "Error en evaluación" in result.reasoning


class TestHeuristicDoesNotInheritStaleHistory:
    """Regresión: la capa heurística combinaba prior_user_text + mensaje
    actual, así que una keyword de alarma en un turno viejo seguía
    matcheando en cada turno siguiente sin relación con ella. Ahora la
    heurística evalúa SOLO el mensaje actual (ver detect_risk)."""

    def test_unrelated_followup_does_not_inherit_high_keyword(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"level": "low", "flags": [], "action": "educational_answer", '
                        '"reasoning": "mensaje no relacionado con el sangrado previo"}'
            ))]
        )
        history = [{"role": "user", "content": "estoy sangrando mucho"}]
        result = detect_risk("gracias por la info, hasta luego", history=history)

        # Si la heurística siguiera combinando historial, "sangrando mucho"
        # habría matcheado de nuevo y nunca habría llegado a llamar al LLM.
        mock_groq_client.chat.completions.create.assert_called_once()
        assert result.level == "low"

    def test_unrelated_followup_after_medium_keyword_also_isolated(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"level": "low", "flags": [], "action": "educational_answer", '
                        '"reasoning": "pregunta nueva sin relacion"}'
            ))]
        )
        history = [{"role": "user", "content": "tengo la presión alta"}]
        result = detect_risk("¿qué alimentos debo evitar?", history=history)

        mock_groq_client.chat.completions.create.assert_called_once()
        assert result.level == "low"

    def test_current_message_keyword_still_triggers_heuristic(self, mock_groq_client):
        """El fix no debe romper la detección normal: una keyword en el
        mensaje ACTUAL (sin necesidad de historial) sigue disparando la
        heurística de inmediato, sin llamar al LLM."""
        history = [{"role": "user", "content": "hola, tengo una duda"}]
        result = detect_risk("estoy sangrando mucho", history=history)

        mock_groq_client.chat.completions.create.assert_not_called()
        assert result.level == "high"
        assert result.used_heuristic is True


class TestLlmStillReceivesRecentContext:
    """La capa LLM sí conserva contexto reciente (a diferencia de la
    heurística) — para combinar síntomas que siguen vigentes, p. ej. tras
    una pregunta de aclaración. Ver SYSTEM_PROMPT: el LLM decide si el
    mensaje actual sigue relacionado o no."""

    def test_llm_prompt_includes_recent_prior_user_text(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"level": "medium", "flags": ["dolor_moderado"], '
                        '"action": "medical_consultation", "reasoning": "cuadro combinado"}'
            ))]
        )
        history = [{"role": "user", "content": "tengo hinchazon en los pies"}]
        detect_risk("y también me duele mucho la cabeza", history=history)

        call_kwargs = mock_groq_client.chat.completions.create.call_args.kwargs
        user_message = call_kwargs["messages"][1]["content"]
        assert "hinchazon" in user_message
        assert "me duele mucho la cabeza" in user_message
