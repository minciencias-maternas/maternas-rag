"""
test_chain_notification.py — _run_notification() en src/rag/chain.py:
deduplicación por episodio (vía risk_episodes) + secuencia de mensajes
enviada al notifier.

ToolRegistry.execute se mockea para capturar los kwargs con los que se
llamaría a notify_risk(), sin tocar SMTP real.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.classifiers.risk_detector import RiskResult
from src.rag import chain, risk_episodes


@pytest.fixture(autouse=True)
def clean_episodes():
    risk_episodes._episodes.clear()
    yield
    risk_episodes._episodes.clear()


@pytest.fixture
def mock_execute(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(chain.ToolRegistry, "execute", mock)
    return mock


def _risk(level: str, flags: list[str] | None = None) -> RiskResult:
    return RiskResult(level=level, flags=flags or [], action="urgent_care", reasoning="r")


class TestHighRiskDedup:
    def test_first_high_risk_notifies(self, mock_execute):
        notified = chain._run_notification(
            "estoy sangrando mucho", "sintomas_embarazo",
            _risk("high", ["hemorragia"]), history=[], session_id="s1",
        )
        assert notified is True
        mock_execute.assert_called_once()

    def test_repeated_same_signal_does_not_renotify(self, mock_execute):
        risk = _risk("high", ["hemorragia"])
        chain._run_notification("m1", "x", risk, history=[], session_id="s1")
        notified_again = chain._run_notification("m2 no relacionado", "x", risk, history=[], session_id="s1")
        assert notified_again is False
        mock_execute.assert_called_once()   # solo la primera vez

    def test_new_flag_renotifies(self, mock_execute):
        chain._run_notification("m1", "x", _risk("high", ["hemorragia"]), history=[], session_id="s1")
        notified2 = chain._run_notification(
            "m2", "x", _risk("high", ["eclampsia_convulsion"]), history=[], session_id="s1"
        )
        assert notified2 is True
        assert mock_execute.call_count == 2

    def test_low_risk_clears_episode_then_same_signal_renotifies(self, mock_execute):
        risk_high = _risk("high", ["hemorragia"])
        chain._run_notification("m1", "x", risk_high, history=[], session_id="s1")
        chain._run_notification("gracias", "x", _risk("low"), history=[], session_id="s1")
        notified_again = chain._run_notification("m1 de nuevo", "x", risk_high, history=[], session_id="s1")
        assert notified_again is True
        assert mock_execute.call_count == 2

    def test_low_risk_never_notifies(self, mock_execute):
        notified = chain._run_notification("gracias", "x", _risk("low"), history=[], session_id="s1")
        assert notified is False
        mock_execute.assert_not_called()

    def test_without_session_id_always_notifies(self, mock_execute):
        risk = _risk("high", ["hemorragia"])
        chain._run_notification("m1", "x", risk, history=[], session_id=None)
        chain._run_notification("m1 otra vez", "x", risk, history=[], session_id=None)
        assert mock_execute.call_count == 2   # sin sesión no hay como deduplicar


class TestMediumRiskDedupAndLlmDecision:
    def test_medium_llm_says_no_does_not_notify_nor_commit(self, monkeypatch, mock_execute):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="NO"))
        ]
        monkeypatch.setattr(chain, "_get_client", lambda: mock_client)

        risk = _risk("medium", ["fiebre_moderada"])
        notified = chain._run_notification("tengo fiebre", "x", risk, history=[], session_id="s1")

        assert notified is False
        mock_execute.assert_not_called()

        # Como el LLM dijo NO, is_new_signal() no debe haber quedado
        # comprometido — el mismo riesgo puede re-evaluarse (y notificar)
        # en el turno siguiente.
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="YES"))
        ]
        notified2 = chain._run_notification("tengo fiebre todavia", "x", risk, history=[], session_id="s1")
        assert notified2 is True
        mock_execute.assert_called_once()

    def test_medium_dedup_skips_llm_call_entirely(self, monkeypatch, mock_execute):
        """Si el episodio ya cubre esta señal, ni siquiera debe llamarse al
        LLM de decisión — el ahorro de esa llamada es parte del punto."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="YES"))
        ]
        monkeypatch.setattr(chain, "_get_client", lambda: mock_client)

        risk = _risk("medium", ["fiebre_moderada"])
        chain._run_notification("m1", "x", risk, history=[], session_id="s1")
        assert mock_client.chat.completions.create.call_count == 1

        notified2 = chain._run_notification("m2 no relacionado", "x", risk, history=[], session_id="s1")
        assert notified2 is False
        assert mock_client.chat.completions.create.call_count == 1   # no llamó de nuevo


class TestConversationSequence:
    def test_conversation_includes_history_and_current_message_last(self, mock_execute):
        history = [
            {"role": "user", "content": "tengo hinchazon"},
            {"role": "assistant", "content": "cuentame mas"},
        ]
        chain._run_notification(
            "ahora estoy sangrando mucho", "x", _risk("high", ["hemorragia"]),
            history=history, session_id="s1",
        )
        conv = mock_execute.call_args.kwargs["conversation"]
        assert conv[0] == {"role": "user", "content": "tengo hinchazon"}
        assert conv[1] == {"role": "assistant", "content": "cuentame mas"}
        assert conv[-1] == {"role": "user", "content": "ahora estoy sangrando mucho"}

    def test_conversation_is_capped_to_last_n_turns(self, mock_execute):
        history = [{"role": "user", "content": f"m{i}"} for i in range(30)]
        chain._run_notification(
            "actual", "x", _risk("high", ["hemorragia"]), history=history, session_id="s1",
        )
        conv = mock_execute.call_args.kwargs["conversation"]
        assert len(conv) == chain.NOTIFICATION_HISTORY_CAP + 1   # +1 por el mensaje actual
        assert conv[0]["content"] == f"m{30 - chain.NOTIFICATION_HISTORY_CAP}"

    def test_conversation_without_history_is_just_current_message(self, mock_execute):
        chain._run_notification(
            "estoy sangrando", "x", _risk("high", ["hemorragia"]), history=[], session_id="s1",
        )
        conv = mock_execute.call_args.kwargs["conversation"]
        assert conv == [{"role": "user", "content": "estoy sangrando"}]
