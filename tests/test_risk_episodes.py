"""
test_risk_episodes.py — Deduplicación de notificaciones de riesgo por
episodio (src/rag/risk_episodes.py).

API de dos fases: is_new_signal() es de solo lectura (aparte de la poda por
inactividad), commit() es lo único que registra un aviso como enviado.
register_low() cierra el episodio de inmediato.
"""

from __future__ import annotations

import pytest

from src.rag import risk_episodes


@pytest.fixture(autouse=True)
def clean_registry():
    risk_episodes._episodes.clear()
    yield
    risk_episodes._episodes.clear()


class TestNoSessionId:
    def test_always_new_signal_without_session(self):
        assert risk_episodes.is_new_signal(None, "high", ["hemorragia"]) is True
        risk_episodes.commit(None, "high", ["hemorragia"])
        # Sin session_id no hay estado que guardar — sigue siendo "nuevo".
        assert risk_episodes.is_new_signal(None, "high", ["hemorragia"]) is True

    def test_commit_and_register_low_are_noop_without_session(self):
        risk_episodes.commit(None, "high", ["hemorragia"])
        risk_episodes.register_low(None)
        assert risk_episodes._episodes == {}


class TestFirstSignal:
    def test_no_prior_episode_is_new_signal(self):
        assert risk_episodes.is_new_signal("s1", "medium", ["presion_alta_leve"]) is True

    def test_is_new_signal_does_not_mutate_state(self):
        """Peek puro: llamarlo repetidas veces sin commit() no crea dedup."""
        risk_episodes.is_new_signal("s1", "high", ["hemorragia"])
        risk_episodes.is_new_signal("s1", "high", ["hemorragia"])
        assert risk_episodes.is_new_signal("s1", "high", ["hemorragia"]) is True


class TestDedupAfterCommit:
    def test_same_signal_after_commit_is_not_new(self):
        risk_episodes.commit("s1", "high", ["hemorragia"])
        assert risk_episodes.is_new_signal("s1", "high", ["hemorragia"]) is False

    def test_subset_of_committed_flags_is_not_new(self):
        risk_episodes.commit("s1", "high", ["hemorragia", "dolor_intenso"])
        assert risk_episodes.is_new_signal("s1", "high", ["hemorragia"]) is False

    def test_new_flag_not_in_committed_set_is_new_signal(self):
        risk_episodes.commit("s1", "high", ["hemorragia"])
        assert risk_episodes.is_new_signal("s1", "high", ["eclampsia_convulsion"]) is True

    def test_escalation_from_medium_to_high_is_new_signal(self):
        risk_episodes.commit("s1", "medium", ["presion_alta_leve"])
        assert risk_episodes.is_new_signal("s1", "high", ["presion_alta_leve"]) is True

    def test_deescalation_with_same_flags_is_not_new(self):
        risk_episodes.commit("s1", "high", ["hemorragia"])
        assert risk_episodes.is_new_signal("s1", "medium", ["hemorragia"]) is False

    def test_empty_flags_both_sides_not_new_unless_escalated(self):
        risk_episodes.commit("s1", "medium", [])
        assert risk_episodes.is_new_signal("s1", "medium", []) is False
        assert risk_episodes.is_new_signal("s1", "high", []) is True

    def test_sessions_are_independent(self):
        risk_episodes.commit("s1", "high", ["hemorragia"])
        assert risk_episodes.is_new_signal("s2", "high", ["hemorragia"]) is True


class TestRegisterLow:
    def test_low_clears_episode_so_same_signal_is_new_again(self):
        risk_episodes.commit("s1", "high", ["hemorragia"])
        assert risk_episodes.is_new_signal("s1", "high", ["hemorragia"]) is False

        risk_episodes.register_low("s1")

        assert risk_episodes.is_new_signal("s1", "high", ["hemorragia"]) is True

    def test_register_low_without_prior_episode_is_safe(self):
        risk_episodes.register_low("s1")   # no debe lanzar
        assert "s1" not in risk_episodes._episodes


class TestCommitDoesNotHappenOnPeek:
    def test_medium_llm_says_no_leaves_episode_untouched(self):
        """Simula el flujo real: is_new_signal() dice que sí vale la pena
        preguntarle al LLM, pero el LLM decide NO notificar — commit() nunca
        se llama, así que el mismo riesgo debe seguir siendo "nuevo" después."""
        assert risk_episodes.is_new_signal("s1", "medium", ["fiebre_moderada"]) is True
        # (el caller decide NO notificar, no llama a commit())
        assert risk_episodes.is_new_signal("s1", "medium", ["fiebre_moderada"]) is True
