"""
test_bm25_index.py — Índice BM25 (capa léxica del retrieval híbrido, Config F).

Usa fake_store_factory (conftest.py) para tener un FAISSStore real y chico,
igual que test_store_admin.py. bm25_index.py no carga su propio
metadata.pkl: reusa src.rag.retriever._get_store(), así que basta con
inyectar el store fake en ese singleton (src.rag.retriever._store).

Los globals de bm25_index (_bm25_index, _bm25_docs, _built_at_mutation_seq)
son un singleton de módulo — se resetean antes y después de cada test para
que un test no vea el índice construido por el anterior.
"""

from __future__ import annotations

import pytest

import src.rag.bm25_index as bm25_index
import src.rag.retriever as retriever
from src.rag.bm25_index import _tokenize, search_bm25


@pytest.fixture(autouse=True)
def reset_bm25_singleton():
    bm25_index._bm25_index = None
    bm25_index._bm25_docs = []
    bm25_index._built_at_mutation_seq = None
    retriever._store = None
    yield
    bm25_index._bm25_index = None
    bm25_index._bm25_docs = []
    bm25_index._built_at_mutation_seq = None
    retriever._store = None


def _inject_store(store) -> None:
    """Inyecta un FAISSStore fake en el singleton que consume bm25_index."""
    retriever._store = store


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_removes_stopwords_es_en():
    tokens = _tokenize("la preeclampsia and the eclampsia")
    assert "la" not in tokens
    assert "and" not in tokens
    assert "the" not in tokens
    assert "preeclampsia" in tokens
    assert "eclampsia" in tokens


def test_tokenize_folds_accents():
    # "preeclámpsia" (con tilde, como puede escribirlo un usuario de
    # Telegram) debe tokenizar igual que "preeclampsia" (sin tilde).
    assert _tokenize("preeclámpsia") == _tokenize("preeclampsia")


def test_tokenize_drops_short_tokens():
    tokens = _tokenize("un ok de si va la preeclampsia")
    assert all(len(t) > 2 for t in tokens)


def test_tokenize_empty_string_returns_empty_list():
    assert _tokenize("") == []
    assert _tokenize("   ") == []


# ---------------------------------------------------------------------------
# search_bm25 — corpus, filtro de fuente y de 'active'
# ---------------------------------------------------------------------------

def test_search_bm25_empty_query_returns_empty(fake_store_factory):
    store = fake_store_factory({
        "doc0": [{"source_dataset": "maternaqaes_lm", "text": "preeclampsia hipertension proteinuria"}],
    })
    _inject_store(store)
    assert search_bm25("") == []
    assert search_bm25("   ") == []


def test_search_bm25_no_lexical_corpus_returns_empty(fake_store_factory):
    # Solo medmcqa/medqa_* en el store — LEXICAL_SOURCES = {maternaqaes_lm, upload}
    store = fake_store_factory({
        "doc0": [{"source_dataset": "medmcqa", "text": "preeclampsia hipertension proteinuria"}],
    })
    _inject_store(store)
    assert search_bm25("preeclampsia") == []


# Distractores con vocabulario disjunto: con corpus muy chico y el termino
# buscado presente en (casi) todos los docs, el IDF de BM25 se vuelve
# negativo (formula clasica Robertson-Sparck-Jones) y el resultado se
# filtra por el "score <= 0" de search_bm25 — no es un bug de produccion
# (irrelevante sobre los ~5.400 docs reales), pero exige que los fixtures
# de test tengan suficiente vocabulario distinto para no gatillarlo.
_DISTRACTORS = {
    "d1": [{"source_dataset": "maternaqaes_lm", "text": "el control prenatal reduce la mortalidad materna"}],
    "d2": [{"source_dataset": "maternaqaes_lm", "text": "diabetes gestacional y niveles de glucosa"}],
    "d3": [{"source_dataset": "maternaqaes_lm", "text": "parto vaginal frente a cesarea electiva"}],
    "d4": [{"source_dataset": "maternaqaes_lm", "text": "protocolo de vacunacion durante el embarazo"}],
}


def test_search_bm25_excludes_non_lexical_sources_even_with_lexical_match(fake_store_factory):
    store = fake_store_factory({
        **_DISTRACTORS,
        "es":  [{"source_dataset": "maternaqaes_lm", "text": "preeclampsia hipertension proteinuria embarazo"}],
        "en":  [{"source_dataset": "medqa_us", "text": "preeclampsia hipertension proteinuria pregnancy"}],
    })
    _inject_store(store)
    results = search_bm25("preeclampsia hipertension", k=10)
    assert len(results) == 1
    assert results[0]["source_dataset"] == "maternaqaes_lm"


def test_search_bm25_finds_lexical_match(fake_store_factory):
    store = fake_store_factory({
        **_DISTRACTORS,
        "match": [{"source_dataset": "maternaqaes_lm", "text": "la preeclampsia grave requiere sulfato de magnesio"}],
    })
    _inject_store(store)
    results = search_bm25("sulfato de magnesio", k=10)
    assert len(results) == 1
    assert results[0]["doc_id"] == "match"
    assert results[0]["retrieval"] == "bm25"
    assert "bm25_score" in results[0]


def test_search_bm25_includes_upload_source(fake_store_factory):
    store = fake_store_factory({
        **_DISTRACTORS,
        "up": [{"source_dataset": "upload", "text": "protocolo de lactancia exclusiva"}],
    })
    _inject_store(store)
    results = search_bm25("lactancia exclusiva", k=10)
    assert len(results) == 1
    assert results[0]["source_dataset"] == "upload"


def test_search_bm25_respects_active_flag(fake_store_factory):
    store = fake_store_factory({
        **_DISTRACTORS,
        "doc0": [{"source_dataset": "maternaqaes_lm", "text": "preeclampsia hipertension proteinuria embarazo"}],
    })
    _inject_store(store)

    # Activo: aparece
    assert len(search_bm25("preeclampsia", k=10)) == 1

    # Se desactiva desde el panel de administración: mutation_seq sube y
    # el índice BM25 debe reconstruirse excluyéndolo — no basta con no
    # recargarlo, tiene que dejar de aparecer en la siguiente búsqueda.
    store.update_document_status("doc0", active=False)
    assert search_bm25("preeclampsia", k=10) == []


def test_search_bm25_rebuilds_on_reactivation(fake_store_factory):
    store = fake_store_factory({
        **_DISTRACTORS,
        "doc0": [{"source_dataset": "maternaqaes_lm", "text": "preeclampsia hipertension proteinuria embarazo"}],
    })
    _inject_store(store)

    store.update_document_status("doc0", active=False)
    assert search_bm25("preeclampsia", k=10) == []

    store.update_document_status("doc0", active=True)
    assert len(search_bm25("preeclampsia", k=10)) == 1


def test_search_bm25_respects_k_limit(fake_store_factory):
    docs = {
        **_DISTRACTORS,
        **{
            f"doc{i}": [{"source_dataset": "maternaqaes_lm", "text": f"preeclampsia hipertension proteinuria variante {i}"}]
            for i in range(5)
        },
    }
    store = fake_store_factory(docs)
    _inject_store(store)
    results = search_bm25("preeclampsia hipertension", k=2)
    assert len(results) == 2
