"""
test_retriever_fusion.py — Fusión híbrida denso+BM25 de retriever.py (Config F).

Cubre las piezas puras (_doc_identity, _fuse_rrf, _fuse_weighted) sin tocar
FAISS ni BM25, y retrieve() end-to-end con _retrieve_dense/_retrieve_bm25
monkeypatcheadas — así se prueba la fusión real sin cargar el índice de
253k+ vectores en cada test.

Nota sobre settings: retriever.py hace "from src.settings import settings"
a nivel de módulo, así que conserva su propia referencia al Settings real
sin importar lo que haga el mock_settings autouse de conftest.py (mismo
caso documentado en test_store_admin.py). Se monkeypatchea el atributo
directamente en el objeto real vía "src.rag.retriever.settings.<campo>".
"""

from __future__ import annotations

import pytest

from src.rag.retriever import (
    _doc_identity,
    _fuse_rrf,
    _fuse_weighted,
    retrieve,
)
import src.rag.retriever as retriever


# ---------------------------------------------------------------------------
# _doc_identity
# ---------------------------------------------------------------------------

def test_doc_identity_same_fields_same_key():
    a = {"source_dataset": "maternaqaes_lm", "doc_id": "gpc1", "chunk_id": "c1", "text": "x"}
    b = {"source_dataset": "maternaqaes_lm", "doc_id": "gpc1", "chunk_id": "c1", "text": "distinto texto pero mismo chunk"}
    assert _doc_identity(a) == _doc_identity(b)


def test_doc_identity_different_source_different_key():
    a = {"source_dataset": "maternaqaes_lm", "doc_id": "gpc1", "chunk_id": "00001"}
    b = {"source_dataset": "medmcqa", "doc_id": "gpc1", "chunk_id": "00001"}
    assert _doc_identity(a) != _doc_identity(b)


def test_doc_identity_falls_back_to_text_hash_without_ids():
    a = {"source_dataset": "upload", "text": "contenido A"}
    b = {"source_dataset": "upload", "text": "contenido B"}
    assert _doc_identity(a) != _doc_identity(b)


# ---------------------------------------------------------------------------
# _fuse_rrf
# ---------------------------------------------------------------------------

def _doc(source, doc_id, chunk_id, **extra):
    return {"source_dataset": source, "doc_id": doc_id, "chunk_id": chunk_id,
            "text": f"{doc_id}/{chunk_id}", **extra}


def test_fuse_rrf_marks_hybrid_when_in_both_lists():
    shared = _doc("maternaqaes_lm", "gpc1", "c1")
    dense = [shared]
    bm25 = [_doc("maternaqaes_lm", "gpc1", "c1")]
    fused = _fuse_rrf(dense, bm25, rrf_k=10)
    assert len(fused) == 1
    assert fused[0]["retrieval"] == "hybrid"


def test_fuse_rrf_preserves_dense_only_and_bm25_only_labels():
    dense = [_doc("maternaqaes_lm", "gpc1", "c1", retrieval="dense")]
    bm25 = [_doc("maternaqaes_lm", "gpc2", "c9", retrieval="bm25")]
    fused = _fuse_rrf(dense, bm25, rrf_k=10)
    labels = {f["doc_id"]: f["retrieval"] for f in fused}
    assert labels["gpc1"] == "dense"
    assert labels["gpc2"] == "bm25"


def test_fuse_rrf_ranks_item_in_both_lists_above_single_list_items():
    # Un item que aparece primero en AMBAS listas debe ganarle a un item
    # que solo aparece primero en una — es el punto central de RRF.
    shared = _doc("maternaqaes_lm", "shared", "c1")
    dense_only = _doc("maternaqaes_lm", "dense_only", "c2")
    bm25_only = _doc("maternaqaes_lm", "bm25_only", "c3")

    dense = [shared, dense_only]
    bm25 = [shared, bm25_only]
    fused = _fuse_rrf(dense, bm25, rrf_k=10)
    assert fused[0]["doc_id"] == "shared"


def test_fuse_rrf_deduplicates():
    shared = _doc("maternaqaes_lm", "gpc1", "c1")
    fused = _fuse_rrf([shared], [dict(shared)], rrf_k=10)
    assert len(fused) == 1


def test_fuse_rrf_empty_inputs():
    assert _fuse_rrf([], [], rrf_k=10) == []


# ---------------------------------------------------------------------------
# _fuse_weighted
# ---------------------------------------------------------------------------

def test_fuse_weighted_dense_weight_1_ignores_bm25_order():
    dense = [
        _doc("maternaqaes_lm", "a", "1", score=0.9),
        _doc("maternaqaes_lm", "b", "2", score=0.1),
    ]
    bm25 = [_doc("maternaqaes_lm", "b", "2", bm25_score=100.0)]
    fused = _fuse_weighted(dense, bm25, dense_weight=1.0)
    assert fused[0]["doc_id"] == "a"


def test_fuse_weighted_marks_hybrid():
    shared = _doc("maternaqaes_lm", "gpc1", "c1", score=0.5)
    dense = [shared]
    bm25 = [_doc("maternaqaes_lm", "gpc1", "c1", bm25_score=5.0)]
    fused = _fuse_weighted(dense, bm25, dense_weight=0.5)
    assert fused[0]["retrieval"] == "hybrid"


def test_fuse_weighted_handles_single_candidate_no_division_by_zero():
    # min==max en una lista de un solo elemento — no debe lanzar ZeroDivisionError.
    dense = [_doc("maternaqaes_lm", "a", "1", score=0.7)]
    fused = _fuse_weighted(dense, [], dense_weight=0.5)
    assert len(fused) == 1


# ---------------------------------------------------------------------------
# retrieve() — end-to-end con capas monkeypatcheadas
# ---------------------------------------------------------------------------

def test_retrieve_empty_query_returns_empty():
    assert retrieve("") == []
    assert retrieve("   ") == []


def test_retrieve_respects_k(monkeypatch):
    dense = [_doc("maternaqaes_lm", f"d{i}", str(i)) for i in range(10)]
    monkeypatch.setattr(retriever, "_retrieve_dense", lambda q, pool: dense)
    monkeypatch.setattr(retriever, "_retrieve_bm25", lambda q, pool: [])
    results = retrieve("preeclampsia", k=3)
    assert len(results) == 3


def test_retrieve_never_duplicates_across_dense_and_bm25(monkeypatch):
    shared = _doc("maternaqaes_lm", "gpc1", "c1")
    monkeypatch.setattr(retriever, "_retrieve_dense", lambda q, pool: [shared])
    monkeypatch.setattr(retriever, "_retrieve_bm25", lambda q, pool: [dict(shared)])
    results = retrieve("preeclampsia", k=5)
    assert len(results) == 1
    assert results[0]["retrieval"] == "hybrid"


def test_retrieve_uses_weighted_strategy_when_configured(monkeypatch):
    dense = [_doc("maternaqaes_lm", "a", "1", score=0.9), _doc("maternaqaes_lm", "b", "2", score=0.1)]
    monkeypatch.setattr(retriever, "_retrieve_dense", lambda q, pool: dense)
    monkeypatch.setattr(retriever, "_retrieve_bm25", lambda q, pool: [])
    monkeypatch.setattr(retriever.settings, "rag_fusion_strategy", "weighted")
    results = retrieve("preeclampsia", k=5)
    assert results[0]["doc_id"] == "a"
