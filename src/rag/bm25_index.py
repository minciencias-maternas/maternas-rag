"""
bm25_index.py — Índice BM25 (léxico) sobre el corpus en español del índice FAISS.

Complementa la búsqueda densa: `multilingual-e5-base` ya separa bien
maternaqaes_lm/upload (ES) de medmcqa/medqa_* (EN/ZH) en el ranking denso, así
que BM25 no filtra ruido de otras fuentes — afina *qué pasaje* dentro del
corpus correcto, desempatando por término exacto (fármacos, dosis, cifras,
siglas) donde el embedding difumina. Medido en retrieval_eval.py: BM25 solo ya
supera al denso puro en Recall@5 sobre este corpus (ver
foragents/retrieval_arquitecturas_configs.md, Config F).

Corpus indexado: LEXICAL_SOURCES = {"maternaqaes_lm", "upload"} — ~5.400
fragmentos. Se excluyen intencionalmente medmcqa/medqa_* (97,8% del índice):
son ítems de examen en inglés/chino, no texto libre en español, y no aportan
señal léxica útil para preguntas en español.

El índice se construye en memoria la primera vez que se usa (singleton) a
partir de la metadata ya cargada en el FAISSStore — no relee metadata.pkl del
disco. Se reconstruye automáticamente si `store.mutation_seq` cambia (nuevo
documento subido, o documento activado/desactivado desde el panel), así que
respeta el mismo flag `active` que el retrieval denso.

Construcción: ~1 s, unos pocos MB (mucho menor que los ~10-20 s / ~150 MB de
la versión anterior sobre los 51.804 fragmentos de Multiclinsum — ver
git show a6baf49:src/rag/bm25_index.py).

Uso:
    from src.rag.bm25_index import search_bm25
    results = search_bm25("preeclampsia hipertension proteinuria", k=10)
"""

from __future__ import annotations

import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

LEXICAL_SOURCES = {"maternaqaes_lm", "upload"}

# ---------------------------------------------------------------------------
# Tokenizador simple multilingüe (ES / EN) con plegado de acentos
# ---------------------------------------------------------------------------

_STOPWORDS_ES = {
    "de", "la", "el", "en", "y", "a", "que", "se", "los", "las", "un", "una",
    "por", "con", "para", "del", "al", "es", "su", "sus", "lo", "le", "les",
    "como", "pero", "si", "no", "fue", "una", "este", "esta", "esto",
}

_STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "was", "for",
    "on", "at", "by", "with", "from", "as", "be", "this", "that", "are",
    "were", "it", "he", "she", "we", "they", "has", "had", "have", "not",
}

_STOPWORDS = _STOPWORDS_ES | _STOPWORDS_EN


def _fold_accents(text: str) -> str:
    """Quita diacríticos (á→a, ñ→n vía NFD+filtro de combining marks).

    Necesario porque en Telegram la gente escribe "preeclampsia" tanto con
    tilde como sin ella, y el golden set (bien redactado) no ejercita ese
    caso. Sin esto, "eclampsia" y "eclámpsia" tokenizan distinto.
    """
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def _tokenize(text: str) -> list[str]:
    """Tokeniza texto en minúsculas, pliega acentos, elimina stopwords y
    tokens cortos."""
    folded = _fold_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]+", folded)
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Singleton BM25 — invalidado por mutation_seq del FAISSStore
# ---------------------------------------------------------------------------

_bm25_index = None            # instancia BM25Okapi
_bm25_docs: list[dict] = []   # metadata + text de los fragmentos indexados
_built_at_mutation_seq: int | None = None


def _build_index() -> None:
    """Filtra LEXICAL_SOURCES desde el FAISSStore ya cargado y construye
    el índice BM25. No relee metadata.pkl: reusa store.metadata."""
    global _bm25_index, _bm25_docs, _built_at_mutation_seq

    from rank_bm25 import BM25Okapi

    from src.rag.retriever import _get_store
    from src.ingestion.store import is_active

    store = _get_store()

    logger.info("[BM25] Filtrando corpus lexico desde metadata en memoria...")
    _bm25_docs = [
        doc for doc in store.metadata.values()
        if doc.get("source_dataset") in LEXICAL_SOURCES and is_active(doc)
    ]
    logger.info(f"[BM25] {len(_bm25_docs):,} fragmentos lexicos (maternaqaes_lm + upload)")

    if not _bm25_docs:
        _bm25_index = None
        _built_at_mutation_seq = store.mutation_seq
        logger.warning("[BM25] Corpus lexico vacio — search_bm25() devolvera siempre []")
        return

    corpus = [_tokenize(doc.get("text", "")) for doc in _bm25_docs]
    _bm25_index = BM25Okapi(corpus, k1=1.5, b=0.75)
    _built_at_mutation_seq = store.mutation_seq
    logger.info("[BM25] Indice BM25 construido")


def _get_index():
    """Devuelve el índice BM25, (re)construyéndolo si el store mutó desde
    la última construcción (documento subido/activado/desactivado)."""
    global _bm25_index

    from src.rag.retriever import _get_store

    store = _get_store()
    if _bm25_index is None or _built_at_mutation_seq != store.mutation_seq:
        _build_index()
    return _bm25_index


# ---------------------------------------------------------------------------
# Función pública
# ---------------------------------------------------------------------------

def search_bm25(query: str, k: int = 10) -> list[dict[str, Any]]:
    """
    Busca en el índice BM25 del corpus léxico (maternaqaes_lm + upload).

    Args:
        query: Texto de la query del usuario.
        k:     Máximo de resultados a devolver.

    Returns:
        Lista de dicts ordenada por score BM25 descendente, con clave
        "bm25_score". Vacía si no hay tokens válidos en la query o el
        corpus léxico está vacío.
    """
    if not query or not query.strip():
        return []

    index = _get_index()
    if index is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = index.get_scores(tokens)

    scored = sorted(zip(scores, _bm25_docs), key=lambda x: x[0], reverse=True)

    results = []
    for score, doc in scored[:k]:
        if score <= 0:
            break
        results.append({**doc, "bm25_score": float(score), "retrieval": "bm25"})

    logger.info(
        f"[BM25] query='{query[:50]}' tokens={tokens[:6]} "
        f"-> {len(results)} resultados"
    )
    return results
