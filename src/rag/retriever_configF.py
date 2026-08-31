"""
retriever_configF.py — CONFIG F: hibrido denso (FAISS) + lexico (BM25).

Retoma el hibrido de Config B (commit a6baf49), pero corrigiendo su premisa:
Config B usaba BM25 para *filtrar ruido* (Multiclinsum contaminando el top-k
denso). Medido sobre el indice actual (post Config D, sin textbook ni
Multiclinsum), el ruido ya no es el problema — 100% de los fragmentos
recuperados en el ultimo eval ya venian de maternaqaes_lm. Lo que BM25 aporta
aqui es desempate lexico *dentro* del corpus correcto (farmacos, dosis,
cifras, siglas) donde el embedding difumina.

Medido con src/evaluation/retrieval_eval.py sobre los 238 pares alcanzables
de MaternaQA-es test (chunk_id-oro presente en el indice):

    denso puro            R@5=0.828  R@10=0.882  MRR@10=0.664
    BM25 solo             R@5=0.866  R@10=0.908  MRR@10=0.719
    hibrido RRF k=10      R@5=0.912  R@10=0.958  MRR@10=0.730

Ver foragents/retrieval_arquitecturas_configs.md (Config F) para el detalle
completo, incluyendo el techo de alcanzabilidad (72.6%) que ningun retriever
puede superar sin bajar MIN_CLINICAL_SCORE en ingest_maternaqaes_lm.py.

PARA ACTIVAR CONFIG F (produccion):
    copy src\\rag\\retriever_configF.py src\\rag\\retriever.py

PARA RESTAURAR CONFIG D (denso puro, sin BM25):
    copy src\\rag\\retriever_configD.py src\\rag\\retriever.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.store import FAISSStore
from src.settings import settings

logger = logging.getLogger(__name__)

# Fuentes de la capa densa. Hoy coincide con el 100% de lo que hay en el
# indice (medmcqa, medqa_*, maternaqaes_lm, upload) — no hay textbook ni
# Multiclinsum desde la remocion por licencia (67145ee). Se mantiene el
# filtro explicito como red de seguridad ante una futura ingesta que
# reintroduzca una fuente que no deba pasar por la capa densa.
DENSE_SOURCES = {
    "medmcqa",
    "medqa_us",
    "medqa_taiwan",
    "medqa_mainland",
    "maternaqaes_lm",
    "upload",   # documentos cargados desde el panel de administración
}

# ---------------------------------------------------------------------------
# Singleton FAISS
# ---------------------------------------------------------------------------

_store: FAISSStore | None = None


def _get_store() -> FAISSStore:
    global _store
    if _store is None:
        logger.info("[Retriever] Cargando indice FAISS...")
        _store = FAISSStore.load()
        logger.info(f"[Retriever] Indice listo: {_store.total:,} vectores")
    return _store


# ---------------------------------------------------------------------------
# Etiquetas legibles por dataset
# ---------------------------------------------------------------------------

SOURCE_LABELS = {
    "medmcqa":               "Pregunta medica con explicacion",
    "medqa_us":               "Pregunta de examen medico (ingles)",
    "medqa_taiwan":            "Pregunta de examen medico (chino tradicional)",
    "medqa_mainland":          "Pregunta de examen medico (chino simplificado)",
    "maternaqaes_lm":          "Documento clinico obstetrico en espanol",
    "upload":                  "Documento cargado por el equipo",
}


def source_label(source_dataset: str) -> str:
    return SOURCE_LABELS.get(source_dataset, f"Fuente: {source_dataset}")


def source_path(doc: dict[str, Any]) -> str:
    filename = doc.get("filename") or doc.get("source_pdf") or ""
    chunk_id = doc.get("chunk_id") or ""
    doc_id   = doc.get("doc_id") or ""

    if filename and chunk_id:
        return f"{filename} (chunk {chunk_id})"
    if filename:
        return filename
    if doc_id and chunk_id:
        return f"{doc_id}/{chunk_id}"
    if doc_id:
        return doc_id
    if chunk_id:
        return chunk_id
    return "desconocido"


# ---------------------------------------------------------------------------
# Identidad de un fragmento — para deduplicar entre capa densa y lexica
# ---------------------------------------------------------------------------

def _doc_identity(doc: dict[str, Any]) -> tuple:
    """Clave de deduplicacion. chunk_id solo no basta: no esta garantizado
    unico entre datasets distintos (p.ej. dos fuentes reusando '00027_00').
    Si falta doc_id o chunk_id, cae a un hash del texto para no colapsar
    fragmentos distintos bajo una clave vacia compartida."""
    doc_id   = doc.get("doc_id")
    chunk_id = doc.get("chunk_id")
    if doc_id and chunk_id:
        return (doc.get("source_dataset"), doc_id, chunk_id)
    return (doc.get("source_dataset"), "_text", hash(doc.get("text", "")))


# ---------------------------------------------------------------------------
# Busqueda densa — FAISS sobre DENSE_SOURCES
# ---------------------------------------------------------------------------

def _retrieve_dense(query: str, pool: int) -> list[dict[str, Any]]:
    store = _get_store()
    candidates = store.search(query, k=pool)

    results = []
    for doc in candidates:
        if doc.get("source_dataset", "") in DENSE_SOURCES:
            results.append({**doc, "retrieval": "dense"})

    logger.info(f"[Retriever:dense] {len(results)}/{pool} candidatos")
    return results


# ---------------------------------------------------------------------------
# Busqueda lexica — BM25 sobre el corpus en espanol
# ---------------------------------------------------------------------------

def _retrieve_bm25(query: str, pool: int) -> list[dict[str, Any]]:
    from src.rag.bm25_index import search_bm25

    try:
        results = search_bm25(query, k=pool)
    except ModuleNotFoundError:
        logger.warning("[Retriever:bm25] rank_bm25 no instalado — capa lexica deshabilitada")
        return []
    except Exception:
        logger.exception("[Retriever:bm25] fallo construyendo/consultando el indice BM25")
        return []

    logger.info(f"[Retriever:bm25] {len(results)}/{pool} candidatos")
    return results


# ---------------------------------------------------------------------------
# Fusion — RRF o ponderada, configurables via settings
# ---------------------------------------------------------------------------

def _fuse_rrf(
    dense: list[dict[str, Any]],
    bm25: list[dict[str, Any]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank + 1)) por lista
    en la que aparece el fragmento. No depende de que los scores de FAISS-IP
    y BM25 sean comparables en escala — solo del orden dentro de cada lista.
    """
    fused: dict[tuple, dict[str, Any]] = {}
    scores: dict[tuple, float] = {}

    for rank, doc in enumerate(dense):
        key = _doc_identity(doc)
        fused[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, doc in enumerate(bm25):
        key = _doc_identity(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        if key in fused:
            fused[key] = {**fused[key], "retrieval": "hybrid"}
        else:
            fused[key] = doc

    ordered = sorted(fused.keys(), key=lambda k: scores[k], reverse=True)
    return [fused[k] for k in ordered]


def _fuse_weighted(
    dense: list[dict[str, Any]],
    bm25: list[dict[str, Any]],
    dense_weight: float,
) -> list[dict[str, Any]]:
    """Fusion min-max ponderada sobre los scores nativos de cada lista
    (FAISS score / bm25_score), normalizados por query antes de combinar.
    """
    def _minmax(docs: list[dict[str, Any]], score_key: str) -> dict[tuple, float]:
        if not docs:
            return {}
        vals = [d.get(score_key, 0.0) for d in docs]
        lo, hi = min(vals), max(vals)
        span = hi - lo
        return {
            _doc_identity(d): (0.0 if span == 0 else (d.get(score_key, 0.0) - lo) / span)
            for d in docs
        }

    dense_norm = _minmax(dense, "score")
    bm25_norm  = _minmax(bm25, "bm25_score")

    fused: dict[tuple, dict[str, Any]] = {}
    for doc in dense:
        fused[_doc_identity(doc)] = doc
    for doc in bm25:
        key = _doc_identity(doc)
        if key in fused:
            fused[key] = {**fused[key], "retrieval": "hybrid"}
        else:
            fused[key] = doc

    scores = {
        key: dense_weight * dense_norm.get(key, 0.0) + (1 - dense_weight) * bm25_norm.get(key, 0.0)
        for key in fused
    }
    ordered = sorted(fused.keys(), key=lambda k: scores[k], reverse=True)
    return [fused[k] for k in ordered]


# ---------------------------------------------------------------------------
# Funcion publica
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int | None = None,
    k_bm25: int | None = None,
) -> list[dict[str, Any]]:
    """
    Config F: fusiona candidatos densos (FAISS) y lexicos (BM25) y devuelve
    el top-k final. `k_bm25` se acepta por compatibilidad con la firma
    historica del proyecto pero no se usa — el tamano del pool lexico se
    controla via settings.rag_bm25_pool.
    """
    if not query or not query.strip():
        return []
    if k is None:
        k = settings.rag_top_k

    dense_pool = _retrieve_dense(query, pool=settings.rag_dense_pool)
    bm25_pool  = _retrieve_bm25(query, pool=settings.rag_bm25_pool)

    if settings.rag_fusion_strategy == "weighted":
        fused = _fuse_weighted(dense_pool, bm25_pool, settings.rag_fusion_dense_weight)
    else:
        fused = _fuse_rrf(dense_pool, bm25_pool, settings.rag_fusion_rrf_k)

    results = fused[:k]
    logger.info(
        f"[Retriever] Total: {len(results)} fusionados "
        f"(denso={len(dense_pool)}, bm25={len(bm25_pool)}, config F)"
    )
    return results


# ---------------------------------------------------------------------------
# Formateo del contexto para el LLM
# ---------------------------------------------------------------------------

def format_context(docs: list[dict[str, Any]], max_chars: int = 4000) -> str:
    if not docs:
        return "No se encontraron fragmentos relevantes en la base de conocimiento."

    fragments: list[str] = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "").strip()
        fragment = f"--- Fragmento [{i}] ---\n{text}"

        if total_chars + len(fragment) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                fragments.append(fragment[:remaining] + "...")
            break

        fragments.append(fragment)
        total_chars += len(fragment)

    return "\n\n".join(fragments)
