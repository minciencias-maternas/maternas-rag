"""
ingest_maternaqaes_lm.py — Ingesta el corpus LM de MaternaQA-es al indice FAISS.

Descarga los splits train+validation+test del corpus LM desde GitHub raw:
  - train_lm.jsonl      (1.744 chunks, 52 PDFs, split train)
  - validation_lm.jsonl (101 chunks, 2 PDFs, split validation)
  - test_lm.jsonl       (111 chunks, 3 PDFs, split test)
Por defecto incluye los 3 splits. Usar --exclude-test para excluir test.

Los chunks del LM dataset tienen ~879 tokens en promedio — demasiado densos
para que Ragas pueda verificar statements contra fragmentos especificos.
Se aplica re-chunking con RecursiveCharacterTextSplitter a ~400 tok / 80 overlap
para obtener fragmentos precisos y citables.

Filtro de calidad: solo chunks con clinical_score >= MIN_CLINICAL_SCORE (default 15).
Descarta introducciones, bibliografias y secciones administrativas.

Por que JSONL y no PDFs crudos: ver foragents/qa_technical.md Q27.

Uso:
    # Ingestion normal (train+val+test, re-chunkeado)
    python -m src.ingestion.ingest_maternaqaes_lm

    # Dry-run: estadisticas sin modificar el indice
    python -m src.ingestion.ingest_maternaqaes_lm --dry-run

    # Excluir split test (para evaluacion sin data leakage)
    python -m src.ingestion.ingest_maternaqaes_lm --exclude-test
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
import urllib.request
from pathlib import Path
from typing import Iterator

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.formatters import Document
from src.ingestion.store import FAISSStore
from src.settings import settings

# ---------------------------------------------------------------------------
# URLs raw de GitHub
# ---------------------------------------------------------------------------

BASE_URL = "https://raw.githubusercontent.com/minciencias-maternas/MaternaQA-es/master/datasets/obstetrics/lm"

SPLITS = {
    "train":      f"{BASE_URL}/train_lm.jsonl",
    "validation": f"{BASE_URL}/validation_lm.jsonl",
    "test":       f"{BASE_URL}/test_lm.jsonl",
}

DATASET_ID = "maternaqaes_lm"
BATCH_SIZE  = 128

# Re-chunking: ~400 tokens / 80 overlap (1 token ≈ 4 chars)
CHUNK_SIZE_CHARS    = 400 * 4   # 1600 chars
CHUNK_OVERLAP_CHARS = 80  * 4   # 320 chars
MIN_CHUNK_CHARS     = 100       # descartar fragmentos muy cortos

# Filtro de calidad: solo chunks con clinical_score >= este umbral.
# Descarta introducciones, bibliografias y secciones administrativas.
#
# 15 es el valor historico (produccion hasta 2026-08). Analisis posterior
# (retrieval_eval.py, ver foragents/qa_technical.md) mostro que descarta 90
# de los 328 pares del golden set de evaluacion — su chunk-oro tiene
# clinical_score 8-14 y nunca llega al indice, poniendo un techo duro de
# 72.6% en cualquier metrica de recall sin importar el retriever usado.
# Se mantiene 15 como default para no romper la reproducibilidad de los
# reportes ya publicados (eval_report_config{A..E}_*.md); usar
# --min-clinical-score 8 para la re-ingesta que elimina ese techo.
MIN_CLINICAL_SCORE  = 15

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    separators=["\n\n", "\n", ". ", " "],
    length_function=len,
)


# ---------------------------------------------------------------------------
# Descarga y parseo
# ---------------------------------------------------------------------------

def _download_jsonl(url: str, split_name: str) -> list[dict]:
    """Descarga un JSONL de GitHub raw y lo parsea linea a linea."""
    print(f"  Descargando {split_name}: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            content = resp.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error descargando {url}: {e}")

    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"    [WARN] Linea JSON invalida, omitida: {e}")
    print(f"  {split_name}: {len(records)} chunks descargados")
    return records


def _to_documents(record: dict, min_clinical_score: int = MIN_CLINICAL_SCORE) -> list[Document]:
    """
    Convierte un registro del LM dataset en UNO O MAS Documents
    aplicando re-chunking a ~400 tokens con RecursiveCharacterTextSplitter.

    Filtros aplicados:
      - Texto < 50 chars: descartado
      - clinical_score < min_clinical_score: descartado (intro, biblio, admin)
      - Sub-chunks < MIN_CHUNK_CHARS tras el split: descartados
    """
    text  = record.get("text", "").strip()
    meta  = record.get("metadata", {})
    score = meta.get("clinical_score", 0)

    if not text or len(text) < 50:
        return []
    if score < min_clinical_score:
        return []

    base_meta = {
        "source_dataset": DATASET_ID,
        "language":       "es",
        "doc_id":         meta.get("pdf_id", "unknown"),
        "source_pdf":     meta.get("source_pdf", ""),
        "section_type":   meta.get("section_type", ""),
        "content_role":   meta.get("content_role", ""),
        "topics":         meta.get("topics", []),
        "clinical_score": score,
        "token_estimate": meta.get("token_estimate", 0),
        "lm_split":       meta.get("split", ""),
        "pages":          meta.get("pages", []),
        "parent_chunk_id": meta.get("chunk_id", ""),
    }

    # Si el texto ya es corto (<= CHUNK_SIZE_CHARS) no hace falta splitear
    if len(text) <= CHUNK_SIZE_CHARS:
        return [Document(
            text=text,
            metadata={**base_meta, "chunk_id": meta.get("chunk_id", str(uuid.uuid4()))},
        )]

    # Re-chunking con RecursiveCharacterTextSplitter
    sub_texts = _splitter.split_text(text)
    docs = []
    for idx, sub in enumerate(sub_texts):
        sub = sub.strip()
        if len(sub) < MIN_CHUNK_CHARS:
            continue
        chunk_meta = {**base_meta, "chunk_id": f"{meta.get('chunk_id','x')}_{idx:02d}"}
        docs.append(Document(text=sub, metadata=chunk_meta))
    return docs


# ---------------------------------------------------------------------------
# Verificacion de chunk_ids ya ingestados (evitar duplicados)
# ---------------------------------------------------------------------------

def _get_existing_chunk_ids(store: FAISSStore) -> set[str]:
    """Extrae todos los chunk_ids del dataset maternaqaes_lm ya en el indice."""
    existing = set()
    for meta in store.metadata.values():
        if meta.get("source_dataset") == DATASET_ID:
            cid = meta.get("chunk_id", "")
            if cid:
                existing.add(cid)
    return existing


# ---------------------------------------------------------------------------
# Ingestion principal
# ---------------------------------------------------------------------------

def ingest(
    include_test: bool = True,
    dry_run: bool = False,
    min_clinical_score: int = MIN_CLINICAL_SCORE,
) -> None:
    sep = "=" * 62
    print(sep)
    print("  Ingestando MaternaQA-es LM corpus al indice FAISS")
    print(f"  Splits: train + validation" + (" + test" if include_test else " (sin test)"))
    print(f"  min_clinical_score: {min_clinical_score}" + (
        " (default)" if min_clinical_score == MIN_CLINICAL_SCORE else " (override via CLI)"
    ))
    if dry_run:
        print("  MODO DRY-RUN: no se modificara el indice")
    print(sep)

    # 1. Descargar JSONL
    all_records: list[dict] = []
    splits_to_use = ["train", "validation", "test"] if include_test else ["train", "validation"]

    for split in splits_to_use:
        records = _download_jsonl(SPLITS[split], split)
        all_records.extend(records)

    print(f"\n  Total registros descargados: {len(all_records)}")

    # 2. Convertir a Documents con re-chunking
    documents: list[Document] = []
    skipped_short = 0
    skipped_score = 0
    original_chunks = 0

    for rec in all_records:
        original_chunks += 1
        score = rec.get("metadata", {}).get("clinical_score", 0)
        text  = rec.get("text", "").strip()
        if not text or len(text) < 50:
            skipped_short += 1
            continue
        if score < min_clinical_score:
            skipped_score += 1
            continue
        docs = _to_documents(rec, min_clinical_score=min_clinical_score)
        documents.extend(docs)

    avg_tok = sum(len(d.text) // 4 for d in documents) / max(1, len(documents))
    print(f"  Chunks originales      : {original_chunks}")
    print(f"  Omitidos (texto corto) : {skipped_short}")
    print(f"  Omitidos (score<{min_clinical_score})   : {skipped_score}")
    print(f"  Sub-chunks tras split  : {len(documents)}")
    print(f"  Promedio tokens/chunk  : {avg_tok:.0f} tok")

    if dry_run:
        print(f"\n  [DRY-RUN] Se ingrestarian {len(documents)} sub-chunks.")
        print(f"  Tokens totales estimados: {sum(len(d.text)//4 for d in documents):,}")
        _print_breakdown(documents)
        return

    # 3. Cargar indice existente
    store = FAISSStore.load()
    total_antes = store.total
    print(f"\n  Indice actual: {total_antes:,} vectores")

    # 4. Verificar duplicados
    existing_ids = _get_existing_chunk_ids(store)
    if existing_ids:
        print(f"  Chunks ya ingestados de {DATASET_ID}: {len(existing_ids)}")
        documents = [d for d in documents if d.metadata.get("chunk_id") not in existing_ids]
        print(f"  Chunks nuevos a ingestar: {len(documents)}")

    if not documents:
        print("  Nada que ingestar — todos los chunks ya estan en el indice.")
        return

    # 5. Ingestar en lotes
    print(f"\n  Ingestando {len(documents)} chunks en lotes de {BATCH_SIZE}...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        store.add_documents(batch, batch_size=BATCH_SIZE)

    # 6. Guardar
    store.save()
    total_despues = store.total
    added = total_despues - total_antes

    print(f"\n  Vectores antes : {total_antes:,}")
    print(f"  Vectores despues: {total_despues:,}")
    print(f"  Agregados       : {added:,}")
    _print_breakdown(documents)
    print(f"\n  Indice guardado en: {settings.faiss_store_path}")
    print(sep)


def _print_breakdown(documents: list[Document]) -> None:
    """Muestra distribucion por split y por source_pdf."""
    from collections import Counter

    splits = Counter(d.metadata.get("lm_split", "?") for d in documents)
    pdfs   = Counter(d.metadata.get("source_pdf", "?") for d in documents)

    print("\n  Por split:")
    for split, n in sorted(splits.items()):
        print(f"    {split:<12} {n:>5} chunks")

    print(f"\n  PDFs fuente ({len(pdfs)} unicos) — top 10 por chunks:")
    for pdf, n in pdfs.most_common(10):
        print(f"    {n:>4}  {pdf}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingesta MaternaQA-es LM corpus al indice FAISS",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Por defecto INCLUYE el split test — es parte del corpus de conocimiento.
    # Usar --exclude-test para evaluar sin data leakage (comparacion con baseline).
    parser.add_argument(
        "--exclude-test",
        action="store_true",
        help="Excluir split test (para evaluacion sin data leakage)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar estadisticas, sin modificar el indice",
    )
    parser.add_argument(
        "--min-clinical-score",
        type=int,
        default=MIN_CLINICAL_SCORE,
        help=(
            f"Umbral de clinical_score para incluir un chunk (default {MIN_CLINICAL_SCORE}). "
            "Bajarlo a 8 recupera los 90 pares del golden set de evaluacion cuyo chunk-oro "
            "hoy queda fuera del indice — ver retrieval_eval.py y foragents/qa_technical.md."
        ),
    )
    args = parser.parse_args()

    ingest(
        include_test=not args.exclude_test,
        dry_run=args.dry_run,
        min_clinical_score=args.min_clinical_score,
    )
