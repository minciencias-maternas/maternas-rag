"""
retrieval_eval.py — Evaluacion determinista de retrieval (sin juez LLM).

Complementa a eval_pipeline.py (Ragas), no lo reemplaza. Ragas mide la cadena
completa (retrieval + generacion) con un LLM judge sobre una muestra de
15-20 pares — util para el entregable, pero incapaz de resolver un delta de
retrieval de unos pocos puntos de Recall@5: el propio commit 67145ee midio
~0.25 std de ruido entre corridas con ese tamano de muestra.

Este harness mide *solo* retrieval, sin juez: el acierto es exacto (el
chunk-oro esta o no esta entre los recuperados), cubre los 328 pares del
golden set de MaternaQA-es test, corre en ~1 minuto y no gasta tokens de Groq.
Se usa para elegir estrategia de fusion, tamano de pool y umbrales — decisiones
de ajuste que no deberian tomarse con 15 muestras y una opinion de LLM.

Metodo: el golden set (evaluation_reports/maternaqa_test.jsonl) trae
`chunk_id` del chunk-oro pre-rechunking. La ingesta de maternaqaes_lm
(src/ingestion/ingest_maternaqaes_lm.py) guarda ese mismo id en
`parent_chunk_id` de cada sub-chunk. Un par cuenta como acierto si CUALQUIERA
de los sub-chunks recuperados tiene `parent_chunk_id == chunk_id-oro`.

Techo de alcanzabilidad: no todo par del golden set tiene su chunk-oro en el
indice (ver MIN_CLINICAL_SCORE en ingest_maternaqaes_lm.py). Se reporta
Recall@k tanto sobre los pares ALCANZABLES como sobre el TOTAL — solo el
segundo refleja lo que un usuario real experimenta.

ADVERTENCIA — data leakage estructural, no introducido por este script: los
3 PDFs del split `test` del corpus LM (GPC-Atencion-Prenatal-de-Bajo-Riesgo-
2023.pdf, vol831-1.pdf, 4142_stamped.pdf) son EXACTAMENTE los 3 source_pdf de
los que sale el golden set de evaluacion (maternaqa_test.jsonl) — verificado
por interseccion exacta. `ingest_maternaqaes_lm.py` incluye ese split por
default (`include_test=True`, ya era el default antes de este harness), asi
que el documento fuente de cada pregunta de evaluacion ya esta en el indice.
Esto es una medida de "encuentra el pasaje correcto dentro de documentos ya
indexados" (lo relevante para produccion), NO una medida de generalizacion a
documentos nunca vistos, y NO es comparable sin mas contra el baseline
publicado de MaternaQA-es en eval_pipeline.py. Usar --exclude-test en la
ingestion para una comparacion sin esta leakage.

Uso:
    # Linea base, solo denso
    python -m src.evaluation.retrieval_eval --compare dense

    # Comparar las tres estrategias en una sola corrida (embeddings reusados)
    python -m src.evaluation.retrieval_eval --compare dense,bm25,hybrid

    # Incluir tambien la fusion ponderada
    python -m src.evaluation.retrieval_eval --compare dense,bm25,hybrid,hybrid_weighted

    # Prueba rapida sobre los primeros 30 pares
    python -m src.evaluation.retrieval_eval --compare dense,hybrid --limit 30
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("evaluation_reports")
GOLDEN_PATH = REPORTS_DIR / "maternaqa_test.jsonl"

K_VALUES = [1, 3, 5, 10, 20]
STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_weighted"]


# ---------------------------------------------------------------------------
# Carga del golden set y del indice
# ---------------------------------------------------------------------------

def _load_golden() -> list[dict[str, Any]]:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {GOLDEN_PATH}. Ejecuta src/evaluation/sampler.py "
            "o eval_pipeline.py primero para descargarlo/cachearlo."
        )
    pairs = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def _build_reachability(pairs: list[dict[str, Any]]) -> tuple[set[str], list[dict[str, Any]]]:
    """Devuelve (chunk_ids-oro presentes en el indice, pares alcanzables)."""
    from src.rag.retriever import _get_store

    store = _get_store()
    parent_ids = {
        doc.get("parent_chunk_id")
        for doc in store.metadata.values()
        if doc.get("source_dataset") in ("maternaqaes_lm", "upload") and doc.get("parent_chunk_id")
    }
    reachable = [p for p in pairs if p.get("chunk_id") in parent_ids]
    return parent_ids, reachable


# ---------------------------------------------------------------------------
# Candidatos por estrategia — reusa las funciones internas de retriever.py
# (mismo patron que routes_admin.py/routes_documents.py, que ya importan
# _get_store directamente: este proyecto no separa una API "publica" para
# estas piezas internas).
# ---------------------------------------------------------------------------

def _run_strategy(name: str, queries: list[str], pool: int) -> list[list[dict[str, Any]]]:
    from src.rag import retriever as R

    if name == "dense":
        return [R._retrieve_dense(q, pool=pool) for q in queries]

    if name == "bm25":
        return [R._retrieve_bm25(q, pool=pool) for q in queries]

    if name in ("hybrid", "hybrid_weighted"):
        from src.settings import settings
        results = []
        for q in queries:
            dense = R._retrieve_dense(q, pool=pool)
            bm25 = R._retrieve_bm25(q, pool=pool)
            if name == "hybrid":
                fused = R._fuse_rrf(dense, bm25, settings.rag_fusion_rrf_k)
            else:
                fused = R._fuse_weighted(dense, bm25, settings.rag_fusion_dense_weight)
            results.append(fused)
        return results

    raise ValueError(f"Estrategia desconocida: {name}")


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def _parent_ids_of(doc: dict[str, Any]) -> str | None:
    return doc.get("parent_chunk_id")


def _dedup_parents(candidates: list[dict[str, Any]]) -> list[str]:
    """Ranking de parent_chunk_id, deduplicado preservando el primer rank
    en que aparece cada uno (un padre puede tener varios sub-chunks
    recuperados; solo cuenta su mejor posicion)."""
    out: list[str] = []
    seen: set[str] = set()
    for doc in candidates:
        pid = _parent_ids_of(doc)
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _metrics_for(ranked_lists: list[list[str]], truth: list[str]) -> dict[str, float]:
    n = len(truth)
    out: dict[str, float] = {}
    for k in K_VALUES:
        hits = sum(1 for ranked, t in zip(ranked_lists, truth) if t in ranked[:k])
        out[f"recall_at_{k}"] = hits / n if n else 0.0

    mrr_sum = 0.0
    ndcg_sum = 0.0
    for ranked, t in zip(ranked_lists, truth):
        top10 = ranked[:10]
        if t in top10:
            rank = top10.index(t) + 1
            mrr_sum += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)
    out["mrr_at_10"] = mrr_sum / n if n else 0.0
    out["ndcg_at_10"] = ndcg_sum / n if n else 0.0
    return out


def _scale_to_total(reachable_metrics: dict[str, float], reachability: float) -> dict[str, float]:
    """Metricas escaladas al total del golden set: un par inalcanzable
    cuenta como fallo (0) para cualquier estrategia por definicion, asi que
    escalar por la tasa de alcanzabilidad es exacto, no una aproximacion."""
    return {k: v * reachability for k, v in reachable_metrics.items()}


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def _write_report(
    label: str,
    ts: str,
    n_total: int,
    n_reachable: int,
    per_strategy_reachable: dict[str, dict[str, float]],
    per_strategy_total: dict[str, dict[str, float]],
    elapsed_s: float,
) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(exist_ok=True)
    json_path = REPORTS_DIR / f"retrieval_eval_{label}_{ts}.json"
    md_path   = REPORTS_DIR / f"retrieval_eval_{label}_{ts}.md"

    reachability = n_reachable / n_total if n_total else 0.0

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":    ts,
            "n_total":      n_total,
            "n_reachable":  n_reachable,
            "reachability": reachability,
            "elapsed_s":    round(elapsed_s, 1),
            "reachable":    per_strategy_reachable,
            "total":        per_strategy_total,
        }, f, ensure_ascii=False, indent=2)

    lines = [
        "# Reporte de Evaluacion de Retrieval — Maternas",
        "",
        f"> **Dataset:** MaternaQA-es (split: test)  |  **Fecha:** {ts[:8]}",
        f"> **Pares totales:** {n_total}  |  **Alcanzables:** {n_reachable} "
        f"({reachability*100:.1f}%)  |  **Tiempo:** {elapsed_s:.0f}s",
        "",
        "Sin juez LLM: acierto exacto por `chunk_id`-oro. Ver docstring de "
        "`retrieval_eval.py` para la metodologia y por que Ragas no puede "
        "resolver deltas de este tamano con la muestra que usa hoy.",
        "",
        "> **Data leakage estructural (no introducido por este harness):** los 3 PDFs",
        "> del split `test` del corpus LM son exactamente los 3 `source_pdf` de este",
        "> golden set, y `ingest_maternaqaes_lm.py` los incluye por default. Estas",
        "> cifras miden \"encuentra el pasaje correcto dentro de documentos ya",
        "> indexados\", no generalizacion a documentos nunca vistos. Ver docstring.",
        "",
        "---",
        "",
        f"## Sobre los {n_reachable} pares alcanzables",
        "",
        "| Estrategia | R@1 | R@3 | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in per_strategy_reachable.items():
        lines.append(
            f"| {name} | {m['recall_at_1']:.3f} | {m['recall_at_3']:.3f} | "
            f"{m['recall_at_5']:.3f} | {m['recall_at_10']:.3f} | "
            f"{m['recall_at_20']:.3f} | {m['mrr_at_10']:.3f} | {m['ndcg_at_10']:.3f} |"
        )

    lines += [
        "",
        f"## Sobre los {n_total} pares totales (incluye el techo de alcanzabilidad)",
        "",
        "| Estrategia | R@1 | R@3 | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in per_strategy_total.items():
        lines.append(
            f"| {name} | {m['recall_at_1']:.3f} | {m['recall_at_3']:.3f} | "
            f"{m['recall_at_5']:.3f} | {m['recall_at_10']:.3f} | "
            f"{m['recall_at_20']:.3f} | {m['mrr_at_10']:.3f} | {m['ndcg_at_10']:.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Guia de interpretacion",
        "",
        "| Metrica | Que mide |",
        "|:---|:---|",
        "| `R@k` | En que fraccion de preguntas el chunk-oro esta entre los k fragmentos entregados al LLM |",
        "| `MRR@10` | Que tan arriba en el ranking llega el chunk-oro (1.0 = siempre primero) |",
        "| `nDCG@10` | Como MRR pero penaliza mas suave por posicion (log2) |",
        "",
        "> Los pares NO alcanzables (chunk-oro fuera del indice) cuentan como fallo",
        "> para toda estrategia — es un limite del corpus, no del retriever. Ver",
        "> `--min-clinical-score` en `src/ingestion/ingest_maternaqaes_lm.py`.",
        "",
        f"*Generado por `src/evaluation/retrieval_eval.py` — {ts[:8]}*",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluacion determinista de retrieval (sin juez LLM)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--compare", type=str, default="dense",
        help=f"Lista separada por comas de: {', '.join(STRATEGIES)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limitar a los primeros N pares")
    parser.add_argument("--pool", type=int, default=20, help="Candidatos por lado antes de fusionar")
    parser.add_argument("--label", type=str, default="", help="Sufijo para el nombre de los reportes")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.compare.split(",") if s.strip()]
    for s in strategies:
        if s not in STRATEGIES:
            parser.error(f"Estrategia invalida: '{s}'. Opciones: {', '.join(STRATEGIES)}")

    pairs = _load_golden()
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Golden set: {len(pairs)} pares totales")
    _, reachable = _build_reachability(pairs)
    print(f"Alcanzables (chunk-oro en el indice): {len(reachable)} "
          f"({len(reachable) / len(pairs) * 100:.1f}%)")

    if not reachable:
        print("Ningun par alcanzable — nada que medir. Revisa el indice FAISS.")
        return

    queries = [p["pregunta"] for p in reachable]
    truth   = [p["chunk_id"] for p in reachable]

    t0 = time.time()
    per_strategy_reachable: dict[str, dict[str, float]] = {}
    per_strategy_total: dict[str, dict[str, float]] = {}
    reachability = len(reachable) / len(pairs)

    for strat in strategies:
        print(f"\n[{strat}] recuperando para {len(queries)} preguntas...")
        candidates = _run_strategy(strat, queries, pool=args.pool)
        ranked = [_dedup_parents(c) for c in candidates]
        m = _metrics_for(ranked, truth)
        per_strategy_reachable[strat] = m
        per_strategy_total[strat] = _scale_to_total(m, reachability)
        print(f"[{strat}] R@5={m['recall_at_5']:.3f}  R@10={m['recall_at_10']:.3f}  "
              f"MRR@10={m['mrr_at_10']:.3f}  nDCG@10={m['ndcg_at_10']:.3f}")

    elapsed = time.time() - t0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or "_".join(strategies)
    json_path, md_path = _write_report(
        label, ts, len(pairs), len(reachable),
        per_strategy_reachable, per_strategy_total, elapsed,
    )

    print(f"\nGuardado: {json_path}")
    print(f"Guardado: {md_path}")


if __name__ == "__main__":
    main()
