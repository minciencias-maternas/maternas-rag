"""
eval_pipeline.py — Pipeline de evaluacion RAG con MaternaQA-es y Ragas.

Metricas evaluadas:
  - faithfulness        (Ragas, LLM judge — KEY_1)
  - answer_correctness  (Ragas, LLM judge — KEY_1)
  - answer_relevancy    (Ragas, LLM judge — KEY_2)
  - context_recall      (Ragas, LLM judge — KEY_2)
  - context_precision   (Ragas, LLM judge — KEY_2)
  - latency_s           (medicion de tiempo manual en fase 1)

Estrategia de tokens:
  KEY_1 (GROQ_API_KEY)   → faithfulness + answer_correctness  ~35-50k tokens
  KEY_2 (GROQ_API_KEY_2) → answer_relevancy + context_recall + context_precision ~28-44k tokens
  RunConfig: max_workers=1, max_retries=2, batch_size=1 → sin rafagas concurrentes

Flujo DOS fases (evita conflictos CUDA/CPU):

  FASE 1 (--generate-only):
    Genera respuestas con el chatbot y mide latencia por par.
    → evaluation_reports/eval_raw_<config>_<ts>.json

  FASE 2 (--evaluate-only <raw.json>):
    Evalua con Ragas en 2 grupos secuenciales (KEY_1 luego KEY_2).
    → evaluation_reports/eval_results_<config>_<ts>.json
    → evaluation_reports/eval_report_<config>_<ts>.md

  FASE COMPLETA (default): ejecuta ambas en secuencia.

Ejemplos:
    python src/evaluation/eval_pipeline.py --config configB --sample 20 --generate-only
    python src/evaluation/eval_pipeline.py --evaluate-only evaluation_reports/eval_raw_configB_XXX.json
    python src/evaluation/eval_pipeline.py --config configB --sample 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("evaluation_reports")

RAGAS_METRICS_NAMES = [
    "faithfulness",
    "answer_correctness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]

BASELINE = {
    "faithfulness":       {"train": 0.7726, "test": 0.7132},
    "answer_relevancy":   {"train": 0.6466, "test": 0.5583},
    "answer_correctness": {"train": "N/A",  "test": "N/A"},
    "context_recall":     {"train": "N/A",  "test": "N/A"},
    "context_precision":  {"train": "N/A",  "test": "N/A"},
    "latency_s":          {"train": "N/A",  "test": "N/A"},
}


# ---------------------------------------------------------------------------
# FASE 1: generar respuestas + medir latencia
# ---------------------------------------------------------------------------

def phase_generate(sample_size: int | None, seed: int, config_name: str = "") -> Path:
    """
    Pasa cada pregunta por chat() midiendo latencia real.
    Guarda raw JSON con contextos completos para Ragas.
    """
    from src.rag.chain import chat
    from src.rag.retriever import retrieve
    from src.evaluation.sampler import load_sample

    sep = "=" * 62
    print(sep)
    print("  FASE 1 — Generando respuestas del chatbot Maternas")
    if config_name:
        print(f"  Config : {config_name}")
    print(sep)

    sample = load_sample(seed=seed)
    if sample_size:
        sample = sample[:sample_size]
    print(f"\n  Muestra: {len(sample)} pares  |  seed={seed}\n")

    rows: list[dict] = []
    failed = 0
    latencies: list[float] = []

    for i, item in enumerate(sample, 1):
        pregunta   = item["pregunta"]
        tipo       = item.get("tipo", "?")
        dificultad = item.get("dificultad", "?")
        print(f"  [{i:02d}/{len(sample)}] {tipo:<12} {dificultad:<12} | {pregunta[:50]}...")

        try:
            t0      = time.perf_counter()
            result  = chat(pregunta)
            latency = round(time.perf_counter() - t0, 3)
            latencies.append(latency)

            docs_full     = retrieve(pregunta, k=5)
            contexts_text = [d.get("text", "")[:600] for d in docs_full]

            rows.append({
                "qa_id":               item.get("qa_id", f"item_{i}"),
                "question":            pregunta,
                "answer":              result.answer,
                "contexts":            contexts_text,
                "ground_truth":        item["respuesta"],
                "reference":           item["respuesta"],
                "contexto_fuente":     item.get("contexto_fuente", ""),
                "tipo":                tipo,
                "dificultad":          dificultad,
                "source_pdf":          item.get("source_pdf", ""),
                "topics":              item.get("topics", []),
                "needs_clarification": result.needs_clarification,
                "intent":              result.intent,
                "risk_level":          result.risk_level,
                "latency_s":           latency,
            })
            print(f"         latencia={latency:.2f}s  riesgo={result.risk_level}")

        except Exception as e:
            logger.error(f"Error en item {i} ({pregunta[:40]}): {e}")
            failed += 1

        time.sleep(1.5)

    REPORTS_DIR.mkdir(exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"_{config_name}" if config_name else ""
    raw_path = REPORTS_DIR / f"eval_raw{label}_{ts}.json"

    latency_avg = round(sum(latencies) / len(latencies), 3) if latencies else -1
    latency_p95 = round(sorted(latencies)[int(len(latencies) * 0.95)], 3) if latencies else -1

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":   ts,
            "config":      config_name,
            "seed":        seed,
            "n_total":     len(sample),
            "n_ok":        len(rows),
            "n_failed":    failed,
            "latency_avg": latency_avg,
            "latency_p95": latency_p95,
            "rows":        rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  OK={len(rows)} | Fallidos={failed} | "
          f"Latencia avg={latency_avg}s p95={latency_p95}s")
    print(f"  Guardado: {raw_path}\n")
    return raw_path


# ---------------------------------------------------------------------------
# FASE 2: helpers de Ragas
# ---------------------------------------------------------------------------

def _make_llm(api_key: str, model: str, base_url: str | None = None):
    """
    Crea un LLM judge para Ragas con SLM (llama-3.1-8b o gemma-4-31b).

    is_finished_parser permisivo: acepta finish_reason 'length' ademas de 'stop'.
    Ragas lanza LLMDidNotFinishException si el finish_reason no es reconocido,
    lo que ocurre cuando el SLM genera statements largos en español y se trunca.
    """
    from ragas.llms import LangchainLLMWrapper
    from langchain_core.outputs import LLMResult

    def _is_finished(response: LLMResult) -> bool:
        """Acepta stop, length, MAX_TOKENS, end_turn como terminaciones válidas."""
        VALID = {"stop", "STOP", "length", "MAX_TOKENS", "end_turn", "eos"}
        for g in response.flatten():
            resp = g.generations[0][0]
            finish = None
            if resp.generation_info:
                finish = resp.generation_info.get("finish_reason") or resp.generation_info.get("stop_reason")
            if finish is None and hasattr(resp, "message"):
                meta = getattr(resp.message, "response_metadata", {})
                finish = meta.get("finish_reason") or meta.get("stop_reason")
            # Si no hay info de finish_reason, asumir que terminó bien
            if finish is not None and finish not in VALID:
                return False
        return True

    if base_url:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model, api_key=api_key, base_url=base_url,
            temperature=0, max_tokens=1024,
        )
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, api_key=api_key, temperature=0, max_tokens=1024)

    return LangchainLLMWrapper(llm, is_finished_parser=_is_finished)


def _make_emb(model_name: str):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))


def _run_ragas_group(dataset, metrics: list, llm, emb, label: str):
    """
    Corre evaluate() para un grupo de metricas con RunConfig conservador:
      max_workers=1 → un job a la vez, sin rafagas concurrentes
      max_retries=2 → maximo 2 reintentos (no 10)
      batch_size=1  → procesa un par a la vez
    Retorna DataFrame o None si falla completamente.
    """
    from ragas import evaluate
    from ragas.run_config import RunConfig

    run_cfg = RunConfig(
        max_workers=1,
        max_retries=2,
        max_wait=15,
        timeout=120,
    )
    metric_names = [m.name for m in metrics]
    print(f"\n  [{label}] Metricas: {metric_names}")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=emb,
            run_config=run_cfg,
            batch_size=1,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        ok = {m: int((df[m] >= 0).sum()) for m in metric_names if m in df.columns}
        print(f"  [{label}] Completados: {ok}")
        return df
    except Exception as e:
        logger.error(f"[{label}] Error: {e}")
        print(f"  [{label}] ERROR: {e}")
        return None


# ---------------------------------------------------------------------------
# FASE 2: evaluar con Ragas — 2 keys, metricas divididas, secuencial
# ---------------------------------------------------------------------------

def phase_evaluate(raw_path: Path) -> dict:
    """
    Evalua el raw JSON de fase 1 con 5 metricas Ragas.

    Estrategia de modelos:
      Ambos grupos usan Cerebras gemma-4-31b (~161 tok/llamada, sin cuota diaria estricta).
      Fallback: Groq llama-3.1-8b-instant si no hay CEREBRAS_KEY.

    Cerebras gemma-4-31b probado y verificado:
      - Genera JSON valido para Ragas faithfulness en español
      - ~3-4s por par, sin OutputParserException ni LLMDidNotFinishException
    """
    from ragas.metrics import (
        faithfulness,
        answer_correctness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset
    from src.settings import settings

    CEREBRAS_MODEL    = "gemma-4-31b"
    CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
    GROQ_FALLBACK     = "openai/gpt-oss-20b"  # llama-3.1-8b-instant fue dado de baja por Groq

    sep = "=" * 62
    print(sep)
    print("  FASE 2 — Evaluando con Ragas")

    if settings.cerebras_key:
        print(f"  Judge: Cerebras {CEREBRAS_MODEL} (ambos grupos)")
        llm1 = _make_llm(settings.cerebras_key, CEREBRAS_MODEL, CEREBRAS_BASE_URL)
        llm2 = llm1   # mismo modelo, misma key — sin cuota diaria estricta
        judge_label = f"Cerebras/{CEREBRAS_MODEL}"
    else:
        print(f"  Judge: Groq {GROQ_FALLBACK} (fallback, ambos grupos)")
        key1 = settings.groq_api_key
        key2 = settings.groq_api_key_2 if settings.groq_api_key_2 else key1
        llm1 = _make_llm(key1, GROQ_FALLBACK)
        llm2 = _make_llm(key2, GROQ_FALLBACK)
        judge_label = f"Groq/{GROQ_FALLBACK}"

    print(f"  Grupo 1: faithfulness + answer_correctness")
    print(f"  Grupo 2: answer_relevancy + context_recall + context_precision")
    print(sep)

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    rows        = raw["rows"]
    ts          = raw["timestamp"]
    config_name = raw.get("config", "")
    latency_avg = raw.get("latency_avg", -1)
    latency_p95 = raw.get("latency_p95", -1)

    # Filtrar pares que necesitan clarificacion (needs_clarification=True)
    # Una pregunta de clarificacion no tiene statements verificables
    # contra contexto → faithfulness siempre=0, distorsiona la metrica
    rows_before = len(rows)
    rows = [r for r in rows if not r.get("needs_clarification", False)]
    rows_excl = rows_before - len(rows)
    if rows_excl:
        print(f"\n  Excluidos por needs_clarification: {rows_excl} pares")

    print(f"\n  {len(rows)} pares  |  config={config_name or 'default'}  |  "
          f"latencia avg={latency_avg}s  p95={latency_p95}s\n")

    emb = _make_emb(settings.embedding_model)

    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
            "reference":    r["reference"],
        }
        for r in rows
    ])

    # --- Grupo 1: faithfulness + answer_correctness ---
    print(f"  Paso 1/2 — faithfulness + answer_correctness via {judge_label}...")
    df1 = _run_ragas_group(dataset, [faithfulness, answer_correctness], llm1, emb, "G1")

    print("\n  Pausa 10s entre grupos...")
    time.sleep(10)

    # --- Grupo 2: metricas de contexto y relevancia ---
    print(f"  Paso 2/2 — answer_relevancy + context_recall + context_precision via {judge_label}...")
    df2 = _run_ragas_group(dataset, [answer_relevancy, context_recall, context_precision], llm2, emb, "G2")

    # --- Merge por indice de fila ---
    metrics_by_row: list[dict] = [{} for _ in rows]
    for df in [df1, df2]:
        if df is None:
            continue
        for i in range(len(rows)):
            if i < len(df):
                for m in RAGAS_METRICS_NAMES:
                    if m in df.columns:
                        metrics_by_row[i][m] = float(df.iloc[i][m])

    # Inyectar en rows (latency_s ya viene de fase 1)
    for i, row in enumerate(rows):
        row.update(metrics_by_row[i])

    # --- Agregados ---
    def mean(vals: list) -> float:
        v = [x for x in vals if isinstance(x, float) and x >= 0]
        return round(sum(v) / len(v), 4) if v else -1

    lat_vals = [r.get("latency_s", -1) for r in rows]
    lat_vals = [x for x in lat_vals if x >= 0]

    metrics_global: dict = {m: mean([r.get(m, -1) for r in rows]) for m in RAGAS_METRICS_NAMES}
    metrics_global["latency_avg_s"] = round(sum(lat_vals) / len(lat_vals), 3) if lat_vals else -1
    metrics_global["latency_p95_s"] = round(sorted(lat_vals)[int(len(lat_vals) * 0.95)], 3) if lat_vals else -1
    metrics_global["n_evaluated"]   = len(rows)
    metrics_global["n_failed"]      = raw["n_failed"]

    # --- Por tipo ---
    by_tipo: dict[str, list] = {}
    for r in rows:
        by_tipo.setdefault(r["tipo"], []).append(r)

    metrics_by_tipo = {
        t: {
            "n": len(rs),
            **{m: mean([r.get(m, -1) for r in rs]) for m in RAGAS_METRICS_NAMES},
            "latency_avg_s": round(
                sum(r.get("latency_s", 0) for r in rs if r.get("latency_s", -1) >= 0) /
                max(1, sum(1 for r in rs if r.get("latency_s", -1) >= 0)), 3
            ),
        }
        for t, rs in by_tipo.items()
    }

    # --- Por dificultad ---
    by_dif: dict[str, list] = {}
    for r in rows:
        by_dif.setdefault(r["dificultad"], []).append(r)

    metrics_by_dificultad = {
        d: {
            "n": len(rs),
            **{m: mean([r.get(m, -1) for r in rs]) for m in RAGAS_METRICS_NAMES},
            "latency_avg_s": round(
                sum(r.get("latency_s", 0) for r in rs if r.get("latency_s", -1) >= 0) /
                max(1, sum(1 for r in rs if r.get("latency_s", -1) >= 0)), 3
            ),
        }
        for d, rs in by_dif.items()
    }

    label   = f"_{config_name}" if config_name else ""
    results = {
        "timestamp":             ts,
        "config":                config_name,
        "dataset":               "JhonHander/MaternaQA-es (split: test)",
        "n_sample":              raw["n_total"],
        "n_evaluated":           len(rows),
        "n_failed":              raw["n_failed"],
        "metrics_global":        metrics_global,
        "metrics_by_tipo":       metrics_by_tipo,
        "metrics_by_dificultad": metrics_by_dificultad,
        "rows":                  rows,
    }

    json_path = REPORTS_DIR / f"eval_results{label}_{ts}.json"
    md_path   = REPORTS_DIR / f"eval_report{label}_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    _write_markdown_report(results, md_path)

    # --- Consola ---
    print(f"\n{'─'*62}")
    print("  RESULTADOS GLOBALES")
    print(f"{'─'*62}")
    for k in RAGAS_METRICS_NAMES + ["latency_avg_s", "latency_p95_s"]:
        v   = metrics_global.get(k, -1)
        bar = _ascii_bar(v) if isinstance(v, float) and 0 <= v <= 1 else ""
        print(f"  {k:<24} {str(v):<8} {bar}")
    print(f"\n  JSON    : {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"{'─'*62}\n")

    return results


# ---------------------------------------------------------------------------
# Barra ASCII para consola
# ---------------------------------------------------------------------------

def _ascii_bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.3f}"


# ---------------------------------------------------------------------------
# Reporte Markdown limpio
# ---------------------------------------------------------------------------

def _write_markdown_report(results: dict, path: Path) -> None:
    ts       = results["timestamp"]
    config   = results.get("config") or "default"
    mg       = results["metrics_global"]
    mt       = results["metrics_by_tipo"]
    md_dif   = results["metrics_by_dificultad"]
    n_eval   = results["n_evaluated"]
    n_fail   = results["n_failed"]
    date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"

    def fmt(v, is_time: bool = False) -> str:
        if not isinstance(v, float) or v < 0:
            return "—"
        return f"{v:.2f} s" if is_time else f"{v:.4f}"

    def emoji(v) -> str:
        if not isinstance(v, float) or v < 0:
            return ""
        if v >= 0.80: return "🟢"
        if v >= 0.60: return "🟡"
        return "🔴"

    lines: list[str] = [
        "# Reporte de Evaluacion RAG — Maternas",
        "",
        f"> **Config:** `{config}`  |  **Dataset:** MaternaQA-es (split: test)  |  **Fecha:** {date_str}",
        f"> **Pares evaluados:** {n_eval}  |  **Fallidos en generacion:** {n_fail}",
        "",
        "---",
        "",
        "## Metricas Globales",
        "",
        "| Metrica | Resultado | Baseline test | Estado |",
        "|:--------|----------:|:-------------:|:------:|",
    ]

    for mkey, is_time in [
        ("faithfulness",       False),
        ("answer_correctness", False),
        ("answer_relevancy",   False),
        ("context_recall",     False),
        ("context_precision",  False),
        ("latency_avg_s",      True),
        ("latency_p95_s",      True),
    ]:
        val    = mg.get(mkey, -1)
        bl     = BASELINE.get(mkey, {}).get("test", "N/A") if not is_time else "—"
        bl_str = f"{bl:.4f}" if isinstance(bl, float) else str(bl)
        lines.append(f"| `{mkey}` | {fmt(val, is_time)} | {bl_str} | {emoji(val) if not is_time else ''} |")

    lines += [
        "",
        "> **Referencia:** baseline publicado en MaternaQA-es (JhonHander/MaternaQA-es)  ",
        "> 🟢 ≥ 0.80 · 🟡 0.60–0.80 · 🔴 < 0.60",
        "",
        "---",
        "",
        "## Por Tipo de Pregunta",
        "",
        "| Tipo | N | Faithfulness | Ans. Correct. | Ans. Relev. | Ctx. Recall | Ctx. Prec. | Latencia avg |",
        "|:-----|--:|----------:|----------:|--------:|--------:|-------:|----------:|",
    ]
    for tipo, m in sorted(mt.items(), key=lambda x: -x[1].get("faithfulness", 0)):
        lines.append(
            f"| {tipo} | {m['n']} "
            f"| {fmt(m.get('faithfulness'))} "
            f"| {fmt(m.get('answer_correctness'))} "
            f"| {fmt(m.get('answer_relevancy'))} "
            f"| {fmt(m.get('context_recall'))} "
            f"| {fmt(m.get('context_precision'))} "
            f"| {fmt(m.get('latency_avg_s'), True)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Por Dificultad",
        "",
        "| Dificultad | N | Faithfulness | Ans. Correct. | Ans. Relev. | Ctx. Recall | Ctx. Prec. | Latencia avg |",
        "|:-----------|--:|----------:|----------:|--------:|--------:|-------:|----------:|",
    ]
    for dif, m in sorted(md_dif.items()):
        lines.append(
            f"| {dif} | {m['n']} "
            f"| {fmt(m.get('faithfulness'))} "
            f"| {fmt(m.get('answer_correctness'))} "
            f"| {fmt(m.get('answer_relevancy'))} "
            f"| {fmt(m.get('context_recall'))} "
            f"| {fmt(m.get('context_precision'))} "
            f"| {fmt(m.get('latency_avg_s'), True)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Guia de Interpretacion",
        "",
        "| Metrica | Que mide | Nota |",
        "|:--------|:---------|:-----|",
        "| `faithfulness` | La respuesta esta respaldada por los fragmentos recuperados | Alto = el LLM no inventa |",
        "| `answer_correctness` | Que tan correcta es la respuesta vs el ground truth | Combina F1 semantico + factual |",
        "| `answer_relevancy` | La respuesta responde bien la pregunta formulada | Independiente de las fuentes |",
        "| `context_recall` | El retrieval capturo el contexto necesario para responder | Bajo si corpus no tiene los docs |",
        "| `context_precision` | Los fragmentos recuperados son precisos y utiles | Alto = poco ruido en contexto |",
        "| `latency_avg_s` | Tiempo promedio de respuesta end-to-end | Incluye embeddings + FAISS + LLM |",
        "| `latency_p95_s` | Percentil 95 de latencia | Representa el peor caso tipico |",
        "",
        "> **context\\_recall y context\\_precision bajos** son esperados: el corpus RAG",
        "> (textbooks EN, MedMCQA, Multiclinsum) no incluye los PDFs de MaternaQA-es",
        "> (GPC Colombia, revistas de obstetricia). Mejoraran al ingestar esos documentos.",
        "",
        "---",
        "",
        "## Comparacion con Linea Base",
        "",
        "| Sistema | Faithfulness | Answer Relevancy |",
        "|:--------|:------------:|:----------------:|",
        f"| **{config}** (este reporte) | {fmt(mg.get('faithfulness'))} | {fmt(mg.get('answer_relevancy'))} |",
        "| MaternaQA-es baseline (train) | 0.7726 | 0.6466 |",
        "| MaternaQA-es baseline (test)  | 0.7132 | 0.5583 |",
        "",
        "---",
        f"*Generado por `src/evaluation/eval_pipeline.py` — {date_str}*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Reporte MD guardado: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluacion RAG Maternas — 5 metricas Ragas + latencia",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--sample", type=int, default=20,
        help="Pares a evaluar (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config", type=str, default="",
        help="Etiqueta de config (ej: configA, configB)",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Solo fase 1: generar respuestas y medir latencia",
    )
    parser.add_argument(
        "--evaluate-only", type=str, default=None, metavar="RAW_JSON",
        help="Solo fase 2: evaluar un raw JSON ya generado",
    )
    args = parser.parse_args()

    if args.evaluate_only:
        phase_evaluate(Path(args.evaluate_only))
    elif args.generate_only:
        phase_generate(args.sample, args.seed, config_name=args.config)
    else:
        raw_path = phase_generate(args.sample, args.seed, config_name=args.config)
        phase_evaluate(raw_path)
