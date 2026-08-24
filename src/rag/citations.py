"""
citations.py — Nombre legible de las fuentes citadas por el chat.

Separado de retriever.py a propósito: retriever.py es un snapshot
intercambiable (ver su docstring — "PARA ACTIVAR CONFIG D: copy
retriever_configD.py retriever.py"), así que una restauración de config
borraría este formateo en silencio si viviera ahí.

Solo funciones puras sobre los dicts de metadata que ya devuelve
store.search() / retrieve(). Sin dependencias de Streamlit ni de FastAPI.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Nombre del documento
# ---------------------------------------------------------------------------

_MEDQA_LABELS = {
    "medqa_us":       "MedQA (EE. UU.)",
    "medqa_taiwan":   "MedQA (Taiwán)",
    "medqa_mainland": "MedQA (China continental)",
}


def document_name(doc: dict[str, Any]) -> str:
    """Nombre legible del documento del que salió el fragmento.

    Para ~73% del índice (medmcqa/medqa_*) no hay archivo fuente: son
    ítems de examen médico, no documentos. Ahí se degrada a
    dataset + subject/topic en vez de inventar un nombre de archivo.
    Para maternaqaes_lm y upload sí hay un PDF/txt real y se usa su
    nombre, sin prettificar guiones ni underscores: algunos doc_id son
    identificadores reales (p. ej. "0120-5633-rcca-30-5-286", un id
    SciELO) que una limpieza cosmética destruiría.
    """
    source = doc.get("source_dataset", "")

    if source == "upload":
        filename = doc.get("filename", "")
        if filename:
            return Path(filename).stem
        return doc.get("doc_id") or "Documento cargado"

    if source == "maternaqaes_lm":
        source_pdf = doc.get("source_pdf", "")
        if source_pdf:
            return Path(source_pdf).stem
        return doc.get("doc_id") or "Documento clínico obstétrico"

    if source == "medmcqa":
        subject = (doc.get("subject") or "").strip()
        topic = (doc.get("topic") or "").strip()
        if subject and topic:
            return f"MedMCQA · {subject} — {topic}"
        if subject:
            return f"MedMCQA · {subject}"
        return "MedMCQA"

    if source in _MEDQA_LABELS:
        subject = (doc.get("subject") or "").strip()
        label = _MEDQA_LABELS[source]
        return f"{label} · {subject}" if subject else label

    return source or "Fuente desconocida"


# ---------------------------------------------------------------------------
# Localizador (página/páginas) dentro del documento
# ---------------------------------------------------------------------------

def document_locator(doc: dict[str, Any]) -> str:
    """'pág. 12' / 'págs. 2-3' / 'págs. 2, 5, 9' desde metadata['pages'].

    Devuelve "" si no hay información de página (la mayoría del índice:
    medmcqa/medqa_* no tiene paginación, son ítems de examen).
    """
    pages = doc.get("pages")
    if not pages:
        return ""
    try:
        nums = sorted({int(p) for p in pages})
    except (TypeError, ValueError):
        return ""
    if not nums:
        return ""

    if len(nums) == 1:
        return f"pág. {nums[0]}"

    # Rango contiguo → "págs. 2-3"; disperso → "págs. 2, 5, 9"
    if nums == list(range(nums[0], nums[-1] + 1)):
        return f"págs. {nums[0]}-{nums[-1]}"
    return "págs. " + ", ".join(str(n) for n in nums)


# ---------------------------------------------------------------------------
# Contexto para el LLM
# ---------------------------------------------------------------------------

def format_context(docs: list[dict[str, Any]], max_chars: int = 4000) -> str:
    """Igual que retriever.format_context, pero encabeza cada pasaje con
    el nombre del documento en vez de "Fragmento [n]"."""
    if not docs:
        return "No se encontraron fuentes relevantes en la base de conocimiento."

    fragments: list[str] = []
    total_chars = 0

    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "").strip()
        name = document_name(doc)
        locator = document_locator(doc)
        header = f"--- [{i}] {name}" + (f" · {locator}" if locator else "") + " ---"
        fragment = f"{header}\n{text}"

        if total_chars + len(fragment) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                fragments.append(fragment[:remaining] + "...")
            break

        fragments.append(fragment)
        total_chars += len(fragment)

    return "\n\n".join(fragments)


# ---------------------------------------------------------------------------
# Bloque de referencias al final de la respuesta
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d+)\]")

# gpt-oss-120b (a diferencia de llama-3.3-70b) cita de forma intermitente con
# corchetes CJK de ancho completo ("【1】", U+3010/U+3011) en vez de ASCII
# ("[1]") — mismo prompt, mismo turno, sin patrón previsible. Sin esto,
# _CITATION_RE nunca matchea esas citas y build_reference_block() las
# descarta en silencio (ver qa_technical.md Q33).
_FULLWIDTH_BRACKETS_RE = re.compile(r"【(\d+)】")


def normalize_citation_brackets(text: str) -> str:
    """Convierte citas en corchetes CJK de ancho completo a ASCII estándar,
    para que _CITATION_RE las reconozca sin importar qué estilo haya
    emitido el LLM en ese turno."""
    return _FULLWIDTH_BRACKETS_RE.sub(r"[\1]", text)


def build_reference_block(answer: str, docs: list[dict[str, Any]]) -> str:
    """Arma el bloque 'Fuentes:' agrupando citas [n] del mismo documento.

    Ignora números citados por el LLM que no correspondan a ningún doc
    recuperado (fuera de rango). Devuelve "" si no hay citas válidas.
    """
    cited = sorted(
        {int(m) for m in _CITATION_RE.findall(answer) if 1 <= int(m) <= len(docs)}
    )
    if not cited:
        return ""

    # Agrupar por nombre de documento, conservando el orden de primera cita
    groups: dict[str, dict[str, Any]] = {}
    for n in cited:
        doc = docs[n - 1]
        name = document_name(doc)
        if name not in groups:
            groups[name] = {"nums": [], "locators": []}
        groups[name]["nums"].append(n)
        locator = document_locator(doc)
        if locator and locator not in groups[name]["locators"]:
            groups[name]["locators"].append(locator)

    ordered_names = sorted(groups.keys(), key=lambda name: groups[name]["nums"][0])

    lines = ["---", "Fuentes:"]
    for name in ordered_names:
        info = groups[name]
        markers = "".join(f"[{n}]" for n in info["nums"])
        line = f"{markers} {name}"
        if info["locators"]:
            # "págs. 2-3" + "págs. 45" → "págs. 2-3, 45"
            page_nums: list[str] = []
            for loc in info["locators"]:
                page_nums.append(loc.replace("págs. ", "").replace("pág. ", ""))
            plural = "págs." if len(info["locators"]) > 1 or "-" in page_nums[0] or "," in page_nums[0] else "pág."
            line += f" · {plural} {', '.join(page_nums)}"
        lines.append(line)

    return "\n".join(lines)
