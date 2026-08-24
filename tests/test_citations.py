"""
test_citations.py — Nombre legible de las fuentes citadas por el chat.

Los casos de metadata cubren las seis formas reales verificadas contra
faiss_store/metadata.pkl (253.455 vectores): medmcqa con y sin topic,
medqa_us/taiwan/mainland, maternaqaes_lm con y sin source_pdf, upload.
"""

from src.rag.citations import (
    build_reference_block,
    document_locator,
    document_name,
    format_context,
    normalize_citation_brackets,
)


# ---------------------------------------------------------------------------
# document_name
# ---------------------------------------------------------------------------

def test_document_name_upload_uses_filename_stem():
    doc = {"source_dataset": "upload", "filename": "protocolo_lactancia.txt", "doc_id": "protocolo_lactancia"}
    assert document_name(doc) == "protocolo_lactancia"


def test_document_name_upload_falls_back_to_doc_id_without_filename():
    doc = {"source_dataset": "upload", "doc_id": "protocolo_lactancia"}
    assert document_name(doc) == "protocolo_lactancia"


def test_document_name_maternaqaes_lm_uses_source_pdf_stem():
    doc = {"source_dataset": "maternaqaes_lm", "source_pdf": "GPC_533_Embarazo_AETSA_compl.pdf",
           "doc_id": "gpc533"}
    assert document_name(doc) == "GPC_533_Embarazo_AETSA_compl"


def test_document_name_maternaqaes_lm_preserves_scielo_style_id():
    # "0120-5633-rcca-30-5-286" es un identificador SciELO real del corpus:
    # prettificar guiones lo destruiría.
    doc = {"source_dataset": "maternaqaes_lm", "source_pdf": "0120-5633-rcca-30-5-286.pdf"}
    assert document_name(doc) == "0120-5633-rcca-30-5-286"


def test_document_name_maternaqaes_lm_falls_back_to_doc_id_without_source_pdf():
    doc = {"source_dataset": "maternaqaes_lm", "doc_id": "0120_5633_rcca_30_5_286"}
    assert document_name(doc) == "0120_5633_rcca_30_5_286"


def test_document_name_medmcqa_with_subject_and_topic():
    doc = {"source_dataset": "medmcqa", "subject": "Anatomy", "topic": "Urinary tract"}
    assert document_name(doc) == "MedMCQA · Anatomy — Urinary tract"


def test_document_name_medmcqa_without_topic():
    doc = {"source_dataset": "medmcqa", "subject": "Anatomy", "topic": ""}
    assert document_name(doc) == "MedMCQA · Anatomy"


def test_document_name_medmcqa_without_subject_or_topic():
    doc = {"source_dataset": "medmcqa"}
    assert document_name(doc) == "MedMCQA"


def test_document_name_medqa_us():
    doc = {"source_dataset": "medqa_us", "subject": "step2&3"}
    assert document_name(doc) == "MedQA (EE. UU.) · step2&3"


def test_document_name_medqa_taiwan():
    doc = {"source_dataset": "medqa_taiwan", "subject": "taiwanese_test_Q"}
    assert document_name(doc) == "MedQA (Taiwán) · taiwanese_test_Q"


def test_document_name_medqa_mainland_without_subject():
    doc = {"source_dataset": "medqa_mainland"}
    assert document_name(doc) == "MedQA (China continental)"


def test_document_name_unknown_source_returns_raw_dataset():
    doc = {"source_dataset": "textbook"}
    assert document_name(doc) == "textbook"


def test_document_name_never_returns_desconocido():
    doc = {}
    assert document_name(doc) != "desconocido"
    assert document_name(doc) == "Fuente desconocida"


# ---------------------------------------------------------------------------
# document_locator
# ---------------------------------------------------------------------------

def test_document_locator_no_pages():
    assert document_locator({}) == ""
    assert document_locator({"pages": []}) == ""


def test_document_locator_single_page():
    assert document_locator({"pages": [12]}) == "pág. 12"


def test_document_locator_contiguous_range():
    assert document_locator({"pages": [2, 3]}) == "págs. 2-3"


def test_document_locator_non_contiguous():
    assert document_locator({"pages": [2, 5, 9]}) == "págs. 2, 5, 9"


def test_document_locator_unordered_input_is_sorted():
    assert document_locator({"pages": [9, 2, 5]}) == "págs. 2, 5, 9"


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

def test_format_context_empty():
    assert "No se encontraron" in format_context([])


def test_format_context_never_says_fragmento():
    docs = [
        {"source_dataset": "maternaqaes_lm", "source_pdf": "Manual-Obstetricia.pdf",
         "pages": [2, 3], "text": "contenido clinico"},
        {"source_dataset": "medmcqa", "subject": "Anatomy", "topic": "Urinary tract",
         "text": "pregunta de examen"},
    ]
    ctx = format_context(docs)
    assert "fragmento" not in ctx.lower()
    assert "Manual-Obstetricia" in ctx
    assert "MedMCQA · Anatomy — Urinary tract" in ctx
    assert "[1]" in ctx and "[2]" in ctx


def test_format_context_respects_max_chars():
    docs = [{"source_dataset": "upload", "filename": "doc.txt", "text": "x" * 5000}]
    ctx = format_context(docs, max_chars=200)
    assert len(ctx) < 400  # header + truncated text + "..."


# ---------------------------------------------------------------------------
# build_reference_block
# ---------------------------------------------------------------------------

def test_build_reference_block_no_citations_returns_empty():
    docs = [{"source_dataset": "medmcqa", "subject": "Anatomy"}]
    assert build_reference_block("Respuesta sin citas.", docs) == ""


def test_build_reference_block_ignores_out_of_range_numbers():
    docs = [{"source_dataset": "medmcqa", "subject": "Anatomy"}]
    assert build_reference_block("Cita inventada [7].", docs) == ""


def test_build_reference_block_single_citation():
    docs = [{"source_dataset": "medmcqa", "subject": "Anatomy", "topic": "Kidney"}]
    block = build_reference_block("Dato respaldado [1].", docs)
    assert "Fuentes:" in block
    assert "[1] MedMCQA · Anatomy — Kidney" in block


def test_build_reference_block_groups_same_document():
    docs = [
        {"source_dataset": "maternaqaes_lm", "source_pdf": "Manual-Obstetricia.pdf", "pages": [2, 3]},
        {"source_dataset": "medmcqa", "subject": "Anatomy"},
        {"source_dataset": "maternaqaes_lm", "source_pdf": "Manual-Obstetricia.pdf", "pages": [45]},
    ]
    answer = "Primero esto [1]. Después esto [3]. Y esto otro [2]."
    block = build_reference_block(answer, docs)
    lines = block.splitlines()
    manual_line = next(l for l in lines if "Manual-Obstetricia" in l)
    assert "[1]" in manual_line and "[3]" in manual_line
    assert "2-3, 45" in manual_line
    assert any("MedMCQA · Anatomy" in l for l in lines)


def test_build_reference_block_never_says_fragmento():
    docs = [{"source_dataset": "medmcqa", "subject": "Anatomy"}]
    block = build_reference_block("Dato [1].", docs)
    assert "fragmento" not in block.lower()


# ---------------------------------------------------------------------------
# normalize_citation_brackets
# ---------------------------------------------------------------------------
# gpt-oss-120b cita de forma intermitente con corchetes CJK de ancho completo
# ("【1】") en vez de ASCII ("[1]"), sin patrón previsible entre turnos —
# ver qa_technical.md Q34. Sin normalizar, _CITATION_RE nunca matchea esas
# citas y build_reference_block() las descarta en silencio.

def test_normalize_citation_brackets_converts_fullwidth_to_ascii():
    assert normalize_citation_brackets("Dato respaldado【1】.") == "Dato respaldado[1]."


def test_normalize_citation_brackets_handles_multiple_citations():
    text = "Esto【1】 y también esto【2】."
    assert normalize_citation_brackets(text) == "Esto[1] y también esto[2]."


def test_normalize_citation_brackets_leaves_ascii_untouched():
    text = "Ya viene en ASCII [1] y [2]."
    assert normalize_citation_brackets(text) == text


def test_build_reference_block_recognizes_citation_after_normalizing():
    docs = [{"source_dataset": "medmcqa", "subject": "Anatomy", "topic": "Kidney"}]
    answer = normalize_citation_brackets("Dato respaldado【1】.")
    block = build_reference_block(answer, docs)
    assert "Fuentes:" in block
    assert "[1] MedMCQA · Anatomy — Kidney" in block
