import pytest

from app.retrieval import SearchFilters, format_context, invalid_labels
from tests.conftest import COOK, SOLAR, WIND, make_pdf


def index_docs(services):
    ing = services.ingestion
    ing.ingest_bytes("solar.txt", SOLAR.encode(), ["energy", "solar"])
    ing.ingest_bytes("wind.txt", WIND.encode(), ["energy", "wind"])
    ing.ingest_bytes("cooking.txt", COOK.encode(), ["food"])


def test_semantic_retrieval_finds_right_doc(services):
    index_docs(services)
    res = services.retriever.retrieve("What is the payback period for rooftop solar?", k=3)
    assert not res.insufficient_evidence
    assert res.citations[0].filename == "solar.txt"
    assert [c.label for c in res.citations] == [f"S{i}" for i in range(1, len(res.citations) + 1)]


def test_citation_snippet_is_exact_source_text(services):
    index_docs(services)
    c = services.retriever.retrieve("sourdough starter fermentation", k=1).citations[0]
    assert c.snippet in COOK and c.filename == "cooking.txt"


def test_metadata_filters(services):
    index_docs(services)
    r = services.retriever.retrieve("electricity", k=5, filters=SearchFilters(tags=["wind"]))
    assert {c.filename for c in r.citations} == {"wind.txt"}
    r = services.retriever.retrieve("electricity", k=5, filters=SearchFilters(filenames=["solar.txt"]))
    assert {c.filename for c in r.citations} == {"solar.txt"}


def test_page_numbers_from_pdf(services):
    services.ingestion.ingest_bytes("report.pdf", make_pdf(["Intro about nothing relevant here.",
                                                            "Geothermal heat pumps reduce heating bills."]))
    c = services.retriever.retrieve("geothermal heat pumps heating bills", k=1).citations[0]
    assert c.page == 2


def test_insufficient_evidence_when_irrelevant(services):
    index_docs(services)
    services.retriever.min_score = 0.4
    res = services.retriever.retrieve("quantum chromodynamics gluon confinement")
    assert res.insufficient_evidence and res.citations == [] and res.note


def test_insufficient_evidence_when_empty_index(services):
    res = services.retriever.retrieve("anything")
    assert res.insufficient_evidence and "No documents" in res.note


def test_empty_query_rejected(services):
    with pytest.raises(ValueError):
        services.retriever.retrieve("  ")


def test_ingestion_is_idempotent(services):
    a = services.ingestion.ingest_bytes("solar.txt", SOLAR.encode())
    n = len(services.store)
    b = services.ingestion.ingest_bytes("renamed.txt", SOLAR.encode())
    assert b.already_indexed and b.doc_id == a.doc_id and len(services.store) == n


def test_invalid_label_detection(services):
    index_docs(services)
    cites = services.retriever.retrieve("solar electricity", k=2).citations
    ctx = format_context(cites)
    assert "[S1]" in ctx and "solar.txt" in ctx
    assert invalid_labels("Claim [S1] and another [S99].", cites) == ["S99"]
    assert invalid_labels("No citations here.", cites) == []
