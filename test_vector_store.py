import numpy as np
import pytest

from app.embeddings.hashing import HashingEmbedder
from app.ingestion.models import Chunk
from app.retrieval import VectorStore


def mk(doc, i, text, page=1, tags=None):
    return Chunk(f"{doc}-{i}", doc, f"{doc}.txt", page, i, text, {"tags": tags or []})


@pytest.fixture
def emb():
    return HashingEmbedder(128)


def populated(emb, path=None):
    store = VectorStore(emb.dim, emb.name, path)
    chunks = [mk("a", 0, "solar panel photovoltaic electricity"), mk("a", 1, "wind turbine blades", 2, ["energy"]),
              mk("b", 0, "bread flour water yeast")]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    return store


def test_search_ranks_relevant_first(emb):
    store = populated(emb)
    res = store.search(emb.embed(["photovoltaic solar electricity"])[0], 3)
    assert res[0][0].chunk_id == "a-0"
    assert res[0][1] > res[-1][1]


def test_predicate_filters_before_topk(emb):
    store = populated(emb)
    res = store.search(emb.embed(["solar"])[0], 1, lambda c: c.doc_id == "b")
    assert [c.doc_id for c, _ in res] == ["b"]


def test_delete_document(emb):
    store = populated(emb)
    assert store.delete_document("a") == 2
    assert len(store) == 1 and not store.has_document("a")
    assert store.search(emb.embed(["bread"])[0], 5)[0][0].doc_id == "b"
    assert store.delete_document("a") == 0


def test_persistence_roundtrip(emb, tmp_path):
    populated(emb, tmp_path)
    loaded = VectorStore.load_or_create(tmp_path, emb.dim, emb.name)
    assert len(loaded) == 3
    assert loaded.search(emb.embed(["wind turbine"])[0], 1)[0][0].chunk_id == "a-1"


def test_embedder_mismatch_is_refused(emb, tmp_path):
    populated(emb, tmp_path)
    with pytest.raises(RuntimeError, match="Re-index"):
        VectorStore.load_or_create(tmp_path, emb.dim, "some-other-model")


def test_bad_dimensions_rejected(emb):
    store = VectorStore(emb.dim, emb.name)
    with pytest.raises(ValueError):
        store.add([mk("x", 0, "t")], np.zeros((1, 7), dtype=np.float32))


def test_empty_store_search(emb):
    assert VectorStore(emb.dim, emb.name).search(emb.embed(["x"])[0], 3) == []
