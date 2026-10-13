import pytest

from app.ingestion.chunker import chunk_pages, split_text


def test_short_text_single_chunk():
    assert split_text("Hello world.", 100, 10) == ["Hello world."]


def test_empty_text_no_chunks():
    assert split_text("   ", 100, 10) == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} ends here." for i in range(60))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    # overlap: consecutive chunks share some text
    assert any(chunks[i][-15:] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_covers_all_text():
    text = " ".join(f"Word{i}" for i in range(500))
    joined = " ".join(split_text(text, 120, 20))
    assert all(f"Word{i}" in joined for i in range(500))


def test_invalid_params():
    with pytest.raises(ValueError):
        split_text("abc", 50, 50)


def test_chunk_pages_preserves_page_and_ids():
    pages = [(1, "A" * 10), (3, "B " * 300)]
    chunks = chunk_pages(pages, doc_id="d1", filename="f.txt", chunk_size=100, overlap=10, metadata={"tags": ["x"]})
    assert chunks[0].page == 1 and chunks[-1].page == 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[0].metadata == {"tags": ["x"]}
