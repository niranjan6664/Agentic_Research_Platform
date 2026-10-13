import pytest

from app.ingestion.loaders import IngestionError, load_pages, sanitize_filename, validate_upload
from tests.conftest import make_pdf

MB = 1024 * 1024


def test_sanitize_strips_paths():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("C:\\x\\y\\report v1.pdf") == "report v1.pdf"


@pytest.mark.parametrize("name,data,msg", [
    ("a.exe", b"abc", "Unsupported"),
    ("a.txt", b"", "empty"),
    ("a.txt", b"x" * (2 * MB), "limit"),
    ("a.pdf", b"not a pdf", "not a valid PDF"),
    ("a.txt", b"abc\x00def", "binary"),
])
def test_validation_rejects(name, data, msg):
    with pytest.raises(IngestionError, match=msg):
        validate_upload(name, data, 1 * MB)


def test_load_text():
    assert load_pages("a.txt", b"Hello\r\n\r\n\r\n\r\nWorld") == [(1, "Hello\n\nWorld")]


def test_load_pdf_pages():
    pages = load_pages("a.pdf", make_pdf(["First page text.", "Second page text."]))
    assert [p for p, _ in pages] == [1, 2]
    assert "Second page" in pages[1][1]


def test_corrupt_pdf_gives_clean_error():
    with pytest.raises(IngestionError):
        load_pages("a.pdf", b"%PDF-1.4 this is garbage")


def test_blank_text_rejected():
    with pytest.raises(IngestionError, match="No extractable text"):
        load_pages("a.txt", b"   \n  ")
