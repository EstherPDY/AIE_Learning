import pytest
from pydantic import ValidationError

from ai_eng_lab.models import Chunk, Document


def test_document_defaults(sample_doc: Document) -> None:
    assert len(sample_doc.id) == 32
    assert sample_doc.metadata == {}
    assert sample_doc.created_at.tzinfo is not None


def test_document_ids_are_unique() -> None:
    a = Document(title="a", content="x")
    b = Document(title="b", content="y")
    assert a.id != b.id


def test_metadata_is_not_shared() -> None:
    a = Document(title="a", content="x")
    b = Document(title="b", content="y")
    a.metadata["page"] = 1
    assert b.metadata == {}


def test_title_is_stripped() -> None:
    doc = Document(title="  hello  ", content="x")
    assert doc.title == "hello"


@pytest.mark.parametrize("bad_title", ["", "   ", "x" * 201])
def test_invalid_title_raises(bad_title: str) -> None:
    with pytest.raises(ValidationError):
        Document(title=bad_title, content="x")


def test_empty_content_raises() -> None:
    with pytest.raises(ValidationError):
        Document(title="t", content="")


def test_valid_chunk(sample_doc: Document) -> None:
    chunk = Chunk(document_id=sample_doc.id, index=0, text=sample_doc.content[0:3], start=0, end=3)
    assert chunk.text == "RAG"
    assert chunk.document_id == sample_doc.id


@pytest.mark.parametrize(
    ("start", "end", "text", "message"),
    [
        (5, 2, "abc", "end must be greater than start"),
        (0, 5, "abc", "text length must equal end - start"),
        (-1, 2, "abc", "greater than or equal to 0"),
    ],
)
def test_invalid_chunk_raises(start: int, end: int, text: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Chunk(document_id="d1", index=0, text=text, start=start, end=end)


def test_negative_index_raises() -> None:
    with pytest.raises(ValidationError):
        Chunk(document_id="d1", index=-1, text="abc", start=0, end=3)
