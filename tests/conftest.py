import pytest

from ai_eng_lab.models import Document


@pytest.fixture
def sample_doc() -> Document:
    return Document(
        title="Intro to RAG",
        content="RAG combines retrieval with generation.",
        source="notes.md",
    )
