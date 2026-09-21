"""Integračne testuje import, tenant filter a úplné mazanie dokumentu v Qdrante."""

from qdrant_client import QdrantClient

from doc_assistant.chunking import SectionAwareChunker
from doc_assistant.document_store import DocumentStore
from doc_assistant.services import DocumentService
from doc_assistant.vector_store import QdrantVectorStore


class FakeEmbeddings:
    """Deterministicky prevádza text na trojrozmerné testovacie vektory."""

    dimension = 3

    def embed_documents(self, texts):
        """Prijme texty a vráti jeden testovací vektor na každý text."""
        return [[1.0, min(len(text) / 100.0, 1.0), 0.0] for text in texts]

    def embed_query(self, text):
        """Prijme otázku a vráti pevný vektor pre similarity search."""
        return [1.0, 0.0, 0.0]


def test_ingest_search_and_delete_stay_consistent(tmp_path) -> None:
    """Prijme temp adresár a overí konzistentný ingest, vyhľadanie a zmazanie."""
    source = tmp_path / "source.csv"
    source.write_text("topic,value\nwarranty,24 months\n", encoding="utf-8")
    document_store = DocumentStore(tmp_path / "files", tmp_path / "data")
    vectors = QdrantVectorStore(
        QdrantClient(path=str(tmp_path / "qdrant")),
        "test_chunks",
        FakeEmbeddings(),
    )
    service = DocumentService(
        documents=document_store,
        vectors=vectors,
        chunker=SectionAwareChunker(),
        tenant_id="tenant-a",
        ocr_mode="reject",
    )

    record = service.add(source)

    assert record.chunk_count >= 1
    assert len(service.list()) == 1
    assert vectors.search("warranty", k=5, tenant_id="tenant-a")
    assert vectors.search("warranty", k=5, tenant_id="tenant-b") == []

    service.delete(record.id)

    assert service.list() == []
    assert vectors.search("warranty", k=5, tenant_id="tenant-a") == []
    assert not record.path.exists()
