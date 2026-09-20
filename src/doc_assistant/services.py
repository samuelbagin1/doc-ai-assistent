from __future__ import annotations

from pathlib import Path

from doc_assistant.chunking import SectionAwareChunker
from doc_assistant.document_store import DocumentStore
from doc_assistant.domain import DocumentRecord
from doc_assistant.vector_store import QdrantVectorStore


class DocumentService:
    def __init__(
        self,
        *,
        documents: DocumentStore,
        vectors: QdrantVectorStore,
        chunker: SectionAwareChunker,
        tenant_id: str,
        ocr_mode: str,
    ) -> None:
        self.documents = documents
        self.vectors = vectors
        self.chunker = chunker
        self.tenant_id = tenant_id
        self.ocr_mode = ocr_mode

    def add(self, source: Path) -> DocumentRecord:
        record, chunks = self.documents.prepare(
            source,
            chunker=self.chunker,
            tenant_id=self.tenant_id,
            ocr_mode=self.ocr_mode,
        )
        try:
            self.vectors.add(chunks)
            self.documents.commit(record)
        except Exception:
            self.vectors.delete_document(record.id, tenant_id=self.tenant_id)
            self.documents.rollback_file(record)
            raise
        return record

    def delete(self, document_id: str) -> DocumentRecord:
        record = self.documents.get(document_id, tenant_id=self.tenant_id)
        if record is None:
            raise KeyError("Dokument neexistuje alebo k nemu nemáte prístup.")
        self.vectors.delete_document(document_id, tenant_id=self.tenant_id)
        return self.documents.delete_record(document_id, tenant_id=self.tenant_id)

    def list(self) -> list[DocumentRecord]:
        return self.documents.list(tenant_id=self.tenant_id)

