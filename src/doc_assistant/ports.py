from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from doc_assistant.domain import AssistantAnswer, DraftAnswer, RetrievedChunk, Verification


class EmbeddingPort(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RetrieverPort(Protocol):
    def search(self, query: str, *, k: int, tenant_id: str) -> list[RetrievedChunk]: ...


class RAGModelPort(Protocol):
    def draft(self, question: str, evidence: Sequence[RetrievedChunk]) -> DraftAnswer: ...

    def verify(
        self, question: str, draft: DraftAnswer, evidence: Sequence[RetrievedChunk]
    ) -> Verification: ...

    def rewrite_query(
        self, question: str, evidence: Sequence[RetrievedChunk], reason: str
    ) -> str: ...


class WebSearchPort(Protocol):
    def search(self, question: str) -> AssistantAnswer: ...
