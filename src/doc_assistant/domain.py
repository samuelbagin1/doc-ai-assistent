from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    filename: str
    stored_path: str
    sha256: str
    media_type: str
    chunk_count: int
    created_at: str
    tenant_id: str = "local"

    @property
    def path(self) -> Path:
        return Path(self.stored_path)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    filename: str
    text: str
    page_start: int | None
    page_end: int | None
    section: str | None
    tenant_id: str = "local"

    @property
    def reference(self) -> str:
        pages = ""
        if self.page_start is not None:
            pages = f", s. {self.page_start}"
            if self.page_end and self.page_end != self.page_start:
                pages = f", s. {self.page_start}–{self.page_end}"
        section = f", § {self.section}" if self.section else ""
        return f"{self.filename}{pages}{section}"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class DraftAnswer:
    text: str
    cited_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Verification:
    faithful: bool
    sufficient: bool
    citation_coverage: float
    faithfulness: float
    answer_relevance: float
    unsupported_claims: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Source:
    label: str
    kind: Literal["document", "web"]
    locator: str
    excerpt: str = ""


@dataclass(frozen=True, slots=True)
class AssistantAnswer:
    text: str
    confidence: float
    sources: tuple[Source, ...]
    route: Literal["documents", "web", "abstain"]
    abstained: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

