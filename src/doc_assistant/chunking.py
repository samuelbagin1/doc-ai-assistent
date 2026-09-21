"""Z textových strán vytvára sekčné chunky s metadátami pre Qdrant a citácie."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from doc_assistant.domain import Chunk


@dataclass(frozen=True, slots=True)
class PageText:
    """Vstupná strana dokumentu; obsahuje číslo strany a extrahovaný text."""

    page: int | None
    text: str


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class SectionAwareChunker:
    """Najprv rešpektuje Markdown sekcie, potom rekurzívne delí dlhý text."""

    def __init__(self, chunk_size: int = 1400, chunk_overlap: int = 180) -> None:
        """Prijme maximálnu dĺžku a prekryv chunkov; pripraví splitter bez výstupu."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(
        self,
        pages: Iterable[PageText],
        *,
        document_id: str,
        filename: str,
        tenant_id: str,
    ) -> list[Chunk]:
        """Prijme strany a identitu dokumentu; vráti chunky s názvom sekcie a stranou."""
        chunks: list[Chunk] = []
        current_section: str | None = None
        for page in pages:
            sections = self._sections(page.text, current_section)
            for section, body in sections:
                current_section = section or current_section
                for text in self._recursive_split(body):
                    cleaned = text.strip()
                    if not cleaned:
                        continue
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            document_id=document_id,
                            filename=filename,
                            text=cleaned,
                            page_start=page.page,
                            page_end=page.page,
                            section=section,
                            tenant_id=tenant_id,
                        )
                    )
        return chunks

    def _sections(self, text: str, inherited: str | None) -> list[tuple[str | None, str]]:
        """Rozdelí text pri Markdown nadpisoch; vráti dvojice názvu sekcie a jej tela."""
        result: list[tuple[str | None, str]] = []
        title = inherited
        buffer: list[str] = []
        for line in text.splitlines():
            match = _HEADING.match(line)
            if match:
                if buffer and any(item.strip() for item in buffer):
                    result.append((title, "\n".join(buffer)))
                title = match.group(2).strip()
                buffer = [line]
            else:
                buffer.append(line)
        if buffer and any(item.strip() for item in buffer):
            result.append((title, "\n".join(buffer)))
        return result or [(inherited, text)]

    def _recursive_split(self, text: str) -> list[str]:
        """Prijme dlhý text; vráti podreťazce rešpektujúce veľkosť chunku."""
        if len(text) <= self.chunk_size:
            return [text]
        return self._split_with_separators(text, ("\n\n", "\n", ". ", " "))

    def _split_with_separators(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Prijme text a poradie oddeľovačov; vráti rekurzívne rozdelené kúsky."""
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            step = self.chunk_size - self.chunk_overlap
            return [text[index : index + self.chunk_size] for index in range(0, len(text), step)]

        separator, *rest = separators
        pieces = text.split(separator)
        if len(pieces) == 1:
            return self._split_with_separators(text, tuple(rest))

        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = piece if not current else current + separator + piece
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.extend(self._split_with_separators(current, tuple(rest)))
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = overlap + separator + piece
            else:
                chunks.extend(self._split_with_separators(piece, tuple(rest)))
                current = ""
        if current:
            chunks.extend(self._split_with_separators(current, tuple(rest)))
        return chunks
