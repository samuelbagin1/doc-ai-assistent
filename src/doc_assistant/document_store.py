"""Ukladá pôvodné dokumenty, číta ich strany a spravuje manifest súborov."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import tempfile
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import anydoc
from pypdf import PdfReader, PdfWriter

from doc_assistant.chunking import PageText, SectionAwareChunker
from doc_assistant.domain import Chunk, DocumentRecord


class DocumentStore:
    """Spravuje fyzické súbory a atómovo zapisovaný JSON manifest dokumentov."""

    def __init__(self, files_dir: Path, data_dir: Path) -> None:
        """Prijme adresáre dokumentov a dát; pripraví ich a cestu k manifestu."""
        self.files_dir = files_dir
        self.manifest_path = data_dir / "documents.json"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

    def list(self, *, tenant_id: str) -> list[DocumentRecord]:
        """Prijme tenant ID a vráti jeho dokumenty od najnovšieho záznamu."""
        records = [DocumentRecord(**item) for item in self._read_manifest()]
        return sorted(
            (record for record in records if record.tenant_id == tenant_id),
            key=lambda record: record.created_at,
            reverse=True,
        )

    def get(self, document_id: str, *, tenant_id: str) -> DocumentRecord | None:
        """Prijme ID dokumentu a tenantu; vráti záznam alebo None."""
        return next(
            (r for r in self.list(tenant_id=tenant_id) if r.id == document_id),
            None,
        )

    def prepare(
        self,
        source: Path,
        *,
        chunker: SectionAwareChunker,
        tenant_id: str,
        ocr_mode: str,
    ) -> tuple[DocumentRecord, list[Chunk]]:
        """Skopíruje zdroj, extrahuje strany a vráti záznam s chunkmi pred commitom."""
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Súbor neexistuje: {source}")
        digest = self._sha256(source)
        duplicate = next(
            (record for record in self.list(tenant_id=tenant_id) if record.sha256 == digest),
            None,
        )
        if duplicate:
            raise ValueError(f"Dokument už je uložený ako {duplicate.filename} ({duplicate.id[:8]}).")

        document_id = str(uuid.uuid4())
        safe_name = self._safe_filename(source.name)
        destination = self.files_dir / f"{document_id}__{safe_name}"
        shutil.copy2(source, destination)
        try:
            pages = self._extract_pages(destination, ocr_mode=ocr_mode)
            chunks = chunker.split(
                pages,
                document_id=document_id,
                filename=source.name,
                tenant_id=tenant_id,
            )
            if not chunks:
                raise ValueError("Z dokumentu sa nepodarilo získať žiadny text.")
            record = DocumentRecord(
                id=document_id,
                filename=source.name,
                stored_path=str(destination),
                sha256=digest,
                media_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream",
                chunk_count=len(chunks),
                created_at=datetime.now(UTC).isoformat(),
                tenant_id=tenant_id,
            )
            return record, chunks
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def commit(self, record: DocumentRecord) -> None:
        """Prijme pripravený záznam a uloží ho do manifestu; nič nevracia."""
        items = self._read_manifest()
        items.append(asdict(record))
        self._write_manifest(items)

    def rollback_file(self, record: DocumentRecord) -> None:
        """Prijme neúspešne importovaný záznam a odstráni jeho fyzickú kópiu."""
        record.path.unlink(missing_ok=True)

    def delete_record(self, document_id: str, *, tenant_id: str) -> DocumentRecord:
        """Odstráni záznam a súbor daného tenantu; vráti odstránený záznam."""
        records = self._read_manifest()
        target = next(
            (
                DocumentRecord(**record)
                for record in records
                if record["id"] == document_id and record.get("tenant_id", "local") == tenant_id
            ),
            None,
        )
        if target is None:
            raise KeyError(f"Dokument {document_id} neexistuje alebo k nemu nemáte prístup.")
        self._write_manifest([record for record in records if record["id"] != document_id])
        target.path.unlink(missing_ok=True)
        return target

    def _extract_pages(self, path: Path, *, ocr_mode: str) -> list[PageText]:
        """Prijme súbor a OCR režim; vráti text po jednotlivých PDF stranách."""
        if path.suffix.lower() != ".pdf":
            return [PageText(page=1, text=anydoc.to_markdown(path, ocr=ocr_mode))]

        reader = PdfReader(path)
        pages: list[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)
            with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as temporary:
                writer.write(temporary)
                temporary.seek(0)
                markdown = anydoc.to_markdown_bytes(temporary.read(), "pdf", ocr=ocr_mode)
            pages.append(PageText(page=index, text=markdown))
        return pages

    def _read_manifest(self) -> list[dict]:
        """Bez vstupu načíta JSON manifest a vráti zoznam záznamov."""
        if not self.manifest_path.exists():
            return []
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Manifest dokumentov je poškodený: {error}") from error
        if not isinstance(value, list):
            raise TypeError("Manifest dokumentov musí obsahovať JSON zoznam.")
        return value

    def _write_manifest(self, records: list[dict]) -> None:
        """Prijme zoznam záznamov a atómovo prepíše manifest; nič nevracia."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Prijme cestu k súboru a vráti jeho SHA-256 hash ako hex reťazec."""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Prijme pôvodný názov a vráti bezpečný názov pre uloženú kópiu."""
        safe = "".join(character for character in name if character.isalnum() or character in ".-_")
        return safe or "document"
