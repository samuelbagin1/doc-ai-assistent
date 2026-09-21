"""Testuje zachovanie strán a sekcií pri delení dokumentového textu na chunky."""

from doc_assistant.chunking import PageText, SectionAwareChunker


def test_chunker_preserves_page_section_and_size() -> None:
    """Prijme dlhú testovaciu stranu a overí počet, sekciu a limit chunkov."""
    text = "# Prvá sekcia\n\n" + ("Dôležitá veta. " * 120)
    chunker = SectionAwareChunker(chunk_size=300, chunk_overlap=30)

    chunks = chunker.split(
        [PageText(page=7, text=text)],
        document_id="doc-1",
        filename="pravidla.pdf",
        tenant_id="tenant-a",
    )

    assert len(chunks) > 1
    assert all(chunk.page_start == 7 for chunk in chunks)
    assert all(chunk.section == "Prvá sekcia" for chunk in chunks)
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert all(chunk.document_id == "doc-1" for chunk in chunks)


def test_chunker_inherits_heading_on_next_page() -> None:
    """Prijme dve strany a overí prenos názvu sekcie na druhú stranu."""
    chunks = SectionAwareChunker().split(
        [
            PageText(page=1, text="# Zmluvné podmienky\nPrvá strana."),
            PageText(page=2, text="Pokračovanie bez opakovaného nadpisu."),
        ],
        document_id="doc",
        filename="zmluva.pdf",
        tenant_id="local",
    )

    assert chunks[-1].section == "Zmluvné podmienky"
    assert chunks[-1].page_start == 2
