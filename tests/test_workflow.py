"""Testuje vetvenie LangGraph workflowu bez reálnych providerov a API kľúčov."""

from doc_assistant.domain import (
    AnswerClaim,
    AssistantAnswer,
    Chunk,
    DraftAnswer,
    RetrievedChunk,
    Source,
    Verification,
)
from doc_assistant.workflow import RAGWorkflow


def evidence(score: float = 0.95) -> RetrievedChunk:
    """Prijme similarity skóre a vráti testovací chunk so zdrojovými metadátami."""
    return RetrievedChunk(
        chunk=Chunk(
            id="chunk-1",
            document_id="doc-1",
            filename="manual.pdf",
            text="Záručná lehota je 24 mesiacov.",
            page_start=4,
            page_end=4,
            section="Záruka",
        ),
        score=score,
    )


class FakeRetriever:
    """Vracia pripravené chunky a počíta počet vyhľadávacích pokusov."""

    def __init__(self, results):
        """Prijme výsledky, ktoré má každý search vrátiť."""
        self.results = results
        self.calls = 0

    def search(self, query, *, k, tenant_id):
        """Prijme dopyt, limit a tenant; vráti pripravené chunky a zvýši počítadlo."""
        self.calls += 1
        return self.results


class FakeModel:
    """Vytvorí pevný návrh odpovede a doplňujúci query pre testy grafu."""

    def __init__(self, valid=True):
        """Prijme kompatibilný valid flag a vynuluje počet query rewrite."""
        self.valid = valid
        self.rewrites = 0

    def draft(self, question, items):
        """Prijme otázku a dôkazy; vráti odpoveď s jedným citovaným tvrdením."""
        return DraftAnswer(
            "Záruka je 24 mesiacov.",
            ("chunk-1",),
            (AnswerClaim("Záruka je 24 mesiacov.", ("chunk-1",)),),
        )

    def rewrite_query(self, question, items, reason):
        """Prijme dôvod retry a vráti pevný doplňujúci query reťazec."""
        self.rewrites += 1
        return "záručná lehota dĺžka"


class FakeVerifier:
    """Vráti vopred zvolený pozitívny alebo negatívny verifikačný verdikt."""

    def __init__(self, valid=True):
        """Prijme boolean určujúci úspešnosť nasledujúcej verifikácie."""
        self.valid = valid

    def verify(self, question, draft, items):
        """Prijme otázku, návrh a dôkazy; vráti pripravené skóre a dôvod."""
        return Verification(
            faithful=self.valid,
            sufficient=self.valid,
            citation_coverage=1.0 if self.valid else 0.0,
            faithfulness=1.0 if self.valid else 0.0,
            answer_relevance=1.0 if self.valid else 0.0,
            reason="Chýba údaj." if not self.valid else "",
        )

class FakeWeb:
    """Simuluje webový fallback s jedným externým citovaným zdrojom."""

    def search(self, question):
        """Prijme otázku a vráti pevnú webovú odpoveď s URL."""
        return AssistantAnswer(
            text="Webová odpoveď.",
            confidence=0.8,
            sources=(Source("Úrad", "web", "https://example.test/source"),),
            route="web",
        )


def test_workflow_returns_document_answer_with_source() -> None:
    """Overí, že úspešný Jev verdikt pustí internú odpoveď s dokumentovou citáciou."""
    workflow = RAGWorkflow(
        retriever=FakeRetriever([evidence()]),
        model=FakeModel(valid=True),
        verifier=FakeVerifier(valid=True),
        web_search=None,
        tenant_id="local",
    )

    answer = workflow.ask("Aká je záruka?")

    assert answer.route == "documents"
    assert answer.confidence >= 0.72
    assert answer.sources[0].locator == "manual.pdf, s. 4, § Záruka"


def test_workflow_retries_then_uses_web() -> None:
    """Overí dva retrieval pokusy a následný webový fallback po odmietnutí."""
    retriever = FakeRetriever([evidence()])
    model = FakeModel(valid=False)
    workflow = RAGWorkflow(
        retriever=retriever,
        model=model,
        verifier=FakeVerifier(valid=False),
        web_search=FakeWeb(),
        tenant_id="local",
        max_retrieval_attempts=2,
    )

    answer = workflow.ask("Otázka bez interného dôkazu")

    assert retriever.calls == 2
    assert model.rewrites == 1
    assert answer.route == "web"


def test_workflow_abstains_without_web() -> None:
    """Overí, že bez dôkazov a webu graf skončí abstenciou bez zdroja."""
    workflow = RAGWorkflow(
        retriever=FakeRetriever([]),
        model=FakeModel(valid=False),
        verifier=FakeVerifier(valid=False),
        web_search=None,
        tenant_id="local",
        max_retrieval_attempts=1,
    )

    answer = workflow.ask("Nezodpovedateľná otázka")

    assert answer.abstained is True
    assert answer.route == "abstain"
    assert answer.sources == ()
