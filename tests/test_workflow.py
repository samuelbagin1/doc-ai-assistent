from doc_assistant.domain import (
    AssistantAnswer,
    Chunk,
    DraftAnswer,
    RetrievedChunk,
    Source,
    Verification,
)
from doc_assistant.workflow import RAGWorkflow


def evidence(score: float = 0.95) -> RetrievedChunk:
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
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def search(self, query, *, k, tenant_id):
        self.calls += 1
        return self.results


class FakeModel:
    def __init__(self, valid=True):
        self.valid = valid
        self.rewrites = 0

    def draft(self, question, items):
        return DraftAnswer("Záruka je 24 mesiacov.", ("chunk-1",))

    def verify(self, question, draft, items):
        return Verification(
            faithful=self.valid,
            sufficient=self.valid,
            citation_coverage=1.0 if self.valid else 0.0,
            faithfulness=1.0 if self.valid else 0.0,
            answer_relevance=1.0 if self.valid else 0.0,
            reason="Chýba údaj." if not self.valid else "",
        )

    def rewrite_query(self, question, items, reason):
        self.rewrites += 1
        return "záručná lehota dĺžka"


class FakeWeb:
    def search(self, question):
        return AssistantAnswer(
            text="Webová odpoveď.",
            confidence=0.8,
            sources=(Source("Úrad", "web", "https://example.test/source"),),
            route="web",
        )


def test_workflow_returns_document_answer_with_source() -> None:
    workflow = RAGWorkflow(
        retriever=FakeRetriever([evidence()]),
        model=FakeModel(valid=True),
        web_search=None,
        tenant_id="local",
    )

    answer = workflow.ask("Aká je záruka?")

    assert answer.route == "documents"
    assert answer.confidence >= 0.72
    assert answer.sources[0].locator == "manual.pdf, s. 4, § Záruka"


def test_workflow_retries_then_uses_web() -> None:
    retriever = FakeRetriever([evidence()])
    model = FakeModel(valid=False)
    workflow = RAGWorkflow(
        retriever=retriever,
        model=model,
        web_search=FakeWeb(),
        tenant_id="local",
        max_retrieval_attempts=2,
    )

    answer = workflow.ask("Otázka bez interného dôkazu")

    assert retriever.calls == 2
    assert model.rewrites == 1
    assert answer.route == "web"


def test_workflow_abstains_without_web() -> None:
    workflow = RAGWorkflow(
        retriever=FakeRetriever([]),
        model=FakeModel(valid=False),
        web_search=None,
        tenant_id="local",
        max_retrieval_attempts=1,
    )

    answer = workflow.ask("Nezodpovedateľná otázka")

    assert answer.abstained is True
    assert answer.route == "abstain"
    assert answer.sources == ()

