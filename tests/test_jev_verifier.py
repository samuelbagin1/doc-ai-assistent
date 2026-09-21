"""Overuje prevod TypeSafe Choice/Noul odpovedí na bezpečný RAG verdikt."""

from typesafe_sdk import SystemOneResponse

from doc_assistant.domain import AnswerClaim, Chunk, DraftAnswer, RetrievedChunk
from doc_assistant.jev_verifier import JevVerifier


class FakeClient:
    """Vráti predpripravenú Jev odpoveď a uchová vstup pre asercie."""

    def __init__(self, response):
        """Prijme falošnú odpoveď a pripraví prázdne miesto na poslednú požiadavku."""
        self.response = response
        self.request = None

    def system_one(self, **kwargs):
        """Uloží parametre požiadavky a vráti falošnú odpoveď bez siete."""
        self.request = kwargs
        return self.response


def fixture_data():
    """Vráti otázku, jediné citované tvrdenie a odpovedajúci dokumentový chunk."""
    chunk = Chunk("chunk-1", "doc-1", "manual.pdf", "Záruka je 24 mesiacov.", 4, 4, "Záruka")
    draft = DraftAnswer(
        text="Záruka je 24 mesiacov.",
        cited_chunk_ids=("chunk-1",),
        claims=(AnswerClaim("Záruka je 24 mesiacov.", ("chunk-1",)),),
    )
    return "Aká je záruka?", draft, [RetrievedChunk(chunk, 0.95)]


def fake_response(*, choice="supports", confidence=0.93, complete=0.95):
    """Prijme verdikt a vráti skutočný dátový typ odpovede TypeSafe SDK."""
    return SystemOneResponse.model_validate(
        {
            "model": "jev-1.13.0",
            "answers": {
                "relation_0": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": confidence,
                    "probabilities": {"supports": 0.94 if choice == "supports" else 0.03},
                },
                "sufficient": {"type": "noul", "noul": complete},
                "relevant": {"type": "noul", "noul": 0.98},
                "claims_complete": {"type": "noul", "noul": 0.96},
            },
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
    )


def test_jev_accepts_supported_claim() -> None:
    """Citované tvrdenie s vysokou istotou sa prijme a tokeny sa započítajú."""
    verifier = JevVerifier(model="jev-1.13.0", api_key="test")
    verifier.client = FakeClient(fake_response())
    result = verifier.verify(*fixture_data())

    assert result.faithful and result.sufficient
    assert result.citation_coverage == 1.0
    assert verifier.usage() == (100, 10)
    assert "relation_0" in verifier.client.request["questions"]


def test_jev_rejects_low_confidence_support() -> None:
    """Podporný verdikt pod prahom sa odmietne namiesto vydania odpovede."""
    verifier = JevVerifier(model="jev-1.13.0", api_key="test", min_confidence=0.8)
    verifier.client = FakeClient(fake_response(confidence=0.5))
    result = verifier.verify(*fixture_data())

    assert not result.faithful
    assert result.unsupported_claims == ("Záruka je 24 mesiacov.",)


def test_jev_rejects_uncited_claim_without_api_call() -> None:
    """Chýbajúca citácia spôsobí fail-closed verdikt bez odoslania dokumentu API."""
    question, draft, evidence = fixture_data()
    draft = DraftAnswer(draft.text, (), (AnswerClaim(draft.text, ()),))
    verifier = JevVerifier(model="jev-1.13.0", api_key="test")
    verifier.client = FakeClient(fake_response())

    result = verifier.verify(question, draft, evidence)

    assert not result.faithful
    assert verifier.client.request is None


def test_jev_rejects_incomplete_sdk_result() -> None:
    """Chýbajúci typovaný verdikt sa zmení na abstenciu, nie na schválenie odpovede."""
    verifier = JevVerifier(model="jev-1.13.0", api_key="test")
    verifier.client = FakeClient(
        SystemOneResponse.model_validate(
            {"model": "jev-1.13.0", "answers": {}, "usage": {}}
        )
    )

    result = verifier.verify(*fixture_data())

    assert not result.faithful
    assert not result.sufficient
    assert "úplný" in result.reason


def test_jev_rejects_non_finite_score() -> None:
    """NaN v pravdepodobnosti nesmie obísť prah dôveryhodnosti."""
    response = fake_response(confidence=float("nan"))
    verifier = JevVerifier(model="jev-1.13.0", api_key="test")
    verifier.client = FakeClient(response)

    result = verifier.verify(*fixture_data())

    assert not result.faithful
