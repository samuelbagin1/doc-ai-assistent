"""Kontroluje bezpečné spracovanie tvrdení a citácií z návrhu DeepSeek."""

from doc_assistant.domain import Chunk, RetrievedChunk
from doc_assistant.providers import DeepSeekRAGModel


def evidence():
    """Bez vstupu vráti jediný testovací chunk s platným identifikátorom."""
    chunk = Chunk("chunk-1", "doc-1", "manual.pdf", "Záruka je 24 mesiacov.", 4, 4, "Záruka")
    return [RetrievedChunk(chunk, 0.95)]


def model_with_response(data):
    """Prijme falošný JSON návrh a vráti DeepSeek adaptér bez sieťového volania."""
    model = DeepSeekRAGModel(model="deepseek-flash", base_url="https://example.test", api_key="test")
    model._chat_json = lambda _system, _user: data
    return model


def test_draft_keeps_only_retrieved_chunk_ids() -> None:
    """Neznáme ID sa nikdy nedostane medzi finálne citácie návrhu."""
    model = model_with_response(
        {
            "answer": "Záruka je 24 mesiacov.",
            "claims": [
                {"text": "Záruka je 24 mesiacov.", "cited_chunk_ids": ["chunk-1", "invented"]}
            ],
        }
    )

    draft = model.draft("Aká je záruka?", evidence())

    assert draft.cited_chunk_ids == ("chunk-1",)
    assert draft.claims[0].cited_chunk_ids == ("chunk-1",)


def test_draft_with_more_than_six_claims_fails_closed() -> None:
    """Nadlimitný návrh nemá čiastočný zoznam tvrdení, takže ho Jev neakceptuje."""
    model = model_with_response(
        {
            "answer": "Dlhá odpoveď",
            "claims": [
                {"text": f"Tvrdenie {index}", "cited_chunk_ids": ["chunk-1"]}
                for index in range(7)
            ],
        }
    )

    draft = model.draft("Otázka", evidence())

    assert draft.claims == ()
    assert draft.cited_chunk_ids == ()
