"""Kontroluje JSON volania GPT-5.6 Luna a bezpečné spracovanie jeho citácií."""

from types import SimpleNamespace

from doc_assistant.domain import Chunk, RetrievedChunk
from doc_assistant.providers import OpenAIAnswerModel


def evidence():
    """Bez vstupu vráti jediný testovací chunk s platným identifikátorom."""
    chunk = Chunk("chunk-1", "doc-1", "manual.pdf", "Záruka je 24 mesiacov.", 4, 4, "Záruka")
    return [RetrievedChunk(chunk, 0.95)]


def model_with_response(data):
    """Prijme falošný JSON návrh a vráti OpenAI adaptér bez sieťového volania."""
    model = OpenAIAnswerModel(model="gpt-5.6-luna", api_key="test")
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


def test_answer_model_uses_luna_compatible_request() -> None:
    """Chat požiadavka používa Lunu, nízke reasoning a JSON režim bez temperature."""
    model = OpenAIAnswerModel(model="gpt-5.6-luna", api_key="test")
    captured = {}

    def create(**kwargs):
        """Prijme parametre Chat Completions a vráti falošnú JSON odpoveď."""
        captured.update(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"query":"záruka"}'))],
        )

    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    result = model._chat_json('Vráť JSON s kľúčom "query".', "Otázka")

    assert result == {"query": "záruka"}
    assert captured["model"] == "gpt-5.6-luna"
    assert captured["reasoning_effort"] == "low"
    assert captured["response_format"] == {"type": "json_object"}
    assert "temperature" not in captured
    assert model.usage() == (12, 5)
