from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from doc_assistant.domain import (
    AssistantAnswer,
    DraftAnswer,
    RetrievedChunk,
    Source,
    Verification,
)


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Model nevrátil platný JSON: {error}") from error
    if not isinstance(value, dict):
        raise TypeError("Model musí vrátiť JSON objekt.")
    return value


class OpenAIEmbeddings:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._dimension = 3072 if model == "text-embedding-3-large" else 1536

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LocalEmbeddings:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Lokálne embeddingy vyžadujú: pip install -e '.[local-embeddings]'"
            ) from error
        self.model = SentenceTransformer(model_name)
        self._dimension = int(self.model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [f"passage: {text}" for text in texts]
        return self.model.encode(prepared, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [f"query: {text}"], normalize_embeddings=True
        )[0].tolist()


class DeepSeekRAGModel:
    def __init__(self, *, model: str, base_url: str, api_key: str | None = None) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )
        self.input_tokens = 0
        self.output_tokens = 0

    def reset_usage(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def usage(self) -> tuple[int, int]:
        return self.input_tokens, self.output_tokens

    def draft(self, question: str, evidence: Sequence[RetrievedChunk]) -> DraftAnswer:
        context = self._context(evidence)
        data = self._chat_json(
            "Si presný RAG asistent. Použi iba dodané dôkazy. Každé faktické tvrdenie musí "
            "byť podložené. Ak dôkazy nestačia, otvorene to povedz. Vráť JSON so schémou "
            '{"answer":"...","cited_chunk_ids":["..."]}. Nevymýšľaj identifikátory.',
            f"OTÁZKA:\n{question}\n\nDÔKAZY:\n{context}",
        )
        valid_ids = {item.chunk.id for item in evidence}
        citations = tuple(
            str(value) for value in data.get("cited_chunk_ids", []) if str(value) in valid_ids
        )
        return DraftAnswer(text=str(data.get("answer", "")).strip(), cited_chunk_ids=citations)

    def verify(
        self, question: str, draft: DraftAnswer, evidence: Sequence[RetrievedChunk]
    ) -> Verification:
        context = self._context(evidence)
        data = self._chat_json(
            "Si nezávislý fact-checker. Posúď iba vzťah odpovede k dôkazom, nie svoje znalosti. "
            "Vráť JSON: faithful (bool), sufficient (bool), citation_coverage (0..1), "
            "faithfulness (0..1), answer_relevance (0..1), unsupported_claims (array), reason.",
            f"OTÁZKA:\n{question}\n\nODPOVEĎ:\n{draft.text}\n\nDÔKAZY:\n{context}",
        )
        return Verification(
            faithful=bool(data.get("faithful", False)),
            sufficient=bool(data.get("sufficient", False)),
            citation_coverage=self._rate(data.get("citation_coverage")),
            faithfulness=self._rate(data.get("faithfulness")),
            answer_relevance=self._rate(data.get("answer_relevance")),
            unsupported_claims=tuple(str(item) for item in data.get("unsupported_claims", [])),
            reason=str(data.get("reason", "")),
        )

    def rewrite_query(
        self, question: str, evidence: Sequence[RetrievedChunk], reason: str
    ) -> str:
        data = self._chat_json(
            "Vytvor jeden presný vyhľadávací dopyt do vektorovej databázy, ktorý doplní chýbajúci "
            'dôkaz. Vráť iba JSON {"query":"..."}.',
            f"Pôvodná otázka: {question}\nDôvod nedostatočnosti: {reason}\n"
            f"Doterajšie dôkazy:\n{self._context(evidence)}",
        )
        query = str(data.get("query", "")).strip()
        return query or question

    def _chat_json(self, system: str, user: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if response.usage:
            self.input_tokens += int(response.usage.prompt_tokens or 0)
            self.output_tokens += int(response.usage.completion_tokens or 0)
        content = response.choices[0].message.content or "{}"
        return _json_object(content)

    @staticmethod
    def _context(evidence: Sequence[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[chunk_id={item.chunk.id}; zdroj={item.chunk.reference}; score={item.score:.3f}]\n"
            f"{item.chunk.text}"
            for item in evidence
        )

    @staticmethod
    def _rate(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0


class OpenAIWebSearch:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def search(self, question: str) -> AssistantAnswer:
        response = self.client.responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            include=["web_search_call.action.sources"],
            instructions=(
                "Odpovedz po slovensky. Použi aktuálne webové zdroje, uveď citácie a jasne označ "
                "neistotu. Ak spoľahlivý zdroj nenájdeš, povedz, že odpoveď nie je možné overiť."
            ),
            input=question,
        )
        sources = self._extract_sources(response)
        confidence = 0.78 if sources else 0.0
        usage = getattr(response, "usage", None)
        return AssistantAnswer(
            text=response.output_text.strip(),
            confidence=confidence,
            sources=tuple(sources),
            route="web" if sources else "abstain",
            abstained=not bool(sources),
            reason="Webové vyhľadávanie nevrátilo overiteľnú citáciu." if not sources else "",
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

    @staticmethod
    def _extract_sources(response: Any) -> list[Source]:
        found: dict[str, Source] = {}
        for output in getattr(response, "output", []) or []:
            if getattr(output, "type", "") == "message":
                for content in getattr(output, "content", []) or []:
                    for annotation in getattr(content, "annotations", []) or []:
                        url = getattr(annotation, "url", None)
                        if url:
                            found[url] = Source(
                                label=getattr(annotation, "title", None) or url,
                                kind="web",
                                locator=url,
                            )
            if getattr(output, "type", "") == "web_search_call":
                action = getattr(output, "action", None)
                for item in getattr(action, "sources", []) or []:
                    url = getattr(item, "url", None)
                    if url:
                        found[url] = Source(
                            label=getattr(item, "title", None) or url,
                            kind="web",
                            locator=url,
                        )
        return list(found.values())
