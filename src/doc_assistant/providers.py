"""Adaptéry pre embeddingy, tvorbu odpovede cez OpenAI a webové vyhľadávanie."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from doc_assistant.api_resilience import ModelCallGate
from doc_assistant.domain import (
    AnswerClaim,
    AssistantAnswer,
    DraftAnswer,
    RetrievedChunk,
    Source,
)


def _json_object(text: str) -> dict[str, Any]:
    """Prijme text modelu a vráti jeho JSON objekt alebo vyvolá chybu formátu."""
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
    """Vytvára dokumentové a otázkové vektory cez OpenAI embeddings API."""

    def __init__(
        self, model: str, api_key: str | None = None, gate: ModelCallGate | None = None
    ) -> None:
        """Prijme model, kľúč a retry gate; pripraví klienta bez interných retries."""
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), max_retries=0)
        self.gate = gate or ModelCallGate()
        self._dimension = 3072 if model == "text-embedding-3-large" else 1536

    @property
    def dimension(self) -> int:
        """Bez vstupu vráti rozmer vektorov zvoleného embedding modelu."""
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Prijme texty a vráti embedding vektory v rovnakom poradí."""
        response = self.gate.call(
            lambda: self.client.embeddings.create(model=self.model, input=list(texts)),
            provider="OpenAI embeddings",
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Prijme text otázky a vráti jej jediný embedding vektor."""
        return self.embed_documents([text])[0]


class LocalEmbeddings:
    """Prevádza text na vektory lokálnym multilingual SentenceTransformer modelom."""

    def __init__(self, model_name: str) -> None:
        """Prijme názov lokálneho modelu a načíta jeho váhy a dimenziu."""
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
        """Bez vstupu vráti dimenziu lokálneho embedding modelu."""
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Prijme dokumentové texty a vráti normalizované vektory."""
        prepared = [f"passage: {text}" for text in texts]
        return self.model.encode(prepared, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Prijme otázku a vráti normalizovaný vyhľadávací vektor."""
        return self.model.encode(
            [f"query: {text}"], normalize_embeddings=True
        )[0].tolist()


class OpenAIAnswerModel:
    """Používa GPT-5.6 Luna na návrh tvrdení a doplňujúci vyhľadávací dopyt."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        gate: ModelCallGate | None = None,
    ) -> None:
        """Prijme model, OpenAI kľúč a retry gate; pripraví klienta bez interných retries."""
        self.model = model
        self.gate = gate or ModelCallGate()
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_retries=0,
        )
        self.input_tokens = 0
        self.output_tokens = 0

    def reset_usage(self) -> None:
        """Pred novou otázkou vynuluje počítadlá tokenov; nič nevracia."""
        self.input_tokens = 0
        self.output_tokens = 0

    def usage(self) -> tuple[int, int]:
        """Bez vstupu vráti súčet vstupných a výstupných tokenov."""
        return self.input_tokens, self.output_tokens

    def draft(self, question: str, evidence: Sequence[RetrievedChunk]) -> DraftAnswer:
        """Prijme otázku a chunky; vráti odpoveď s citovanými atómovými tvrdeniami."""
        context = self._context(evidence)
        data = self._chat_json(
            "Si presný RAG asistent. Použi iba dodané dôkazy. Každé faktické tvrdenie musí "
            "byť podložené. Rozdeľ odpoveď na najviac 6 samostatne overiteľných tvrdení. "
            "Ak dôkazy nestačia, otvorene to povedz. Vráť JSON so schémou "
            '{"answer":"...","claims":[{"text":"jedno tvrdenie",'
            '"cited_chunk_ids":["chunk-id"]}]}. Každé faktické tvrdenie z answer musí byť '
            "v claims. Nevymýšľaj identifikátory.",
            f"OTÁZKA:\n{question}\n\nDÔKAZY:\n{context}",
        )
        valid_ids = {item.chunk.id for item in evidence}
        raw_claims = data.get("claims")
        claims: list[AnswerClaim] = []
        if isinstance(raw_claims, list) and len(raw_claims) <= 6:
            for item in raw_claims:
                if not isinstance(item, dict):
                    continue
                raw_ids = item.get("cited_chunk_ids")
                ids = (
                    tuple(str(value) for value in raw_ids if str(value) in valid_ids)
                    if isinstance(raw_ids, list)
                    else ()
                )
                text = str(item.get("text", "")).strip()
                if text:
                    claims.append(AnswerClaim(text=text, cited_chunk_ids=ids))
        citations = tuple(dict.fromkeys(id_ for claim in claims for id_ in claim.cited_chunk_ids))
        return DraftAnswer(
            text=str(data.get("answer", "")).strip(),
            cited_chunk_ids=citations,
            claims=tuple(claims),
        )

    def rewrite_query(
        self, question: str, evidence: Sequence[RetrievedChunk], reason: str
    ) -> str:
        """Prijme otázku, dôkazy a dôvod; vráti preformulovaný retrieval dopyt."""
        data = self._chat_json(
            "Vytvor jeden presný vyhľadávací dopyt do vektorovej databázy, ktorý doplní chýbajúci "
            'dôkaz. Vráť iba JSON {"query":"..."}.',
            f"Pôvodná otázka: {question}\nDôvod nedostatočnosti: {reason}\n"
            f"Doterajšie dôkazy:\n{self._context(evidence)}",
        )
        query = str(data.get("query", "")).strip()
        return query or question

    def _chat_json(self, system: str, user: str) -> dict[str, Any]:
        """Prijme systémový a používateľský prompt; vráti JSON z OpenAI API."""
        response = self.gate.call(
            lambda: self.client.chat.completions.create(
                model=self.model,
                reasoning_effort="low",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            ),
            provider="OpenAI answer",
        )
        if response.usage:
            self.input_tokens += int(response.usage.prompt_tokens or 0)
            self.output_tokens += int(response.usage.completion_tokens or 0)
        content = response.choices[0].message.content or "{}"
        return _json_object(content)

    @staticmethod
    def _context(evidence: Sequence[RetrievedChunk]) -> str:
        """Prijme skórované chunky a vráti ich text s ID a zdrojovými metadátami."""
        return "\n\n".join(
            f"[chunk_id={item.chunk.id}; zdroj={item.chunk.reference}; score={item.score:.3f}]\n"
            f"{item.chunk.text}"
            for item in evidence
        )

class OpenAIWebSearch:
    """Používa Responses API na webový fallback s URL citáciami."""

    def __init__(
        self, model: str, api_key: str | None = None, gate: ModelCallGate | None = None
    ) -> None:
        """Prijme model, kľúč a retry gate; pripraví OpenAI klienta."""
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), max_retries=0)
        self.gate = gate or ModelCallGate()

    def search(self, question: str) -> AssistantAnswer:
        """Prijme otázku a vráti webovú odpoveď s citáciami alebo abstenciu."""
        response = self.gate.call(
            lambda: self.client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                include=["web_search_call.action.sources"],
                instructions=(
                    "Odpovedz po slovensky. Použi aktuálne webové zdroje, uveď citácie a jasne "
                    "označ neistotu. Ak spoľahlivý zdroj nenájdeš, povedz, že odpoveď nie je "
                    "možné overiť."
                ),
                input=question,
            ),
            provider="OpenAI web search",
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
        """Prijme API odpoveď a vráti unikátne URL z anotácií a webového nástroja."""
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
