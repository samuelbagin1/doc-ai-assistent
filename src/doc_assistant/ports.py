"""Rozhrania adaptérov; opisujú vstupy a výstupy generovania, verifikácie a vyhľadávania."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from doc_assistant.domain import AssistantAnswer, DraftAnswer, RetrievedChunk, Verification


class EmbeddingPort(Protocol):
    """Prevádza text na vektory; každý adapter musí mať nemennú dimenziu."""

    @property
    def dimension(self) -> int:
        """Bez vstupu vráti rozmer embedding vektora."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Prijme dokumentové texty a vráti vektory v rovnakom poradí."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Prijme otázku a vráti jediný vyhľadávací vektor."""
        ...


class RetrieverPort(Protocol):
    """Vyhľadáva dôkazy s povinným tenant filtrom."""

    def search(self, query: str, *, k: int, tenant_id: str) -> list[RetrievedChunk]:
        """Prijme dopyt, limit a tenant; vráti zoradené relevantné chunky."""
        ...


class RAGModelPort(Protocol):
    """Generuje návrh odpovede a doplňujúci vyhľadávací dopyt."""

    def draft(self, question: str, evidence: Sequence[RetrievedChunk]) -> DraftAnswer:
        """Prijme otázku a dôkazy; vráti odpoveď s atómovými tvrdeniami."""
        ...

    def rewrite_query(
        self, question: str, evidence: Sequence[RetrievedChunk], reason: str
    ) -> str:
        """Prijme otázku, dôkazy a dôvod odmietnutia; vráti nový query reťazec."""
        ...


class VerificationPort(Protocol):
    """Overí návrh odpovede voči dokladom a vráti rozhodnutie pre workflow."""

    def verify(
        self, question: str, draft: DraftAnswer, evidence: Sequence[RetrievedChunk]
    ) -> Verification:
        """Prijme otázku, návrh a chunky; vráti dôkazový verdikt."""
        ...


class WebSearchPort(Protocol):
    """Vyhľadá externé zdroje, keď interná kolekcia nestačí."""

    def search(self, question: str) -> AssistantAnswer:
        """Prijme otázku a vráti webovú odpoveď s URL alebo abstenciu."""
        ...
