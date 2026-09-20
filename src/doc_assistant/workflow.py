from __future__ import annotations

import time
from dataclasses import replace
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from doc_assistant.domain import (
    AssistantAnswer,
    DraftAnswer,
    RetrievedChunk,
    Source,
    Verification,
)
from doc_assistant.ports import RAGModelPort, RetrieverPort, WebSearchPort


class WorkflowState(TypedDict, total=False):
    question: str
    query: str
    evidence: list[RetrievedChunk]
    draft: DraftAnswer
    verification: Verification
    attempts: int
    answer: AssistantAnswer
    started_at: float


class RAGWorkflow:
    def __init__(
        self,
        *,
        retriever: RetrieverPort,
        model: RAGModelPort,
        web_search: WebSearchPort | None,
        tenant_id: str,
        top_k: int = 5,
        max_retrieval_attempts: int = 2,
        min_retrieval_score: float = 0.25,
        min_answer_confidence: float = 0.72,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.web_search = web_search
        self.tenant_id = tenant_id
        self.top_k = top_k
        self.max_retrieval_attempts = max_retrieval_attempts
        self.min_retrieval_score = min_retrieval_score
        self.min_answer_confidence = min_answer_confidence
        self.graph = self._build_graph()

    def ask(self, question: str) -> AssistantAnswer:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Otázka nesmie byť prázdna.")
        reset_usage = getattr(self.model, "reset_usage", None)
        if callable(reset_usage):
            reset_usage()
        final = self.graph.invoke(
            {
                "question": cleaned,
                "query": cleaned,
                "evidence": [],
                "attempts": 0,
                "started_at": time.perf_counter(),
            }
        )
        answer = final["answer"]
        usage = getattr(self.model, "usage", None)
        if callable(usage):
            input_tokens, output_tokens = usage()
            answer = replace(
                answer,
                input_tokens=answer.input_tokens + input_tokens,
                output_tokens=answer.output_tokens + output_tokens,
            )
        return answer

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("draft", self._draft)
        graph.add_node("verify", self._verify)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("finalize_documents", self._finalize_documents)
        graph.add_node("web_search", self._web_search)
        graph.add_node("abstain", self._abstain)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "draft")
        graph.add_edge("draft", "verify")
        graph.add_conditional_edges(
            "verify",
            self._route_after_verification,
            {
                "answer": "finalize_documents",
                "retry": "rewrite",
                "web": "web_search",
                "abstain": "abstain",
            },
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("finalize_documents", END)
        graph.add_edge("web_search", END)
        graph.add_edge("abstain", END)
        return graph.compile()

    def _retrieve(self, state: WorkflowState) -> WorkflowState:
        new_items = self.retriever.search(
            state["query"], k=self.top_k, tenant_id=self.tenant_id
        )
        merged = {item.chunk.id: item for item in state.get("evidence", [])}
        for item in new_items:
            if item.score >= self.min_retrieval_score:
                existing = merged.get(item.chunk.id)
                if existing is None or item.score > existing.score:
                    merged[item.chunk.id] = item
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return {"evidence": ranked[: self.top_k * 2], "attempts": state["attempts"] + 1}

    def _draft(self, state: WorkflowState) -> WorkflowState:
        if not state["evidence"]:
            return {"draft": DraftAnswer(text="", cited_chunk_ids=())}
        return {"draft": self.model.draft(state["question"], state["evidence"])}

    def _verify(self, state: WorkflowState) -> WorkflowState:
        if not state["evidence"] or not state["draft"].text:
            return {
                "verification": Verification(
                    faithful=False,
                    sufficient=False,
                    citation_coverage=0.0,
                    faithfulness=0.0,
                    answer_relevance=0.0,
                    reason="Nenašli sa použiteľné interné dôkazy.",
                )
            }
        return {
            "verification": self.model.verify(
                state["question"], state["draft"], state["evidence"]
            )
        }

    def _rewrite(self, state: WorkflowState) -> WorkflowState:
        return {
            "query": self.model.rewrite_query(
                state["question"],
                state["evidence"],
                state["verification"].reason,
            )
        }

    def _route_after_verification(
        self, state: WorkflowState
    ) -> Literal["answer", "retry", "web", "abstain"]:
        confidence = self._confidence(state)
        verification = state["verification"]
        if (
            verification.faithful
            and verification.sufficient
            and not verification.unsupported_claims
            and confidence >= self.min_answer_confidence
            and state["draft"].cited_chunk_ids
        ):
            return "answer"
        if state["attempts"] < self.max_retrieval_attempts:
            return "retry"
        if self.web_search is not None:
            return "web"
        return "abstain"

    def _finalize_documents(self, state: WorkflowState) -> WorkflowState:
        evidence = {item.chunk.id: item for item in state["evidence"]}
        sources = tuple(
            Source(
                label=evidence[chunk_id].chunk.filename,
                kind="document",
                locator=evidence[chunk_id].chunk.reference,
                excerpt=evidence[chunk_id].chunk.text[:280].strip(),
            )
            for chunk_id in state["draft"].cited_chunk_ids
            if chunk_id in evidence
        )
        return {
            "answer": AssistantAnswer(
                text=state["draft"].text,
                confidence=self._confidence(state),
                sources=sources,
                route="documents",
                latency_ms=self._elapsed_ms(state),
            )
        }

    def _web_search(self, state: WorkflowState) -> WorkflowState:
        if self.web_search is None:
            return self._abstain(state)
        result = self.web_search.search(state["question"])
        if result.abstained or result.confidence < self.min_answer_confidence or not result.sources:
            return self._abstain(
                state,
                reason=result.reason or "Webový výsledok nemal dostatočnú istotu alebo citácie.",
            )
        return {
            "answer": AssistantAnswer(
                text=result.text,
                confidence=result.confidence,
                sources=result.sources,
                route="web",
                latency_ms=self._elapsed_ms(state),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        }

    def _abstain(
        self, state: WorkflowState, reason: str | None = None
    ) -> WorkflowState:
        verification = state.get("verification")
        detail = reason or (verification.reason if verification else "")
        return {
            "answer": AssistantAnswer(
                text=(
                    "Nemám dostatok spoľahlivých a citovateľných podkladov na bezpečnú odpoveď."
                ),
                confidence=self._confidence(state) if verification else 0.0,
                sources=(),
                route="abstain",
                abstained=True,
                reason=detail,
                latency_ms=self._elapsed_ms(state),
            )
        }

    def _confidence(self, state: WorkflowState) -> float:
        verification = state.get("verification")
        if verification is None:
            return 0.0
        scores = [max(0.0, min(1.0, item.score)) for item in state.get("evidence", [])[:3]]
        retrieval = sum(scores) / len(scores) if scores else 0.0
        value = (
            0.25 * retrieval
            + 0.35 * verification.faithfulness
            + 0.20 * verification.answer_relevance
            + 0.20 * verification.citation_coverage
        )
        return round(max(0.0, min(1.0, value)), 3)

    @staticmethod
    def _elapsed_ms(state: WorkflowState) -> float:
        return round((time.perf_counter() - state["started_at"]) * 1000, 1)
