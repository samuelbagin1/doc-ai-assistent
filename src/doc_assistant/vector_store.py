"""Ukladá a vyhľadáva dokumentové chunky v Qdrante s tenant izoláciou."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from doc_assistant.domain import Chunk, RetrievedChunk
from doc_assistant.ports import EmbeddingPort


class QdrantVectorStore:
    """Adaptér Qdrantu pre zápis, similarity search a mazanie podľa dokumentu."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embeddings: EmbeddingPort,
    ) -> None:
        """Prijme Qdrant klienta, názov kolekcie a embedding adapter; nič nevracia."""
        self.client = client
        self.collection = collection
        self.embeddings = embeddings

    def ensure_collection(self) -> None:
        """Vytvorí chýbajúcu kolekciu podľa dimenzie embeddingov; nič nevracia."""
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.embeddings.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Prijme chunky, vytvorí embeddingy a uloží vektory s metadátami."""
        if not chunks:
            return
        self.ensure_collection()
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "text": chunk.text,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": chunk.section,
                    "tenant_id": chunk.tenant_id,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query: str, *, k: int, tenant_id: str) -> list[RetrievedChunk]:
        """Prijme dopyt, limit a tenant; vráti skórované chunky len z daného tenantu."""
        if not self.client.collection_exists(self.collection):
            return []
        result = self.client.query_points(
            collection_name=self.collection,
            query=self.embeddings.embed_query(query),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id),
                    )
                ]
            ),
            limit=k,
            with_payload=True,
        )
        return [self._to_retrieved(point) for point in result.points]

    def delete_document(self, document_id: str, *, tenant_id: str) -> None:
        """Odstráni body zvoleného dokumentu a tenantu; nič nevracia."""
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        ),
                        models.FieldCondition(
                            key="tenant_id", match=models.MatchValue(value=tenant_id)
                        ),
                    ]
                )
            ),
            wait=True,
        )

    @staticmethod
    def _to_retrieved(point: models.ScoredPoint) -> RetrievedChunk:
        """Prijme bod Qdrantu a vráti doménový chunk so similarity skóre."""
        payload = point.payload or {}
        return RetrievedChunk(
            chunk=Chunk(
                id=str(point.id),
                document_id=str(payload["document_id"]),
                filename=str(payload["filename"]),
                text=str(payload["text"]),
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                section=payload.get("section"),
                tenant_id=str(payload.get("tenant_id", "local")),
            ),
            score=float(point.score),
        )
