"""Zaznamenáva metriky otázok, tokenov, latencie a používateľský feedback do SQLite."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from doc_assistant.domain import AssistantAnswer


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    """Agregované počty otázok, abstencií, tokenov a negatívnych hodnotení."""

    questions: int
    abstentions: int
    average_latency_ms: float
    input_tokens: int
    output_tokens: int
    negative_feedback: int


class MetricsStore:
    """SQLite úložisko interakcií a súhrnných prevádzkových metrík."""

    def __init__(self, path: Path) -> None:
        """Prijme cestu k SQLite databáze, vytvorí schému a nič nevracia."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                route TEXT NOT NULL,
                confidence REAL NOT NULL,
                abstained INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                feedback INTEGER,
                feedback_note TEXT
            )
            """
        )
        self.connection.commit()

    def record(
        self, question: str, answer: AssistantAnswer, *, tenant_id: str, user_id: str
    ) -> str:
        """Uloží otázku, odpoveď a identitu; vráti ID interakcie pre feedback."""
        interaction_id = str(uuid.uuid4())
        self.connection.execute(
            """
            INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                interaction_id,
                datetime.now(UTC).isoformat(),
                tenant_id,
                user_id,
                question,
                answer.text,
                answer.route,
                answer.confidence,
                int(answer.abstained),
                answer.latency_ms,
                answer.input_tokens,
                answer.output_tokens,
            ),
        )
        self.connection.commit()
        return interaction_id

    def feedback(self, interaction_id: str, value: int, note: str = "") -> None:
        """Prijme ID, hodnotenie ±1 a poznámku; aktualizuje interakciu."""
        if value not in {-1, 1}:
            raise ValueError("Feedback musí byť -1 alebo 1.")
        cursor = self.connection.execute(
            "UPDATE interactions SET feedback = ?, feedback_note = ? WHERE id = ?",
            (value, note, interaction_id),
        )
        self.connection.commit()
        if cursor.rowcount == 0:
            raise KeyError("Interakcia neexistuje.")

    def summary(self, *, tenant_id: str) -> MetricsSummary:
        """Prijme tenant ID a vráti jeho agregované metriky."""
        row = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(abstained), 0), COALESCE(AVG(latency_ms), 0),
                   COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),
                   COALESCE(SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END), 0)
            FROM interactions WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchone()
        assert row is not None
        return MetricsSummary(
            questions=int(row[0]),
            abstentions=int(row[1]),
            average_latency_ms=round(float(row[2]), 1),
            input_tokens=int(row[3]),
            output_tokens=int(row[4]),
            negative_feedback=int(row[5]),
        )
