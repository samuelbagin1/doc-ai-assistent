"""Testuje ukladanie interakcií a výpočet metrík s používateľským feedbackom."""

from doc_assistant.domain import AssistantAnswer
from doc_assistant.observability import MetricsStore


def test_metrics_and_feedback(tmp_path) -> None:
    """Prijme temp databázu a overí počty, tokeny, latenciu a negatívny feedback."""
    store = MetricsStore(tmp_path / "metrics.sqlite3")
    answer = AssistantAnswer(
        text="Odpoveď",
        confidence=0.9,
        sources=(),
        route="documents",
        latency_ms=250.0,
        input_tokens=10,
        output_tokens=5,
    )
    interaction_id = store.record("Otázka", answer, tenant_id="t1", user_id="u1")
    store.feedback(interaction_id, -1, "Chýba detail")

    summary = store.summary(tenant_id="t1")

    assert summary.questions == 1
    assert summary.average_latency_ms == 250.0
    assert summary.input_tokens + summary.output_tokens == 15
    assert summary.negative_feedback == 1
