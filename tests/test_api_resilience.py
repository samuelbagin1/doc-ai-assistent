"""Overuje časovanie opakovaných API volaní, trvalé kvóty a limit súbežnosti."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import httpx2
import pytest
from typesafe_sdk import TypeSafeRateLimitError

from doc_assistant.api_resilience import ModelCallGate


def rate_limit(body=None, retry_after=None):
    """Z voliteľného tela a hlavičky vytvorí testovaciu TypeSafe 429 výnimku."""
    headers = httpx2.Headers({"retry-after": retry_after} if retry_after else {})
    return TypeSafeRateLimitError(429, body or {"error": "rate limit"}, headers)


def test_retries_after_ten_then_sixty_seconds() -> None:
    """Pri dvoch dočasných chybách vráti tretí výsledok a čaká presne 10 + 60 sekúnd."""
    waits = []
    attempts = 0
    gate = ModelCallGate(sleep=waits.append)

    def operation():
        """Simuluje dve dočasné chyby a potom vráti úspešný výsledok."""
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise rate_limit()
        return "ok"

    assert gate.call(operation, provider="test") == "ok"
    assert waits == [10.0, 60.0]
    assert attempts == 3


def test_retry_after_can_extend_wait() -> None:
    """Serverová Retry-After hlavička predĺži prvý interval, ak žiada viac než 10 s."""
    waits = []
    attempts = 0
    gate = ModelCallGate(sleep=waits.append)

    def operation():
        """Simuluje prvý rate limit s dlhšou Retry-After hlavičkou."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise rate_limit(retry_after="30")
        return "ok"

    assert gate.call(operation, provider="test") == "ok"
    assert waits == [30.0]


def test_permanent_quota_is_not_retried() -> None:
    """Vyčerpaný kredit sa okamžite propaguje bez čakania a ďalších API požiadaviek."""
    waits = []
    attempts = 0
    gate = ModelCallGate(sleep=waits.append)

    def operation():
        """Simuluje trvalé vyčerpanie kreditu a zvýši počítadlo pokusov."""
        nonlocal attempts
        attempts += 1
        raise rate_limit({"error": {"code": "credit_balance_exhausted"}})

    with pytest.raises(TypeSafeRateLimitError):
        gate.call(operation, provider="test")
    assert attempts == 1
    assert waits == []


def test_semaphore_serializes_concurrent_calls() -> None:
    """Dve vlákna dostanú výsledok, no naraz beží len jedna modelová požiadavka."""
    active = 0
    maximum = 0
    lock = Lock()
    gate = ModelCallGate(max_concurrent=1)

    def operation():
        """Zmeria počet aktívnych vlákien a vráti úspech po krátkej pauze."""
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        sleep(0.02)
        with lock:
            active -= 1
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: gate.call(operation, provider="test"), range(2)))
    assert results == ["ok", "ok"]
    assert maximum == 1
