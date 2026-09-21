"""Riadi súbežnosť a časované opakovanie dočasne zlyhaných modelových API volaní."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import BoundedSemaphore
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

_RESULT = TypeVar("_RESULT")
_LOG = logging.getLogger(__name__)
_PERMANENT_LIMIT_CODES = {
    "credit_balance_exhausted",
    "insufficient_quota",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}


def _error_code(error: Exception) -> str | None:
    """Načíta kód chyby z výnimky alebo JSON tela; vstupom je API výnimka, výstupom kód."""
    code = getattr(error, "code", None)
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        details = nested if isinstance(nested, dict) else body
        code = code or details.get("code") or details.get("type")
    return str(code) if code is not None else None


def _retry_after_seconds(error: Exception) -> float:
    """Vráti čas z Retry-After hlavičky v sekundách; bez platnej hlavičky vráti nulu."""
    retry_after_ms = getattr(error, "retry_after_ms", None)
    if retry_after_ms is not None:
        return max(0.0, float(retry_after_ms) / 1000.0)
    response = getattr(error, "response", None)
    headers: Mapping[str, str] | None = getattr(error, "headers", None)
    if headers is None and response is not None:
        headers = getattr(response, "headers", None)
    if headers is None:
        return 0.0
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return 0.0
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        return max(0.0, (target - datetime.now(UTC)).total_seconds())


def _is_retryable(error: Exception) -> bool:
    """Rozlíši dočasný rate limit/preťaženie od trvalého vyčerpania kreditu."""
    if _error_code(error) in _PERMANENT_LIMIT_CODES:
        return False
    return isinstance(
        error,
        (
            RateLimitError,
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            TypeSafeRateLimitError,
            TypeSafeInternalServerError,
            TypeSafeAPIConnectionError,
        ),
    )


class ModelCallGate:
    """Obmedzí súbežné volania semaforom a vykoná najviac dva časované retry."""

    def __init__(
        self,
        max_concurrent: int = 1,
        delays: tuple[float, float] = (10.0, 60.0),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Prijme kapacitu semafora, časy čakania a testovateľnú sleep funkciu; nič nevracia."""
        if max_concurrent < 1 or any(delay < 0 for delay in delays):
            raise ValueError("Kapacita semafora musí byť kladná a retry časy nezáporné.")
        self._semaphore = BoundedSemaphore(max_concurrent)
        self._delays = delays
        self._sleep = sleep

    def call(self, operation: Callable[[], _RESULT], *, provider: str) -> _RESULT:
        """Zavolá bezargumentovú operáciu; vráti výsledok alebo poslednú API výnimku."""
        for attempt in range(len(self._delays) + 1):
            try:
                with self._semaphore:
                    return operation()
            except (
                RateLimitError,
                InternalServerError,
                APIConnectionError,
                APITimeoutError,
                TypeSafeRateLimitError,
                TypeSafeInternalServerError,
                TypeSafeAPIConnectionError,
            ) as error:
                if attempt >= len(self._delays) or not _is_retryable(error):
                    raise
                delay = max(self._delays[attempt], _retry_after_seconds(error))
                _LOG.warning(
                    "%s API je dočasne nedostupné; pokus %s/%s o %.0f s",
                    provider,
                    attempt + 2,
                    len(self._delays) + 1,
                    math.ceil(delay),
                )
                self._sleep(delay)
        raise AssertionError("Retry slučka sa musí ukončiť výsledkom alebo výnimkou.")
