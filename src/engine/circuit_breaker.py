# Copyright 2026 Preethi Andichamy
# Licensed under the Apache License, Version 2.0
"""
Circuit breaker for adapter calls.

Wraps any callable with three-state protection:

  CLOSED    Normal operation. Failures are counted; if ``failure_threshold``
            consecutive failures occur the circuit trips to OPEN.

  OPEN      Fast-fail mode. Calls are rejected immediately with
            ``CircuitOpenError`` without touching the downstream service.
            After ``half_open_timeout`` seconds the circuit moves to HALF_OPEN
            to test whether the service has recovered.

  HALF_OPEN One probe call is allowed through. If it succeeds the circuit
            resets to CLOSED. If it fails the circuit returns to OPEN and
            the cooldown timer restarts.

State transitions
-----------------

    CLOSED ──(threshold consecutive failures)──► OPEN
    OPEN   ──(half_open_timeout elapsed)───────► HALF_OPEN
    HALF_OPEN ──(probe succeeds)────────────────► CLOSED
    HALF_OPEN ──(probe fails)───────────────────► OPEN

Thread safety
-------------
A ``threading.Lock`` guards every state read/write so the breaker is safe to
use from the HealthWatcher background threads and FastAPI request handlers
concurrently.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, TypeVar

from src.clouds.base import AdapterTimeoutError, AdapterUnavailableError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Exceptions that count as a circuit-breaker failure. Request-level errors
# (4xx) are not included — a backend that actively rejects a payload is still
# *reachable* and should not trip the breaker.
_TRIPPABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    AdapterTimeoutError,
    AdapterUnavailableError,
    ConnectionError,
    TimeoutError,
    OSError,
)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class CircuitOpenError(Exception):
    """
    Raised when ``CircuitBreaker.call()`` is invoked while the circuit is OPEN.

    Callers should treat this the same as ``AdapterUnavailableError``: the
    server is considered unreachable and the request should be rerouted or
    rejected rather than retried immediately.
    """

    def __init__(self, server_id: str, retry_after: float) -> None:
        super().__init__(
            f"Circuit OPEN for server '{server_id}' — "
            f"retry after {retry_after:.1f}s"
        )
        self.server_id   = server_id
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """
    Thread-safe circuit breaker for a single downstream server.

    Args:
        server_id:          Logical name used in log messages and exceptions.
        failure_threshold:  Consecutive failures required to open the circuit
                            (default 5).
        timeout:            Per-call timeout in seconds enforced on top of any
                            timeout the wrapped function already applies.
                            Set to ``None`` to disable (default 30s).
        half_open_timeout:  Seconds to wait in OPEN state before allowing one
                            probe through (default 60s).

    Usage::

        cb = CircuitBreaker(server_id="on-prem-01")

        try:
            result = cb.call(adapter.enqueue, request)
        except CircuitOpenError:
            # fast-fail — don't wait for a real timeout
            raise AdapterUnavailableError("circuit open", server_id="on-prem-01")
        except AdapterTimeoutError:
            # real timeout on a CLOSED/HALF_OPEN call
            raise
    """

    def __init__(
        self,
        server_id: str,
        failure_threshold: int = 5,
        timeout: float | None = 30.0,
        half_open_timeout: float = 60.0,
    ) -> None:
        self.server_id         = server_id
        self.failure_threshold = failure_threshold
        self.timeout           = timeout
        self.half_open_timeout = half_open_timeout

        self._state:             CircuitState = CircuitState.CLOSED
        self._failure_count:     int          = 0
        self._last_failure_time: float        = 0.0
        self._lock                            = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition OPEN→HALF_OPEN on read)."""
        with self._lock:
            return self._current_state()

    def call(self, func: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """
        Invoke *func* with circuit-breaker protection.

        - If OPEN and the cooldown has **not** elapsed: raises ``CircuitOpenError``
          immediately without calling *func*.
        - If OPEN and the cooldown **has** elapsed: transitions to HALF_OPEN and
          lets one probe through.
        - If HALF_OPEN: one probe is allowed; success resets to CLOSED, failure
          returns to OPEN.
        - If CLOSED: normal call; failures increment the counter.

        Args:
            func:   The callable to invoke (e.g. ``adapter.enqueue``).
            *args:  Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            Whatever *func* returns.

        Raises:
            CircuitOpenError:        Circuit is OPEN and cooldown has not elapsed.
            AdapterTimeoutError:     Call timed out (circuit was CLOSED/HALF_OPEN).
            AdapterUnavailableError: Backend was unreachable (circuit was CLOSED/HALF_OPEN).
            Any exception raised by *func* that is not in ``_TRIPPABLE_EXCEPTIONS``
            is re-raised without affecting the failure counter.
        """
        with self._lock:
            state = self._current_state()

            if state == CircuitState.OPEN:
                retry_after = self._seconds_until_half_open()
                logger.debug(
                    "circuit: OPEN fast-fail server=%s retry_after=%.1fs",
                    self.server_id, retry_after,
                )
                raise CircuitOpenError(self.server_id, retry_after)

            if state == CircuitState.HALF_OPEN:
                logger.info(
                    "circuit: HALF_OPEN probe server=%s", self.server_id
                )

        # ── Execute the call (lock released) ──────────────────────────
        try:
            result = func(*args, **kwargs)
        except _TRIPPABLE_EXCEPTIONS as exc:
            self._record_failure(exc)
            raise
        except Exception:
            # Non-trippable errors (e.g. AdapterRequestError / ValueError)
            # do not count against the circuit.
            raise

        self._record_success()
        return result

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED. Useful in tests or ops tooling."""
        with self._lock:
            self._transition(CircuitState.CLOSED, reason="manual reset")
            self._failure_count = 0

    # ------------------------------------------------------------------
    # Internal helpers (all called with lock held unless noted)
    # ------------------------------------------------------------------

    def _current_state(self) -> CircuitState:
        """
        Return the effective state, auto-transitioning OPEN→HALF_OPEN when
        the cooldown has elapsed.  Must be called with ``self._lock`` held.
        """
        if (
            self._state == CircuitState.OPEN
            and self._seconds_until_half_open() <= 0
        ):
            self._transition(CircuitState.HALF_OPEN, reason="cooldown elapsed")
        return self._state

    def _seconds_until_half_open(self) -> float:
        """Remaining cooldown seconds. Negative once elapsed."""
        return (self._last_failure_time + self.half_open_timeout) - time.monotonic()

    def _record_success(self) -> None:
        with self._lock:
            if self._state in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
                if self._state == CircuitState.HALF_OPEN:
                    logger.info(
                        "circuit: probe succeeded — closing server=%s",
                        self.server_id,
                    )
                self._transition(CircuitState.CLOSED, reason="success")
                self._failure_count = 0

    def _record_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "circuit: probe failed — returning to OPEN server=%s exc=%s",
                    self.server_id, exc,
                )
                self._transition(CircuitState.OPEN, reason="half-open probe failed")

            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                logger.warning(
                    "circuit: threshold reached (%d/%d) — opening server=%s exc=%s",
                    self._failure_count, self.failure_threshold,
                    self.server_id, exc,
                )
                self._transition(CircuitState.OPEN, reason="failure threshold reached")

            else:
                logger.debug(
                    "circuit: failure %d/%d server=%s exc=%s",
                    self._failure_count, self.failure_threshold,
                    self.server_id, exc,
                )

    def _transition(self, new_state: CircuitState, *, reason: str) -> None:
        """Record a state change. Must be called with ``self._lock`` held."""
        if new_state != self._state:
            logger.info(
                "circuit: %s → %s server=%s reason=%s",
                self._state.value, new_state.value,
                self.server_id, reason,
            )
        self._state = new_state

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(server_id={self.server_id!r}, "
            f"state={self._state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )
