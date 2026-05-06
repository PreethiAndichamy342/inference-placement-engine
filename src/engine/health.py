# Copyright 2026 Preethi Andichamy
# Licensed under the Apache License, Version 2.0
"""
Background health watcher for multi-cloud inference server pools.

HealthWatcher spawns one daemon thread per registered adapter. Each thread
polls its adapter on a fixed interval, calls health_check(), and atomically
updates the corresponding CloudServer.status so the router always reads
a current view without blocking on I/O.

Thread safety
-------------
- ``CloudServer.status`` updates are GIL-atomic for simple attribute assignment
  on CPython, but we also hold a per-server ``threading.Lock`` during the
  read-modify-write cycle to be correct under all Python implementations.
- ``get_healthy_servers()`` acquires each server's lock before reading status
  so callers never observe a partially updated server.
- The adapter registry itself is protected by a single ``threading.RLock``
  so adapters can be registered/deregistered while polling is running.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from src.clouds.base import AdapterError, CloudAdapter
from src.engine.models import CloudServer, ServerStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal bookkeeping record
# ---------------------------------------------------------------------------


@dataclass
class _AdapterEntry:
    """Pairs a CloudAdapter with its managed CloudServer and sync primitives."""

    adapter: CloudAdapter
    server: CloudServer
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_checked_at: float = 0.0        # monotonic timestamp
    consecutive_failures: int = 0


# ---------------------------------------------------------------------------
# HealthWatcher
# ---------------------------------------------------------------------------


class HealthWatcher:
    """
    Continuously polls each registered (adapter, server) pair in the background
    and keeps ``CloudServer.status`` up to date.

    Usage::

        watcher = HealthWatcher(interval=30)
        watcher.register(on_prem_adapter, on_prem_server)
        watcher.register(aws_adapter, aws_server)
        watcher.start()

        # Router reads fresh status via:
        healthy = watcher.get_healthy_servers()

        # On shutdown:
        watcher.stop()

    Args:
        interval:               Poll interval in seconds (default 30).
        failure_threshold:      Consecutive failures before a server is marked
                                UNAVAILABLE rather than DEGRADED (default 3).
        probe_timeout_sentinel: If health_check() itself raises (adapter bug),
                                treat the server as UNAVAILABLE after this many
                                consecutive exceptions (default same as
                                failure_threshold).
    """

    def __init__(
        self,
        interval: float = 30.0,
        failure_threshold: int = 3,
    ) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")

        self._interval = interval
        self._failure_threshold = failure_threshold

        self._entries: list[_AdapterEntry] = []
        self._registry_lock = threading.RLock()

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._started = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, adapter: CloudAdapter, server: CloudServer) -> None:
        """
        Add an (adapter, server) pair to the watch list.

        Can be called before or after ``start()``. If called after ``start()``,
        a new polling thread is spawned immediately for the new entry.

        Args:
            adapter: The CloudAdapter implementation to poll.
            server:  The CloudServer whose ``status`` field will be updated.
        """
        entry = _AdapterEntry(adapter=adapter, server=server)
        with self._registry_lock:
            self._entries.append(entry)
            if self._started:
                self._spawn_thread(entry)
        logger.info(
            "health: registered server=%s env=%s",
            server.server_id,
            server.cloud_env.value,
        )

    def deregister(self, server_id: str) -> bool:
        """
        Remove the entry with the given server_id from the watch list.

        The polling thread for that entry will exit on its next wake-up.
        Returns True if an entry was found and removed, False otherwise.
        """
        with self._registry_lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.server.server_id != server_id]
            removed = len(self._entries) < before
        if removed:
            logger.info("health: deregistered server=%s", server_id)
        return removed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start background polling threads for all currently registered adapters.

        Calling ``start()`` more than once is a no-op.
        """
        if self._started:
            return

        self._stop_event.clear()
        self._started = True

        with self._registry_lock:
            entries = list(self._entries)

        for entry in entries:
            self._spawn_thread(entry)

        logger.info(
            "health: watcher started — interval=%.1fs threads=%d",
            self._interval,
            len(entries),
        )

    def stop(self, timeout: float = 10.0) -> None:
        """
        Signal all polling threads to exit and wait for them to finish.

        Args:
            timeout: Maximum seconds to wait for each thread to join.
        """
        if not self._started:
            return

        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("health: thread %s did not stop within %.1fs", t.name, timeout)

        self._threads.clear()
        self._started = False
        logger.info("health: watcher stopped")

    def __enter__(self) -> HealthWatcher:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Public query
    # ------------------------------------------------------------------

    def get_healthy_servers(self) -> list[CloudServer]:
        """
        Return a snapshot of all servers whose current status is HEALTHY.

        Acquires each server's lock briefly to ensure a consistent read.
        Safe to call from any thread at any time.
        """
        healthy: list[CloudServer] = []
        with self._registry_lock:
            entries = list(self._entries)

        for entry in entries:
            with entry.lock:
                if entry.server.status == ServerStatus.HEALTHY:
                    healthy.append(entry.server)
        return healthy

    def get_all_servers(self) -> list[CloudServer]:
        """Return a snapshot of every registered server regardless of status."""
        with self._registry_lock:
            return [e.server for e in self._entries]

    def get_server_status(self, server_id: str) -> ServerStatus | None:
        """
        Return the current status of a specific server, or None if not registered.
        """
        with self._registry_lock:
            entries = list(self._entries)

        for entry in entries:
            if entry.server.server_id == server_id:
                with entry.lock:
                    return entry.server.status
        return None

    # ------------------------------------------------------------------
    # Background polling internals
    # ------------------------------------------------------------------

    def _spawn_thread(self, entry: _AdapterEntry) -> None:
        t = threading.Thread(
            target=self._poll_loop,
            args=(entry,),
            name=f"health-{entry.server.server_id}",
            daemon=True,
        )
        self._threads.append(t)
        t.start()

    def _poll_loop(self, entry: _AdapterEntry) -> None:
        """
        Main loop for a single polling thread.

        Runs until ``_stop_event`` is set or the entry is deregistered.
        Polls immediately on first iteration so the server status is known
        before the first interval elapses.
        """
        server_id = entry.server.server_id
        logger.debug("health: poll loop started for server=%s", server_id)

        first_poll = True
        while not self._stop_event.is_set():
            # Check if this entry is still registered (deregister() removes it).
            with self._registry_lock:
                still_registered = any(
                    e.server.server_id == server_id for e in self._entries
                )
            if not still_registered:
                logger.debug("health: server=%s deregistered, exiting poll loop", server_id)
                break

            if not first_poll:
                # Sleep in short increments so stop_event is noticed promptly.
                self._interruptible_sleep(self._interval)
                if self._stop_event.is_set():
                    break

            first_poll = False
            self._probe(entry)

        logger.debug("health: poll loop exited for server=%s", server_id)

    def _probe(self, entry: _AdapterEntry) -> None:
        """Execute one health_check() call and update server.status and p99_latency_ms."""
        server_id = entry.server.server_id
        try:
            t0 = time.monotonic()
            status = entry.adapter.health_check()
            probe_ms = (time.monotonic() - t0) * 1000
            entry.last_checked_at = time.monotonic()

            # Feed the health-check round-trip into the adapter's latency deque
            # so p99_latency_ms reflects real network RTT to the backend.
            # Then read back the current p99 (returns 0.0 when deque is empty).
            try:
                if hasattr(entry.adapter, "_latency_deque"):
                    entry.adapter._latency_deque.append(probe_ms)
                p99 = entry.adapter.get_latency_p99()
            except Exception:  # noqa: BLE001
                p99 = None

            with entry.lock:
                previous = entry.server.status
                entry.server.status = status
                if p99 is not None:
                    entry.server.p99_latency_ms = p99

            if status == ServerStatus.HEALTHY:
                if entry.consecutive_failures > 0:
                    logger.info(
                        "health: server=%s recovered → HEALTHY "
                        "(was failing for %d consecutive poll(s))",
                        server_id,
                        entry.consecutive_failures,
                    )
                entry.consecutive_failures = 0
            else:
                entry.consecutive_failures += 1
                logger.warning(
                    "health: server=%s status=%s (consecutive_failures=%d)",
                    server_id,
                    status.value,
                    entry.consecutive_failures,
                )

            if previous != status:
                logger.info(
                    "health: server=%s status changed %s → %s",
                    server_id,
                    previous.value,
                    status.value,
                )

        except AdapterError as exc:
            # health_check() should not raise per the contract, but guard anyway.
            entry.consecutive_failures += 1
            new_status = (
                ServerStatus.UNAVAILABLE
                if entry.consecutive_failures >= self._failure_threshold
                else ServerStatus.DEGRADED
            )
            with entry.lock:
                entry.server.status = new_status
            logger.error(
                "health: server=%s adapter raised %s — setting status=%s "
                "(consecutive_failures=%d): %s",
                server_id,
                type(exc).__name__,
                new_status.value,
                entry.consecutive_failures,
                exc,
            )

        except Exception as exc:  # noqa: BLE001
            # Catch-all: a buggy adapter must not crash the watcher thread.
            entry.consecutive_failures += 1
            with entry.lock:
                entry.server.status = ServerStatus.UNAVAILABLE
            logger.exception(
                "health: server=%s unexpected error in health_check() "
                "(consecutive_failures=%d): %s",
                server_id,
                entry.consecutive_failures,
                exc,
            )

    def _interruptible_sleep(self, duration: float) -> None:
        """Sleep for *duration* seconds but wake early if stop_event is set."""
        deadline = time.monotonic() + duration
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Wake at most every 0.5s to check the stop event.
            self._stop_event.wait(timeout=min(remaining, 0.5))
