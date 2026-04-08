"""
Abstract base class for cloud / on-prem adapter implementations.

Each concrete adapter wraps one hosting environment (AWS, GCP, on-prem vLLM,
etc.) and exposes a uniform interface so the placement router can interact
with any backend without knowing its internals.

Adapter contract
----------------
- enqueue        Submit a request and return an opaque job/request ID.
- get_queue_depth  Current number of requests waiting to be processed.
- get_latency_p99  99th-percentile end-to-end latency in milliseconds,
                   sourced from the adapter's own metrics endpoint.
- get_active_connections  Number of in-flight inference connections right now.
- health_check   Perform a live probe and return the node's ServerStatus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.engine.models import InferenceRequest, ServerStatus


class CloudAdapter(ABC):
    """
    Abstract interface that every cloud/on-prem backend must implement.

    Concrete subclasses are responsible for all transport details (HTTP,
    gRPC, SDK calls, etc.) and must translate backend-specific errors into
    the standard exceptions documented below.

    Exceptions
    ----------
    Implementations should raise:
      - ``AdapterUnavailableError``  when the backend cannot be reached.
      - ``AdapterRequestError``      when the backend rejects the request.
      - ``AdapterTimeoutError``      when a call exceeds its configured timeout.

    These are defined in this module so callers can catch them without
    importing concrete adapter packages.
    """

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every concrete adapter
    # ------------------------------------------------------------------

    @abstractmethod
    def enqueue(self, request: InferenceRequest) -> str:
        """
        Submit *request* to the backend inference server.

        Args:
            request: The InferenceRequest to be executed.

        Returns:
            An opaque string identifier (job ID, request ID, etc.) that
            callers can use to poll for results or correlate log entries.

        Raises:
            AdapterUnavailableError: Backend is unreachable.
            AdapterRequestError:     Backend rejected the payload (4xx).
            AdapterTimeoutError:     Call exceeded the adapter's timeout.
        """

    @abstractmethod
    def get_queue_depth(self) -> int:
        """
        Return the number of requests currently waiting in the server queue.

        Returns:
            Non-negative integer queue depth. Returns 0 when the backend
            does not expose queue metrics rather than raising.

        Raises:
            AdapterUnavailableError: Backend is unreachable.
        """

    @abstractmethod
    def get_latency_p99(self) -> float:
        """
        Return the server's current 99th-percentile inference latency in ms.

        The value should be sourced from the backend's own metrics endpoint
        (e.g. Prometheus /metrics, vLLM /metrics, cloud-native monitoring)
        so it reflects real observed latency rather than a synthetic probe.

        Returns:
            Latency in milliseconds as a float. Returns 0.0 when the backend
            does not expose a p99 metric rather than raising.

        Raises:
            AdapterUnavailableError: Backend is unreachable.
        """

    @abstractmethod
    def get_active_connections(self) -> int:
        """
        Return the number of inference requests currently being processed.

        Returns:
            Non-negative integer connection count.

        Raises:
            AdapterUnavailableError: Backend is unreachable.
        """

    @abstractmethod
    def health_check(self) -> ServerStatus:
        """
        Perform a live probe against the backend and return its status.

        Implementations must not raise — all probe failures should be
        translated into ``ServerStatus.UNAVAILABLE`` or ``ServerStatus.DEGRADED``
        and returned so callers always get a usable status value.

        Returns:
            ServerStatus reflecting the current node health.
        """

    # ------------------------------------------------------------------
    # Optional hook — subclasses may override
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Release any persistent resources held by this adapter (connection
        pools, SDK clients, etc.).

        The default implementation is a no-op. Subclasses that maintain
        persistent connections should override this and call it on shutdown.
        """


# ---------------------------------------------------------------------------
# Adapter-specific exceptions
# ---------------------------------------------------------------------------


class AdapterError(Exception):
    """Base class for all adapter errors."""

    def __init__(self, message: str, server_id: str = "") -> None:
        super().__init__(message)
        self.server_id = server_id


class AdapterUnavailableError(AdapterError):
    """Raised when the backend cannot be reached (network, DNS, etc.)."""


class AdapterRequestError(AdapterError):
    """Raised when the backend rejects the request (4xx response)."""

    def __init__(self, message: str, status_code: int, server_id: str = "") -> None:
        super().__init__(message, server_id=server_id)
        self.status_code = status_code


class AdapterTimeoutError(AdapterError):
    """Raised when a call to the backend exceeds the configured timeout."""
