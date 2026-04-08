"""
On-premises adapter for a locally hosted vLLM inference server.

Communicates with vLLM's OpenAI-compatible HTTP API:
  POST /v1/completions        — submit an inference request
  GET  /health                — liveness probe
  GET  /metrics               — Prometheus text exposition (optional)

vLLM does not expose a native queue-depth or active-connections endpoint,
so those metrics are scraped from the Prometheus /metrics page when
available and fall back gracefully to 0 when the page is absent.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.clouds.base import (
    AdapterRequestError,
    AdapterTimeoutError,
    AdapterUnavailableError,
    CloudAdapter,
)
from src.engine.models import InferenceRequest, ServerStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metric name constants (vLLM >= 0.4)
# ---------------------------------------------------------------------------
_METRIC_QUEUE_DEPTH = "vllm:num_requests_waiting"
_METRIC_RUNNING = "vllm:num_requests_running"
# vLLM exposes p99 via a histogram; fall back to the _sum/_count estimate.
_METRIC_E2E_P99 = "vllm:e2e_request_latency_seconds"


class OnPremAdapter(CloudAdapter):
    """
    Concrete CloudAdapter for a local vLLM server.

    Args:
        base_url:       Root URL of the vLLM server, e.g. ``"http://localhost:8000"``.
                        No trailing slash.
        model_id:       The model name to pass in every completion request,
                        e.g. ``"meta-llama/Llama-3-8b-Instruct"``.
        server_id:      Logical identifier of this node (used in log messages
                        and error context).
        timeout:        Per-request HTTP timeout in seconds (default 30).
        max_tokens:     Default ``max_tokens`` sent with each completion call.
        connect_retries: Number of connection-level retries on transient errors.
        api_key:        Optional bearer token if the vLLM server has auth enabled.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        server_id: str = "on_prem",
        timeout: float = 30.0,
        max_tokens: int = 512,
        connect_retries: int = 2,
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._server_id = server_id
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._api_key = api_key

        self._session = self._build_session(connect_retries)

    # ------------------------------------------------------------------
    # CloudAdapter interface
    # ------------------------------------------------------------------

    def enqueue(self, request: InferenceRequest) -> str:
        """
        POST the inference payload to ``/v1/completions`` and return the
        vLLM-assigned request ID from the response body.

        The ``payload`` field of InferenceRequest is expected to contain either:
          - ``{"prompt": "<text>", ...}``     for raw-text completion, or
          - ``{"messages": [...], ...}``      for chat-style input (passed as-is).

        Any extra keys in ``payload`` (temperature, top_p, stop, etc.) are
        forwarded verbatim to vLLM so callers retain full control.
        """
        body = self._build_completion_body(request)
        url = f"{self._base_url}/v1/completions"

        try:
            resp = self._session.post(url, json=body, timeout=self._timeout)
        except requests.exceptions.Timeout as exc:
            raise AdapterTimeoutError(
                f"POST {url} timed out after {self._timeout}s",
                server_id=self._server_id,
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise AdapterUnavailableError(
                f"Cannot reach vLLM server at {self._base_url}: {exc}",
                server_id=self._server_id,
            ) from exc

        if resp.status_code >= 400:
            raise AdapterRequestError(
                f"vLLM rejected request: HTTP {resp.status_code} — {resp.text[:256]}",
                status_code=resp.status_code,
                server_id=self._server_id,
            )

        data: dict[str, Any] = resp.json()
        request_id: str = data.get("id") or request.request_id
        logger.debug(
            "on_prem: enqueued request=%s → vllm_id=%s server=%s",
            request.request_id,
            request_id,
            self._server_id,
        )
        return request_id

    def get_queue_depth(self) -> int:
        """
        Scrape ``vllm:num_requests_waiting`` from ``/metrics``.

        Returns 0 if the metrics endpoint is unavailable rather than raising,
        so the router can still use this adapter with degraded metric fidelity.
        """
        metrics = self._fetch_prometheus_metrics()
        if metrics is None:
            return 0
        return int(self._parse_gauge(metrics, _METRIC_QUEUE_DEPTH) or 0)

    def get_latency_p99(self) -> float:
        """
        Estimate p99 latency in milliseconds from vLLM's Prometheus histogram.

        vLLM exposes ``vllm:e2e_request_latency_seconds_bucket`` histogram
        buckets. We use the ``_sum / _count`` mean as a proxy when a true
        p99 quantile is not available in the text format. Returns 0.0 on any
        metric scrape failure.
        """
        metrics = self._fetch_prometheus_metrics()
        if metrics is None:
            return 0.0

        total_seconds = self._parse_gauge(metrics, f"{_METRIC_E2E_P99}_sum")
        count = self._parse_gauge(metrics, f"{_METRIC_E2E_P99}_count")

        if not total_seconds or not count or count == 0:
            return 0.0

        mean_ms = (total_seconds / count) * 1000
        # Apply a conservative p99 ≈ 2× mean heuristic when bucket data is
        # unavailable. Callers relying on this should prefer _sum/_count mean
        # and treat the value as approximate.
        return round(mean_ms * 2, 2)

    def get_active_connections(self) -> int:
        """
        Scrape ``vllm:num_requests_running`` from ``/metrics``.

        Returns 0 on scrape failure.
        """
        metrics = self._fetch_prometheus_metrics()
        if metrics is None:
            return 0
        return int(self._parse_gauge(metrics, _METRIC_RUNNING) or 0)

    def health_check(self) -> ServerStatus:
        """
        GET ``/health`` and map the HTTP status code to a ServerStatus.

        vLLM returns:
          200  — server is ready
          503  — server is loading / unhealthy

        Any connection or timeout error is treated as UNAVAILABLE.
        Never raises.
        """
        url = f"{self._base_url}/health"
        try:
            t0 = time.monotonic()
            resp = self._session.get(url, timeout=self._timeout)
            elapsed_ms = (time.monotonic() - t0) * 1000

            if resp.status_code == 200:
                logger.debug(
                    "on_prem: health=HEALTHY server=%s latency=%.1fms",
                    self._server_id,
                    elapsed_ms,
                )
                return ServerStatus.HEALTHY

            if resp.status_code == 503:
                logger.warning(
                    "on_prem: health=DEGRADED server=%s HTTP 503",
                    self._server_id,
                )
                return ServerStatus.DEGRADED

            logger.warning(
                "on_prem: health=UNAVAILABLE server=%s unexpected HTTP %d",
                self._server_id,
                resp.status_code,
            )
            return ServerStatus.UNAVAILABLE

        except requests.exceptions.Timeout:
            logger.warning(
                "on_prem: health=UNAVAILABLE server=%s — /health timed out",
                self._server_id,
            )
            return ServerStatus.UNAVAILABLE

        except requests.exceptions.ConnectionError:
            logger.warning(
                "on_prem: health=UNAVAILABLE server=%s — connection refused",
                self._server_id,
            )
            return ServerStatus.UNAVAILABLE

    def close(self) -> None:
        """Close the underlying HTTP session and release connection pools."""
        self._session.close()
        logger.debug("on_prem: HTTP session closed for server=%s", self._server_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session(self, retries: int) -> requests.Session:
        session = requests.Session()

        retry_policy = Retry(
            total=retries,
            backoff_factor=0.3,
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_policy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.headers.update({"Content-Type": "application/json"})
        if self._api_key:
            session.headers.update({"Authorization": f"Bearer {self._api_key}"})

        return session

    def _build_completion_body(self, request: InferenceRequest) -> dict[str, Any]:
        """Merge adapter defaults with caller-supplied payload fields."""
        body: dict[str, Any] = {
            "model": self._model_id,
            "max_tokens": self._max_tokens,
            **request.payload,  # caller overrides take precedence
        }
        # Ensure the model field always reflects the adapter's configured model.
        body["model"] = self._model_id
        return body

    def _fetch_prometheus_metrics(self) -> str | None:
        """
        GET ``/metrics`` and return the raw Prometheus text body.

        Returns None on any error so metric-dependent methods can fall back
        gracefully without polluting caller logic with try/except.
        """
        url = f"{self._base_url}/metrics"
        try:
            resp = self._session.get(url, timeout=5.0)
            if resp.status_code == 200:
                return resp.text
            logger.debug(
                "on_prem: /metrics returned HTTP %d for server=%s",
                resp.status_code,
                self._server_id,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logger.debug(
                "on_prem: /metrics unreachable for server=%s: %s",
                self._server_id,
                exc,
            )
            return None

    @staticmethod
    def _parse_gauge(metrics_text: str, metric_name: str) -> float | None:
        """
        Extract the value of a Prometheus gauge or counter by name from raw
        Prometheus text-format exposition.

        Only matches unquoted, label-free lines of the form::

            metric_name <value>
            metric_name <value> <timestamp>

        Returns None if the metric is not found.
        """
        pattern = rf"^{re.escape(metric_name)}\s+([\d.eE+\-]+)"
        match = re.search(pattern, metrics_text, re.MULTILINE)
        if match:
            return float(match.group(1))
        return None
