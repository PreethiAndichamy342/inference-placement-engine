"""
FastAPI application for the multi-cloud healthcare inference placement engine.

Endpoints
---------
POST /route         Accept an InferenceRequest, run it through PlacementRouter,
                    return a RoutingDecision as JSON.
GET  /health        Return app liveness status and count of healthy servers.
GET  /metrics       Return per-server stats for all registered servers.
POST /de-identify   De-identify free text and return anonymised version + token count.

Startup / shutdown
------------------
A FastAPI lifespan context manager wires up HealthWatcher and PlacementRouter
once at startup and tears them down cleanly on shutdown. Both are stored in
``app.state`` so they are accessible from request handlers without globals.

Configuration is read from environment variables (see _Settings below).
In production, set these via a .env file, Kubernetes secrets, or your
cloud's secret manager — never hard-code them.
"""

from __future__ import annotations

import logging
import os
import time as _time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import redis as _redis

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    CircuitServerStatus,
    CircuitStatusResponse,
    DeIdentifyRequest,
    DeIdentifyResponse,
    ErrorDetail,
    ForceHealthPollResult,
    HealthCheckResult,
    HealthResponse,
    LogEntry,
    LogsResponse,
    LogsStatsResponse,
    MetricsResponse,
    RouteRequest,
    RouteResponse,
    ScoreEntry,
    ServerLogsResponse,
    ServerMetrics,
    ServerSummary,
    TestPromptsResponse,
)
from src.cache.cache import InferenceCache, PHICacheViolation
from src.clouds.base import AdapterError
import src.demo_servers as demo_servers
from src.clouds.on_prem import OllamaAdapter, OnPremAdapter
from src.engine.health import HealthWatcher
from src.engine.models import (
    CloudEnv,
    CloudServer,
    DataSensitivity,
    InferenceRequest,
    InferenceTaskType,
    RoutingStrategy,
    ServerStatus,
)
from src.engine.policy import PolicyEngine
from src.engine.router import PlacementRouter
from src.phi.de_identifier import DeIdentifier
from src.phi.vault import PHIVault
from src.api.test_data import get_sample_prompts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings (read from environment at startup)
# ---------------------------------------------------------------------------


class _Settings:
    health_poll_interval: float = float(os.getenv("HEALTH_POLL_INTERVAL", "30"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = _Settings()
logging.basicConfig(level=settings.log_level)


# ---------------------------------------------------------------------------
# Server / adapter registry
# ---------------------------------------------------------------------------


def _build_server_pool() -> list[tuple[CloudServer, OllamaAdapter]]:
    """
    Return (CloudServer, OllamaAdapter) pairs from the demo server registry.

    All three simulated environments (aws-sim, gcp-sim, on-prem) are
    registered so the HealthWatcher and PlacementRouter share the same
    CloudServer objects — health-status updates are immediately visible
    to the router.
    """
    return [
        (server, demo_servers.adapters[server.server_id])
        for server in demo_servers.servers
    ]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start background services on startup, shut them down on exit."""
    pairs = _build_server_pool()
    servers = [s for s, _ in pairs]

    watcher = HealthWatcher(interval=settings.health_poll_interval)
    for server, adapter in pairs:
        watcher.register(adapter, server)
    watcher.start()

    router = PlacementRouter(
        servers=servers,
        policy_engine=PolicyEngine(),
        default_strategy=RoutingStrategy.COMPLIANCE_FIRST,
    )

    # PHI de-identification pipeline — initialised once, shared across requests
    de_identifier = DeIdentifier()
    phi_vault     = PHIVault()

    # Inference result cache — Redis-backed, PHI-gated.
    # Redis is optional: if the instance is unreachable at startup we log a
    # warning and disable caching rather than refusing to start.
    cache: InferenceCache | None = None
    _redis_host = os.getenv("REDIS_HOST", "localhost")
    _redis_port = int(os.getenv("REDIS_PORT", "6379"))
    try:
        _redis_client = _redis.Redis(
            host=_redis_host, port=_redis_port, decode_responses=True,
            socket_connect_timeout=2,
        )
        _redis_client.ping()
        cache = InferenceCache(_redis_client)
        logger.info("cache: Redis connected at %s:%d", _redis_host, _redis_port)
    except _redis.RedisError as _exc:
        logger.warning(
            "cache: Redis unavailable at %s:%d (%s) — caching disabled",
            _redis_host, _redis_port, _exc,
        )

    app.state.watcher       = watcher
    app.state.router        = router
    app.state.servers       = servers
    app.state.log: deque[LogEntry] = deque(maxlen=500)
    app.state.de_identifier = de_identifier
    app.state.phi_vault     = phi_vault
    app.state.cache         = cache

    logger.info(
        "placement engine started — servers=%d poll_interval=%.0fs",
        len(servers),
        settings.health_poll_interval,
    )

    yield  # application runs here

    watcher.stop()
    for _, adapter in pairs:
        adapter.close()
    logger.info("placement engine shut down cleanly")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Multi-Cloud Healthcare Inference Placement Engine",
    description=(
        "HIPAA-aware placement router that selects the best cloud or on-prem "
        "inference server for each request based on compliance rules, latency, "
        "cost, and load."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the dashboard SPA (index.html + css/ + js/) at /dashboard.
# html=True makes StaticFiles serve index.html for directory requests.
_dashboard_dir = Path(__file__).parent.parent.parent / "dashboard"
app.mount("/dashboard", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(AdapterError)
async def adapter_error_handler(request: Request, exc: AdapterError) -> JSONResponse:
    logger.error("AdapterError: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=ErrorDetail(
            error="backend_unavailable",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorDetail(error="validation_error", detail=str(exc)).model_dump(),
    )


# ---------------------------------------------------------------------------
# POST /route
# ---------------------------------------------------------------------------


@app.post(
    "/route",
    response_model=RouteResponse,
    status_code=status.HTTP_200_OK,
    summary="Route an inference request to the best eligible server",
    responses={
        422: {"model": ErrorDetail, "description": "Invalid request body"},
        502: {"model": ErrorDetail, "description": "All backends unavailable"},
    },
)
async def route_request(body: RouteRequest, request: Request) -> RouteResponse:
    """
    Accept an inference request, apply HIPAA compliance policy, and return
    the routing decision including which server was selected (or why none was).

    If the payload contains a ``prompt`` field, it is run through the PHI
    de-identifier before the request is forwarded to the router.  The entity
    map is encrypted and stored in PHIVault keyed by request_id.

    Routing decisions are cached in Redis (keyed on model_id + sensitivity +
    sha256(payload)) for non-PHI requests so repeated identical requests skip
    the scoring pass. PHI and PHI_STRICT requests are never cached.
    """
    router:        PlacementRouter = request.app.state.router
    de_identifier: DeIdentifier    = request.app.state.de_identifier
    phi_vault:     PHIVault        = request.app.state.phi_vault
    cache: InferenceCache | None   = request.app.state.cache

    # De-identify prompt text if present
    payload = dict(body.payload)
    deid_entity_count = 0
    phi_entity_breakdown: dict = {}
    if "prompt" in payload and isinstance(payload["prompt"], str):
        result = de_identifier.de_identify(payload["prompt"])
        payload["prompt"] = result.anonymized_text
        deid_entity_count = result.entity_count
        phi_entity_breakdown = result.entities_by_type
        if result.entity_map:
            logger.debug(
                "route: de-identified %d entities from prompt for tenant=%s",
                result.entity_count, body.tenant_id,
            )

    inference_request = InferenceRequest(
        model_id=body.model_id,
        payload=payload,
        tenant_id=body.tenant_id,
        task_type=InferenceTaskType(body.task_type),
        data_sensitivity=DataSensitivity(body.data_sensitivity),
        max_latency_ms=body.max_latency_ms,
        priority=body.priority,
        region_hint=body.region_hint,
        metadata=body.metadata,
    )

    # Store encrypted entity map — keyed by the new request_id
    if deid_entity_count:
        phi_vault.store(inference_request.request_id, result.entity_map)

    # ── Cache read (skip for PHI tiers) ──────────────────────────────────
    if cache is not None:
        try:
            cached = cache.get(inference_request)
            if cached is not None:
                logger.debug(
                    "route: cache HIT request=%s", inference_request.request_id
                )
                return RouteResponse.model_validate(cached)
        except PHICacheViolation:
            pass  # expected for phi/phi_strict — silently skip
        except _redis.RedisError as exc:
            logger.warning("route: cache read failed: %s", exc)

    strategy = RoutingStrategy(body.strategy)
    decision = router.route(inference_request, strategy=strategy)

    # Append PHI-safe log entry (payload and prompt text are never stored)
    selected_server = decision.selected_server
    request.app.state.log.append(
        LogEntry(
            timestamp=datetime.now(tz=timezone.utc),
            request_id=decision.request_id,
            tenant_id=inference_request.tenant_id,
            data_sensitivity=inference_request.data_sensitivity.value,
            strategy_used=decision.strategy_used.value,
            selected_server_id=selected_server.server_id if selected_server else None,
            cloud_env=selected_server.cloud_env.value if selected_server else None,
            routing_latency_ms=decision.routing_latency_ms,
            rejected=decision.rejected,
            rejection_reason=decision.rejection_reason,
            phi_entities_detected=deid_entity_count,
            candidate_count=len(decision.candidate_servers),
            score_breakdown=decision.score_breakdown,
            phi_entity_breakdown=phi_entity_breakdown,
        )
    )

    if decision.rejected:
        logger.warning(
            "route: request=%s rejected — %s",
            inference_request.request_id,
            decision.rejection_reason,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=decision.rejection_reason or "No eligible server found",
        )

    selected = decision.selected_server
    response = RouteResponse(
        request_id=decision.request_id,
        rejected=decision.rejected,
        rejection_reason=decision.rejection_reason,
        strategy_used=decision.strategy_used.value,
        selected_server=ServerSummary(
            server_id=selected.server_id,
            cloud_env=selected.cloud_env.value,
            region=selected.region,
            endpoint=selected.endpoint,
            status=selected.status.value,
        ),
        candidate_count=len(decision.candidate_servers),
        score_breakdown={
            sid: ScoreEntry(**scores)
            for sid, scores in decision.score_breakdown.items()
        },
        routing_latency_ms=decision.routing_latency_ms,
        decided_at=decision.decided_at,
        phi_entities_detected=deid_entity_count,
    )

    # ── Cache write (skip for PHI tiers) ──────────────────────────────────
    if cache is not None:
        try:
            cache.set(inference_request, response.model_dump(mode="json"))
            logger.debug(
                "route: cache SET request=%s", inference_request.request_id
            )
        except PHICacheViolation:
            pass  # expected for phi/phi_strict — silently skip
        except _redis.RedisError as exc:
            logger.warning("route: cache write failed: %s", exc)

    return response


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Return app liveness and healthy server count",
)
async def health(request: Request) -> HealthResponse:
    """
    Returns ``status: ok`` when at least one server is healthy.
    Returns ``status: degraded`` when no servers are healthy (still 200 so
    load-balancer health checks don't immediately pull the instance).
    """
    watcher: HealthWatcher = request.app.state.watcher
    healthy_servers = watcher.get_healthy_servers()
    all_servers = watcher.get_all_servers()

    app_status = "ok" if healthy_servers else "degraded"

    return HealthResponse(
        status=app_status,
        healthy_server_count=len(healthy_servers),
        total_server_count=len(all_servers),
        checked_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Return per-server stats for all registered servers",
)
async def metrics(request: Request) -> MetricsResponse:
    """
    Returns a snapshot of load, latency, cost, and status for every server
    registered with the HealthWatcher.
    """
    watcher: HealthWatcher = request.app.state.watcher
    all_servers: list[CloudServer] = watcher.get_all_servers()

    return MetricsResponse(
        servers=[
            ServerMetrics(
                server_id=s.server_id,
                cloud_env=s.cloud_env.value,
                region=s.region,
                status=s.status.value,
                current_load=s.current_load,
                avg_latency_ms=s.avg_latency_ms,
                cost_per_token=s.cost_per_token,
                gpu_count=s.gpu_count,
                gpu_type=s.gpu_type,
            )
            for s in all_servers
        ],
        collected_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /logs
# ---------------------------------------------------------------------------


@app.get(
    "/logs",
    response_model=LogsResponse,
    summary="Query the in-memory routing decision log (last 500 entries)",
)
async def get_logs(
    request: Request,
    sensitivity: Optional[str] = Query(None, description="Filter by data_sensitivity tier"),
    tenant_id: Optional[str] = Query(None, description="Filter by exact tenant_id"),
    cloud_env: Optional[str] = Query(None, description="Filter by cloud environment"),
    rejected: Optional[bool] = Query(None, description="true = rejected only, false = accepted only"),
    limit: int = Query(50, ge=1, le=500, description="Max entries to return (default 50)"),
    search: Optional[str] = Query(None, description="Match against request_id or tenant_id (case-insensitive)"),
) -> LogsResponse:
    """
    Return routing decision log entries. No payload or PHI fields are ever stored.

    Entries are returned newest-first. Use ``limit`` to control page size (max 500).
    """
    entries: list[LogEntry] = list(request.app.state.log)

    if sensitivity is not None:
        entries = [e for e in entries if e.data_sensitivity == sensitivity]
    if tenant_id is not None:
        entries = [e for e in entries if e.tenant_id == tenant_id]
    if cloud_env is not None:
        entries = [e for e in entries if e.cloud_env == cloud_env]
    if rejected is not None:
        entries = [e for e in entries if e.rejected == rejected]
    if search:
        needle = search.lower()
        entries = [
            e for e in entries
            if needle in e.request_id.lower() or needle in e.tenant_id.lower()
        ]

    total = len(entries)
    # Return newest first
    entries = list(reversed(entries))[:limit]

    return LogsResponse(entries=entries, total=total, returned=len(entries))


# ---------------------------------------------------------------------------
# GET /logs/stats
# ---------------------------------------------------------------------------


@app.get(
    "/logs/stats",
    response_model=LogsStatsResponse,
    summary="Aggregated counts from the routing log grouped by sensitivity and cloud_env",
)
async def get_logs_stats(request: Request) -> LogsStatsResponse:
    """
    Returns total counts and breakdowns by ``data_sensitivity`` and ``cloud_env``
    across all entries currently in the in-memory log.
    """
    entries: list[LogEntry] = list(request.app.state.log)

    by_sensitivity: dict[str, int] = {}
    by_cloud_env: dict[str, int] = {}
    rejected_count = 0

    for e in entries:
        by_sensitivity[e.data_sensitivity] = by_sensitivity.get(e.data_sensitivity, 0) + 1
        if e.cloud_env:
            by_cloud_env[e.cloud_env] = by_cloud_env.get(e.cloud_env, 0) + 1
        if e.rejected:
            rejected_count += 1

    return LogsStatsResponse(
        total=len(entries),
        rejected_count=rejected_count,
        by_sensitivity=by_sensitivity,
        by_cloud_env=by_cloud_env,
    )


# ---------------------------------------------------------------------------
# GET /circuit-status
# ---------------------------------------------------------------------------


@app.get(
    "/circuit-status",
    response_model=CircuitStatusResponse,
    summary="Return circuit breaker state for all registered servers",
)
async def circuit_status() -> CircuitStatusResponse:
    """
    Returns the circuit breaker state for every adapter in the demo server
    registry: CLOSED (normal), OPEN (fast-failing), or HALF_OPEN (probing).
    """
    results: list[CircuitServerStatus] = []

    for server in demo_servers.servers:
        adapter = demo_servers.adapters.get(server.server_id)
        if adapter is None:
            continue
        cb = adapter._circuit
        last_fail = cb._last_failure_time if cb._failure_count > 0 else None
        results.append(
            CircuitServerStatus(
                server_id=server.server_id,
                state=cb.state.value,
                consecutive_failures=cb._failure_count,
                failure_threshold=cb.failure_threshold,
                last_failure_time=last_fail,
            )
        )

    return CircuitStatusResponse(
        servers=results,
        collected_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# POST /de-identify
# ---------------------------------------------------------------------------


@app.post(
    "/de-identify",
    response_model=DeIdentifyResponse,
    status_code=status.HTTP_200_OK,
    summary="De-identify free text and return anonymised version with token count",
    responses={
        422: {"model": ErrorDetail, "description": "Invalid request body"},
    },
)
async def de_identify(body: DeIdentifyRequest, request: Request) -> DeIdentifyResponse:
    """
    Run the input text through the PHI de-identifier and return:
    - ``anonymized_text`` — original text with PHI replaced by tokens
    - ``entity_count``    — total number of PHI entities detected
    - ``entities_by_type`` — per-type breakdown (PERSON, DATE, SSN, etc.)

    The entity map (token → original value) is **not** returned here — callers
    that need re-identification should use ``POST /route`` which stores the map
    securely in PHIVault.
    """
    de_identifier: DeIdentifier = request.app.state.de_identifier
    result = de_identifier.de_identify(body.text)

    return DeIdentifyResponse(
        anonymized_text=result.anonymized_text,
        entity_count=result.entity_count,
        entities_by_type=result.entities_by_type,
    )


# ---------------------------------------------------------------------------
# GET /test-prompts
# ---------------------------------------------------------------------------


@app.get(
    "/test-prompts",
    response_model=TestPromptsResponse,
    summary="Return sample prompts grouped by data_sensitivity tier",
)
async def test_prompts() -> TestPromptsResponse:
    """
    Returns fabricated sample prompts for each sensitivity tier, suitable for
    exercising the de-identification pipeline and routing logic from the
    dashboard's test panel.

    All PHI in ``phi`` and ``phi_strict`` tiers is entirely fabricated.
    """
    return TestPromptsResponse(prompts=get_sample_prompts())


# ---------------------------------------------------------------------------
# GET /health-check/{server_id}
# ---------------------------------------------------------------------------


@app.get(
    "/health-check/{server_id}",
    response_model=HealthCheckResult,
    summary="Directly probe a server's health endpoint and return the result",
)
async def health_check_server(server_id: str) -> HealthCheckResult:
    """
    Immediately calls the adapter's ``health_check()`` for the given server
    and returns the raw result with timing. Does **not** update the server's
    persisted status — use ``POST /force-health-poll/{server_id}`` for that.
    """
    adapter = demo_servers.adapters.get(server_id)
    if adapter is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No adapter registered for server_id '{server_id}'",
        )

    start = _time.monotonic()
    error_msg: str | None = None
    probe_status: ServerStatus = ServerStatus.UNAVAILABLE
    try:
        probe_status = adapter.health_check()
    except Exception as exc:
        error_msg = str(exc)

    latency_ms = (_time.monotonic() - start) * 1000

    return HealthCheckResult(
        server_id=server_id,
        status=probe_status.value,
        latency_ms=round(latency_ms, 3),
        error=error_msg,
        checked_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /server-logs/{server_id}
# ---------------------------------------------------------------------------


@app.get(
    "/server-logs/{server_id}",
    response_model=ServerLogsResponse,
    summary="Return routing log entries that were dispatched to a specific server",
)
async def server_logs(
    server_id: str,
    request: Request,
    limit: int = Query(20, ge=1, le=200, description="Max entries to return"),
) -> ServerLogsResponse:
    """
    Filters the in-memory routing log to entries where ``selected_server_id``
    matches the given server. Returns newest first.
    """
    all_entries: list[LogEntry] = list(request.app.state.log)
    matched = [e for e in all_entries if e.selected_server_id == server_id]
    total = len(matched)
    # Newest first
    matched = list(reversed(matched))[:limit]

    return ServerLogsResponse(server_id=server_id, entries=matched, total=total)


# ---------------------------------------------------------------------------
# POST /force-health-poll/{server_id}
# ---------------------------------------------------------------------------


@app.post(
    "/force-health-poll/{server_id}",
    response_model=ForceHealthPollResult,
    summary="Trigger an immediate health check and update the server's persisted status",
)
async def force_health_poll(server_id: str) -> ForceHealthPollResult:
    """
    Calls ``adapter.health_check()`` for the given server, updates
    ``server.status`` in place (same object the router reads), and returns
    the previous and new status values. Use this to force a recovery check
    without waiting for the next background poll interval.
    """
    adapter = demo_servers.adapters.get(server_id)
    server  = next((s for s in demo_servers.servers if s.server_id == server_id), None)

    if adapter is None or server is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No server registered with server_id '{server_id}'",
        )

    previous_status = server.status

    start = _time.monotonic()
    latency_ms: float | None = None
    try:
        new_status_enum = adapter.health_check()
        latency_ms = round((_time.monotonic() - start) * 1000, 3)
        server.status = new_status_enum
    except Exception as exc:
        latency_ms = round((_time.monotonic() - start) * 1000, 3)
        server.status = ServerStatus.UNAVAILABLE
        new_status_enum = ServerStatus.UNAVAILABLE
        logger.warning(
            "force-health-poll: server=%s probe failed: %s", server_id, exc
        )

    logger.info(
        "force-health-poll: server=%s %s → %s (%.1f ms)",
        server_id,
        previous_status.value,
        new_status_enum.value,
        latency_ms or 0,
    )

    return ForceHealthPollResult(
        server_id=server_id,
        previous_status=previous_status.value,
        new_status=new_status_enum.value,
        latency_ms=latency_ms,
        polled_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# GET /dashboard
# ---------------------------------------------------------------------------


@app.get("/dashboard", include_in_schema=False)
async def dashboard() -> RedirectResponse:
    """Redirect bare /dashboard to the StaticFiles mount for backwards compatibility."""
    return RedirectResponse(url="/dashboard/")
