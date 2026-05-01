"""
FastAPI application for the multi-cloud healthcare inference placement engine.

Endpoints
---------
POST /route    Accept an InferenceRequest, run it through PlacementRouter,
               return a RoutingDecision as JSON.
GET  /health   Return app liveness status and count of healthy servers.
GET  /metrics  Return per-server stats for all registered servers.

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
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response

from src.api.schemas import (
    ErrorDetail,
    HealthResponse,
    LogEntry,
    LogsResponse,
    LogsStatsResponse,
    MetricsResponse,
    RouteRequest,
    RouteResponse,
    ScoreEntry,
    ServerMetrics,
    ServerSummary,
)
from src.clouds.base import AdapterError
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings (read from environment at startup)
# ---------------------------------------------------------------------------


class _Settings:
    health_poll_interval: float = float(os.getenv("HEALTH_POLL_INTERVAL", "30"))
    on_prem_base_url: str = os.getenv("ON_PREM_BASE_URL", "http://localhost:8000")
    on_prem_model_id: str = os.getenv("ON_PREM_MODEL_ID", "llama3-med")
    on_prem_api_key: str | None = os.getenv("ON_PREM_API_KEY")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = _Settings()
logging.basicConfig(level=settings.log_level)


# ---------------------------------------------------------------------------
# Server / adapter registry
# ---------------------------------------------------------------------------


def _build_server_pool() -> list[tuple[CloudServer, OnPremAdapter]]:
    """
    Construct the initial server pool and their paired adapters.

    Extend this function (or replace it with a config-file loader) to add
    AWS, GCP, or additional on-prem nodes. Each entry is a (CloudServer,
    adapter) pair so the HealthWatcher and PlacementRouter share the exact
    same CloudServer objects — status updates from the watcher are
    immediately visible to the router.
    """
    on_prem_server = CloudServer(
        server_id="on-prem-01",
        cloud_env=CloudEnv.ON_PREM,
        region="local",
        endpoint=f"{settings.on_prem_base_url}/v1/completions",
        supported_models={settings.on_prem_model_id},
        max_sensitivity=DataSensitivity.PHI_STRICT,
        has_baa=True,
        gpu_count=1,
        gpu_type="A100",
        status=ServerStatus.HEALTHY,
    )
    on_prem_adapter = OllamaAdapter(
        base_url=settings.on_prem_base_url,
        model_id=settings.on_prem_model_id,
        server_id=on_prem_server.server_id,
        api_key=settings.on_prem_api_key,
    )
    return [(on_prem_server, on_prem_adapter)]


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

    app.state.watcher = watcher
    app.state.router = router
    app.state.servers = servers
    app.state.log: deque[LogEntry] = deque(maxlen=500)

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
    """
    router: PlacementRouter = request.app.state.router

    inference_request = InferenceRequest(
        model_id=body.model_id,
        payload=body.payload,
        tenant_id=body.tenant_id,
        task_type=InferenceTaskType(body.task_type),
        data_sensitivity=DataSensitivity(body.data_sensitivity),
        max_latency_ms=body.max_latency_ms,
        priority=body.priority,
        region_hint=body.region_hint,
        metadata=body.metadata,
    )

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
    return RouteResponse(
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
    )


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
# GET /dashboard
# ---------------------------------------------------------------------------


@app.get("/dashboard", include_in_schema=False)
async def dashboard() -> Response:
    """Serve the CloudWatch-style log viewer dashboard."""
    html_path = Path(__file__).parent.parent.parent / "dashboard" / "index.html"
    return Response(content=html_path.read_text(), media_type="text/html")
