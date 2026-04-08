"""
Pydantic schemas for the inference placement engine API.

All request/response bodies are defined here so main.py stays focused on
routing logic. Schemas mirror the domain models in src/engine/models.py but
are decoupled from them — the API surface can evolve independently of the
internal dataclasses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared enums (string literals — Pydantic validates membership automatically)
# ---------------------------------------------------------------------------

# Re-export the string values accepted by the API. Using Literal keeps OpenAPI
# docs accurate without coupling schemas to the internal Enum classes.
from typing import Literal

DataSensitivityLiteral = Literal[
    "public", "internal", "sensitive", "phi", "phi_strict"
]

InferenceTaskTypeLiteral = Literal[
    "clinical_nlp",
    "medical_imaging",
    "risk_scoring",
    "drug_interaction",
    "genomics",
    "general",
]

RoutingStrategyLiteral = Literal[
    "compliance_first",
    "latency_optimized",
    "cost_optimized",
    "round_robin",
    "least_loaded",
]

CloudEnvLiteral = Literal["aws", "gcp", "azure", "on_prem"]

ServerStatusLiteral = Literal["healthy", "degraded", "unavailable", "draining"]


# ---------------------------------------------------------------------------
# POST /route — request body
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    """
    Body for ``POST /route``.

    The ``payload`` field is forwarded verbatim to the inference backend, so
    callers own its schema (prompt text, messages list, generation params, etc.).
    """

    model_config = {"extra": "forbid"}

    model_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the model to invoke, e.g. 'llama3-med'.",
        examples=["llama3-med"],
    )
    payload: dict[str, Any] = Field(
        ...,
        description="Opaque inference payload forwarded to the backend (prompt, messages, etc.).",
        examples=[{"prompt": "Summarise the patient discharge note.", "max_tokens": 256}],
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the requesting organisation or tenant.",
        examples=["tenant-hospital-a"],
    )
    task_type: InferenceTaskTypeLiteral = Field(
        "general",
        description="Broad category of the ML inference workload.",
    )
    data_sensitivity: DataSensitivityLiteral = Field(
        "internal",
        description="HIPAA-aligned sensitivity tier of the request payload.",
    )
    max_latency_ms: float | None = Field(
        None,
        gt=0,
        description="Soft SLA latency ceiling in milliseconds (None = best-effort).",
        examples=[500.0],
    )
    priority: int = Field(
        5,
        ge=1,
        le=10,
        description="Request priority 1 (lowest) – 10 (highest).",
    )
    region_hint: str | None = Field(
        None,
        description="Preferred cloud region, e.g. 'us-east-1'. Advisory only.",
        examples=["us-east-1"],
    )
    strategy: RoutingStrategyLiteral = Field(
        "compliance_first",
        description="Routing strategy the placement engine should apply.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for tracing or audit.",
    )

    @field_validator("payload")
    @classmethod
    def payload_not_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("payload must not be empty")
        return v


# ---------------------------------------------------------------------------
# POST /route — response body
# ---------------------------------------------------------------------------


class ServerSummary(BaseModel):
    """Condensed server info embedded in a routing response."""

    server_id: str
    cloud_env: CloudEnvLiteral
    region: str
    endpoint: str
    status: ServerStatusLiteral


class ScoreEntry(BaseModel):
    """Per-server numeric scores captured at routing time."""

    current_load: float
    avg_latency_ms: float
    cost_per_token: float
    gpu_count: float


class RouteResponse(BaseModel):
    """Response body for ``POST /route``."""

    request_id: str = Field(..., description="Echoed from the incoming request.")
    rejected: bool = Field(..., description="True when no eligible server was found.")
    rejection_reason: str | None = Field(
        None,
        description="Human-readable reason for rejection. Null when not rejected.",
    )
    strategy_used: RoutingStrategyLiteral
    selected_server: ServerSummary | None = Field(
        None,
        description="The chosen server. Null when rejected=true.",
    )
    candidate_count: int = Field(
        ...,
        description="Total number of servers evaluated before selection.",
    )
    score_breakdown: dict[str, ScoreEntry] = Field(
        default_factory=dict,
        description="Per-server metric snapshot captured at routing time.",
    )
    routing_latency_ms: float = Field(
        ...,
        description="Wall-clock time the router itself took in milliseconds.",
    )
    decided_at: datetime


# ---------------------------------------------------------------------------
# GET /health — response body
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: Literal["ok", "degraded"] = Field(
        ...,
        description="'ok' when at least one server is healthy, otherwise 'degraded'.",
    )
    healthy_server_count: int
    total_server_count: int
    checked_at: datetime


# ---------------------------------------------------------------------------
# GET /metrics — response body
# ---------------------------------------------------------------------------


class ServerMetrics(BaseModel):
    """Per-server stats returned by ``GET /metrics``."""

    server_id: str
    cloud_env: CloudEnvLiteral
    region: str
    status: ServerStatusLiteral
    current_load: float = Field(..., ge=0.0, le=1.0)
    avg_latency_ms: float = Field(..., ge=0.0)
    cost_per_token: float = Field(..., ge=0.0)
    gpu_count: int = Field(..., ge=0)
    gpu_type: str | None


class MetricsResponse(BaseModel):
    """Response body for ``GET /metrics``."""

    servers: list[ServerMetrics]
    collected_at: datetime


# ---------------------------------------------------------------------------
# Generic error envelope
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Standard error response body used for all 4xx/5xx responses."""

    error: str
    detail: str | None = None
    request_id: str | None = None
