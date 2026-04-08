"""
Domain models for the multi-cloud healthcare inference placement engine.

All PHI/PII sensitivity levels follow HIPAA classification conventions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CloudEnv(str, Enum):
    """Target cloud / hosting environment for inference workloads."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ON_PREM = "on_prem"


class DataSensitivity(str, Enum):
    """
    HIPAA-aligned data sensitivity tiers that gate cloud placement.

    PHI (Protected Health Information) and PII must never leave
    environments that lack a signed BAA or equivalent compliance posture.

    Tiers (ascending sensitivity):
      PUBLIC     – no PII/PHI; any cloud env is acceptable.
      INTERNAL   – de-identified or aggregated; public cloud allowed,
                   but prefer private subnets.
      SENSITIVE  – limited data set (dates, zip-5); BAA required.
      PHI        – full PHI; BAA + HIPAA-compliant env required.
      PHI_STRICT – PHI plus additional contractual/regulatory overlay
                   (e.g. 42 CFR Part 2 substance-use records); on-prem
                   or dedicated tenancy only.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    PHI = "phi"
    PHI_STRICT = "phi_strict"


class InferenceTaskType(str, Enum):
    """Broad category of the ML inference task."""

    CLINICAL_NLP = "clinical_nlp"          # Named-entity / ICD coding
    MEDICAL_IMAGING = "medical_imaging"    # Radiology, pathology vision models
    RISK_SCORING = "risk_scoring"          # Readmission / sepsis risk
    DRUG_INTERACTION = "drug_interaction"  # Pharmacovigilance
    GENOMICS = "genomics"                  # Variant calling / annotation
    GENERAL = "general"                    # Catch-all


class ServerStatus(str, Enum):
    """Operational health of a CloudServer instance."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DRAINING = "draining"           # Graceful shutdown in progress


class RoutingStrategy(str, Enum):
    """High-level algorithm used to select the placement target."""

    COMPLIANCE_FIRST = "compliance_first"   # Eliminate non-compliant envs first
    LATENCY_OPTIMIZED = "latency_optimized"
    COST_OPTIMIZED = "cost_optimized"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InferenceRequest:
    """
    Represents an incoming request to run ML inference on healthcare data.

    Attributes:
        request_id:       Unique identifier (auto-generated if omitted).
        task_type:        Category of inference workload.
        model_id:         Identifier of the model to invoke (e.g. "llama3-med").
        payload:          Raw input payload; kept opaque at this layer.
        data_sensitivity: HIPAA sensitivity tier of the payload data.
        max_latency_ms:   Soft SLA ceiling in milliseconds (None = best-effort).
        priority:         Integer priority; higher value = higher priority (1–10).
        region_hint:      Preferred cloud region, e.g. "us-east-1" (optional).
        tenant_id:        Identifier of the requesting organisation or tenant.
        created_at:       UTC timestamp of request creation.
        metadata:         Arbitrary key-value pairs for tracing / audit.
    """

    model_id: str
    payload: dict[str, Any]
    tenant_id: str

    task_type: InferenceTaskType = InferenceTaskType.GENERAL
    data_sensitivity: DataSensitivity = DataSensitivity.INTERNAL
    max_latency_ms: float | None = None
    priority: int = 5
    region_hint: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be between 1 and 10, got {self.priority}")
        if self.max_latency_ms is not None and self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")

    @property
    def requires_baa(self) -> bool:
        """True when the sensitivity tier mandates a signed BAA."""
        return self.data_sensitivity in (
            DataSensitivity.PHI,
            DataSensitivity.PHI_STRICT,
            DataSensitivity.SENSITIVE,
        )

    @property
    def on_prem_only(self) -> bool:
        """True when the data must remain on-premises."""
        return self.data_sensitivity == DataSensitivity.PHI_STRICT


@dataclass
class CloudServer:
    """
    Represents a single inference-capable server/node within a cloud environment.

    Attributes:
        server_id:          Unique identifier for this node.
        cloud_env:          The cloud or hosting environment it belongs to.
        region:             Cloud region identifier (e.g. "us-central1").
        endpoint:           Base URL / address for inference requests.
        supported_models:   Set of model IDs this node can serve.
        max_sensitivity:    Highest data sensitivity tier this node is cleared for.
        has_baa:            Whether the env has a signed Business Associate Agreement.
        gpu_count:          Number of GPUs available (0 = CPU-only node).
        gpu_type:           GPU model string, e.g. "A100", "T4" (None if CPU-only).
        status:             Operational health status.
        current_load:       Fraction of capacity in use (0.0–1.0).
        avg_latency_ms:     Rolling average inference latency in milliseconds.
        cost_per_token:     Estimated cost in USD per 1 K tokens (for LLMs).
        tags:               Arbitrary labels for policy matching.
    """

    server_id: str
    cloud_env: CloudEnv
    region: str
    endpoint: str
    supported_models: set[str]

    max_sensitivity: DataSensitivity = DataSensitivity.INTERNAL
    has_baa: bool = False
    gpu_count: int = 0
    gpu_type: str | None = None
    status: ServerStatus = ServerStatus.HEALTHY
    current_load: float = 0.0
    avg_latency_ms: float = 0.0
    cost_per_token: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.current_load <= 1.0:
            raise ValueError(f"current_load must be in [0, 1], got {self.current_load}")

    @property
    def is_available(self) -> bool:
        return self.status == ServerStatus.HEALTHY and self.current_load < 1.0

    def can_serve(self, request: InferenceRequest) -> bool:
        """
        Returns True if this server is eligible to handle the given request.

        Checks:
          1. Server is available (healthy + not fully loaded).
          2. Requested model is supported.
          3. Server's clearance tier covers the request's sensitivity.
          4. BAA requirement is satisfied when needed.
          5. On-prem restriction is respected for PHI_STRICT data.
        """
        if not self.is_available:
            return False
        if request.model_id not in self.supported_models:
            return False

        sensitivity_order = list(DataSensitivity)
        if sensitivity_order.index(self.max_sensitivity) < sensitivity_order.index(
            request.data_sensitivity
        ):
            return False

        if request.requires_baa and not self.has_baa:
            return False

        if request.on_prem_only and self.cloud_env != CloudEnv.ON_PREM:
            return False

        return True


@dataclass
class RoutingDecision:
    """
    The output of the placement engine for a single InferenceRequest.

    Attributes:
        request_id:        The originating request's ID.
        selected_server:   The server chosen to handle the request (None if rejected).
        strategy_used:     The routing strategy that produced this decision.
        rejected:          True when no eligible server was found.
        rejection_reason:  Human-readable reason for rejection (if rejected).
        candidate_servers: All servers evaluated before selection.
        score_breakdown:   Per-server scoring details keyed by server_id.
        decided_at:        UTC timestamp when the decision was made.
        routing_latency_ms: Wall-clock time the router itself took in milliseconds.
        metadata:          Arbitrary key-value pairs for tracing / audit.
    """

    request_id: str
    strategy_used: RoutingStrategy

    selected_server: CloudServer | None = None
    rejected: bool = False
    rejection_reason: str | None = None
    candidate_servers: list[CloudServer] = field(default_factory=list)
    score_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    decided_at: datetime = field(default_factory=datetime.utcnow)
    routing_latency_ms: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rejected and self.selected_server is not None:
            raise ValueError("A rejected decision cannot have a selected_server.")
        if not self.rejected and self.selected_server is None:
            raise ValueError(
                "A non-rejected decision must have a selected_server. "
                "Set rejected=True if no server could be found."
            )

    @property
    def cloud_env(self) -> CloudEnv | None:
        """Convenience accessor for the selected server's cloud environment."""
        return self.selected_server.cloud_env if self.selected_server else None

    @property
    def endpoint(self) -> str | None:
        """Convenience accessor for the selected server's endpoint URL."""
        return self.selected_server.endpoint if self.selected_server else None
