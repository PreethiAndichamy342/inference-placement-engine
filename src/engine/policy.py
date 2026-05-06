# Copyright 2026 Preethi Andichamy
# Licensed under the Apache License, Version 2.0
"""
Policy engine for HIPAA-compliant cloud placement decisions.

Each rule is an independent, named check. The engine runs all rules in order
and collects per-server violations so callers can explain why a server was
rejected — useful for audit trails required under HIPAA §164.312.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from src.engine.models import (
    CloudEnv,
    CloudServer,
    DataSensitivity,
    InferenceRequest,
    ServerStatus,
)

logger = logging.getLogger(__name__)

# Ordered list of DataSensitivity tiers (lowest → highest).
_SENSITIVITY_ORDER: list[DataSensitivity] = list(DataSensitivity)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PolicyViolation:
    """A single rule failure for one server."""

    rule_name: str
    reason: str


@dataclass
class ServerEvaluation:
    """Full policy evaluation result for a single CloudServer."""

    server: CloudServer
    violations: list[PolicyViolation] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return len(self.violations) == 0

    @property
    def violation_summary(self) -> str:
        if self.eligible:
            return "eligible"
        return "; ".join(f"[{v.rule_name}] {v.reason}" for v in self.violations)


@dataclass
class PolicyResult:
    """
    Aggregate outcome of running the PolicyEngine against a candidate pool.

    Attributes:
        eligible_servers:   Servers that passed every rule.
        evaluations:        Per-server evaluation details (eligible + rejected).
        request_id:         The originating InferenceRequest ID.
    """

    request_id: str
    evaluations: list[ServerEvaluation] = field(default_factory=list)

    @property
    def eligible_servers(self) -> list[CloudServer]:
        return [e.server for e in self.evaluations if e.eligible]

    @property
    def rejected_servers(self) -> list[CloudServer]:
        return [e.server for e in self.evaluations if not e.eligible]

    def violations_for(self, server_id: str) -> list[PolicyViolation]:
        for ev in self.evaluations:
            if ev.server.server_id == server_id:
                return ev.violations
        return []


# ---------------------------------------------------------------------------
# Rule type
# ---------------------------------------------------------------------------

# A rule accepts (server, request) and returns a PolicyViolation if the server
# fails the rule, or None if it passes.
Rule = Callable[[CloudServer, InferenceRequest], PolicyViolation | None]


# ---------------------------------------------------------------------------
# Individual compliance rules
# ---------------------------------------------------------------------------


def _rule_availability(
    server: CloudServer, request: InferenceRequest  # noqa: ARG001
) -> PolicyViolation | None:
    """Server must be HEALTHY and not fully loaded."""
    if server.status != ServerStatus.HEALTHY:
        return PolicyViolation(
            rule_name="availability",
            reason=f"server status is '{server.status.value}', expected 'healthy'",
        )
    if server.current_load >= 1.0:
        return PolicyViolation(
            rule_name="availability",
            reason=f"server is at full capacity (load={server.current_load:.2f})",
        )
    return None


def _rule_model_support(
    server: CloudServer, request: InferenceRequest
) -> PolicyViolation | None:
    """Server must advertise support for the requested model."""
    if request.model_id not in server.supported_models:
        return PolicyViolation(
            rule_name="model_support",
            reason=(
                f"model '{request.model_id}' is not in the server's supported set "
                f"{sorted(server.supported_models)}"
            ),
        )
    return None


def _rule_sensitivity_clearance(
    server: CloudServer, request: InferenceRequest
) -> PolicyViolation | None:
    """
    HIPAA compliance gate — sensitivity tier.

    The server's max_sensitivity ceiling must be >= the request's
    data_sensitivity. Servers with a lower clearance tier are rejected.
    """
    server_idx = _SENSITIVITY_ORDER.index(server.max_sensitivity)
    request_idx = _SENSITIVITY_ORDER.index(request.data_sensitivity)
    if server_idx < request_idx:
        return PolicyViolation(
            rule_name="sensitivity_clearance",
            reason=(
                f"server clearance '{server.max_sensitivity.value}' is insufficient "
                f"for request sensitivity '{request.data_sensitivity.value}'"
            ),
        )
    return None


def _rule_baa_required(
    server: CloudServer, request: InferenceRequest
) -> PolicyViolation | None:
    """
    HIPAA BAA gate — §164.308(b).

    Requests carrying SENSITIVE, PHI, or PHI_STRICT data require the
    hosting environment to have a signed Business Associate Agreement.
    """
    if request.requires_baa and not server.has_baa:
        return PolicyViolation(
            rule_name="baa_required",
            reason=(
                f"data sensitivity '{request.data_sensitivity.value}' requires a BAA "
                "but this server's environment does not have one"
            ),
        )
    return None


def _rule_on_prem_restriction(
    server: CloudServer, request: InferenceRequest
) -> PolicyViolation | None:
    """
    PHI_STRICT gate — 42 CFR Part 2 / contractual on-prem mandates.

    PHI_STRICT data must never leave on-premises infrastructure regardless
    of BAA status or any other clearance.
    """
    if request.on_prem_only and server.cloud_env != CloudEnv.ON_PREM:
        return PolicyViolation(
            rule_name="on_prem_restriction",
            reason=(
                f"data sensitivity '{request.data_sensitivity.value}' mandates "
                f"on-prem hosting, but server is in '{server.cloud_env.value}'"
            ),
        )
    return None


def _rule_phi_public_cloud_guard(
    server: CloudServer, request: InferenceRequest
) -> PolicyViolation | None:
    """
    Additional PHI safeguard for public cloud environments.

    PHI data sent to a public cloud (AWS / GCP / Azure) must land on a
    server that explicitly carries a BAA *and* has a max_sensitivity of
    PHI or higher. This double-checks that cloud operators haven't
    misconfigured a node (e.g. has_baa=True but max_sensitivity=INTERNAL).
    """
    if request.data_sensitivity not in (DataSensitivity.PHI, DataSensitivity.PHI_STRICT):
        return None
    if server.cloud_env == CloudEnv.ON_PREM:
        return None  # On-prem is governed by the on_prem_restriction rule

    phi_cleared = _SENSITIVITY_ORDER.index(server.max_sensitivity) >= _SENSITIVITY_ORDER.index(
        DataSensitivity.PHI
    )
    if not (server.has_baa and phi_cleared):
        return PolicyViolation(
            rule_name="phi_public_cloud_guard",
            reason=(
                "public-cloud server must have both has_baa=True and "
                f"max_sensitivity >= PHI; got has_baa={server.has_baa}, "
                f"max_sensitivity='{server.max_sensitivity.value}'"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

# Default ordered rule pipeline.  Order matters: cheaper / most-selective
# checks run first to fail fast.
DEFAULT_RULES: list[Rule] = [
    _rule_availability,
    _rule_model_support,
    _rule_on_prem_restriction,       # Hard geographic constraint first
    _rule_baa_required,              # Legal / contractual gate
    _rule_sensitivity_clearance,     # Clearance tier
    _rule_phi_public_cloud_guard,    # Belt-and-suspenders PHI double-check
]


class PolicyEngine:
    """
    Evaluates a pool of CloudServers against a set of compliance rules and
    returns only those eligible to handle the given InferenceRequest.

    Usage::

        engine = PolicyEngine()
        result = engine.evaluate(servers, request)
        eligible = result.eligible_servers   # list[CloudServer]

    Custom rules can be appended at construction time::

        engine = PolicyEngine(extra_rules=[my_tenant_rule])

    All rule failures are recorded in PolicyResult.evaluations so that the
    router layer can include them in audit logs without re-running policy.
    """

    def __init__(self, extra_rules: list[Rule] | None = None) -> None:
        self._rules: list[Rule] = list(DEFAULT_RULES)
        if extra_rules:
            self._rules.extend(extra_rules)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        servers: list[CloudServer],
        request: InferenceRequest,
    ) -> PolicyResult:
        """
        Run all rules against every server and return a PolicyResult.

        Args:
            servers: Candidate pool of CloudServer instances.
            request: The InferenceRequest to be placed.

        Returns:
            PolicyResult with eligible_servers populated and full per-server
            violation details available for audit logging.
        """
        result = PolicyResult(request_id=request.request_id)

        for server in servers:
            evaluation = self._evaluate_server(server, request)
            result.evaluations.append(evaluation)

            if evaluation.eligible:
                logger.debug(
                    "policy: server=%s ELIGIBLE for request=%s",
                    server.server_id,
                    request.request_id,
                )
            else:
                logger.info(
                    "policy: server=%s REJECTED for request=%s — %s",
                    server.server_id,
                    request.request_id,
                    evaluation.violation_summary,
                )

        eligible_count = len(result.eligible_servers)
        logger.info(
            "policy: %d/%d servers eligible for request=%s (sensitivity=%s)",
            eligible_count,
            len(servers),
            request.request_id,
            request.data_sensitivity.value,
        )

        return result

    def add_rule(self, rule: Rule) -> None:
        """Append a custom rule to the end of the pipeline at runtime."""
        self._rules.append(rule)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_server(
        self,
        server: CloudServer,
        request: InferenceRequest,
    ) -> ServerEvaluation:
        evaluation = ServerEvaluation(server=server)
        for rule in self._rules:
            violation = rule(server, request)
            if violation is not None:
                evaluation.violations.append(violation)
        return evaluation
