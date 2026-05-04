"""
Placement router for the multi-cloud healthcare inference engine.

PlacementRouter delegates compliance filtering to PolicyEngine, then applies
a pluggable selection strategy to pick the single best eligible server and
returns a fully populated RoutingDecision.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from src.engine.models import (
    CloudServer,
    InferenceRequest,
    RoutingDecision,
    RoutingStrategy,
)
from src.engine.policy import PolicyEngine, PolicyResult

logger = logging.getLogger(__name__)

# A selector takes the eligible server list and the request, and returns the
# chosen server (or None if the list is empty).
Selector = Callable[[list[CloudServer], InferenceRequest], CloudServer | None]


# ---------------------------------------------------------------------------
# Built-in selection strategies
# ---------------------------------------------------------------------------


def _select_latency_optimized(
    servers: list[CloudServer],
    request: InferenceRequest,  # noqa: ARG001
) -> CloudServer | None:
    """Pick the server with the lowest rolling-average latency."""
    if not servers:
        return None
    return min(servers, key=lambda s: s.p99_latency_ms)


def _select_cost_optimized(
    servers: list[CloudServer],
    request: InferenceRequest,  # noqa: ARG001
) -> CloudServer | None:
    """Pick the server with the lowest cost-per-token."""
    if not servers:
        return None
    return min(servers, key=lambda s: s.cost_per_token)


def _select_least_loaded(
    servers: list[CloudServer],
    request: InferenceRequest,  # noqa: ARG001
) -> CloudServer | None:
    """Pick the server with the lowest current load fraction."""
    if not servers:
        return None
    return min(servers, key=lambda s: s.current_load)


def _select_compliance_first(
    servers: list[CloudServer],
    request: InferenceRequest,  # noqa: ARG001
) -> CloudServer | None:
    """
    Return the first eligible server in the list as-is.

    Because PolicyEngine already eliminated non-compliant servers, any
    remaining server is compliant. This strategy is useful when you want
    deterministic placement (e.g. primary / failover ordering controlled
    by the caller's server list ordering).
    """
    return servers[0] if servers else None


# Map every RoutingStrategy to its selector function.
_STRATEGY_SELECTORS: dict[RoutingStrategy, Selector] = {
    RoutingStrategy.LATENCY_OPTIMIZED: _select_latency_optimized,
    RoutingStrategy.COST_OPTIMIZED:    _select_cost_optimized,
    RoutingStrategy.LEAST_LOADED:      _select_least_loaded,
    RoutingStrategy.COMPLIANCE_FIRST:  _select_compliance_first,
}


# ---------------------------------------------------------------------------
# PlacementRouter
# ---------------------------------------------------------------------------


class PlacementRouter:
    """
    Routes an InferenceRequest to the best available CloudServer.

    Responsibilities:
      1. Run PolicyEngine to filter the candidate pool down to compliant servers.
      2. Apply the requested RoutingStrategy to select the single best server.
      3. Return a RoutingDecision that includes the selection, the full
         candidate list, per-server policy violations, and routing latency —
         all fields needed for HIPAA audit logging downstream.

    Usage::

        router = PlacementRouter(servers)
        decision = router.route(request)
        if decision.rejected:
            handle_no_eligible_server(decision.rejection_reason)
        else:
            send_to(decision.endpoint, request.payload)

    Custom routing strategies can be registered at construction time::

        router = PlacementRouter(servers, custom_strategies={
            RoutingStrategy.ROUND_ROBIN: my_round_robin_selector,
        })

    Args:
        servers:            The full pool of CloudServer instances to route across.
        policy_engine:      Optional pre-configured PolicyEngine; a default one is
                            created if omitted.
        default_strategy:   Strategy used when the request does not specify one.
                            Defaults to COMPLIANCE_FIRST.
        custom_strategies:  Additional or override strategy → selector mappings.
    """

    def __init__(
        self,
        servers: list[CloudServer],
        policy_engine: PolicyEngine | None = None,
        default_strategy: RoutingStrategy = RoutingStrategy.COMPLIANCE_FIRST,
        custom_strategies: dict[RoutingStrategy, Selector] | None = None,
    ) -> None:
        self._servers = servers
        self._policy = policy_engine or PolicyEngine()
        self._default_strategy = default_strategy
        self._selectors: dict[RoutingStrategy, Selector] = {**_STRATEGY_SELECTORS}
        if custom_strategies:
            self._selectors.update(custom_strategies)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        request: InferenceRequest,
        strategy: RoutingStrategy | None = None,
        candidate_override: list[CloudServer] | None = None,
    ) -> RoutingDecision:
        """
        Evaluate and select a server for *request*.

        Args:
            request:            The InferenceRequest to place.
            strategy:           Overrides the router's default_strategy for this
                                call. If None, default_strategy is used.
            candidate_override: Use this server pool instead of the router's own
                                ``servers`` list. Useful for region-scoped routing.

        Returns:
            RoutingDecision — always returned, never raises. Callers must check
            ``decision.rejected`` before using ``decision.selected_server``.
        """
        chosen_strategy = strategy or self._default_strategy
        pool = candidate_override if candidate_override is not None else self._servers

        t_start = time.monotonic()

        policy_result: PolicyResult = self._policy.evaluate(pool, request)
        eligible = policy_result.eligible_servers

        selected = self._select(eligible, request, chosen_strategy)

        routing_latency_ms = (time.monotonic() - t_start) * 1000

        if selected is None:
            rejection_reason = self._build_rejection_reason(pool, policy_result)
            logger.warning(
                "router: NO eligible server for request=%s strategy=%s "
                "candidates=%d eligible=%d — %s",
                request.request_id,
                chosen_strategy.value,
                len(pool),
                len(eligible),
                rejection_reason,
            )
            return RoutingDecision(
                request_id=request.request_id,
                strategy_used=chosen_strategy,
                selected_server=None,
                rejected=True,
                rejection_reason=rejection_reason,
                candidate_servers=pool,
                score_breakdown=self._build_score_breakdown(pool),
                routing_latency_ms=routing_latency_ms,
            )

        logger.info(
            "router: request=%s → server=%s (%s) strategy=%s latency_ms=%.2f",
            request.request_id,
            selected.server_id,
            selected.cloud_env.value,
            chosen_strategy.value,
            routing_latency_ms,
        )
        return RoutingDecision(
            request_id=request.request_id,
            strategy_used=chosen_strategy,
            selected_server=selected,
            rejected=False,
            candidate_servers=pool,
            score_breakdown=self._build_score_breakdown(pool),
            routing_latency_ms=routing_latency_ms,
        )

    def update_servers(self, servers: list[CloudServer]) -> None:
        """Replace the candidate pool (e.g. after a health-check refresh)."""
        self._servers = servers

    def register_strategy(self, strategy: RoutingStrategy, selector: Selector) -> None:
        """Register or override a strategy selector at runtime."""
        self._selectors[strategy] = selector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select(
        self,
        eligible: list[CloudServer],
        request: InferenceRequest,
        strategy: RoutingStrategy,
    ) -> CloudServer | None:
        selector = self._selectors.get(strategy)
        if selector is None:
            logger.error(
                "router: unknown strategy '%s', falling back to COMPLIANCE_FIRST",
                strategy.value,
            )
            selector = _select_compliance_first

        selected = selector(eligible, request)
        return selected

    def _build_rejection_reason(
        self,
        pool: list[CloudServer],
        policy_result: PolicyResult,
    ) -> str:
        if not pool:
            return "no servers in candidate pool"

        reasons: list[str] = []
        for server in pool:
            violations = policy_result.violations_for(server.server_id)
            if violations:
                first = violations[0]
                reasons.append(f"{server.server_id}: [{first.rule_name}] {first.reason}")

        if not reasons:
            return "all servers rejected by policy (no specific violation recorded)"

        # Surface up to three representative reasons to keep the message concise.
        sample = reasons[:3]
        suffix = f" (+{len(reasons) - 3} more)" if len(reasons) > 3 else ""
        return "; ".join(sample) + suffix

    @staticmethod
    def _build_score_breakdown(
        pool: list[CloudServer],
    ) -> dict[str, dict[str, float]]:
        """
        Capture raw metrics for every candidate so the RoutingDecision carries
        enough data for downstream observability / audit without re-querying servers.
        """
        return {
            s.server_id: {
                "current_load":   s.current_load,
                "p99_latency_ms": s.p99_latency_ms,
                "cost_per_token": s.cost_per_token,
                "gpu_count":      float(s.gpu_count),
            }
            for s in pool
        }
