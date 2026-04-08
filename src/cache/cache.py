"""
Redis-backed inference result cache.

PHI safety contract
-------------------
Requests with DataSensitivity.PHI or PHI_STRICT must NEVER be cached —
caching PHI results would create a secondary store that is hard to audit,
purge on patient request (HIPAA Right of Access / Right to Delete), or
protect with the same access controls as the primary data store.

Any attempt to cache a PHI request raises ``PHICacheViolation`` (a subclass
of ValueError) immediately, before any Redis I/O takes place.

Cache key design
----------------
Keys are formed as::

    inference:v1:<model_id>:<sensitivity>:<sha256(canonical_json(payload))>

- The payload is serialised to canonical JSON (sorted keys, no whitespace)
  before hashing so key stability is independent of dict insertion order.
- SHA-256 of the payload is used — raw payload text never appears in the key.
- model_id and sensitivity tier are included so the same payload sent under
  different models or sensitivity contexts produces different cache entries.
- The ``inference:v1:`` namespace prefix makes it easy to flush all entries
  with a single ``SCAN + DEL`` without affecting other Redis keyspaces.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis

from src.engine.models import DataSensitivity, InferenceRequest

logger = logging.getLogger(__name__)

# Sensitivity tiers that are unconditionally banned from the cache.
_BANNED_TIERS: frozenset[DataSensitivity] = frozenset(
    [DataSensitivity.PHI, DataSensitivity.PHI_STRICT]
)

_KEY_PREFIX = "inference:v1"
_DEFAULT_TTL = 900  # 15 minutes


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PHICacheViolation(ValueError):
    """
    Raised when a caller attempts to cache or read a PHI/PHI_STRICT request.

    Inherits from ValueError so it surfaces as a programming error rather than
    a recoverable runtime condition.
    """

    def __init__(self, sensitivity: DataSensitivity) -> None:
        super().__init__(
            f"Caching is prohibited for data sensitivity '{sensitivity.value}'. "
            "PHI and PHI_STRICT requests must never be written to or read from cache."
        )
        self.sensitivity = sensitivity


# ---------------------------------------------------------------------------
# InferenceCache
# ---------------------------------------------------------------------------


class InferenceCache:
    """
    Redis-backed cache for inference results.

    PHI-gated: ``get``, ``set``, and ``invalidate`` all raise
    ``PHICacheViolation`` when called with a PHI or PHI_STRICT request,
    so callers do not need to check sensitivity before calling.

    Args:
        redis_client:   A configured ``redis.Redis`` instance. The caller owns
                        the client lifecycle (connection pool, TLS, auth, etc.).
        default_ttl:    Default TTL in seconds applied when ``set()`` is called
                        without an explicit ttl argument (default 900 / 15 min).
        key_prefix:     Namespace prefix for all cache keys. Override in tests to
                        avoid collisions with production data.

    Usage::

        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        cache = InferenceCache(client)

        result = cache.get(request)
        if result is None:
            result = run_inference(request)
            cache.set(request, result, ttl=300)
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        default_ttl: int = _DEFAULT_TTL,
        key_prefix: str = _KEY_PREFIX,
    ) -> None:
        self._redis = redis_client
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, request: InferenceRequest) -> dict[str, Any] | None:
        """
        Look up a cached inference result for *request*.

        Args:
            request: The InferenceRequest to look up.

        Returns:
            The cached result dict if a cache hit occurs, otherwise None.

        Raises:
            PHICacheViolation: If the request's sensitivity is PHI or PHI_STRICT.
            redis.RedisError:  On Redis connectivity or protocol errors.
        """
        self._assert_not_phi(request)
        key = self._make_key(request)

        raw = self._redis.get(key)
        if raw is None:
            logger.debug("cache: MISS key=%s request=%s", key, request.request_id)
            return None

        result: dict[str, Any] = json.loads(raw)
        logger.debug("cache: HIT  key=%s request=%s", key, request.request_id)
        return result

    def set(
        self,
        request: InferenceRequest,
        result: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """
        Store *result* in the cache under the key derived from *request*.

        Args:
            request: The originating InferenceRequest (used to derive the key).
            result:  The inference result dict to cache. Must be JSON-serialisable.
            ttl:     Time-to-live in seconds. Defaults to ``default_ttl``.

        Raises:
            PHICacheViolation: If the request's sensitivity is PHI or PHI_STRICT.
            ValueError:        If *result* is not JSON-serialisable.
            redis.RedisError:  On Redis connectivity or protocol errors.
        """
        self._assert_not_phi(request)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        key = self._make_key(request)

        try:
            payload_bytes = json.dumps(result, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Inference result for request={request.request_id} is not "
                f"JSON-serialisable: {exc}"
            ) from exc

        self._redis.setex(key, effective_ttl, payload_bytes)
        logger.debug(
            "cache: SET  key=%s request=%s ttl=%ds",
            key,
            request.request_id,
            effective_ttl,
        )

    def invalidate(self, request: InferenceRequest) -> bool:
        """
        Delete the cached entry for *request*, if it exists.

        Args:
            request: The InferenceRequest whose cache entry should be removed.

        Returns:
            True if a key was deleted, False if there was no entry to delete.

        Raises:
            PHICacheViolation: If the request's sensitivity is PHI or PHI_STRICT.
            redis.RedisError:  On Redis connectivity or protocol errors.
        """
        self._assert_not_phi(request)
        key = self._make_key(request)
        deleted = self._redis.delete(key)
        if deleted:
            logger.info(
                "cache: INVALIDATED key=%s request=%s",
                key,
                request.request_id,
            )
        else:
            logger.debug(
                "cache: INVALIDATE no-op (key not found) key=%s request=%s",
                key,
                request.request_id,
            )
        return bool(deleted)

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    def _make_key(self, request: InferenceRequest) -> str:
        """
        Build a stable, opaque cache key for *request*.

        Format::

            <prefix>:<model_id>:<sensitivity>:<sha256(canonical_payload)>

        The payload is serialised with sorted keys and no extra whitespace
        (canonical JSON) before hashing, ensuring key stability regardless of
        Python dict insertion order.
        """
        canonical_payload = json.dumps(
            request.payload, sort_keys=True, separators=(",", ":")
        )
        payload_hash = hashlib.sha256(canonical_payload.encode()).hexdigest()
        return (
            f"{self._key_prefix}"
            f":{request.model_id}"
            f":{request.data_sensitivity.value}"
            f":{payload_hash}"
        )

    # ------------------------------------------------------------------
    # PHI guard
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_not_phi(request: InferenceRequest) -> None:
        if request.data_sensitivity in _BANNED_TIERS:
            raise PHICacheViolation(request.data_sensitivity)
