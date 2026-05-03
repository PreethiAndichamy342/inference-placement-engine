import pytest
from unittest.mock import MagicMock

from src.cache.cache import InferenceCache, PHICacheViolation
from src.engine.models import DataSensitivity, InferenceRequest


def _cache(redis=None):
    return InferenceCache(redis or MagicMock())


def _req(**kwargs):
    defaults = dict(model_id="tinyllama:latest", payload={"prompt": "test"}, tenant_id="t")
    defaults.update(kwargs)
    return InferenceRequest(**defaults)


def test_phi_raises_cache_violation_on_set():
    """set() must raise PHICacheViolation before touching Redis for PHI requests."""
    cache = _cache()
    request = _req(data_sensitivity=DataSensitivity.PHI)

    with pytest.raises(PHICacheViolation):
        cache.set(request, {"result": "data"})

    # Redis must never be called — PHI data must not be written
    cache._redis.setex.assert_not_called()


def test_cache_miss_returns_none():
    """get() must return None when the key is absent in Redis."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    cache = _cache(redis_mock)

    result = cache.get(_req())

    assert result is None


def test_cache_key_stable_for_same_payload():
    """Keys for equivalent payloads must match regardless of dict insertion order."""
    cache = _cache()

    # Same payload, different request_ids (auto-generated UUIDs)
    key1 = cache._make_key(_req(payload={"a": "1", "b": "2"}))
    key2 = cache._make_key(_req(payload={"b": "2", "a": "1"}))

    assert key1 == key2
