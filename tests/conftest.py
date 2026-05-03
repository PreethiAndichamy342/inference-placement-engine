import pytest
from src.engine.models import (
    CloudEnv,
    CloudServer,
    DataSensitivity,
    InferenceRequest,
    ServerStatus,
)


@pytest.fixture
def mock_server():
    """Healthy AWS server cleared for INTERNAL data, no BAA."""
    return CloudServer(
        server_id="test-server",
        cloud_env=CloudEnv.AWS,
        region="us-east-1",
        endpoint="http://localhost:11434",
        supported_models={"tinyllama:latest"},
        max_sensitivity=DataSensitivity.INTERNAL,
        has_baa=False,
        status=ServerStatus.HEALTHY,
        current_load=0.3,
    )


@pytest.fixture
def mock_request():
    """PUBLIC inference request — no BAA or on-prem requirements."""
    return InferenceRequest(
        model_id="tinyllama:latest",
        payload={"prompt": "What is hypertension?"},
        tenant_id="test-tenant",
        data_sensitivity=DataSensitivity.PUBLIC,
    )
