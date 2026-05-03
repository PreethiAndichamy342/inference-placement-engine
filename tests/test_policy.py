from src.engine.models import (
    CloudEnv,
    CloudServer,
    DataSensitivity,
    InferenceRequest,
)
from src.engine.policy import PolicyEngine


def test_phi_rejected_when_no_baa():
    """PHI request must be rejected if the server lacks a signed BAA."""
    server = CloudServer(
        server_id="aws-no-baa",
        cloud_env=CloudEnv.AWS,
        region="us-east-1",
        endpoint="http://localhost:11434",
        supported_models={"tinyllama:latest"},
        max_sensitivity=DataSensitivity.PHI,
        has_baa=False,
    )
    request = InferenceRequest(
        model_id="tinyllama:latest",
        payload={"prompt": "Patient record"},
        tenant_id="hospital",
        data_sensitivity=DataSensitivity.PHI,
    )

    result = PolicyEngine().evaluate([server], request)

    assert len(result.eligible_servers) == 0
    violations = result.violations_for("aws-no-baa")
    assert any(v.rule_name == "baa_required" for v in violations)


def test_phi_strict_rejected_on_aws():
    """PHI_STRICT data must stay on-prem — even a BAA-holding AWS server is rejected."""
    server = CloudServer(
        server_id="aws-baa",
        cloud_env=CloudEnv.AWS,
        region="us-east-1",
        endpoint="http://localhost:11434",
        supported_models={"tinyllama:latest"},
        max_sensitivity=DataSensitivity.PHI_STRICT,
        has_baa=True,
    )
    request = InferenceRequest(
        model_id="tinyllama:latest",
        payload={"prompt": "42 CFR Part 2 record"},
        tenant_id="hospital",
        data_sensitivity=DataSensitivity.PHI_STRICT,
    )

    result = PolicyEngine().evaluate([server], request)

    assert len(result.eligible_servers) == 0
    violations = result.violations_for("aws-baa")
    assert any(v.rule_name == "on_prem_restriction" for v in violations)


def test_public_passes_on_healthy_server(mock_server, mock_request):
    """A PUBLIC request on a healthy server with sufficient clearance is eligible."""
    result = PolicyEngine().evaluate([mock_server], mock_request)

    assert mock_server in result.eligible_servers
