from src.engine.models import (
    CloudEnv,
    CloudServer,
    DataSensitivity,
    InferenceRequest,
    RoutingStrategy,
    ServerStatus,
)
from src.engine.policy import PolicyEngine
from src.engine.router import PlacementRouter


def _router(servers):
    return PlacementRouter(servers=servers, policy_engine=PolicyEngine())


def test_least_loaded_picks_lowest_load():
    """LEAST_LOADED must select the server with the smaller current_load."""
    servers = [
        CloudServer(
            server_id="heavy", cloud_env=CloudEnv.AWS, region="us-east-1",
            endpoint="http://a", supported_models={"m"}, current_load=0.8,
        ),
        CloudServer(
            server_id="light", cloud_env=CloudEnv.AWS, region="us-east-1",
            endpoint="http://b", supported_models={"m"}, current_load=0.2,
        ),
    ]
    request = InferenceRequest(model_id="m", payload={"p": "x"}, tenant_id="t")

    decision = _router(servers).route(request, strategy=RoutingStrategy.LEAST_LOADED)

    assert not decision.rejected
    assert decision.selected_server.server_id == "light"


def test_rejects_when_all_servers_unhealthy(mock_request):
    """Router must return a rejected decision when every candidate is unavailable."""
    servers = [
        CloudServer(
            server_id="s1", cloud_env=CloudEnv.AWS, region="r",
            endpoint="http://a", supported_models={"tinyllama:latest"},
            status=ServerStatus.UNAVAILABLE,
        ),
        CloudServer(
            server_id="s2", cloud_env=CloudEnv.GCP, region="r",
            endpoint="http://b", supported_models={"tinyllama:latest"},
            status=ServerStatus.UNAVAILABLE,
        ),
    ]

    decision = _router(servers).route(mock_request)

    assert decision.rejected
    assert decision.selected_server is None
    assert decision.rejection_reason is not None


def test_phi_strict_routes_to_on_prem_only():
    """PHI_STRICT request must be placed on the on-prem node, never on AWS."""
    servers = [
        CloudServer(
            server_id="aws-node", cloud_env=CloudEnv.AWS, region="us-east-1",
            endpoint="http://a", supported_models={"m"},
            max_sensitivity=DataSensitivity.PHI_STRICT, has_baa=True,
        ),
        CloudServer(
            server_id="on-prem", cloud_env=CloudEnv.ON_PREM, region="local",
            endpoint="http://b", supported_models={"m"},
            max_sensitivity=DataSensitivity.PHI_STRICT, has_baa=True,
        ),
    ]
    request = InferenceRequest(
        model_id="m", payload={"p": "x"}, tenant_id="hospital",
        data_sensitivity=DataSensitivity.PHI_STRICT,
    )

    decision = _router(servers).route(request, strategy=RoutingStrategy.COMPLIANCE_FIRST)

    assert not decision.rejected
    assert decision.selected_server.server_id == "on-prem"
    assert decision.selected_server.cloud_env == CloudEnv.ON_PREM
