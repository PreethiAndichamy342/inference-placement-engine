"""
Demo server registry for local development.

Defines three simulated cloud environments, each backed by an Ollama instance
serving tinyllama. Start all three with:

    ./scripts/start_demo.sh

Ports:
    11434  aws-sim   (simulates AWS)
    11435  gcp-sim   (simulates GCP)
    11436  on-prem   (simulates on-premises)

Usage::

    from src.demo_servers import servers, adapters

    # servers: list[CloudServer]   — pass to PlacementRouter
    # adapters: dict[str, OnPremAdapter] — keyed by server_id
"""

from src.clouds.on_prem import OnPremAdapter
from src.engine.models import CloudEnv, CloudServer, DataSensitivity

_MODEL = "tinyllama:latest"

# ---------------------------------------------------------------------------
# CloudServer definitions
# ---------------------------------------------------------------------------

aws_server = CloudServer(
    server_id="aws-sim",
    cloud_env=CloudEnv.AWS,
    region="us-east-1",
    endpoint="http://localhost:11434",
    supported_models={_MODEL},
    max_sensitivity=DataSensitivity.INTERNAL,
    has_baa=False,
    gpu_count=0,
    cost_per_token=0.0002,
    tags={"env": "demo", "provider": "aws"},
)

gcp_server = CloudServer(
    server_id="gcp-sim",
    cloud_env=CloudEnv.GCP,
    region="us-central1",
    endpoint="http://localhost:11435",
    supported_models={_MODEL},
    max_sensitivity=DataSensitivity.INTERNAL,
    has_baa=False,
    gpu_count=0,
    cost_per_token=0.00018,
    tags={"env": "demo", "provider": "gcp"},
)

on_prem_server = CloudServer(
    server_id="on-prem",
    cloud_env=CloudEnv.ON_PREM,
    region="local",
    endpoint="http://localhost:11436",
    supported_models={_MODEL},
    max_sensitivity=DataSensitivity.PHI_STRICT,
    has_baa=True,
    gpu_count=0,
    cost_per_token=0.0,
    tags={"env": "demo", "provider": "on_prem"},
)

# Ordered list ready to pass to PlacementRouter
servers: list[CloudServer] = [aws_server, gcp_server, on_prem_server]

# ---------------------------------------------------------------------------
# OnPremAdapter instances (one per simulated environment)
# ---------------------------------------------------------------------------

aws_adapter = OnPremAdapter(
    base_url="http://localhost:11434",
    model_id=_MODEL,
    server_id="aws-sim",
)

gcp_adapter = OnPremAdapter(
    base_url="http://localhost:11435",
    model_id=_MODEL,
    server_id="gcp-sim",
)

on_prem_adapter = OnPremAdapter(
    base_url="http://localhost:11436",
    model_id=_MODEL,
    server_id="on-prem",
)

# Keyed by server_id for easy lookup: adapters[decision.selected_server.server_id]
adapters: dict[str, OnPremAdapter] = {
    "aws-sim": aws_adapter,
    "gcp-sim": gcp_adapter,
    "on-prem": on_prem_adapter,
}
