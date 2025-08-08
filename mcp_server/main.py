import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal

from mcp_server.protocol_manager.protocol_manager import ProtocolManager
from mcp_server.orchestration_engine.orchestration_engine import OrchestrationEngine
from mcp_server.resource_manager.resource_manager import ResourceManager

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialization ---
logger.info("Initializing mCP Server...")
protocol_manager = ProtocolManager()
resource_manager = ResourceManager()
orchestration_engine = OrchestrationEngine(protocol_manager, resource_manager)
logger.info("mCP Server initialized.")

app = FastAPI(
    title="mCP Server",
    description="A server for orchestrating cognitive protocols based on the SIM-ONE framework.",
    version="0.2.0",
)

# --- Data Models ---
class WorkflowRequest(BaseModel):
    protocol_names: List[str]
    initial_data: Dict[str, Any]
    coordination_mode: Literal['Sequential', 'Parallel'] = Field(
        'Sequential',
        description="The coordination mode for executing the protocols."
    )

# --- API Endpoints ---
@app.post("/execute", tags=["Orchestration"])
async def execute_workflow(request: WorkflowRequest) -> Dict[str, Any]:
    """
    Executes a cognitive workflow.
    """
    logger.info(f"Received workflow request: {request.protocol_names} (Mode: {request.coordination_mode})")
    result = await orchestration_engine.execute_workflow(
        request.protocol_names,
        request.initial_data,
        request.coordination_mode
    )
    logger.info(f"Workflow finished.")
    return result

@app.get("/protocols", tags=["Protocols"])
async def list_protocols() -> Dict[str, Any]:
    """
    Lists the available protocols.
    """
    return protocol_manager.protocols

@app.get("/", tags=["Status"])
async def root():
    return {"message": "mCP Server is running."}
