import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

from mcp_server.protocol_manager.protocol_manager import ProtocolManager
from mcp_server.orchestration_engine.orchestration_engine import OrchestrationEngine
from mcp_server.resource_manager.resource_manager import ResourceManager
from mcp_server.session_manager.session_manager import SessionManager
from mcp_server.workflow_template_manager.workflow_template_manager import WorkflowTemplateManager

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialization ---
logger.info("Initializing mCP Server...")
protocol_manager = ProtocolManager()
resource_manager = ResourceManager()
session_manager = SessionManager()
workflow_template_manager = WorkflowTemplateManager()
orchestration_engine = OrchestrationEngine(protocol_manager, resource_manager)
logger.info("mCP Server initialized.")

app = FastAPI(
    title="mCP Server",
    description="A server for orchestrating cognitive protocols based on the SIM-ONE framework.",
    version="0.5.0",
)

# --- Data Models ---
class WorkflowRequest(BaseModel):
    # Now, a client can provide either a template name or a direct list of protocols
    template_name: Optional[str] = Field(None, description="The name of the workflow template to use.")
    protocol_names: Optional[List[str]] = Field(None, description="A direct list of protocol names to execute.")
    coordination_mode: Optional[Literal['Sequential', 'Parallel']] = Field(
        'Sequential',
        description="The coordination mode for executing the protocols (if not using a template)."
    )
    initial_data: Dict[str, Any]
    session_id: Optional[str] = Field(None, description="The ID of the conversational session.")

class WorkflowResponse(BaseModel):
    session_id: str
    results: Dict[str, Any]
    resource_usage: Dict[str, Any]
    error: Optional[str] = None

# --- API Endpoints ---
@app.post("/execute", response_model=WorkflowResponse, tags=["Orchestration"])
async def execute_workflow(request: WorkflowRequest) -> WorkflowResponse:
    logger.info(f"Received workflow request for session: {request.session_id}")

    protocols_to_run: List[str] = []
    mode: Literal['Sequential', 'Parallel'] = 'Sequential'

    if request.template_name:
        template = workflow_template_manager.get_template(request.template_name)
        if not template:
            return WorkflowResponse(session_id=request.session_id or "", results={}, resource_usage={}, error=f"Template '{request.template_name}' not found.")
        protocols_to_run = template.get("protocols", [])
        mode = template.get("mode", "Sequential")
    elif request.protocol_names:
        protocols_to_run = request.protocol_names
        mode = request.coordination_mode or 'Sequential'
    else:
        return WorkflowResponse(session_id=request.session_id or "", results={}, resource_usage={}, error="Either 'template_name' or 'protocol_names' must be provided.")

    sid = request.session_id or session_manager.create_session()
    history = session_manager.get_history(sid) or []
    workflow_context = request.initial_data.copy()
    workflow_context['history'] = history

    result = await orchestration_engine.execute_workflow(protocols_to_run, workflow_context, mode)

    current_turn = {"user_request": request.dict(exclude={'session_id'}), "server_response": result}
    history.append(current_turn)
    session_manager.update_history(sid, history)

    return WorkflowResponse(
        session_id=sid,
        results=result.get("results", {}),
        resource_usage=result.get("resource_usage", {}),
        error=result.get("error")
    )

# ... (other endpoints remain the same) ...
@app.get("/protocols", tags=["Protocols"])
async def list_protocols() -> Dict[str, Any]:
    return protocol_manager.protocols

@app.get("/templates", tags=["Workflows"])
async def list_templates() -> Dict[str, Any]:
    return workflow_template_manager.templates

@app.get("/session/{session_id}", tags=["Sessions"])
async def get_session_history(session_id: str):
    history = session_manager.get_history(session_id)
    if history is None:
        return {"error": "Session not found."}
    return {"session_id": session_id, "history": history}

@app.get("/", tags=["Status"])
async def root():
    return {"message": "mCP Server is running."}
