import logging
import time
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

from mcp_server.protocol_manager.protocol_manager import ProtocolManager
from mcp_server.orchestration_engine.orchestration_engine import OrchestrationEngine
from mcp_server.resource_manager.resource_manager import ResourceManager
from mcp_server.session_manager.session_manager import SessionManager
from mcp_server.memory_manager.memory_manager import MemoryManager
from mcp_server.workflow_template_manager.workflow_template_manager import WorkflowTemplateManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Initializing mCP Server...")
protocol_manager = ProtocolManager()
resource_manager = ResourceManager()
session_manager = SessionManager()
memory_manager = MemoryManager()
# CRITICAL FIX: Pass the correct manager instances to the engine
orchestration_engine = OrchestrationEngine(protocol_manager, resource_manager, memory_manager)
logger.info("mCP Server initialized.")

app = FastAPI(title="mCP Server", version="0.8.1") # Bump version for bugfix

# ... (rest of the file is the same, I will paste it for completeness) ...

class WorkflowRequest(BaseModel):
    template_name: Optional[str] = Field(None)
    protocol_names: Optional[List[str]] = Field(None)
    coordination_mode: Optional[Literal['Sequential', 'Parallel']] = Field('Sequential')
    initial_data: Dict[str, Any]
    session_id: Optional[str] = Field(None)
    latency_budget_ms: Optional[int] = Field(None)

class WorkflowResponse(BaseModel):
    session_id: str
    results: Dict[str, Any]
    resource_usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float

@app.post("/execute", response_model=WorkflowResponse, tags=["Orchestration"])
async def execute_workflow(request: WorkflowRequest) -> WorkflowResponse:
    start_time = time.time()

    sid = request.session_id or session_manager.create_session()
    workflow_context = request.initial_data.copy()
    workflow_context['session_id'] = sid
    workflow_context['latency_info'] = {
        "budget_ms": request.latency_budget_ms,
        "start_time": start_time
    }

    final_context = {}
    error = None

    try:
        if request.template_name:
            template = workflow_template_manager.get_template(request.template_name)
            if not template:
                error = f"Template '{request.template_name}' not found."
            elif "workflow" in template:
                final_context = await orchestration_engine.execute_structured_workflow(template["workflow"], workflow_context)
            else:
                final_context = await orchestration_engine.execute_workflow(template.get("protocols", []), workflow_context, template.get("mode", "Sequential"))
        elif request.protocol_names:
            final_context = await orchestration_engine.execute_workflow(request.protocol_names, workflow_context, request.coordination_mode or 'Sequential')
        else:
            error = "Either 'template_name' or 'protocol_names' must be provided."

        if "error" in final_context:
            error = final_context.get("error")

    except Exception as e:
        logger.error(f"Critical error during workflow execution: {e}", exc_info=True)
        error = f"An unexpected server error occurred: {e}"

    if session_manager.redis_client and not error:
        history = session_manager.get_history(sid) or []
        history.append({"user_request": request.model_dump(exclude={'session_id'}), "server_response": final_context})
        session_manager.update_history(sid, history)

    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000

    return WorkflowResponse(
        session_id=sid,
        results=final_context,
        error=error,
        execution_time_ms=execution_time_ms
    )

@app.get("/protocols", tags=["Protocols"])
async def list_protocols() -> Dict[str, Any]:
    return protocol_manager.protocols

@app.get("/templates", tags=["Workflows"])
async def list_templates() -> Dict[str, Any]:
    return workflow_template_manager.templates

@app.get("/session/{session_id}", tags=["Sessions"])
async def get_session_history(session_id: str):
    history = session_manager.get_history(session_id)
    if history is None and session_manager.redis_client:
        return {"error": "Session not found."}
    return {"session_id": session_id, "history": history or []}

@app.get("/", tags=["Status"])
async def root():
    return {"message": "mCP Server is running."}
