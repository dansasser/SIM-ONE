import logging
import time
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from mcp_server.config import settings
from mcp_server.protocol_manager.protocol_manager import ProtocolManager
from mcp_server.orchestration_engine.orchestration_engine import OrchestrationEngine
from mcp_server.resource_manager.resource_manager import ResourceManager # <-- FIX: Re-add missing import
from mcp_server.session_manager.session_manager import SessionManager
from mcp_server.memory_manager.memory_manager import MemoryManager
from mcp_server.workflow_template_manager.workflow_template_manager import WorkflowTemplateManager
from mcp_server.middleware.auth_middleware import get_api_key
from mcp_server.security.input_validator import validate_input_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address)

# --- App Initialization ---
app = FastAPI(title="mCP Server", version=settings.APP_VERSION)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialization of Managers ---
protocol_manager = ProtocolManager()
resource_manager = ResourceManager()
session_manager = SessionManager()
memory_manager = MemoryManager()
workflow_template_manager = WorkflowTemplateManager()
orchestration_engine = OrchestrationEngine(protocol_manager, resource_manager, memory_manager)

# --- API Models ---
class WorkflowRequest(BaseModel):
    template_name: Optional[str] = Field(None)
    protocol_names: Optional[List[str]] = Field(None)
    coordination_mode: Optional[Literal['Sequential', 'Parallel']] = Field('Sequential')
    initial_data: Dict[str, Any]
    session_id: Optional[str] = Field(None)

class WorkflowResponse(BaseModel):
    session_id: str
    results: Dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: float

# --- API Endpoints ---
@app.post("/execute", response_model=WorkflowResponse, tags=["Orchestration"], dependencies=[Depends(get_api_key)])
@limiter.limit("20/minute") # Increased limit slightly
async def execute_workflow(request: Request, workflow_request: WorkflowRequest) -> WorkflowResponse:
    validate_input_data(workflow_request.initial_data)
    start_time = time.time()
    sid = workflow_request.session_id or session_manager.create_session()
    workflow_context = workflow_request.initial_data.copy()
    workflow_context['session_id'] = sid
    error = None
    final_context = {}
    try:
        workflow_def = []
        if workflow_request.template_name:
            template = workflow_template_manager.get_template(workflow_request.template_name)
            if not template:
                error = f"Template '{workflow_request.template_name}' not found."
            else:
                if "workflow" in template:
                    workflow_def = template["workflow"]
                else:
                    protocols = template.get("protocols", [])
                    if template.get("mode") == "Parallel":
                        workflow_def = [{"parallel": [{"step": name} for name in protocols]}]
                    else:
                        workflow_def = [{"step": name} for name in protocols]
        elif workflow_request.protocol_names:
            if workflow_request.coordination_mode == "Parallel":
                workflow_def = [{"parallel": [{"step": name} for name in workflow_request.protocol_names]}]
            else:
                workflow_def = [{"step": name} for name in workflow_request.protocol_names]
        else:
            error = "Either 'template_name' or 'protocol_names' must be provided."
        if not error:
            final_context = await orchestration_engine.execute_workflow(workflow_def, workflow_context)
            if "error" in final_context:
                error = final_context.pop("error")
    except Exception as e:
        if isinstance(e, HTTPException): raise
        error = f"An unexpected server error occurred: {e}"
    if session_manager.redis_client and not error:
        history = session_manager.get_history(sid) or []
        history.append({"user_request": workflow_request.model_dump(exclude={'session_id'}), "server_response": final_context})
        session_manager.update_history(sid, history)
    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    return WorkflowResponse(session_id=sid, results=final_context, error=error, execution_time_ms=execution_time_ms)

# ... (other endpoints are the same) ...
@app.get("/", tags=["Status"])
async def root(): return {"message": "mCP Server is running."}
@app.get("/protocols", tags=["Protocols"])
async def list_protocols(): return protocol_manager.protocols
@app.get("/templates", tags=["Workflows"])
async def list_templates(): return workflow_template_manager.templates
@app.get("/session/{session_id}", tags=["Sessions"])
async def get_session_history(session_id: str):
    history = session_manager.get_history(session_id)
    if history is None and session_manager.redis_client:
        return {"error": "Session not found."}
    return {"session_id": session_id, "history": history or []}
