import logging
import time
from fastapi import FastAPI, Depends, Request, HTTPException
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
from mcp_server.middleware.security_headers_middleware import SecurityHeadersMiddleware
from mcp_server.security.advanced_validator import advanced_validate_input
from mcp_server.security.error_handler import add_exception_handlers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address)

# --- App Initialization ---
app = FastAPI(title="mCP Server", version=settings.APP_VERSION)
app.state.limiter = limiter
add_exception_handlers(app) # Add custom exception handlers
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
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
from mcp_server.middleware.auth_middleware import get_api_key, RoleChecker

@app.post("/execute", response_model=WorkflowResponse, tags=["Orchestration"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
@limiter.limit("20/minute") # Increased limit slightly
async def execute_workflow(
    request: Request,
    workflow_request: WorkflowRequest,
    user: dict = Depends(get_api_key),
) -> WorkflowResponse:
    advanced_validate_input(workflow_request.model_dump())
    start_time = time.time()
    sid = workflow_request.session_id or session_manager.create_session(user_id=user["user_id"])
    workflow_context = workflow_request.initial_data.copy()
    workflow_context['session_id'] = sid
    workflow_context = workflow_request.initial_data.copy()
    workflow_context['session_id'] = sid

    workflow_def = []
    if workflow_request.template_name:
        template = workflow_template_manager.get_template(workflow_request.template_name)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{workflow_request.template_name}' not found.")

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
        raise HTTPException(status_code=400, detail="Either 'template_name' or 'protocol_names' must be provided.")

    final_context = await orchestration_engine.execute_workflow(workflow_def, workflow_context)

    if "error" in final_context:
        # Handle errors returned from the orchestration engine
        error_detail = final_context.pop("error")
        # We can choose to log this or handle it as a specific type of response
        # For now, we'll return it in the response, but not as a 500 error
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        return WorkflowResponse(session_id=sid, results=final_context, error=error_detail, execution_time_ms=execution_time_ms)

    if session_manager.redis_client:
        history = session_manager.get_history(sid) or []
        history.append({"user_request": workflow_request.model_dump(exclude={'session_id'}), "server_response": final_context})
        session_manager.update_history(sid, history)

    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    return WorkflowResponse(session_id=sid, results=final_context, error=None, execution_time_ms=execution_time_ms)

# ... (other endpoints are the same) ...
@app.get("/", tags=["Status"])
async def root(): return {"message": "mCP Server is running."}

@app.get("/protocols", tags=["Protocols"], dependencies=[Depends(RoleChecker(["admin", "user", "read-only"]))])
async def list_protocols(): return protocol_manager.protocols

@app.get("/templates", tags=["Workflows"], dependencies=[Depends(RoleChecker(["admin", "user", "read-only"]))])
async def list_templates(): return workflow_template_manager.templates

@app.get("/session/{session_id}", tags=["Sessions"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def get_session_history(session_id: str, user: dict = Depends(get_api_key)):
    session_owner = session_manager.get_session_owner(session_id)

    if user["role"] != "admin" and (not session_owner or user["user_id"] != session_owner):
        # To avoid leaking information about which sessions exist, we can return 403
        # for both "not found" and "not authorized" for non-admin users.
        # Admins will get a 404 if the session truly doesn't exist.
        if user["role"] == "admin" and not session_owner:
            raise HTTPException(status_code=404, detail="Session not found.")
        raise HTTPException(status_code=403, detail="Not authorized to view this session.")

    history = session_manager.get_history(session_id)
    return {"session_id": session_id, "history": history or []}
