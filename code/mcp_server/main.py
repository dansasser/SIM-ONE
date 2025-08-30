import logging
import time
import asyncio
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
from mcp_server.memory_manager.memory_consolidation import MemoryConsolidationEngine
from mcp_server.database.database_manager import initialize_database_manager, db_manager
from mcp_server.logging_config import setup_logging

setup_logging()
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

# --- Background Tasks ---
async def run_memory_consolidation():
    consolidation_engine = MemoryConsolidationEngine()
    while True:
        logger.info("Starting periodic memory consolidation...")
        try:
            # In a real multi-tenant system, you'd get all active sessions
            # For now, we don't have a central session list, so this is a placeholder.
            # A better approach would be to get sessions from the session_manager.
            all_sessions = session_manager.get_all_session_ids()
            for session_id in all_sessions:
                # remove "session:" prefix
                session_id = session_id.replace("session:", "")
                consolidation_engine.run_consolidation_cycle(session_id)
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}", exc_info=True)

        await asyncio.sleep(3600) # Run every hour

@app.on_event("startup")
async def startup_event():
    # Initialize database system
    await initialize_database_manager()
    logger.info(f"Database initialized: {db_manager.get_database_type().value}")
    
    # Start background tasks
    asyncio.create_task(run_memory_consolidation())

@app.on_event("shutdown")
async def shutdown_event():
    # Close database connections
    await db_manager.close()
    logger.info("Database connections closed")

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

# --- Health Check Endpoints ---
class HealthStatus(BaseModel):
    status: str
    services: Dict[str, str]

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

from mcp_server.metrics.metrics_collector import MetricsCollector

@app.get("/health/detailed", response_model=HealthStatus, tags=["Health"])
async def health_check_detailed():
    services = {
        "database": "ok",
        "redis": "ok"
    }
    status = "ok"

    # Check database connection using new database manager
    try:
        db_health = await db_manager.health_check()
        if db_health["status"] == "healthy":
            services["database"] = f"ok ({db_manager.get_database_type().value})"
        else:
            services["database"] = f"error ({db_health.get('message', 'unknown')})"
            status = "error"
    except Exception as e:
        logger.error(f"Health check failed for database: {e}")
        services["database"] = "error"
        status = "error"

    # Check Redis connection
    try:
        if not session_manager.redis_client or not session_manager.redis_client.ping():
            raise Exception("Failed to ping Redis.")
    except Exception as e:
        logger.error(f"Health check failed for Redis: {e}")
        services["redis"] = "error"
        status = "error"

    return HealthStatus(status=status, services=services)

@app.get("/metrics", tags=["Health"], dependencies=[Depends(RoleChecker(["admin"]))])
async def get_metrics():
    """
    Returns a collection of system and application metrics.
    Protected and only accessible by users with the 'admin' role.
    """
    collector = MetricsCollector()
    return collector.get_all_metrics()

# --- Database Management Endpoints ---
from mcp_server.database.backup_manager import backup_manager

@app.get("/database/info", tags=["Database"], dependencies=[Depends(RoleChecker(["admin"]))])
async def get_database_info():
    """Get database configuration and status information."""
    health = await db_manager.health_check()
    return {
        "database_type": db_manager.get_database_type().value,
        "health": health,
        "features": {
            "connection_pooling": db_manager.is_postgresql(),
            "full_text_search": db_manager.is_postgresql(),
            "json_support": db_manager.is_postgresql(),
            "uuid_support": db_manager.is_postgresql(),
            "async_support": db_manager.is_postgresql()
        }
    }

@app.post("/database/backup", tags=["Database"], dependencies=[Depends(RoleChecker(["admin"]))])
async def create_database_backup():
    """Create a manual database backup."""
    try:
        backup_info = await backup_manager.create_backup("manual")
        return {
            "success": True,
            "backup": backup_info,
            "message": f"Backup created successfully: {backup_info['filename']}"
        }
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

@app.get("/database/backups", tags=["Database"], dependencies=[Depends(RoleChecker(["admin"]))])
async def list_database_backups():
    """List all available database backups."""
    try:
        backups = backup_manager.list_backups()
        return {
            "backups": backups,
            "count": len(backups),
            "total_size": sum(b["size_bytes"] for b in backups)
        }
    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")

@app.post("/database/restore/{backup_filename}", tags=["Database"], dependencies=[Depends(RoleChecker(["admin"]))])
async def restore_database_backup(backup_filename: str):
    """Restore database from a backup file. USE WITH CAUTION - this will replace current data."""
    try:
        restore_info = await backup_manager.restore_backup(backup_filename)
        return {
            "success": True,
            "restore": restore_info,
            "message": f"Database restored successfully from: {backup_filename}"
        }
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

@app.delete("/database/backups/cleanup", tags=["Database"], dependencies=[Depends(RoleChecker(["admin"]))])
async def cleanup_old_backups():
    """Clean up old backups according to retention policy."""
    try:
        cleanup_result = backup_manager.cleanup_old_backups()
        return {
            "success": True,
            "cleanup": cleanup_result,
            "message": f"Cleanup completed: {cleanup_result['removed_count']} backups removed"
        }
    except Exception as e:
        logger.error(f"Backup cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
