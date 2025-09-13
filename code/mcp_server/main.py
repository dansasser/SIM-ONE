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
from mcp_server.middleware.auth_middleware import get_api_key, RoleChecker
from mcp_server.middleware.security_headers_middleware import SecurityHeadersMiddleware
from mcp_server.security.advanced_validator import advanced_validate_input
from mcp_server.security.error_handler import add_exception_handlers
from mcp_server.memory_manager.memory_consolidation import MemoryConsolidationEngine
from mcp_server.database.database_manager import initialize_database_manager, db_manager
from mcp_server.database.performance_monitor import get_performance_monitor
from mcp_server.database.connection_monitor import get_connection_monitor
from mcp_server.database.search_engine import get_search_engine
from mcp_server.database.vector_search import get_vector_engine
from mcp_server.database.schema_manager import get_schema_manager
from mcp_server.database.analytics_engine import get_analytics_engine
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
    
    # Initialize advanced database features
    try:
        # Fetch managers concurrently
        search_engine_coro = get_search_engine()
        vector_engine_coro = get_vector_engine()
        schema_manager_coro = get_schema_manager()
        connection_monitor_coro = get_connection_monitor()

        search_engine, vector_engine, schema_manager, connection_monitor = await asyncio.gather(
            search_engine_coro, vector_engine_coro, schema_manager_coro, connection_monitor_coro
        )

        # Initialize services concurrently
        await asyncio.gather(
            search_engine.initialize_search_indexes(),
            vector_engine.initialize_vector_storage(),
            schema_manager.initialize(),
            connection_monitor.start_monitoring(),
        )

        logger.info("Advanced database features initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize advanced database features: {e}")
    
    # Start background tasks
    asyncio.create_task(run_memory_consolidation())

@app.on_event("shutdown")
async def shutdown_event():
    # Stop monitoring services
    try:
        connection_monitor = await get_connection_monitor()
        await connection_monitor.stop_monitoring()
        logger.info("Connection monitoring stopped")
    except Exception as e:
        logger.error(f"Error stopping connection monitoring: {e}")
    
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
    governance_summary: Optional[Dict[str, Any]] = None

# --- API Endpoints ---
from mcp_server.middleware.auth_middleware import get_api_key, RoleChecker

@app.post("/execute", response_model=WorkflowResponse, tags=["Orchestration"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
@limiter.limit(settings.RATE_LIMIT_EXECUTE)
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

    # Prepare summarized governance diagnostics (non-sensitive)
    governance = final_context.get("governance") or {}
    quality = governance.get("quality") or {}
    quality_scores = {}
    if isinstance(quality, dict):
        for p, q in quality.items():
            if isinstance(q, dict) and "quality_score" in q:
                quality_scores[p] = q.get("quality_score")
    coherence = governance.get("coherence")
    is_coherent = None
    if isinstance(coherence, dict) and "is_coherent" in coherence:
        is_coherent = bool(coherence.get("is_coherent"))
    governance_summary = {"quality_scores": quality_scores, "is_coherent": is_coherent}

    if "error" in final_context:
        # Handle errors returned from the orchestration engine
        error_detail = final_context.pop("error")
        # We can choose to log this or handle it as a specific type of response
        # For now, we'll return it in the response, but not as a 500 error
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        # Audit summary on error
        logging.getLogger("audit").info({
            "event": "execute_completed",
            "user_id": user.get("user_id"),
            "session_id": sid,
            "governance_summary": governance_summary,
            "error": error_detail
        })
        return WorkflowResponse(session_id=sid, results=final_context, error=error_detail, execution_time_ms=execution_time_ms, governance_summary=governance_summary)

    if session_manager.redis_client:
        history = session_manager.get_history(sid) or []
        history.append({"user_request": workflow_request.model_dump(exclude={'session_id'}), "server_response": final_context})
        session_manager.update_history(sid, history)

    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    # Audit successful execution summary
    logging.getLogger("audit").info({
        "event": "execute_completed",
        "user_id": user.get("user_id"),
        "session_id": sid,
        "governance_summary": governance_summary,
        "error": None
    })
    return WorkflowResponse(session_id=sid, results=final_context, error=None, execution_time_ms=execution_time_ms, governance_summary=governance_summary)

# ... (other endpoints are the same) ...
@app.get("/", tags=["Status"])
async def root(): return {"message": "mCP Server is running."}

@app.get("/protocols", tags=["Protocols"], dependencies=[Depends(RoleChecker(["admin", "user", "read-only"]))])
@limiter.limit(settings.RATE_LIMIT_PROTOCOLS)
async def list_protocols(): return protocol_manager.protocols

@app.get("/templates", tags=["Workflows"], dependencies=[Depends(RoleChecker(["admin", "user", "read-only"]))])
@limiter.limit(settings.RATE_LIMIT_TEMPLATES)
async def list_templates(): return workflow_template_manager.templates

@app.get("/session/{session_id}", tags=["Sessions"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
@limiter.limit(settings.RATE_LIMIT_SESSION)
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
@limiter.limit(settings.RATE_LIMIT_METRICS)
async def get_metrics():
    """
    Returns a collection of system and application metrics.
    Protected and only accessible by users with the 'admin' role.
    """
    collector = MetricsCollector()
    return collector.get_all_metrics()

# --- Admin: API Key Lifecycle ---
from pydantic import BaseModel

class CreateApiKeyRequest(BaseModel):
    api_key: str
    role: Literal['admin','user','read-only'] = 'user'
    user_id: Optional[str] = None

@app.get("/admin/api-keys", tags=["Admin"], dependencies=[Depends(RoleChecker(["admin"]))])
@limiter.limit("20/minute")
async def list_api_keys():
    from mcp_server.security.key_manager import load_api_keys
    # Return without hashes for safety
    keys = load_api_keys()
    return [{"role": k.get("role"), "user_id": k.get("user_id") } for k in keys]

@app.post("/admin/api-keys", tags=["Admin"], dependencies=[Depends(RoleChecker(["admin"]))])
@limiter.limit("10/minute")
async def create_api_key(req: CreateApiKeyRequest):
    from mcp_server.security import key_manager
    user_id = req.user_id or f"user_{hash(req.api_key) & 0xffffffff:08x}"  # stable synthetic id
    key_manager.add_api_key(req.api_key, req.role, user_id)
    return {"success": True, "role": req.role, "user_id": user_id}

@app.delete("/admin/api-keys/{user_id}", tags=["Admin"], dependencies=[Depends(RoleChecker(["admin"]))])
@limiter.limit("10/minute")
async def delete_api_key(user_id: str):
    from mcp_server.security.key_manager import remove_api_key_by_user_id
    removed = remove_api_key_by_user_id(user_id)
    if not removed:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"success": True, "user_id": user_id}

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

# === Advanced Database Features API Endpoints ===

# Performance Monitoring Endpoints
@app.get("/database/performance/summary", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def get_performance_summary(time_window_hours: int = 24):
    """Get database performance summary over a time window."""
    try:
        performance_monitor = await get_performance_monitor()
        summary = performance_monitor.get_performance_summary(time_window_hours)
        return {"success": True, "performance_summary": summary}
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/performance/slow-queries", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin"]))])
async def get_slow_queries_report(limit: int = 10):
    """Get report of slowest database queries."""
    try:
        performance_monitor = await get_performance_monitor()
        slow_queries = performance_monitor.get_slow_queries_report(limit)
        return {"success": True, "slow_queries": slow_queries}
    except Exception as e:
        logger.error(f"Failed to get slow queries report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/performance/recommendations", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin"]))])
async def get_optimization_recommendations():
    """Get database optimization recommendations based on performance analysis."""
    try:
        performance_monitor = await get_performance_monitor()
        recommendations = performance_monitor.get_optimization_recommendations()
        return {"success": True, "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Failed to get optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Connection Monitoring Endpoints
@app.get("/database/connections/status", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def get_connection_status():
    """Get current database connection status and metrics."""
    try:
        connection_monitor = await get_connection_monitor()
        metrics = connection_monitor.get_current_metrics()
        return {"success": True, "connection_metrics": metrics}
    except Exception as e:
        logger.error(f"Failed to get connection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/connections/summary", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin"]))])
async def get_connection_summary(time_window_minutes: int = 60):
    """Get connection performance summary over a time window."""
    try:
        connection_monitor = await get_connection_monitor()
        summary = connection_monitor.get_metrics_summary(time_window_minutes)
        return {"success": True, "connection_summary": summary}
    except Exception as e:
        logger.error(f"Failed to get connection summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/connections/health-test", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin"]))])
async def run_connection_health_test():
    """Run comprehensive database connection health test."""
    try:
        connection_monitor = await get_connection_monitor()
        test_results = await connection_monitor.run_connection_health_test()
        return {"success": True, "health_test": test_results}
    except Exception as e:
        logger.error(f"Failed to run health test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/connections/optimization", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin"]))])
async def get_pool_optimization_analysis():
    """Get connection pool optimization recommendations."""
    try:
        connection_monitor = await get_connection_monitor()
        analysis = await connection_monitor.optimize_pool_configuration()
        return {"success": True, "optimization_analysis": analysis}
    except Exception as e:
        logger.error(f"Failed to get optimization analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Search Endpoints
@app.post("/database/search/memories", tags=["Memory Search"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def search_memories_advanced(request: Request):
    """Perform advanced memory search with multiple filters and ranking."""
    try:
        data = await request.json()
        query = data.get('query', '')
        session_id = data.get('session_id')
        entity_filter = data.get('entity_filter')
        memory_type_filter = data.get('memory_type_filter')
        min_salience = data.get('min_salience', 0.0)
        limit = data.get('limit', 20)
        offset = data.get('offset', 0)
        
        search_engine = await get_search_engine()
        results = await search_engine.search_memories(
            query=query,
            session_id=session_id,
            entity_filter=entity_filter,
            memory_type_filter=memory_type_filter,
            min_salience=min_salience,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "results": [
                {
                    "memory_id": r.memory_id,
                    "content": r.content,
                    "entity_name": r.entity_name,
                    "relevance_score": r.relevance_score,
                    "match_type": r.match_type,
                    "matched_terms": r.matched_terms,
                    "emotional_salience": r.emotional_salience,
                    "timestamp": r.timestamp.isoformat()
                } for r in results
            ],
            "total_results": len(results),
            "query": query
        }
    except Exception as e:
        logger.error(f"Advanced search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/search/suggestions", tags=["Memory Search"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def get_search_suggestions(partial_query: str, limit: int = 10):
    """Get search suggestions based on partial query."""
    try:
        search_engine = await get_search_engine()
        suggestions = await search_engine.get_search_suggestions(partial_query, limit)
        return {"success": True, "suggestions": suggestions}
    except Exception as e:
        logger.error(f"Failed to get search suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vector Similarity Search Endpoints
@app.post("/database/search/similarity", tags=["Memory Search"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def similarity_search_memories(request: Request):
    """Perform similarity-based memory search using vector embeddings."""
    try:
        data = await request.json()
        query_text = data.get('query_text', '')
        session_id = data.get('session_id')
        top_k = data.get('top_k', 10)
        similarity_threshold = data.get('similarity_threshold', 0.7)
        
        vector_engine = await get_vector_engine()
        results = await vector_engine.similarity_search(
            query_text=query_text,
            session_id=session_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        return {
            "success": True,
            "results": [
                {
                    "memory_id": r.memory_id,
                    "content": r.content,
                    "entity_name": r.entity_name,
                    "similarity_score": r.similarity_score,
                    "emotional_salience": r.emotional_salience,
                    "timestamp": r.timestamp.isoformat()
                } for r in results
            ],
            "total_results": len(results),
            "query_text": query_text
        }
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/vectors/rebuild", tags=["Memory Search"], dependencies=[Depends(RoleChecker(["admin"]))])
async def rebuild_memory_embeddings(session_id: Optional[str] = None):
    """Rebuild vector embeddings for all memories."""
    try:
        vector_engine = await get_vector_engine()
        result = await vector_engine.rebuild_embeddings(session_id)
        return {"success": True, "rebuild_result": result}
    except Exception as e:
        logger.error(f"Failed to rebuild embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/vectors/statistics", tags=["Memory Search"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def get_vector_statistics():
    """Get vector storage and similarity search statistics."""
    try:
        vector_engine = await get_vector_engine()
        stats = vector_engine.get_vector_statistics()
        return {"success": True, "vector_statistics": stats}
    except Exception as e:
        logger.error(f"Failed to get vector statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Schema Management Endpoints
@app.get("/database/schema/status", tags=["Database Management"], dependencies=[Depends(RoleChecker(["admin"]))])
async def get_schema_status():
    """Get current database schema status and migration information."""
    try:
        schema_manager = await get_schema_manager()
        status = schema_manager.get_schema_status()
        return {"success": True, "schema_status": status}
    except Exception as e:
        logger.error(f"Failed to get schema status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/schema/migrate", tags=["Database Management"], dependencies=[Depends(RoleChecker(["admin"]))])
async def migrate_database_to_latest():
    """Execute all pending migrations to bring database to latest version."""
    try:
        schema_manager = await get_schema_manager()
        result = await schema_manager.migrate_to_latest()
        return {"success": True, "migration_result": result}
    except Exception as e:
        logger.error(f"Failed to migrate database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/schema/rollback/{target_version}", tags=["Database Management"], dependencies=[Depends(RoleChecker(["admin"]))])
async def rollback_database_schema(target_version: str):
    """Rollback database schema to a specific version. USE WITH EXTREME CAUTION."""
    try:
        schema_manager = await get_schema_manager()
        result = await schema_manager.rollback_migration(target_version)
        return {"success": True, "rollback_result": result}
    except Exception as e:
        logger.error(f"Failed to rollback database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Endpoints
@app.get("/database/analytics/report", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin"]))])
async def get_comprehensive_analytics_report(session_id: Optional[str] = None):
    """Generate comprehensive database analytics report."""
    try:
        analytics_engine = await get_analytics_engine()
        report = await analytics_engine.generate_analytics_report(session_id)
        return {"success": True, "analytics_report": report}
    except Exception as e:
        logger.error(f"Failed to generate analytics report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/analytics/memory-patterns", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin", "user"]))])
async def analyze_memory_patterns(session_id: Optional[str] = None, time_window_hours: int = 24):
    """Analyze memory storage patterns and usage statistics."""
    try:
        analytics_engine = await get_analytics_engine()
        patterns = await analytics_engine.analyze_memory_patterns(session_id, time_window_hours)
        return {
            "success": True,
            "memory_patterns": {
                "total_memories": patterns.total_memories,
                "memories_by_entity": patterns.memories_by_entity,
                "memories_by_session": patterns.memories_by_session,
                "memories_by_protocol": patterns.memories_by_protocol,
                "emotional_salience_distribution": patterns.emotional_salience_distribution,
                "memory_type_distribution": patterns.memory_type_distribution,
                "average_memory_length": patterns.average_memory_length,
                "recent_activity": patterns.recent_activity
            }
        }
    except Exception as e:
        logger.error(f"Failed to analyze memory patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/analytics/growth-trend", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin"]))])
async def get_memory_growth_trend(days: int = 30):
    """Analyze memory growth trends over time."""
    try:
        analytics_engine = await get_analytics_engine()
        trend = await analytics_engine.get_memory_growth_trend(days)
        return {"success": True, "growth_trend": trend}
    except Exception as e:
        logger.error(f"Failed to get growth trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/analytics/session-patterns", tags=["Database Analytics"], dependencies=[Depends(RoleChecker(["admin"]))])
async def analyze_session_patterns():
    """Analyze session-based memory patterns and usage."""
    try:
        analytics_engine = await get_analytics_engine()
        patterns = await analytics_engine.analyze_session_patterns()
        return {"success": True, "session_patterns": patterns}
    except Exception as e:
        logger.error(f"Failed to analyze session patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))
