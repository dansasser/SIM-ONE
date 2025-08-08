import logging
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Literal, AsyncGenerator, Optional

from mcp_server.protocol_manager.protocol_manager import ProtocolManager
from mcp_server.resource_manager.resource_manager import ResourceManager
from mcp_server.memory_manager.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class OrchestrationEngine:
    """
    Orchestrates the execution of cognitive workflows.
    """

    def __init__(self, protocol_manager: ProtocolManager, resource_manager: ResourceManager, memory_manager: MemoryManager):
        self.protocol_manager = protocol_manager
        self.resource_manager = resource_manager
        self.memory_manager = memory_manager
        self.executor = ThreadPoolExecutor()

    async def _batch_fetch_memory(self, session_id: Optional[str]) -> List[Dict[str, Any]]:
        """Performs a single memory fetch for the entire workflow."""
        if not session_id or not self.memory_manager.redis_client:
            return []
        logger.info(f"Performing batch memory pull for session {session_id}")
        return self.memory_manager.get_all_memories(session_id)

    async def execute_structured_workflow(self, workflow: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a complex, structured workflow that can include loops.
        """
        current_context = context.copy()

        # BATCH MEMORY PULL: Fetch all memories once at the start.
        current_context['batch_memory'] = await self._batch_fetch_memory(current_context.get('session_id'))

        for item in workflow:
            if "step" in item:
                protocol_name = item["step"]
                logger.info(f"Executing step: {protocol_name}")
                try:
                    res = await self._execute_protocol(protocol_name, current_context)
                    current_context[protocol_name] = res["result"]
                except Exception as e:
                    logger.error(f"Error in step {protocol_name}: {e}")
                    current_context["error"] = str(e)
                    break

            elif "loop" in item:
                # ... (loop logic is the same) ...
                loop_count = item["loop"]
                loop_steps = item.get("steps", [])
                logger.info(f"Entering loop for {loop_count} iterations.")
                for i in range(loop_count):
                    logger.info(f"  - Loop iteration {i+1}/{loop_count}")
                    current_context = await self.execute_structured_workflow(loop_steps, current_context)
                    if "error" in current_context:
                        logger.error("  - Error in loop, breaking.")
                        break

        return current_context

    async def _execute_protocol(self, protocol_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # ... (same as before) ...
        protocol = self.protocol_manager.get_protocol(protocol_name)
        if not protocol:
            raise ValueError(f"Protocol '{protocol_name}' not found.")

        with self.resource_manager.profile(protocol_name) as metrics:
            execute_method = getattr(protocol, 'execute')
            if inspect.iscoroutinefunction(execute_method):
                result = await execute_method(data)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(self.executor, execute_method, data)

        return {"result": result, "metrics": metrics}

    # ... (old methods for simple workflows remain for compatibility) ...
    async def execute_workflow(self, protocol_names: List[str], initial_data: Dict[str, Any], coordination_mode: Literal['Sequential', 'Parallel'] = 'Sequential') -> Dict[str, Any]:
        # ... (same as before) ...
        workflow = [{"step": name} for name in protocol_names]
        final_context = await self.execute_structured_workflow(workflow, initial_data)

        results = {}
        error = final_context.get("error")
        if not error:
            for protocol_name in protocol_names:
                if protocol_name in final_context:
                    results[protocol_name] = final_context[protocol_name]

        return {"results": results, "resource_usage": {}, "error": error}

    # ... (other helper methods are effectively deprecated) ...
    async def _execute_parallel(self, protocol_names: List[str], initial_data: Dict[str, Any]) -> Dict[str, Any]:
        # ... (same as before) ...
        workflow_results = {"results": {}, "resource_usage": {}}
        tasks = [self._execute_protocol(name, initial_data) for name in protocol_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            protocol_name = protocol_names[i]
            if isinstance(res, Exception):
                workflow_results["results"][protocol_name] = {"error": str(res)}
            else:
                workflow_results["results"][protocol_name] = res["result"]
                workflow_results["resource_usage"][protocol_name] = res["metrics"]
        return workflow_results
