import logging
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from mcp_server.protocol_manager.protocol_manager import ProtocolManager
from mcp_server.resource_manager.resource_manager import ResourceManager
from mcp_server.memory_manager.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class OrchestrationEngine:
    """
    A simplified, robust engine that executes a structured cognitive workflow.
    """

    def __init__(self, protocol_manager: ProtocolManager, resource_manager: ResourceManager, memory_manager: MemoryManager):
        self.protocol_manager = protocol_manager
        self.resource_manager = resource_manager
        self.memory_manager = memory_manager
        self.executor = ThreadPoolExecutor()

    async def execute_workflow(self, workflow_def: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        The single entry point for executing any workflow.
        """
        # --- Batch Memory Pull ---
        session_id = context.get('session_id')
        if session_id:
            logger.info(f"Performing batch memory pull for session {session_id}")
            loop = asyncio.get_running_loop()
            context['batch_memory'] = await loop.run_in_executor(None, self.memory_manager.get_all_memories, session_id)
        else:
            context['batch_memory'] = []

        return await self._execute_steps(workflow_def, context)

    async def _execute_steps(self, steps: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively executes a list of workflow steps.
        """
        for item in steps:
            if "step" in item:
                protocol_name = item["step"]
                try:
                    res = await self._execute_protocol(protocol_name, context)
                    context[protocol_name] = res
                except Exception as e:
                    context["error"] = f"Error in protocol {protocol_name}: {e}"
                    break

            elif "parallel" in item:
                parallel_steps = item.get("parallel", [])
                tasks = [self._execute_protocol(step["step"], context) for step in parallel_steps]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, res in enumerate(results):
                    protocol_name = parallel_steps[i]["step"]
                    if isinstance(res, Exception):
                        context[protocol_name] = {"error": str(res)}
                    else:
                        context[protocol_name] = res

            elif "loop" in item:
                loop_count = item["loop"]
                loop_steps = item.get("steps", [])
                for i in range(loop_count):
                    context = await self._execute_steps(loop_steps, context)
                    if "error" in context: break
                    if "RevisorProtocol" in context:
                        revised_text = context.get("RevisorProtocol", {}).get("result", {}).get("revised_draft_text")
                        if revised_text:
                            if "DrafterProtocol" not in context: context["DrafterProtocol"] = {}
                            context["DrafterProtocol"]["draft_text"] = revised_text

        return context

    async def _execute_protocol(self, protocol_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        protocol = self.protocol_manager.get_protocol(protocol_name)
        if not protocol: raise ValueError(f"Protocol '{protocol_name}' not found.")

        with self.resource_manager.profile(protocol_name) as metrics:
            execute_method = getattr(protocol, 'execute')
            if inspect.iscoroutinefunction(execute_method):
                result = await execute_method(data)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(self.executor, execute_method, data)

        return {"result": result, "resource_usage": metrics}
