import logging
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Literal, AsyncGenerator

from mcp_server.protocol_manager.protocol_manager import ProtocolManager
from mcp_server.resource_manager.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class OrchestrationEngine:
    """
    Orchestrates the execution of cognitive workflows.
    """

    def __init__(self, protocol_manager: ProtocolManager, resource_manager: ResourceManager):
        self.protocol_manager = protocol_manager
        self.resource_manager = resource_manager
        self.executor = ThreadPoolExecutor()

    async def execute_structured_workflow(self, workflow: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a complex, structured workflow that can include loops.
        """
        current_context = context.copy()

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
                loop_count = item["loop"]
                loop_steps = item.get("steps", [])
                logger.info(f"Entering loop for {loop_count} iterations.")
                for i in range(loop_count):
                    logger.info(f"  - Loop iteration {i+1}/{loop_count}")

                    # The context passed into the loop is the full, current context
                    loop_context = await self.execute_structured_workflow(loop_steps, current_context)

                    if "error" in loop_context:
                        logger.error("  - Error in loop, breaking.")
                        current_context.update(loop_context) # merge error back
                        break

                    # CRITICAL FIX: Update the canonical draft for the next iteration
                    if "RevisorProtocol" in loop_context and "revised_draft_text" in loop_context["RevisorProtocol"]:
                        logger.info("  - Updating context with revised draft for next loop/step.")
                        # Ensure DrafterProtocol key exists
                        if "DrafterProtocol" not in loop_context:
                            loop_context["DrafterProtocol"] = {}
                        # Set the revised text as the new canonical draft
                        loop_context["DrafterProtocol"]["draft_text"] = loop_context["RevisorProtocol"]["revised_draft_text"]

                    # Merge the results of the loop back into the main context
                    current_context.update(loop_context)

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
        # This now just wraps the structured workflow for simple, non-looping cases
        if coordination_mode == 'Parallel':
            return await self._execute_parallel(protocol_names, initial_data)

        # Sequential is a simple structured workflow
        workflow = [{"step": name} for name in protocol_names]
        # We need to adapt the final output to match the old structure for old tests
        final_context = await self.execute_structured_workflow(workflow, initial_data)

        results = {}
        resource_usage = {}
        error = final_context.get("error")
        if not error:
            for protocol_name in protocol_names:
                if protocol_name in final_context:
                    results[protocol_name] = final_context[protocol_name]
                    # This part is tricky as resource usage is not in the context
                    # For now, we'll just return the results part

        return {"results": results, "resource_usage": {}, "error": error}

    async def _execute_parallel(self, protocol_names: List[str], initial_data: Dict[str, Any]) -> Dict[str, Any]:
        # ... (same as before) ...
        workflow_results = {"results": {}, "resource_usage": {}}
        tasks = [self._execute_protocol(name, initial_data) for name in protocol_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            protocol_name = protocol_names[i]
            if isinstance(res, Exception):
                logger.error(f"    - Error executing protocol '{protocol_name}' in parallel: {res}")
                workflow_results["results"][protocol_name] = {"error": str(res)}
            else:
                workflow_results["results"][protocol_name] = res["result"]
                workflow_results["resource_usage"][protocol_name] = res["metrics"]
        return workflow_results

    async def execute_workflow_stream(self, protocol_names: List[str], initial_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        # ... (same as before) ...
        workflow_context = initial_data.copy()
        for protocol_name in protocol_names:
            logger.info(f"  - Streaming execution for protocol: {protocol_name}")
            try:
                res = await self._execute_protocol(protocol_name, workflow_context)
                workflow_context[protocol_name] = res["result"]
                yield {"protocol": protocol_name, "result": res["result"], "resource_usage": res["metrics"]}
            except Exception as e:
                logger.error(f"    - Error executing protocol '{protocol_name}': {e}")
                yield {"protocol": protocol_name, "error": f"Error in protocol '{protocol_name}': {e}"}
                break

    async def _execute_sequential(self, protocol_names: List[str], initial_data: Dict[str, Any]) -> Dict[str, Any]:
        # This is now a wrapper around the streaming executor
        final_results = {"results": {}, "resource_usage": {}}
        async for step_result in self.execute_workflow_stream(protocol_names, initial_data):
            if "error" in step_result:
                final_results["error"] = step_result["error"]
                break
            protocol_name = step_result["protocol"]
            final_results["results"][protocol_name] = step_result["result"]
            final_results["resource_usage"][protocol_name] = step_result["resource_usage"]
        return final_results
