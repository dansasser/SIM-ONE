import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Literal

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

    async def execute_workflow(
        self,
        protocol_names: List[str],
        initial_data: Dict[str, Any],
        coordination_mode: Literal['Sequential', 'Parallel'] = 'Sequential'
    ) -> Dict[str, Any]:
        logger.info(f"Executing workflow in {coordination_mode} mode: {' -> '.join(protocol_names)}")
        if coordination_mode == 'Sequential':
            return await self._execute_sequential(protocol_names, initial_data)
        elif coordination_mode == 'Parallel':
            return await self._execute_parallel(protocol_names, initial_data)
        else:
            return {"error": f"Unknown coordination mode: {coordination_mode}"}

    def _execute_protocol(self, protocol_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        protocol = self.protocol_manager.get_protocol(protocol_name)
        if not protocol:
            raise ValueError(f"Protocol '{protocol_name}' not found.")

        with self.resource_manager.profile(protocol_name) as metrics:
            # Pass the entire data context to the protocol
            result = protocol.execute(data)

        return {"result": result, "metrics": metrics}

    async def _execute_sequential(self, protocol_names: List[str], initial_data: Dict[str, Any]) -> Dict[str, Any]:
        workflow_context = initial_data.copy()
        final_results = {"results": {}, "resource_usage": {}}
        loop = asyncio.get_running_loop()

        for protocol_name in protocol_names:
            logger.info(f"  - Sequentially executing protocol: {protocol_name}")
            try:
                res = await loop.run_in_executor(
                    self.executor, self._execute_protocol, protocol_name, workflow_context
                )

                # Add the result of this protocol to the context for the next one
                workflow_context[protocol_name] = res["result"]

                # Store the results for the final output
                final_results["results"][protocol_name] = res["result"]
                final_results["resource_usage"][protocol_name] = res["metrics"]
            except Exception as e:
                logger.error(f"    - Error executing protocol '{protocol_name}': {e}")
                final_results['error'] = f"Error in protocol '{protocol_name}': {e}"
                return final_results

        return final_results

    async def _execute_parallel(self, protocol_names: List[str], initial_data: Dict[str, Any]) -> Dict[str, Any]:
        workflow_results = {"results": {}, "resource_usage": {}}
        loop = asyncio.get_running_loop()

        tasks = [
            loop.run_in_executor(self.executor, self._execute_protocol, name, initial_data)
            for name in protocol_names
        ]
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
