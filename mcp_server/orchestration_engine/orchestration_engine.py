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
        # Using a ThreadPoolExecutor to run synchronous protocol code in a non-blocking way
        self.executor = ThreadPoolExecutor()

    async def execute_workflow(
        self,
        protocol_names: List[str],
        initial_data: Dict[str, Any],
        coordination_mode: Literal['Sequential', 'Parallel'] = 'Sequential'
    ) -> Dict[str, Any]:
        """
        Executes a workflow of protocols.

        Args:
            protocol_names: A list of protocol names to execute.
            initial_data: The initial input data for the workflow.
            coordination_mode: The mode of execution ('Sequential' or 'Parallel').

        Returns:
            A dictionary containing the final result.
        """
        logger.info(f"Executing workflow in {coordination_mode} mode: {' -> '.join(protocol_names)}")

        if coordination_mode == 'Sequential':
            return await self._execute_sequential(protocol_names, initial_data)
        elif coordination_mode == 'Parallel':
            return await self._execute_parallel(protocol_names, initial_data)
        else:
            return {"error": f"Unknown coordination mode: {coordination_mode}"}

    def _execute_protocol(self, protocol_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for executing a single protocol with resource profiling.
        """
        protocol = self.protocol_manager.get_protocol(protocol_name)
        if not protocol:
            raise ValueError(f"Protocol '{protocol_name}' not found.")

        with self.resource_manager.profile(protocol_name) as metrics:
            if protocol_name == "ReasoningAndExplanationProtocol":
                facts = data.get("facts", [])
                rules = data.get("rules", [])
                result = protocol.execute(facts, rules)
            else:
                # This will be used by the VVP protocol later
                result = protocol.execute(data)

        return {"result": result, "metrics": metrics}

    async def _execute_sequential(self, protocol_names: List[str], initial_data: Dict[str, Any]) -> Dict[str, Any]:
        current_data = initial_data.copy()
        workflow_results = {"results": {}, "resource_usage": {}}
        loop = asyncio.get_running_loop()

        for protocol_name in protocol_names:
            logger.info(f"  - Sequentially executing protocol: {protocol_name}")
            try:
                # Run the synchronous execute method in a thread pool to avoid blocking
                res = await loop.run_in_executor(
                    self.executor, self._execute_protocol, protocol_name, current_data
                )
                workflow_results["results"][protocol_name] = res["result"]
                workflow_results["resource_usage"][protocol_name] = res["metrics"]
                current_data.update(res["result"])
            except Exception as e:
                logger.error(f"    - Error executing protocol '{protocol_name}': {e}")
                workflow_results['error'] = f"Error in protocol '{protocol_name}': {e}"
                return workflow_results

        return workflow_results

    async def _execute_parallel(self, protocol_names: List[str], initial_data: Dict[str, Any]) -> Dict[str, Any]:
        workflow_results = {"results": {}, "resource_usage": {}}
        loop = asyncio.get_running_loop()

        # Create a task for each protocol execution
        tasks = [
            loop.run_in_executor(self.executor, self._execute_protocol, name, initial_data)
            for name in protocol_names
        ]

        # Wait for all tasks to complete
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
