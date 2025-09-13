import logging
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from mcp_server.protocol_manager.protocol_manager import ProtocolManager
from mcp_server.resource_manager.resource_manager import ResourceManager
from mcp_server.memory_manager.memory_manager import MemoryManager
from mcp_server.cognitive_governance_engine.governance_orchestrator import GovernanceOrchestrator
from mcp_server.cognitive_governance_engine.error_recovery.recovery_strategist import RecoveryStrategist
from mcp_server.cognitive_governance_engine.error_recovery.resilience_monitor import ResilienceMonitor
from mcp_server.config import settings

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
        self.governance = GovernanceOrchestrator()
        self.recovery = RecoveryStrategist(max_retries=2)
        self.resilience_monitor = ResilienceMonitor()

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
                    # Governance: evaluate this step
                    try:
                        delta = self.governance.evaluate_step(protocol_name, context, res.get("result"), context)
                        if "governance" not in context:
                            context["governance"] = {"quality": {}, "warnings": [], "actions": [], "coherence": None}
                        # Merge minimal diagnostics
                        if isinstance(delta.get("quality"), dict) and delta.get("protocol"):
                            context["governance"]["quality"][delta["protocol"]] = delta["quality"]
                        if delta.get("coherence") is not None:
                            context["governance"]["coherence"] = delta["coherence"]
                        # Enforce coherence policy if configured
                        coherence = context["governance"].get("coherence")
                        if settings.GOV_REQUIRE_COHERENCE and isinstance(coherence, dict) and coherence.get("is_coherent") is False:
                            logger.warning("Governance incoherence detected; policy requires coherence. Attempting single retry for %s", protocol_name)
                            audit = logging.getLogger("audit")
                            audit.info({
                                "event": "governance_incoherence_detected",
                                "protocol": protocol_name,
                                "action": "retry_once",
                            })
                            # Single retry of this step
                            res_retry = await self._execute_protocol(protocol_name, context)
                            context[protocol_name] = res_retry
                            delta2 = self.governance.evaluate_step(protocol_name, context, res_retry.get("result"), context)
                            if isinstance(delta2.get("quality"), dict) and delta2.get("protocol"):
                                context["governance"]["quality"][delta2["protocol"]] = delta2["quality"]
                            if delta2.get("coherence") is not None:
                                context["governance"]["coherence"] = delta2["coherence"]
                            coherence2 = context["governance"].get("coherence")
                            if isinstance(coherence2, dict) and coherence2.get("is_coherent") is False:
                                # Abort according to policy
                                msg = "Incoherent workflow output; aborted by governance policy"
                                audit.info({
                                    "event": "governance_abort",
                                    "protocol": protocol_name,
                                    "reason": "incoherence_after_retry"
                                })
                                context["error"] = msg
                                break
                    except Exception as ge:
                        logger.warning(f"Governance evaluation failed for {protocol_name}: {ge}")
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
                        # Governance: evaluate each parallel result
                        try:
                            delta = self.governance.evaluate_step(protocol_name, context, res.get("result"), context)
                            if "governance" not in context:
                                context["governance"] = {"quality": {}, "warnings": [], "actions": [], "coherence": None}
                            if isinstance(delta.get("quality"), dict) and delta.get("protocol"):
                                context["governance"]["quality"][delta["protocol"]] = delta["quality"]
                            if delta.get("coherence") is not None:
                                context["governance"]["coherence"] = delta["coherence"]
                            # For parallel, if policy requires coherence and it's false, annotate and set error
                            coherence = context["governance"].get("coherence")
                            if settings.GOV_REQUIRE_COHERENCE and isinstance(coherence, dict) and coherence.get("is_coherent") is False:
                                logging.getLogger("audit").info({
                                    "event": "governance_abort",
                                    "protocol": protocol_name,
                                    "reason": "incoherence_parallel"
                                })
                                context["error"] = "Incoherent workflow output in parallel execution; aborted by policy"
                                break
                        except Exception as ge:
                            logger.warning(f"Governance evaluation failed for {protocol_name}: {ge}")
            # Common early exit if an error was set by policy or failure
            if "error" in context:
                break

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

        retry_count = 0
        last_exception = None
        while True:
            try:
                with self.resource_manager.profile(protocol_name) as metrics:
                    execute_method = getattr(protocol, 'execute')
                    if inspect.iscoroutinefunction(execute_method):
                        result = await execute_method(data)
                    else:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(self.executor, execute_method, data)
                # mark retry success if we had previous failures
                if retry_count > 0:
                    self.resilience_monitor.record_strategy(protocol_name, "retry", True)
                return {"result": result, "resource_usage": metrics}
            except Exception as e:
                last_exception = e
                strategy = self.recovery.select_strategy(e, protocol_name, retry_count)
                action = strategy.get("strategy")
                # Audit recovery decision
                logging.getLogger("audit").info({
                    "event": "recovery_decision",
                    "protocol": protocol_name,
                    "action": action,
                    "retry_count": retry_count,
                    "reason": strategy.get("reason")
                })
                if action == "retry":
                    retry_count += 1
                    continue
                if action == "use_fallback":
                    # Return fallback data as a valid result
                    fallback = strategy.get("data") or {"status": "fallback"}
                    return {"result": fallback, "resource_usage": {"recovery": "fallback"}, "note": strategy.get("reason")}
                # abort or unknown -> re-raise to let caller handle
                raise last_exception
