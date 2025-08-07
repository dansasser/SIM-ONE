import logging
from typing import List, Dict, Any
from mcp_server.protocol_manager.protocol_manager import ProtocolManager

logger = logging.getLogger(__name__)

class OrchestrationEngine:
    """
    Orchestrates the execution of cognitive workflows.
    """

    def __init__(self, protocol_manager: ProtocolManager):
        self.protocol_manager = protocol_manager

    def execute_workflow(self, protocol_names: List[str], initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a workflow of protocols sequentially.

        Args:
            protocol_names: A list of protocol names to execute in order.
            initial_data: The initial input data for the workflow.

        Returns:
            A dictionary containing the final result.
        """
        current_data = initial_data.copy()
        workflow_results = {}

        logger.info(f"Executing workflow: {' -> '.join(protocol_names)}")

        for protocol_name in protocol_names:
            logger.info(f"  - Executing protocol: {protocol_name}")
            protocol = self.protocol_manager.get_protocol(protocol_name)
            if not protocol:
                logger.error(f"    - Could not load protocol '{protocol_name}'. Aborting workflow.")
                workflow_results['error'] = f"Protocol '{protocol_name}' not found."
                return workflow_results

            try:
                # This is a simplification for the MVP. A real implementation
                # would have a more sophisticated way of passing data between protocols.
                if protocol_name == "ReasoningAndExplanationProtocol":
                    facts = current_data.get("facts", [])
                    rules = current_data.get("rules", [])
                    result = protocol.execute(facts, rules)
                else:
                    result = protocol.execute(current_data)

                workflow_results[protocol_name] = result
                current_data.update(result)
                logger.info(f"    - Protocol '{protocol_name}' executed successfully.")

            except Exception as e:
                logger.error(f"    - Error executing protocol '{protocol_name}': {e}")
                workflow_results['error'] = f"Error in protocol '{protocol_name}': {e}"
                return workflow_results

        return workflow_results

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    pm = ProtocolManager()
    engine = OrchestrationEngine(pm)

    workflow = ["ReasoningAndExplanationProtocol"]
    data = {
        "facts": ["has_feathers", "flies", "lays_eggs"],
        "rules": [
            (["has_feathers"], "is_bird"),
            (["flies", "is_bird"], "is_flying_bird"),
            (["is_bird", "lays_eggs"], "is_oviparous_bird")
        ]
    }

    final_result = engine.execute_workflow(workflow, data)

    logger.info("\nWorkflow finished.")
    logger.info("Final Result:")
    import json
    logger.info(json.dumps(final_result, indent=2))
