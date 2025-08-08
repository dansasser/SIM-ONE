import logging
from typing import Dict, Any

from mcp_server.neural_engine.neural_engine import NeuralEngine

logger = logging.getLogger(__name__)

class SP:
    """
    A simple implementation of the Summarizer Protocol (SP).
    """

    def __init__(self):
        # In a more advanced design, the NeuralEngine might be injected as a dependency.
        # For this implementation, the protocol instantiates it directly.
        self.neural_engine = NeuralEngine()

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the summarization logic.

        Args:
            data: The data from the workflow. It's expected to contain
                  the output of the ReasoningAndExplanationProtocol.

        Returns:
            A dictionary with the generated summary.
        """
        logger.info("Executing SP...")

        rep_results = data.get("ReasoningAndExplanationProtocol", {})
        conclusions = rep_results.get("conclusions", [])
        explanation = rep_results.get("explanation", [])

        if not conclusions:
            return {
                "summary": "No conclusions were provided to summarize.",
                "status": "skipped"
            }

        # Construct a detailed prompt for the LLM
        prompt = (
            "Please provide a concise, human-readable summary of the following logical process.\n\n"
            f"Initial Facts and Rules led to the following explanation steps:\n"
            f"{' -> '.join(explanation)}\n\n"
            f"This process resulted in the following final conclusions:\n"
            f"{', '.join(conclusions)}\n\n"
            "Synthesize this information into a brief summary paragraph."
        )

        summary = self.neural_engine.generate_text(prompt)

        return {
            "summary": summary,
            "status": "success"
        }

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)

    sp_protocol = SP()

    sample_data = {
        "ReasoningAndExplanationProtocol": {
            "conclusions": ["is_bird", "is_flying_bird", "is_oviparous_bird"],
            "explanation": [
                "Initial facts: {'has_feathers', 'flies', 'lays_eggs'}",
                "Rule applied: IF has_feathers THEN is_bird. New fact derived: is_bird",
                "Rule applied: IF flies AND is_bird THEN is_flying_bird. New fact derived: is_flying_bird",
                "Rule applied: IF is_bird AND lays_eggs THEN is_oviparous_bird. New fact derived: is_oviparous_bird"
            ]
        }
    }

    result = sp_protocol.execute(sample_data)
    print("\n--- Summarizer Protocol Result ---")
    print(result['summary'])
