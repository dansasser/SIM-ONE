import logging
from typing import Dict, Any, List

from mcp_server.neural_engine.neural_engine import NeuralEngine

logger = logging.getLogger(__name__)

class DrafterProtocol:
    """
    Takes a list of ideas and research context and generates a first draft.
    """

    def __init__(self):
        self.neural_engine = NeuralEngine()

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the drafting workflow.

        Args:
            data: The input data, expected to contain the output of the IdeatorProtocol.

        Returns:
            A dictionary with the generated draft text.
        """
        ideator_results = data.get("IdeatorProtocol", {})
        ideas = ideator_results.get("ideas")
        research_context = ideator_results.get("research_context", "No research context provided.")

        if not ideas:
            return {"error": "No ideas provided from IdeatorProtocol to draft from."}

        logger.info(f"Drafter: Drafting document based on {len(ideas)} ideas and research context.")

        prompt = (
            "You are a skilled and articulate writer. Your task is to write a detailed and well-structured first draft of a document. "
            "Use the provided key ideas as the main structure for the document, and enrich the content with information from the research material.\n\n"
            "--- Key Ideas ---\n"
        )
        for idea in ideas:
            prompt += f"- {idea}\n"

        prompt += (
            "\n--- Supporting Research Material ---\n"
            f"{research_context}\n"
            "\n--- End of Research Material ---\n\n"
            "Please write a comprehensive draft. Ensure the tone is formal and academic. "
            "Structure the document with clear headings for each key idea."
        )

        draft_text = self.neural_engine.generate_text(prompt, model="gpt-3.5-turbo")

        return {"draft_text": draft_text}

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    drafter = DrafterProtocol()

    sample_data = {
        "IdeatorProtocol": {
            "ideas": [
                "1. Introduce the concept of AI Constitutionalism.",
                "2. Discuss the challenges of implementing global AI treaties.",
                "3. Propose a framework for decentralized AI safety audits."
            ],
            "research_context": "A recent paper from the Stanford Institute for Human-Centered AI suggests that..."
        }
    }

    result = drafter.execute(sample_data)

    print("\n--- Drafter Protocol Result ---")
    print(result.get("draft_text"))
