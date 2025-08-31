import logging
from typing import Dict, Any

from mcp_server.neural_engine.neural_engine import NeuralEngine

logger = logging.getLogger(__name__)

class SP:
    """
    An enhanced implementation of the Summarizer Protocol (SP).
    """

    def __init__(self):
        self.neural_engine = NeuralEngine()

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the summarization logic on the final draft.

        Args:
            data: The workflow context. It looks for the output of the
                  RevisorProtocol or, failing that, the DrafterProtocol.

        Returns:
            A dictionary with the generated summary.
        """
        logger.info("Executing enhanced SP...")

        # Prioritize the revised draft, but fall back to the initial draft.
        revisor_results = data.get("RevisorProtocol", {})
        drafter_results = data.get("DrafterProtocol", {})

        text_to_summarize = revisor_results.get("revised_draft_text") or drafter_results.get("draft_text")

        if not text_to_summarize:
            return {
                "summary": "No draft text was found in the context to summarize.",
                "status": "skipped"
            }

        prompt = (
            "Please provide a concise, polished, and executive-level summary of the following document.\n\n"
            f"--- Document ---\n"
            f"{text_to_summarize}\n\n"
            f"--- End of Document ---\n\n"
            "The summary should be a single, well-written paragraph."
        )

        summary = await self.neural_engine.async_generate_text(prompt)

        return {
            "summary": summary,
            "status": "success"
        }

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)

    sp_protocol = SP()

    # --- Test Case 1: With revised draft ---
    print("--- Test Case 1: With revised draft ---")
    sample_data_revised = {
        "RevisorProtocol": {
            "revised_draft_text": "The sun is a star, a glowing ball of plasma held together by its own gravity. Nuclear fusion in its core produces immense energy."
        }
    }
    result = sp_protocol.execute(sample_data_revised)
    print(result.get("summary"))

    # --- Test Case 2: With initial draft only ---
    print("\n--- Test Case 2: With initial draft only ---")
    sample_data_initial = {
        "DrafterProtocol": {
            "draft_text": "The sun is hot."
        }
    }
    result = sp_protocol.execute(sample_data_initial)
    print(result.get("summary"))

    # --- Test Case 3: No text to summarize ---
    print("\n--- Test Case 3: No text ---")
    result = sp_protocol.execute({})
    print(result.get("summary"))
