import logging
from typing import Dict, Any, List

from mcp_server.neural_engine.neural_engine import NeuralEngine
from mcp_server.neural_engine.prompt_adapters import build_bullets_prompt
from mcp_server.neural_engine.json_guard import ensure_json

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

        # Build strict JSON prompt for exactly 5 bullets summarizing the document
        schema_hint = '{ "bullets": [string, string, string, string, string] }'
        task_text = (
            "Produce exactly 5 bullet points for this document.\n"
            "--- Document ---\n"
            f"{text_to_summarize}\n"
            "--- End of Document ---"
        )
        _, mvlm_prompt = build_bullets_prompt(5, task_text)

        def call_primary() -> str:
            return self.neural_engine.generate_text(mvlm_prompt)

        def call_tight() -> str:
            tight = mvlm_prompt + "\nReturn ONLY valid JSON with exactly 5 items in 'bullets'."
            return self.neural_engine.generate_text(tight)

        def default_json() -> dict:
            return {"bullets": []}

        bullets_json = await self._ensure_json_async(call_primary, call_tight, default_json)
        bullets: List[str] = bullets_json.get("bullets", []) if isinstance(bullets_json, dict) else []
        # Also provide a joined summary paragraph for backward compatibility consumers
        summary_text = " ".join(bullets) if bullets else ""

        return {
            "summary": summary_text or "No summary produced.",
            "bullets": bullets,
            "status": "success" if bullets else "fallback"
        }

    async def _ensure_json_async(self, primary, tight, default_factory):
        # Async wrapper for the sync ensure_json to avoid blocking event loop excessively
        # Run the blocking engine calls in a thread via the async engine API when available
        def as_sync(callable_fn):
            # Wrap the sync generate by delegating to sync method through proxy
            return callable_fn()
        # Use ensure_json directly since engine proxy routes to sync generate_text
        return ensure_json(lambda: as_sync(primary), lambda: as_sync(tight), default_factory)

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
