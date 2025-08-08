import logging
import asyncio
from typing import Dict, Any, List

from mcp_server.neural_engine.neural_engine import NeuralEngine
from mcp_server.rag_manager.rag_manager import RAGManager

logger = logging.getLogger(__name__)

class RevisorProtocol:
    """
    Revises a draft based on feedback, using RAG to gather new information.
    """

    def __init__(self):
        self.neural_engine = NeuralEngine()
        self.rag_manager = RAGManager()

    async def _research_feedback_points(self, feedback: List[str]) -> str:
        # ... (same as before) ...
        logger.info(f"Revisor: Researching {len(feedback)} feedback points.")
        topics_to_research = []
        for point in feedback[:2]:
            if "lacks examples" in point or "fact-check" in point or "verify" in point:
                topics_to_research.append(point)
        if not topics_to_research:
            return "No specific research was triggered by the feedback."
        research_context = await self.rag_manager.perform_research(" ".join(topics_to_research), num_sources=1)
        return research_context

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the revision workflow.
        """
        # FIX: Look for the latest draft, whether from a previous Revisor run or the initial Drafter run.
        revisor_results = data.get("RevisorProtocol", {})
        drafter_results = data.get("DrafterProtocol", {})
        draft_text = revisor_results.get("revised_draft_text") or drafter_results.get("draft_text")

        critic_results = data.get("CriticProtocol", {})
        feedback = critic_results.get("feedback")

        if not draft_text or not feedback:
            return {"error": "Draft text or feedback not provided for revision."}

        # 1. Research the feedback points
        research_context = await self._research_feedback_points(feedback)

        # 2. Construct the prompt for revision
        logger.info("Revisor: Generating revised draft.")
        revision_prompt = (
            "You are a professional editor and writer. Your task is to revise the following draft based on the provided critique. "
            "Use the supplementary research material to enrich the content and address factual inaccuracies.\n\n"
            f"--- Document to Revise ---\n{draft_text}\n\n"
            f"--- Critique ---\n"
        )
        for point in feedback:
            revision_prompt += f"- {point}\n"

        revision_prompt += (
            f"\n--- Supplementary Research ---\n{research_context}\n\n"
            "Please provide the full, revised version of the document, incorporating the feedback and research. "
            "Do not just list the changes; provide the complete, rewritten text."
        )

        # 3. Use the Neural Engine to generate the revised draft
        revised_draft = self.neural_engine.generate_text(revision_prompt)

        return {"revised_draft_text": revised_draft}

async def main():
    # ... (same as before) ...
    logging.basicConfig(level=logging.INFO)
    revisor = RevisorProtocol()
    sample_data = {
        "DrafterProtocol": {"draft_text": "The sun is hot."},
        "CriticProtocol": {"feedback": ["Please add more detail."]}
    }
    try:
        result = await revisor.execute(sample_data)
        print("\n--- Revisor Protocol Result ---")
        if "revised_draft_text" in result:
            print(result["revised_draft_text"])
        else:
            print(result)
    except NameError as e:
        print(f"\nCaught expected error because tools are not in local scope: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())
