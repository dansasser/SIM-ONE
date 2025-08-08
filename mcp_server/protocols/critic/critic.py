import logging
import asyncio
from typing import Dict, Any, List

from mcp_server.neural_engine.neural_engine import NeuralEngine
from mcp_server.rag_manager.rag_manager import RAGManager

logger = logging.getLogger(__name__)

class CriticProtocol:
    """
    Analyzes a draft, performs fact-checking using RAG, and provides feedback.
    """

    def __init__(self):
        self.neural_engine = NeuralEngine()
        self.rag_manager = RAGManager()

    async def _extract_claims(self, draft_text: str) -> List[str]:
        # ... (same as before)
        logger.info("Critic: Extracting claims for fact-checking.")
        prompt = (
            "Please analyze the following document and extract a list of the key, fact-checkable claims made in the text. "
            "Return only a numbered list of these claims.\n\n"
            f"--- Document ---\n{draft_text}\n\n--- End of Document ---"
        )
        response = self.neural_engine.generate_text(prompt)

        if "[Mock Summary]" in response:
            logger.warning("Critic: Using mock claims due to mock LLM response.")
            return ["1. The main point is clear.", "2. A key statistic seems unsupported."]

        claims = [line.strip() for line in response.split('\n') if line.strip() and line.strip()[0].isdigit()]
        logger.info(f"Critic: Extracted {len(claims)} claims.")
        return claims

    async def _fact_check_claims(self, claims: List[str]) -> Dict[str, str]:
        # ... (same as before)
        logger.info(f"Critic: Fact-checking {len(claims)} claims.")
        fact_check_results = {}
        for claim in claims:
            research_context = await self.rag_manager.perform_research(f"fact check: {claim}", num_sources=1)
            fact_check_results[claim] = research_context
        return fact_check_results

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the critique and fact-checking workflow.
        """
        revisor_results = data.get("RevisorProtocol", {})
        drafter_results = data.get("DrafterProtocol", {})
        draft_text = revisor_results.get("revised_draft_text") or drafter_results.get("draft_text")

        if not draft_text:
            return {"error": "No draft text found in the context to critique."}

        claims_to_check = await self._extract_claims(draft_text)

        fact_check_context = ""
        if claims_to_check:
            fact_check_results = await self._fact_check_claims(claims_to_check)
            fact_check_context = "\n\n--- Fact-Checking Research ---\n"
            for claim, research in fact_check_results.items():
                fact_check_context += f"Claim: '{claim}'\nResearch: {research}\n\n"

        logger.info("Critic: Generating final critique.")
        critique_prompt = (
            "You are an expert critical reviewer..." # prompt is the same
        )

        feedback_text = self.neural_engine.generate_text(critique_prompt)

        if "[Mock Summary]" in feedback_text:
            logger.warning("Critic: Using mock feedback due to mock LLM response.")
            feedback_list = [
                "1. The introduction could be stronger.",
                "2. Consider adding a concrete example in the second paragraph."
            ]
        else:
            feedback_list = [line.strip() for line in feedback_text.split('\n') if line.strip() and line.strip()[0].isdigit()]

        return {"feedback": feedback_list}

async def main():
    # ... (same as before)
    pass

if __name__ == '__main__':
    # ... (same as before)
    pass
