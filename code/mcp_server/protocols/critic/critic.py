import logging
import asyncio
from typing import Dict, Any, List

from mcp_server.neural_engine.neural_engine import NeuralEngine
from mcp_server.rag_manager.rag_manager import RAGManager

logger = logging.getLogger(__name__)

class CriticProtocol:
    """
    Analyzes a draft, performs fact-checking, and provides feedback, informed by memory
    and constrained by a latency budget.
    """

    def __init__(self):
        self.neural_engine = NeuralEngine()
        self.rag_manager = RAGManager()

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the critique and fact-checking workflow.
        """
        revisor_results = data.get("RevisorProtocol", {})
        drafter_results = data.get("DrafterProtocol", {})
        draft_text = revisor_results.get("revised_draft_text") or drafter_results.get("draft_text")
        latency_info = data.get("latency_info", {}) # Get latency info from context

        if not draft_text:
            return {"error": "No draft text found in the context to critique."}

        # ... (memory context retrieval is the same) ...

        # 2. Extract claims and fact-check them
        claims_to_check = [f"A key point in the draft is about '{draft_text[:30]}...'"]
        fact_check_context = ""
        # CRITICAL FIX: Pass latency_info to the fact checking method
        fact_check_results = await self._fact_check_claims(claims_to_check, latency_info)
        fact_check_context = "\n\n--- Fact-Checking Research ---\n"
        for claim, research in fact_check_results.items():
            fact_check_context += f"Claim: '{claim}'\nResearch: {research}\n\n"

        # 3. Generate final critique
        logger.info("Critic: Generating final critique with memory context.")
        critique_prompt = (
            "You are an expert critical reviewer..." # Abbreviated for this change
        )

        feedback_text = self.neural_engine.generate_text(critique_prompt)

        if "[Mock Summary]" in feedback_text:
            feedback_list = ["1. Mock critique based on memory.", "2. Mock fact-check passed."]
        else:
            feedback_list = [line.strip() for line in feedback_text.split('\n') if line.strip() and line.strip()[0].isdigit()]

        return {"feedback": feedback_list}

    async def _fact_check_claims(self, claims: List[str], latency_info: Dict) -> Dict[str, str]:
        logger.info(f"Critic: Fact-checking {len(claims)} claims.")
        fact_check_results = {}
        for claim in claims:
            # Pass latency_info to the RAGManager
            research_context = await self.rag_manager.perform_research(f"fact check: {claim}", latency_info, num_sources=1)
            fact_check_results[claim] = research_context
        return fact_check_results

if __name__ == '__main__':
    pass
