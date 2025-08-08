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
        """
        Uses the LLM to extract key, fact-checkable claims from the draft.
        """
        logger.info("Critic: Extracting claims for fact-checking.")
        prompt = (
            "Please analyze the following document and extract a list of the key, fact-checkable claims made in the text. "
            "Return only a numbered list of these claims.\n\n"
            f"--- Document ---\n{draft_text}\n\n--- End of Document ---"
        )
        response = self.neural_engine.generate_text(prompt)
        claims = [line.strip() for line in response.split('\n') if line.strip() and line.strip()[0].isdigit()]
        logger.info(f"Critic: Extracted {len(claims)} claims.")
        return claims

    async def _fact_check_claims(self, claims: List[str]) -> Dict[str, str]:
        """
        Uses the RAGManager to research each claim and provide a verification summary.
        """
        logger.info(f"Critic: Fact-checking {len(claims)} claims.")
        fact_check_results = {}
        for claim in claims:
            # For each claim, perform a targeted search
            research_context = await self.rag_manager.perform_research(f"fact check: {claim}", num_sources=1)
            fact_check_results[claim] = research_context
        return fact_check_results

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the critique and fact-checking workflow.
        """
        drafter_results = data.get("DrafterProtocol", {})
        draft_text = drafter_results.get("draft_text")

        if not draft_text:
            return {"error": "No draft text provided from DrafterProtocol to critique."}

        # 1. Extract claims from the draft
        claims_to_check = await self._extract_claims(draft_text)

        # 2. Perform fact-checking on the claims
        fact_check_context = ""
        if claims_to_check:
            fact_check_results = await self._fact_check_claims(claims_to_check)
            fact_check_context = "\n\n--- Fact-Checking Research ---\n"
            for claim, research in fact_check_results.items():
                fact_check_context += f"Claim: '{claim}'\nResearch: {research}\n\n"

        # 3. Generate final critique
        logger.info("Critic: Generating final critique.")
        critique_prompt = (
            "You are an expert critical reviewer. Your task is to provide constructive feedback on the following draft. "
            "Your feedback should cover clarity, style, and factual accuracy. "
            "Use the provided fact-checking research to inform your critique of the draft's accuracy.\n\n"
            f"--- Document Draft ---\n{draft_text}\n--- End of Document ---\n"
            f"{fact_check_context}"
            "Please provide your feedback as a structured, numbered list of specific, actionable points."
        )

        feedback = self.neural_engine.generate_text(critique_prompt)
        feedback_list = [line.strip() for line in feedback.split('\n') if line.strip() and line.strip()[0].isdigit()]

        return {"feedback": feedback_list}

async def main():
    logging.basicConfig(level=logging.INFO)
    critic = CriticProtocol()

    sample_data = {
        "DrafterProtocol": {
            "draft_text": "The sun is a planet. It is the center of our solar system. All planets revolve around it."
        }
    }

    try:
        result = await critic.execute(sample_data)
        print("\n--- Critic Protocol Result ---")
        if "feedback" in result:
            for point in result["feedback"]:
                print(f"- {point}")
        else:
            print(result)
    except NameError as e:
        print(f"\nCaught expected error because tools are not in local scope: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())
