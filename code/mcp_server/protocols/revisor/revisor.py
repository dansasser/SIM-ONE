import logging
import asyncio
from typing import Dict, Any, List

from mcp_server.neural_engine.neural_engine import NeuralEngine
from mcp_server.rag_manager.rag_manager import RAGManager

logger = logging.getLogger(__name__)

class RevisorProtocol:
    """
    Revises a draft based on feedback, using RAG and pre-fetched memory.
    """

    def __init__(self):
        self.neural_engine = NeuralEngine()
        self.rag_manager = RAGManager()

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the revision workflow.
        """
        revisor_results = data.get("RevisorProtocol", {})
        drafter_results = data.get("DrafterProtocol", {})
        draft_text = revisor_results.get("revised_draft_text") or drafter_results.get("draft_text")

        critic_results = data.get("CriticProtocol", {})
        feedback = critic_results.get("feedback")
        latency_info = data.get("latency_info", {})

        if not draft_text or not feedback:
            return {"error": "Draft text or feedback not provided for revision."}

        # 1. Retrieve pre-fetched conversational memory
        memories = data.get("batch_memory", [])
        memory_context = "No relevant memories found."
        if memories:
            memory_context = "The following memories have been tagged in this conversation. Use them to ensure the revised draft aligns with the user's context and sentiment:\n"
            for mem in memories:
                memory_context += f"- Entity: {mem['entity']}, Emotion: {mem['emotional_state']}\n"

        # 2. Research the feedback points
        research_context = await self._research_feedback_points(feedback, latency_info)

        # 3. Construct the prompt for revision
        logger.info("Revisor: Generating revised draft with memory and research context.")
        revision_prompt = (
            "You are a professional editor. Revise the draft based on the critique. "
            "Use the conversational memory to guide the tone, and the research to improve factual content.\n\n"
            f"--- Conversational Memory ---\n{memory_context}\n\n"
            f"--- Document to Revise ---\n{draft_text}\n\n"
            f"--- Critique ---\n"
        )
        for point in feedback:
            revision_prompt += f"- {point}\n"

        revision_prompt += (
            f"\n--- Supplementary Research ---\n{research_context}\n\n"
            "Please provide the full, rewritten, and polished version of the document."
        )

        # 4. Generate the revised draft
        revised_draft = self.neural_engine.generate_text(revision_prompt)

        return {"revised_draft_text": revised_draft}

    async def _research_feedback_points(self, feedback: List[str], latency_info: Dict) -> str:
        logger.info(f"Revisor: Researching {len(feedback)} feedback points.")
        topics_to_research = []
        for point in feedback[:1]: # Limit to 1 research point to conserve latency budget
            if "lacks examples" in point or "fact-check" in point or "verify" in point:
                topics_to_research.append(point)
        if not topics_to_research:
            return "No specific research was triggered by the feedback."
        research_context = await self.rag_manager.perform_research(" ".join(topics_to_research), latency_info, num_sources=1)
        return research_context

if __name__ == '__main__':
    pass
