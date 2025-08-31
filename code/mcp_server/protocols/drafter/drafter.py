import logging
from typing import Dict, Any, List

from mcp_server.neural_engine.neural_engine import NeuralEngine

logger = logging.getLogger(__name__)

class DrafterProtocol:
    """
    Takes ideas, research, and pre-fetched memory context and generates a first draft.
    """

    def __init__(self):
        self.neural_engine = NeuralEngine()
        # No longer instantiates MemoryManager

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the drafting workflow.
        """
        ideator_results = data.get("IdeatorProtocol", {})
        ideas = ideator_results.get("ideas")
        research_context = ideator_results.get("research_context", "No research context provided.")

        if not ideas:
            return {"error": "No ideas provided from IdeatorProtocol to draft from."}

        # 1. Retrieve pre-fetched conversational memory
        memories = data.get("batch_memory", [])
        memory_context = "No relevant memories found."
        if memories:
            memory_context = "The following memories have been tagged in this conversation. Use them to maintain a consistent tone and context:\n"
            for mem in memories:
                memory_context += f"- Entity: {mem['entity']}, Emotion: {mem['emotional_state']}\n"

        logger.info(f"Drafter: Drafting document with memory context.")

        # 2. Construct the prompt
        prompt = (
            "You are a skilled writer. Write a detailed first draft based on the key ideas. "
            "Use the research material for factual content and the conversational memory to inform the tone and style of the draft.\n\n"
            f"--- Conversational Memory ---\n{memory_context}\n\n"
            f"--- Key Ideas ---\n"
        )
        for idea in ideas:
            prompt += f"- {idea}\n"

        prompt += (
            f"\n--- Supporting Research Material ---\n{research_context}\n\n"
            "Please write a comprehensive and well-structured draft."
        )

        # 3. Generate the draft
        draft_text = await self.neural_engine.async_generate_text(prompt)

        return {"draft_text": draft_text}

if __name__ == '__main__':
    pass
