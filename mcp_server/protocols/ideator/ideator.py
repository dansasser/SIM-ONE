import logging
import asyncio
from typing import Dict, Any, List

from mcp_server.neural_engine.neural_engine import NeuralEngine
from mcp_server.rag_manager.rag_manager import RAGManager

logger = logging.getLogger(__name__)

class IdeatorProtocol:
    """
    Generates ideas on a topic, augmented by web research (RAG).
    """

    def __init__(self):
        self.neural_engine = NeuralEngine()
        self.rag_manager = RAGManager()

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the idea generation workflow.

        Args:
            data: The input data, expected to contain a 'topic'.

        Returns:
            A dictionary with the generated ideas.
        """
        topic = data.get("topic")
        if not topic:
            return {"error": "No topic provided for idea generation."}

        # 1. Perform RAG to get context
        context = await self.rag_manager.perform_research(topic)

        # 2. Construct the prompt with the retrieved context
        prompt = (
            "You are a creative strategist and researcher. "
            f"Based on the following in-depth research material about '{topic}', please brainstorm a comprehensive and insightful list of 5-7 key ideas, angles, and talking points for a document. "
            "Focus on unique insights and avoid generic statements.\n\n"
            f"--- Research Material ---\n"
            f"{context}\n\n"
            f"--- End of Research Material ---\n\n"
            "Please provide the ideas as a numbered list. Each idea should be a complete sentence."
        )

        # 3. Use the Neural Engine to generate ideas
        generated_text = self.neural_engine.generate_text(prompt)

        # 4. Parse the output into a list
        ideas = [line.strip() for line in generated_text.split('\n') if line.strip() and line.strip()[0].isdigit()]

        return {"ideas": ideas, "research_context": context}

async def main():
    logging.basicConfig(level=logging.INFO)
    ideator = IdeatorProtocol()

    data = {"topic": "The role of ethics in large language models"}

    try:
        result = await ideator.execute(data)
        print("\n--- Ideator Protocol Result ---")
        if "ideas" in result:
            print("Generated Ideas:")
            for idea in result["ideas"]:
                print(f"- {idea}")
        else:
            print(result)
    except NameError as e:
        print(f"\nCaught expected error because tools are not in local scope: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())
