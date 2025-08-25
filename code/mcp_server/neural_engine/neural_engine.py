import logging
from openai import OpenAI
from .local_engine import LocalModelEngine
from mcp_server.config import settings # Import the settings object

logger = logging.getLogger(__name__)

class OpenAIEngine:
    """
    An engine for interacting with the OpenAI API.
    """
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAIEngine initialized.")
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY not set in config. OpenAIEngine will use mock responses.")

    def generate_text(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        if not self.client:
            return f"[Mock OpenAI Response]: This is a mock summary."
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error communicating with OpenAI API: {e}")
            return "[Error]: Could not generate text from OpenAI."

def NeuralEngine():
    """
    Factory function that returns the configured neural engine instance.
    """
    if settings.NEURAL_ENGINE_BACKEND == "local":
        logger.info("Using LocalModelEngine backend.")
        return LocalModelEngine(model_path=settings.LOCAL_MODEL_PATH)

    logger.info("Using OpenAIEngine backend.")
    return OpenAIEngine()
