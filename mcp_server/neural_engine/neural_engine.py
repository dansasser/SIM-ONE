import os
import logging
from openai import OpenAI
from .local_engine import LocalModelEngine

logger = logging.getLogger(__name__)

class OpenAIEngine:
    """
    An engine for interacting with the OpenAI API.
    """
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAIEngine initialized.")
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY not set. OpenAIEngine will use mock responses.")

    def generate_text(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        if not self.client:
            logger.info(f"Using mock OpenAI response for prompt: '{prompt[:50]}...'")
            return f"[Mock OpenAI Response]: This is a mock summary."

        try:
            logger.info(f"Sending prompt to OpenAI model {model}...")
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
    backend = os.environ.get("NEURAL_ENGINE_BACKEND", "openai").lower()

    if backend == "local":
        logger.info("Using LocalModelEngine backend.")
        model_path = os.environ.get("LOCAL_MODEL_PATH", "models/llama-3.1-8b.gguf")
        return LocalModelEngine(model_path=model_path)

    # Default to OpenAI
    logger.info("Using OpenAIEngine backend.")
    return OpenAIEngine()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("--- Testing NeuralEngine Factory ---")

    # Test default (OpenAI)
    print("\n1. Testing with default backend (openai):")
    engine_openai = NeuralEngine()
    response_openai = engine_openai.generate_text("Hello OpenAI")
    print(f"Response: {response_openai}")
    assert "Mock OpenAI Response" in response_openai

    # Test local
    print("\n2. Testing with local backend:")
    os.environ["NEURAL_ENGINE_BACKEND"] = "local"
    engine_local = NeuralEngine()
    response_local = engine_local.generate_text("Hello Local")
    print(f"Response: {response_local}")
    assert "Mock Local LLM Response" in response_local

    del os.environ["NEURAL_ENGINE_BACKEND"] # cleanup
