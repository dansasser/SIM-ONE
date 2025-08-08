import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class NeuralEngine:
    """
    A component for interacting with a Large Language Model.
    """

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("Neural Engine initialized with OpenAI API key.")
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY environment variable not set. Neural Engine will use mock responses.")

    def generate_text(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Generates text using the configured LLM.

        Args:
            prompt: The prompt to send to the LLM.
            model: The model to use for generation.

        Returns:
            The generated text.
        """
        if not self.client:
            logger.info(f"Using mock response for prompt: '{prompt[:50]}...'")
            return f"[Mock Summary]: This is a mock summary of the provided text."

        try:
            logger.info(f"Sending prompt to {model}...")
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that synthesizes information."},
                    {"role": "user", "content": prompt}
                ]
            )
            generated_text = response.choices[0].message.content
            logger.info("Received response from LLM.")
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error communicating with OpenAI API: {e}")
            return f"[Error]: Could not generate text due to an API error."

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)

    # To test with a real API key, set the OPENAI_API_KEY environment variable
    # export OPENAI_API_KEY='your-key-here'

    ne = NeuralEngine()

    sample_prompt = "Summarize the following conclusions: Socrates is mortal. All men are mortal."
    summary = ne.generate_text(sample_prompt)
    print(f"\nPrompt: {sample_prompt}")
    print(f"Generated Summary: {summary}")

    # Example without API key (will use mock response)
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n--- Testing without API key ---")
        ne_mock = NeuralEngine()
        summary_mock = ne_mock.generate_text("This is a test prompt.")
        print(f"Generated Mock Summary: {summary_mock}")
