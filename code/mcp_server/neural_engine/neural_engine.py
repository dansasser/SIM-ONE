import logging
import asyncio
from openai import OpenAI
try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - older clients may not have AsyncOpenAI
    AsyncOpenAI = None
from .local_engine import LocalModelEngine
from .mvlm_engine import MVLMEngine
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
            self.async_client = AsyncOpenAI(api_key=self.api_key) if AsyncOpenAI else None
            logger.info("OpenAIEngine initialized.")
        else:
            self.client = None
            self.async_client = None
            logger.warning("OPENAI_API_KEY not set in config. OpenAIEngine will use mock responses.")

    def generate_text(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        if not self.client:
            logger.error("OpenAI client not initialized. Please set OPENAI_API_KEY.")
            raise ValueError("OpenAI client not initialized. API key is missing.")
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

    async def async_generate_text(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Async variant to avoid blocking the event loop."""
        if self.async_client:
            try:
                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Async OpenAI request failed: {e}")
                return "[Error]: Could not generate text from OpenAI."
        # Fallback: run sync method in a thread
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate_text, prompt, model)


class MockEngine:
    """Deterministic mock engine for offline testing and CI."""
    def __init__(self):
        logger.info("MockEngine initialized (deterministic responses).")

    def _mock_for_prompt(self, prompt: str) -> str:
        p = (prompt or "").lower()
        if "list of 5-7" in p or "creative strategist" in p:
            return "[Mock Summary]\n1. Mock idea considering memory.\n2. Mock idea considering research.\n3. Mock idea with governance."
        if "please write a comprehensive" in p or "skilled writer" in p:
            return "Mock Draft: coherent draft using memory and research."
        if "full, rewritten" in p or "professional editor" in p:
            return "Mock Revised Draft: improved clarity, accuracy, and tone."
        if "concise, polished" in p or "executive-level summary" in p:
            return "Mock Summary: an executive summary of the document."
        if "fact check" in p:
            return "Mock Research: sources corroborate the claim."
        return "[Mock Summary] Mock response."

    def generate_text(self, prompt: str, model: str = None) -> str:
        return self._mock_for_prompt(prompt)

    async def async_generate_text(self, prompt: str, model: str = None) -> str:
        # Keep async signature; no blocking
        return self._mock_for_prompt(prompt)

def NeuralEngine():
    """
    Factory function that returns the configured neural engine instance.
    """
    backend = settings.NEURAL_ENGINE_BACKEND
    if backend == "mock":
        logger.info("Using MockEngine backend.")
        return MockEngine()
    if backend == "local":
        logger.info("Using LocalModelEngine backend.")
        engine = LocalModelEngine(model_path=settings.LOCAL_MODEL_PATH)
        # Fallback to mock if local model failed to initialize
        try:
            # probe a trivial call to ensure availability without heavy compute
            if engine.client is None:
                logger.warning("Local model unavailable; falling back to MockEngine.")
                return MockEngine()
        except Exception:
            logger.warning("Local model probe failed; falling back to MockEngine.")
            return MockEngine()
        return engine

    if backend == "mvlm":
        logger.info("Using MVLMEngine backend.")
        try:
            engine = MVLMEngine(model_path=settings.LOCAL_MODEL_PATH)
            # Lightweight probe
            _ = engine.generate_text("ping")
            return engine
        except Exception as e:
            logger.warning(f"MVLMEngine initialization failed; using MockEngine. Error: {e}")
            return MockEngine()

    # Default: OpenAI, but fall back to mock if API key missing
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY missing; using MockEngine instead of OpenAIEngine.")
        return MockEngine()
    logger.info("Using OpenAIEngine backend.")
    return OpenAIEngine()
