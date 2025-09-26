import logging
import asyncio
import time
import random
from openai import OpenAI
try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - older clients may not have AsyncOpenAI
    AsyncOpenAI = None
from .local_engine import LocalModelEngine
from .mvlm_engine import MVLMEngine
from mcp_server.config import settings # Import the settings object
from mcp_server.metrics import governance_metrics as govm

logger = logging.getLogger(__name__)


class _EngineProxy:
    """Singleton proxy that delegates to a concrete engine and supports runtime refresh."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._impl = None
            cls._instance.refresh()
        return cls._instance

    def refresh(self):
        self._impl = _build_engine()

    def generate_text(self, *args, **kwargs):
        return self._impl.generate_text(*args, **kwargs)

    async def async_generate_text(self, *args, **kwargs):
        # Delegate preserving async semantics
        if hasattr(self._impl, 'async_generate_text'):
            return await self._impl.async_generate_text(*args, **kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._impl.generate_text, *args)

class OpenAIEngine:
    """
    An engine for interacting with the OpenAI API.
    """
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.timeout = settings.OPENAI_TIMEOUT_SECONDS
        self.max_retries = max(0, settings.OPENAI_MAX_RETRIES)
        self.json_only = settings.OPENAI_JSON_ONLY
        base_url = getattr(settings, 'OPENAI_API_BASE', None)
        if self.api_key:
            kwargs = {"api_key": self.api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)
            self.async_client = AsyncOpenAI(**kwargs) if AsyncOpenAI else None
            logger.info("OpenAIEngine initialized (model=%s).", self.model)
        else:
            self.client = None
            self.async_client = None
            logger.warning("OPENAI_API_KEY not set in config. OpenAIEngine will use mock responses.")

    def _retry_backoff(self, attempt: int) -> float:
        # simple exponential backoff with jitter
        return min(2 ** attempt, 5) + random.random()

    def _as_messages(self, prompt):
        # If caller passes a dict with messages, use it; else wrap as system+user
        if isinstance(prompt, dict) and "messages" in prompt:
            return prompt["messages"], bool(prompt.get("json_mode", False))
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": str(prompt)},
        ]
        return messages, False

    def generate_text(self, prompt, model: str = None) -> str:
        if not self.client:
            logger.error("OpenAI client not initialized. Please set OPENAI_API_KEY.")
            raise ValueError("OpenAI client not initialized. API key is missing.")
        use_model = model or self.model
        messages, json_mode_flag = self._as_messages(prompt)
        attempts = 0
        while True:
            govm.inc("openai_calls")
            try:
                kwargs = {
                    "model": use_model,
                    "messages": messages,
                }
                # JSON mode if requested
                if self.json_only or json_mode_flag:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                govm.inc("openai_errors")
                attempts += 1
                if attempts > self.max_retries:
                    logger.error("OpenAI request failed after retries: %s", e)
                    return ""
                time.sleep(self._retry_backoff(attempts))

    async def async_generate_text(self, prompt, model: str = None) -> str:
        """Async variant to avoid blocking the event loop."""
        if self.async_client and AsyncOpenAI:
            messages, json_mode_flag = self._as_messages(prompt)
            use_model = model or self.model
            attempts = 0
            while True:
                govm.inc("openai_calls")
                try:
                    kwargs = {
                        "model": use_model,
                        "messages": messages,
                    }
                    if self.json_only or json_mode_flag:
                        kwargs["response_format"] = {"type": "json_object"}
                    response = await self.async_client.chat.completions.create(**kwargs)
                    return (response.choices[0].message.content or "").strip()
                except Exception as e:
                    govm.inc("openai_errors")
                    attempts += 1
                    if attempts > self.max_retries:
                        logger.error("Async OpenAI request failed after retries: %s", e)
                        return ""
                    await asyncio.sleep(self._retry_backoff(attempts))
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

def _build_engine():
    """Internal builder for a concrete engine based on current settings."""
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


def NeuralEngine():
    """Public factory: returns a singleton proxy that can be refreshed at runtime."""
    return _EngineProxy()


def refresh_neural_engine():
    """Refreshes the singleton proxy to pick up new backend/model settings."""
    _EngineProxy().refresh()
