import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class MVLMEngine:
    """
    Placeholder MVLM engine with interface parity.

    Behavior:
    - Until the real MVLM is provided, returns deterministic mock-like outputs to
      keep tests stable and allow wiring validation.
    - Once assets are injected, replace internals to call the actual MVLM.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        # No heavy initialization here to avoid optional dependency failures.
        logger.info("MVLMEngine initialized (scaffold). Model path=%s", model_path)

    def generate_text(self, prompt: str, model: str = None) -> str:
        p = (prompt or "").lower()
        # Deterministic responses for stability, mirroring MockEngine cues
        if "list of 5-7" in p or "creative strategist" in p:
            return "[MVLM Summary]\n1. Idea A.\n2. Idea B.\n3. Idea C."
        if "please write a comprehensive" in p or "skilled writer" in p:
            return "MVLM Draft: coherent draft using governance."
        if "full, rewritten" in p or "professional editor" in p:
            return "MVLM Revised Draft: clarity and accuracy improved."
        if "concise, polished" in p or "executive-level summary" in p:
            return "MVLM Summary: concise executive summary."
        if "fact check" in p:
            return "MVLM Research: sources corroborate key claim."
        return "[MVLM] Mock-compatible response."

    async def async_generate_text(self, prompt: str, model: str = None) -> str:
        # Keep async signature; no blocking
        return self.generate_text(prompt, model)

