import logging
import asyncio
from typing import Optional, Dict

from mcp_server.config import settings

logger = logging.getLogger(__name__)


class MVLMEngine:
    """
    HuggingFace Transformers-backed local MVLM engine with optional model switching.

    - Loads a local HF model from a directory (no network).
    - Supports alias-based switching via `MVLM_MODEL_DIRS` + `ACTIVE_MVLM_MODEL`.
    - Falls back to deterministic stubs if Transformers/Torch are unavailable.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_dir = self._resolve_model_dir(model_path)
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._hf_available = False

        try:
            # Local imports to keep optional dependency
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch  # noqa: F401
            self._hf_available = True
            self.AutoTokenizer = AutoTokenizer
            self.AutoModelForCausalLM = AutoModelForCausalLM
        except Exception as e:
            logger.warning("Transformers/Torch not available: %s. MVLMEngine will use deterministic stubs.", e)
            self._hf_available = False

        if self._hf_available and self.model_dir:
            try:
                self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
                # Ensure pad token to avoid warnings with GPT-2 style tokenizers
                if getattr(self.tokenizer, 'pad_token', None) is None and getattr(self.tokenizer, 'eos_token', None) is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = self.AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                # Optionally load a LoRA adapter if provided
                adapter = getattr(settings, 'MVLM_LORA_ADAPTER_PATH', None)
                if adapter:
                    try:
                        from peft import PeftModel  # type: ignore
                        self.model = PeftModel.from_pretrained(self.model, adapter)
                        logger.info("Loaded LoRA adapter from %s", adapter)
                    except Exception as e:
                        logger.warning("Failed to load LoRA adapter at '%s': %s", adapter, e)
                logger.info("MVLMEngine loaded model from %s", self.model_dir)
            except Exception as e:
                logger.error("Failed to load MVLM model at '%s': %s", self.model_dir, e)
                self._hf_available = False

    def _parse_dirs(self, spec: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for pair in (spec or "").split(','):
            pair = pair.strip()
            if not pair:
                continue
            if ':' in pair:
                alias, path = pair.split(':', 1)
                out[alias.strip()] = path.strip()
        return out

    def _resolve_model_dir(self, model_path: Optional[str]) -> Optional[str]:
        # Priority: explicit arg > ACTIVE_MVLM_MODEL alias > LOCAL_MODEL_PATH
        if model_path:
            return model_path
        dirs = self._parse_dirs(getattr(settings, 'MVLM_MODEL_DIRS', ''))
        active = getattr(settings, 'ACTIVE_MVLM_MODEL', '')
        if active and active in dirs:
            return dirs[active]
        # Fallback to LOCAL_MODEL_PATH if it looks like a directory
        return getattr(settings, 'LOCAL_MODEL_PATH', None)

    def _deterministic_stub(self, prompt: str) -> str:
        p = (prompt or "").lower()
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
        return "[MVLM] Deterministic stub response."

    def generate_text(self, prompt: str, model: str = None) -> str:
        if not self._hf_available or not self.tokenizer or not self.model:
            return self._deterministic_stub(prompt)
        try:
            # Build generation controls from settings
            inputs = self.tokenizer(prompt, return_tensors="pt")
            do_sample = getattr(settings, 'MVLM_DO_SAMPLE', True)
            if getattr(settings, 'MVLM_GREEDY', False):
                do_sample = False
            gen_kwargs = dict(
                max_new_tokens=getattr(settings, 'MVLM_MAX_NEW_TOKENS', 160),
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs.update(
                    temperature=getattr(settings, 'MVLM_TEMPERATURE', 0.8),
                    top_p=getattr(settings, 'MVLM_TOP_P', 0.95),
                )
                top_k = getattr(settings, 'MVLM_TOP_K', 50)
                if top_k:
                    gen_kwargs['top_k'] = top_k
            rep = getattr(settings, 'MVLM_REPETITION_PENALTY', 1.1)
            if rep and rep != 1.0:
                gen_kwargs['repetition_penalty'] = rep
            ngram = getattr(settings, 'MVLM_NO_REPEAT_NGRAM_SIZE', 0)
            if ngram and ngram > 0:
                gen_kwargs['no_repeat_ngram_size'] = ngram

            seed = getattr(settings, 'MVLM_SEED', None)
            if seed is not None:
                try:
                    import torch  # type: ignore
                    torch.manual_seed(int(seed))
                except Exception:
                    pass

            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Return only the generated suffix beyond the prompt for clarity when possible
            if text.startswith(prompt):
                return text[len(prompt):].lstrip()
            return text
        except Exception as e:
            logger.error("MVLM generation failed: %s", e)
            return self._deterministic_stub(prompt)

    async def async_generate_text(self, prompt: str, model: str = None) -> str:
        if not self._hf_available:
            return self._deterministic_stub(prompt)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate_text, prompt, model)
