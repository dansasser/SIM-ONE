from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    """
    Centralized configuration for the mCP Server.
    Loads settings from environment variables and/or a .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_VERSION: str = "1.5.0"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    NEURAL_ENGINE_BACKEND: str = "openai"
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    LOCAL_MODEL_PATH: str = "models/llama-3.1-8b.gguf"
    SERPER_API_KEY: Optional[str] = os.getenv("SERPER_API_KEY")
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(',')
    # RAG concurrency control
    RAG_MAX_CONCURRENCY: int = int(os.getenv("RAG_MAX_CONCURRENCY", 8))
    # Critic fact-check concurrency cap
    CRITIC_FACTCHECK_MAX_CONCURRENCY: int = int(os.getenv("CRITIC_FACTCHECK_MAX_CONCURRENCY", 3))
    # Orchestration controls
    PROTOCOL_TIMEOUT_MS: int = int(os.getenv("PROTOCOL_TIMEOUT_MS", 10000))
    MAX_PARALLEL_PROTOCOLS: int = int(os.getenv("MAX_PARALLEL_PROTOCOLS", 4))
    # Optional per-protocol timeout overrides (comma-separated name:ms)
    # Example: "ReasoningAndExplanationProtocol:15000,EmotionalStateLayerProtocol:8000"
    PROTOCOL_TIMEOUTS_MS: str = os.getenv("PROTOCOL_TIMEOUTS_MS", "")

    # Governance controls
    GOV_ENABLE: bool = os.getenv("GOV_ENABLE", "true").lower() in ("1", "true", "yes")
    GOV_MIN_QUALITY: float = float(os.getenv("GOV_MIN_QUALITY", 0.6))
    GOV_REQUIRE_COHERENCE: bool = os.getenv("GOV_REQUIRE_COHERENCE", "false").lower() in ("1", "true", "yes")

    # Rate limits (per endpoint)
    RATE_LIMIT_EXECUTE: str = os.getenv("RATE_LIMIT_EXECUTE", "20/minute")
    RATE_LIMIT_PROTOCOLS: str = os.getenv("RATE_LIMIT_PROTOCOLS", "60/minute")
    RATE_LIMIT_TEMPLATES: str = os.getenv("RATE_LIMIT_TEMPLATES", "60/minute")
    RATE_LIMIT_SESSION: str = os.getenv("RATE_LIMIT_SESSION", "30/minute")
    RATE_LIMIT_METRICS: str = os.getenv("RATE_LIMIT_METRICS", "10/minute")

    # MVLM local model switching
    # Comma-separated alias:path pairs, e.g. "main:models/mvlm_gpt2/mvlm_final,enhanced:/opt/models/mvlm_v2"
    MVLM_MODEL_DIRS: str = os.getenv("MVLM_MODEL_DIRS", "")
    ACTIVE_MVLM_MODEL: str = os.getenv("ACTIVE_MVLM_MODEL", "")
    # MVLM decoding controls
    MVLM_MAX_NEW_TOKENS: int = int(os.getenv("MVLM_MAX_NEW_TOKENS", 160))
    MVLM_DO_SAMPLE: bool = os.getenv("MVLM_DO_SAMPLE", "true").lower() in ("1", "true", "yes")
    MVLM_GREEDY: bool = os.getenv("MVLM_GREEDY", "false").lower() in ("1", "true", "yes")
    MVLM_TEMPERATURE: float = float(os.getenv("MVLM_TEMPERATURE", 0.8))
    MVLM_TOP_P: float = float(os.getenv("MVLM_TOP_P", 0.95))
    MVLM_TOP_K: int = int(os.getenv("MVLM_TOP_K", 50))
    MVLM_REPETITION_PENALTY: float = float(os.getenv("MVLM_REPETITION_PENALTY", 1.1))
    MVLM_NO_REPEAT_NGRAM_SIZE: int = int(os.getenv("MVLM_NO_REPEAT_NGRAM_SIZE", 3))
    MVLM_SEED: Optional[int] = int(os.getenv("MVLM_SEED")) if os.getenv("MVLM_SEED") else None
    MVLM_LORA_ADAPTER_PATH: Optional[str] = os.getenv("MVLM_LORA_ADAPTER_PATH")

    # OpenAI backend configuration
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE")
    OPENAI_TIMEOUT_SECONDS: int = int(os.getenv("OPENAI_TIMEOUT_SECONDS", 30))
    OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", 2))
    OPENAI_JSON_ONLY: bool = os.getenv("OPENAI_JSON_ONLY", "true").lower() in ("1", "true", "yes")
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", 86400))
    # Redis connection
    REDIS_MODE: str = os.getenv("REDIS_MODE", "standalone")  # standalone | cluster
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    # Per-API-key quota (additional to rate limits). 0 disables.
    API_KEY_QUOTA_PER_MINUTE: int = int(os.getenv("API_KEY_QUOTA_PER_MINUTE", 0))
    # Prometheus metrics exposure
    METRICS_PUBLIC: bool = os.getenv("METRICS_PUBLIC", "false").lower() in ("1", "true", "yes")

settings = Settings()

if __name__ == '__main__':
    print("--- Loaded Configuration ---")
    print(settings.model_dump_json(indent=2))
