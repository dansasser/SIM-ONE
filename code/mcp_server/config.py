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

settings = Settings()

if __name__ == '__main__':
    print("--- Loaded Configuration ---")
    print(settings.model_dump_json(indent=2))
