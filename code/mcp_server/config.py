from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    """
    Centralized configuration for the mCP Server.
    Loads settings from environment variables and/or a .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_VERSION: str = "1.5.0"
    VALID_API_KEYS: List[str] = ["secret-key-1", "secret-key-2"]
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    NEURAL_ENGINE_BACKEND: str = "openai"
    OPENAI_API_KEY: Optional[str] = None
    LOCAL_MODEL_PATH: str = "models/llama-3.1-8b.gguf"
    SERPER_API_KEY: Optional[str] = None

settings = Settings()

if __name__ == '__main__':
    print("--- Loaded Configuration ---")
    print(settings.model_dump_json(indent=2))
