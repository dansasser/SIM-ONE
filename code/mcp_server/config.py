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

settings = Settings()

if __name__ == '__main__':
    print("--- Loaded Configuration ---")
    print(settings.model_dump_json(indent=2))
