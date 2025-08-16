from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    """
    Centralized configuration for the mCP Server.
    Loads settings from environment variables and/or a .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_VERSION: str = "1.5.0"
    VALID_API_KEYS: str = ""  # Will be parsed into a list
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    NEURAL_ENGINE_BACKEND: str = "openai"
    OPENAI_API_KEY: Optional[str] = None
    LOCAL_MODEL_PATH: str = "models/llama-3.1-8b.gguf"
    SERPER_API_KEY: Optional[str] = None

    def get_valid_api_keys(self) -> List[str]:
        """Parse comma-separated API keys into a list.""" 
        if not self.VALID_API_KEYS:
            return []
        return [key.strip() for key in self.VALID_API_KEYS.split(",") if key.strip()]

settings = Settings()

if __name__ == '__main__':
    print("--- Loaded Configuration ---")
    print(settings.model_dump_json(indent=2))
    print("--- Parsed API Keys ---")
    print(settings.get_valid_api_keys())
