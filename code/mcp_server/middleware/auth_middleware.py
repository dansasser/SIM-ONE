import logging
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from mcp_server.config import settings # Import the settings object

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency to validate the API key using centralized configuration.
    """
    if not api_key:
        logger.warning("Missing API Key.")
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials: API Key is missing."
        )
    if api_key not in settings.get_valid_api_keys():
        logger.warning(f"Invalid API Key received: {api_key}")
        raise HTTPException(
            status_code=403,
            detail="Could not validate credentials: Invalid API Key."
        )
    return api_key
