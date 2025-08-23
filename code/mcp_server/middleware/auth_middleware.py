import logging
from fastapi import HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from typing import Dict, Optional

from mcp_server.security import key_manager

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)) -> Dict[str, str]:
    """
    Dependency to validate the API key using the key manager.
    Returns the user's info (role, user_id) if the key is valid.
    """
    if not api_key:
        logger.warning("Missing API Key.")
        raise HTTPException(status_code=403, detail="API Key is missing.")

    user_info = key_manager.validate_api_key(api_key)
    if not user_info:
        logger.warning(f"Invalid API Key received.")
        raise HTTPException(status_code=403, detail="Invalid API Key.")

    return user_info

class RoleChecker:
    """
    Dependency to check if the user has the required role.
    """
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: Dict[str, str] = Depends(get_api_key)):
        if user.get("role") not in self.allowed_roles:
            logger.warning(f"User with role '{user.get('role')}' not authorized for this endpoint.")
            raise HTTPException(status_code=403, detail="Not authorized.")
        return user
