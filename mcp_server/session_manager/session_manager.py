import redis
import json
import uuid
import logging
from typing import List, Dict, Any, Optional

from mcp_server.config import settings # Import the settings object

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages conversational sessions using a Redis backend.
    """

    def __init__(self):
        try:
            self.redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"SessionManager successfully connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to Redis for SessionManager. Sessions will not be persistent. Error: {e}")
            self.redis_client = None

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        if self.redis_client:
            self.update_history(session_id, [])
            logger.info(f"Created new session: {session_id}")
        return session_id

    def get_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        if not self.redis_client:
            return []
        try:
            history_json = self.redis_client.get(session_id)
            if history_json:
                return json.loads(history_json)
            return []
        except Exception as e:
            logger.error(f"Error retrieving history for session {session_id}: {e}")
            return None

    def update_history(self, session_id: str, history: List[Dict[str, Any]]):
        if not self.redis_client:
            return
        try:
            history_json = json.dumps(history)
            self.redis_client.set(session_id, history_json, ex=86400)
        except Exception as e:
            logger.error(f"Error updating history for session {session_id}: {e}")
