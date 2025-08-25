import redis
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from fakeredis import FakeRedis

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
            logger.warning(f"Could not connect to Redis for SessionManager. Using in-memory FakeRedis for sessions. Error: {e}")
            self.redis_client = FakeRedis(decode_responses=True)

    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        if self.redis_client:
            session_data = {
                "user_id": user_id,
                "history": []
            }
            self.redis_client.set(session_id, json.dumps(session_data), ex=86400)
            logger.info(f"Created new session: {session_id} for user: {user_id}")
        return session_id

    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.redis_client:
            return None
        try:
            session_json = self.redis_client.get(session_id)
            if session_json:
                return json.loads(session_json)
            return None
        except Exception as e:
            logger.error(f"Error retrieving session data for session {session_id}: {e}")
            return None

    def get_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        session_data = self.get_session_data(session_id)
        return session_data.get("history", []) if session_data else []

    def get_session_owner(self, session_id: str) -> Optional[str]:
        session_data = self.get_session_data(session_id)
        return session_data.get("user_id") if session_data else None

    def update_history(self, session_id: str, history: List[Dict[str, Any]]):
        if not self.redis_client:
            return
        try:
            session_data = self.get_session_data(session_id)
            if session_data:
                session_data["history"] = history
                history_json = json.dumps(session_data)
                self.redis_client.set(session_id, history_json, ex=86400)
        except Exception as e:
            logger.error(f"Error updating history for session {session_id}: {e}")
