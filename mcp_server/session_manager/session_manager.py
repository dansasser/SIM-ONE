import redis
import json
import uuid
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages conversational sessions using a Redis backend.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=0, decode_responses=True)
            # Ping the server to check the connection
            self.redis_client.ping()
            logger.info(f"Successfully connected to Redis at {host}:{port}")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to Redis at {host}:{port}. Session management will not work. Error: {e}")
            self.redis_client = None

    def create_session(self) -> str:
        """
        Creates a new session and returns the session ID.
        """
        if not self.redis_client:
            return str(uuid.uuid4()) # Return a dummy UUID if no redis

        session_id = str(uuid.uuid4())
        # We can store some initial metadata if we want
        self.update_history(session_id, [])
        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieves the conversation history for a given session ID.
        """
        if not self.redis_client:
            return []

        try:
            history_json = self.redis_client.get(session_id)
            if history_json:
                return json.loads(history_json)
            return [] # Return empty list if session is new or doesn't exist
        except Exception as e:
            logger.error(f"Error retrieving history for session {session_id}: {e}")
            return None

    def update_history(self, session_id: str, history: List[Dict[str, Any]]):
        """
        Updates the conversation history for a given session ID.
        """
        if not self.redis_client:
            return

        try:
            history_json = json.dumps(history)
            # We can set an expiration time for sessions, e.g., 24 hours
            self.redis_client.set(session_id, history_json, ex=86400)
        except Exception as e:
            logger.error(f"Error updating history for session {session_id}: {e}")

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)

    sm = SessionManager()

    if sm.redis_client:
        # Create a new session
        sid = sm.create_session()
        print(f"New session ID: {sid}")

        # Get initial history (should be empty)
        hist = sm.get_history(sid)
        print(f"Initial history: {hist}")

        # Add some turns to the history
        hist.append({"user_request": "Hello", "server_response": "Hi there!"})
        hist.append({"user_request": "How are you?", "server_response": "I am a machine, I have no feelings."})
        sm.update_history(sid, hist)
        print("Updated history.")

        # Get the updated history
        hist_updated = sm.get_history(sid)
        print(f"Updated history: {json.dumps(hist_updated, indent=2)}")
    else:
        print("Cannot run example because Redis connection failed.")
