import redis
import json
import logging
from typing import List, Dict, Any

from mcp_server.config import settings # Import the settings object

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages the storage and retrieval of structured, emotionally-tagged memories.
    """

    def __init__(self):
        try:
            self.redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=1, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"MemoryManager successfully connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"MemoryManager could not connect to Redis. Memory functions will be disabled. Error: {e}")
            self.redis_client = None

    def _get_session_key(self, session_id: str) -> str:
        return f"memory:{session_id}"

    def add_memories(self, session_id: str, memories: List[Dict[str, Any]]):
        if not self.redis_client or not memories:
            return
        session_key = self._get_session_key(session_id)
        try:
            pipeline = self.redis_client.pipeline()
            for memory in memories:
                entity = memory.get("entity")
                if entity:
                    pipeline.hset(session_key, entity, json.dumps(memory))
            pipeline.execute()
            logger.info(f"Added {len(memories)} new memories to session {session_id}.")
        except Exception as e:
            logger.error(f"Error adding memories for session {session_id}: {e}")

    def get_all_memories(self, session_id: str) -> List[Dict[str, Any]]:
        if not self.redis_client:
            return []
        session_key = self._get_session_key(session_id)
        try:
            all_memories_raw = self.redis_client.hgetall(session_key)
            return [json.loads(mem_json) for mem_json in all_memories_raw.values()]
        except Exception as e:
            logger.error(f"Error retrieving all memories for session {session_id}: {e}")
            return []
