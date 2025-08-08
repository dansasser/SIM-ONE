import redis
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages the storage and retrieval of structured, emotionally-tagged memories.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=1, decode_responses=True) # Use DB 1 for memory
            self.redis_client.ping()
            logger.info(f"MemoryManager successfully connected to Redis at {host}:{port}")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"MemoryManager could not connect to Redis. Memory functions will be disabled. Error: {e}")
            self.redis_client = None

    def _get_session_key(self, session_id: str) -> str:
        """Helper to create a consistent Redis key for a session's memory."""
        return f"memory:{session_id}"

    def add_memories(self, session_id: str, memories: List[Dict[str, Any]]):
        """
        Adds a list of memory objects to the session's memory store.
        """
        if not self.redis_client or not memories:
            return

        session_key = self._get_session_key(session_id)
        try:
            # Use a Redis Hash to store memories, with the entity as the field.
            # This allows for efficient lookups and prevents duplicate entities.
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
        """
        Retrieves all memories for a given session.
        """
        if not self.redis_client:
            return []

        session_key = self._get_session_key(session_id)
        try:
            all_memories_raw = self.redis_client.hgetall(session_key)
            return [json.loads(mem_json) for mem_json in all_memories_raw.values()]
        except Exception as e:
            logger.error(f"Error retrieving all memories for session {session_id}: {e}")
            return []

if __name__ == '__main__':
    # Example Usage
    import uuid
    logging.basicConfig(level=logging.INFO)

    mm = MemoryManager()

    if mm.redis_client:
        session_id = str(uuid.uuid4())
        print(f"--- Testing MemoryManager with session ID: {session_id} ---")

        # 1. Add some memories
        memories_to_add = [
            {"entity": "Jules", "emotional_state": "positive", "salience": 2},
            {"entity": "Google", "emotional_state": "neutral", "salience": 0}
        ]
        mm.add_memories(session_id, memories_to_add)

        # 2. Retrieve all memories
        all_mems = mm.get_all_memories(session_id)
        print("\nRetrieved all memories:")
        print(json.dumps(all_mems, indent=2))
        assert len(all_mems) == 2

        # 3. Add another memory (should update if entity exists, but here it's new)
        mm.add_memories(session_id, [{"entity": "AI", "emotional_state": "neutral", "salience": 1}])
        all_mems_updated = mm.get_all_memories(session_id)
        print("\nRetrieved updated memories:")
        print(json.dumps(all_mems_updated, indent=2))
        assert len(all_mems_updated) == 3

        # Clean up the test data
        mm.redis_client.delete(mm._get_session_key(session_id))
        print(f"\nCleaned up test data for session {session_id}.")
    else:
        print("Cannot run MemoryManager example because Redis connection failed.")
