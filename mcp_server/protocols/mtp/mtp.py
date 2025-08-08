import logging
from typing import Dict, Any, List, Set

from mcp_server.memory_manager.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class MTP:
    """
    An enhanced implementation of the Memory Tagger Protocol (MTP).
    It now persists emotionally-tagged memories using the MemoryManager.
    """
    def __init__(self):
        self.memory_manager = MemoryManager()

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts entities, tags them with emotional context, and saves them to memory.
        """
        user_input = data.get("user_input", "")
        esl_results = data.get("EmotionalStateLayerProtocol", {})
        session_id = data.get("session_id")

        logger.info(f"MTP: Analyzing input for session {session_id} with context: {esl_results}")

        if not user_input or not session_id:
            return {"status": "skipped", "reason": "Missing user_input or session_id."}

        # Get existing entities from memory to avoid re-tagging
        all_memories = self.memory_manager.get_all_memories(session_id)
        existing_entities = {mem.get("entity") for mem in all_memories}

        # Simple entity extraction
        words = user_input.split()
        potential_entities = {word.strip(".,?!") for i, word in enumerate(words) if i > 0 and word.istitle()}
        new_entities = list(potential_entities - existing_entities)

        if not new_entities:
            return {"status": "no_new_entities", "tagged_count": 0}

        # Create structured memory objects
        emotional_state = esl_results.get("emotional_state", "unknown")
        salience = esl_results.get("salience", 0)

        new_memories = []
        for entity in new_entities:
            new_memories.append({
                "entity": entity,
                "emotional_state": emotional_state,
                "salience": salience,
                "source_input": user_input
            })

        # Persist the new memories
        self.memory_manager.add_memories(session_id, new_memories)

        logger.info(f"MTP: Persisted {len(new_memories)} new memories for session {session_id}.")

        return {"status": "success", "tagged_count": len(new_memories)}

if __name__ == '__main__':
    # Example Usage
    import uuid
    logging.basicConfig(level=logging.INFO)

    mtp = MTP()

    if mtp.memory_manager.redis_client:
        session_id = str(uuid.uuid4())
        print(f"--- Testing Enhanced MTP with session: {session_id} ---")

        sample_data = {
            "user_input": "I think Google is a fascinating company.",
            "EmotionalStateLayerProtocol": {"emotional_state": "positive", "salience": 1},
            "session_id": session_id
        }

        result = mtp.execute(sample_data)
        print(f"Execution result: {result}")
        assert result['status'] == 'success' and result['tagged_count'] == 1

        # Verify that the memory was saved
        memories = mtp.memory_manager.get_all_memories(session_id)
        print(f"Retrieved memories: {memories}")
        assert len(memories) == 1
        assert memories[0]['entity'] == 'Google'
        assert memories[0]['emotional_state'] == 'positive'

        # Clean up
        mtp.memory_manager.redis_client.delete(f"memory:{session_id}")
    else:
        print("Cannot run MTP example because Redis connection failed.")
