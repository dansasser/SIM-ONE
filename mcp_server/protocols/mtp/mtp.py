import logging
import json
from typing import Dict, Any, List, Set

from mcp_server.memory_manager.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class MTP:
    """
    Memory Tagger Protocol (MTP) updated to handle sophisticated,
    multi-dimensional emotional data from the new ESLProtocol.
    """
    def __init__(self):
        self.memory_manager = MemoryManager()

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts entities, tags them with rich emotional context from ESL,
        and saves them to memory.
        """
        user_input = data.get("user_input", "")
        esl_results = data.get("EmotionalStateLayerProtocol", {})
        session_id = data.get("session_id")

        logger.info(f"MTP: Analyzing input for session {session_id} with context: {esl_results}")

        if not user_input or not session_id:
            return {"status": "skipped", "reason": "Missing user_input or session_id."}

        # Simple entity extraction (find capitalized words)
        # This is a naive implementation and could be improved with a real NER model.
        words = user_input.split()
        potential_entities = {word.strip(".,?!") for word in words if word.istitle()}

        if not potential_entities:
            return {"status": "no_new_entities", "tagged_count": 0}

        # Get the rich emotional data from the new ESL output
        detected_emotions = esl_results.get("detected_emotions", [])
        overall_valence = esl_results.get("valence", "neutral")

        if not detected_emotions:
            logger.info("MTP: No emotions detected by ESL. Skipping memory creation.")
            return {"status": "skipped", "reason": "No emotions detected."}

        # Get existing entities from memory to avoid re-tagging the same entity
        # in the same way, although this logic could be more sophisticated.
        all_memories = self.memory_manager.get_all_memories(session_id)
        existing_entities = {mem.get("entity") for mem in all_memories}
        new_entities = list(potential_entities - existing_entities)

        if not new_entities:
            return {"status": "no_new_entities_found", "tagged_count": 0}

        new_memories = []
        for entity in new_entities:
            new_memories.append({
                "entity": entity,
                "emotions": json.dumps(detected_emotions),  # Serialize list of dicts to a JSON string
                "overall_valence": overall_valence,
                "source_input": user_input
            })

        self.memory_manager.add_memories(session_id, new_memories)

        logger.info(f"MTP: Persisted {len(new_memories)} new memories for session {session_id}.")

        return {"status": "success", "tagged_count": len(new_memories)}

if __name__ == '__main__':
    import uuid
    logging.basicConfig(level=logging.INFO)

    mtp = MTP()

    # The MemoryManager uses Redis, so we need a connection to run the test.
    if mtp.memory_manager.redis_client:
        session_id = str(uuid.uuid4())
        print(f"--- Testing Updated MTP with session: {session_id} ---")

        # Sample output from the new, sophisticated ESL
        sample_esl_output = {
            "emotional_state": "gratitude",
            "valence": "positive",
            "intensity": 0.75,
            "salience": 0.57,
            "confidence": 0.9,
            "detected_emotions": [
                {
                    "emotion": "gratitude",
                    "intensity": 0.75,
                    "confidence": 0.9,
                    "dimension": "social",
                    "valence": "positive"
                }
            ],
            "analysis_type": "linguistic_patterns",
            "ml_ready": True
        }

        sample_data = {
            "user_input": "I am so grateful for the help from OpenAI.",
            "EmotionalStateLayerProtocol": sample_esl_output,
            "session_id": session_id
        }

        result = mtp.execute(sample_data)
        print(f"Execution result: {result}")
        assert result['status'] == 'success' and result['tagged_count'] == 1

        # Verify that the memory was saved correctly
        memories = mtp.memory_manager.get_all_memories(session_id)
        print(f"Retrieved memories: {memories}")
        assert len(memories) == 1

        saved_memory = memories[0]
        assert saved_memory['entity'] == 'OpenAI'
        assert saved_memory['overall_valence'] == 'positive'

        # Check the stored emotions
        retrieved_emotions = json.loads(saved_memory['emotions'])
        print(f"Retrieved and deserialized emotions: {retrieved_emotions}")
        assert isinstance(retrieved_emotions, list)
        assert len(retrieved_emotions) == 1
        assert retrieved_emotions[0]['emotion'] == 'gratitude'

        # Clean up the test data from Redis
        mtp.memory_manager.redis_client.delete(f"memory:{session_id}")
        print(f"--- Test complete and cleaned up for session: {session_id} ---")
    else:
        print("Cannot run MTP example because Redis connection is not available.")
