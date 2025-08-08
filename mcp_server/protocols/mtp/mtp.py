import logging
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)

class MTP:
    """
    A simple implementation of the Memory Tagger Protocol (MTP).
    """

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts entities from the user input and adds them to memory.

        Args:
            data: The input data, expected to contain 'user_input' and 'history'.

        Returns:
            A dictionary with the newly tagged entities.
        """
        user_input = data.get("user_input", "")
        history = data.get("history", [])
        logger.info(f"MTP: Analyzing input for entities: '{user_input[:50]}...'")

        if not user_input:
            return {"newly_tagged_entities": [], "status": "skipped"}

        # Extract existing entities from history
        # In a real system, memory would be more structured. Here we just use a flat list of tags.
        memory: Set[str] = set()
        for turn in history:
            # Assuming MTP results are stored in the response
            if 'server_response' in turn and 'results' in turn['server_response']:
                mtp_res = turn['server_response']['results'].get('MemoryTaggerProtocol', {})
                memory.update(mtp_res.get('newly_tagged_entities', []))

        # A very simple entity extraction: find capitalized words not at the start of a sentence.
        # This is a placeholder for a real NER model.
        words = user_input.split()
        potential_entities = {word.strip(".,?!") for i, word in enumerate(words) if i > 0 and word.istitle()}

        newly_tagged_entities = list(potential_entities - memory)

        logger.info(f"MTP: Found {len(newly_tagged_entities)} new entities: {newly_tagged_entities}")

        return {
            "newly_tagged_entities": newly_tagged_entities,
            "status": "success"
        }

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    mtp_protocol = MTP()

    print("--- Testing MTP Protocol ---")

    # --- Test Case 1: New entities ---
    history1 = []
    data1 = {"user_input": "My name is Jules and I work at Google.", "history": history1}
    result1 = mtp_protocol.execute(data1)
    print(f"Input: '{data1['user_input']}' -> New Entities: {result1['newly_tagged_entities']}")
    assert "Jules" in result1['newly_tagged_entities']
    assert "Google" in result1['newly_tagged_entities']

    # --- Test Case 2: Some existing entities ---
    history2 = [
        {"server_response": {"results": {"MemoryTaggerProtocol": {"newly_tagged_entities": ["Jules"]}}}}
    ]
    data2 = {"user_input": "Jules is still working at Google.", "history": history2}
    result2 = mtp_protocol.execute(data2)
    print(f"Input: '{data2['user_input']}' -> New Entities: {result2['newly_tagged_entities']}")
    assert "Jules" not in result2['newly_tagged_entities']
    assert "Google" in result2['newly_tagged_entities']

    # --- Test Case 3: No new entities ---
    history3 = [
        {"server_response": {"results": {"MemoryTaggerProtocol": {"newly_tagged_entities": ["Jules", "Google"]}}}}
    ]
    data3 = {"user_input": "Yes, Jules from Google.", "history": history3}
    result3 = mtp_protocol.execute(data3)
    print(f"Input: '{data3['user_input']}' -> New Entities: {result3['newly_tagged_entities']}")
    assert not result3['newly_tagged_entities']
