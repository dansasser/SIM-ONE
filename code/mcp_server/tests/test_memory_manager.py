import unittest
import os
import json
from pathlib import Path

from mcp_server.memory_manager.memory_manager import MemoryManager
from mcp_server.database.memory_database import initialize_database, DB_FILE

class TestMemoryManagerWithSalience(unittest.TestCase):

    def setUp(self):
        # Ensure a clean database for each test
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        initialize_database()
        self.memory_manager = MemoryManager()

    def tearDown(self):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)

    def test_emotional_salience_storage(self):
        """
        Tests that memories with emotional content are stored with a higher salience.
        """
        session_id = "test_session_1"
        memories_to_add = [
            {"entity": "Neutral", "source_input": "The sky is blue."},
            {"entity": "Emotional", "source_input": "I am so incredibly happy and thrilled today!"}
        ]
        self.memory_manager.add_memories(session_id, memories_to_add)

        all_memories = self.memory_manager.get_all_memories(session_id)

        neutral_memory = next(m for m in all_memories if m['entity'] == 'Neutral')
        emotional_memory = next(m for m in all_memories if m['entity'] == 'Emotional')

        # The neutral memory should have a low salience (close to 0)
        self.assertLess(neutral_memory['emotional_salience'], 0.1)
        # The emotional memory should have a high salience
        self.assertGreater(emotional_memory['emotional_salience'], 0.6)

    def test_search_prioritizes_salience(self):
        """
        Tests that search results are ranked higher based on emotional salience.
        """
        session_id = "test_session_2"
        memories_to_add = [
            {"entity": "Event", "source_input": "We had a meeting about the project."},
            {"entity": "Event", "source_input": "The project meeting was an amazing, fantastic success!"}
        ]
        self.memory_manager.add_memories(session_id, memories_to_add)

        # Search for a term present in both memories
        search_results = self.memory_manager.search_memories(session_id, "project meeting", top_k=2)

        # The first result should be the more emotional one
        self.assertIn("fantastic success", search_results[0]['content'])
        self.assertIn("had a meeting", search_results[1]['content'])

    def test_memory_isolation(self):
        """
        Tests that memories are isolated between different sessions.
        """
        session_1_id = "session_1"
        session_2_id = "session_2"

        # Add memory to session 1
        self.memory_manager.add_memories(session_1_id, [{"entity": "Project", "source_input": "Session 1 memory."}])

        # Add memory to session 2
        self.memory_manager.add_memories(session_2_id, [{"entity": "Project", "source_input": "Session 2 memory."}])

        # Get memories for session 1
        session_1_memories = self.memory_manager.get_all_memories(session_1_id)
        self.assertEqual(len(session_1_memories), 1)
        self.assertEqual(session_1_memories[0]['content'], "Session 1 memory.")

        # Get memories for session 2
        session_2_memories = self.memory_manager.get_all_memories(session_2_id)
        self.assertEqual(len(session_2_memories), 1)
        self.assertEqual(session_2_memories[0]['content'], "Session 2 memory.")

if __name__ == '__main__':
    unittest.main()
