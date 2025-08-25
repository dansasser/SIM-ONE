import unittest
import os
import json
from pathlib import Path

from mcp_server.memory_manager.memory_manager import MemoryManager
from mcp_server.database.memory_database import initialize_database, DB_FILE, get_db_connection

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

    def test_rehearsal_and_recency(self):
        """
        Tests that rehearsal_count and last_accessed are updated on retrieval.
        """
        session_id = "test_rehearsal"
        self.memory_manager.add_memories(session_id, [{"entity": "Test", "source_input": "rehearsal test"}])

        # Retrieve the memory once
        retrieved_memories = self.memory_manager.search_memories(session_id, "rehearsal")
        self.assertEqual(len(retrieved_memories), 1)

        # Check the database directly
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT rehearsal_count, last_accessed FROM memories WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        self.assertEqual(row['rehearsal_count'], 1)
        self.assertIsNotNone(row['last_accessed'])

    def test_contextual_search_with_actors(self):
        """
        Tests that search results are boosted by matching actors in the context.
        """
        session_id = "test_actors"
        memories = [
            {"entity": "Meeting", "source_input": "Project meeting notes.", "actors": ["Alice"]},
            {"entity": "Meeting", "source_input": "Project meeting notes with Bob.", "actors": ["Alice", "Bob"]}
        ]
        self.memory_manager.add_memories(session_id, memories)

        # Search without context
        results_no_context = self.memory_manager.search_memories(session_id, "meeting")
        # Without context, the order is not guaranteed, so we don't assert order here.

        # Search with context for "Bob"
        context = {"actors": ["Bob"]}
        results_with_context = self.memory_manager.search_memories(session_id, "meeting", context=context)

        # The memory with "Bob" should be ranked first
        self.assertEqual(len(results_with_context), 2)
        self.assertIn("Bob", results_with_context[0]['content'])

if __name__ == '__main__':
    unittest.main()
