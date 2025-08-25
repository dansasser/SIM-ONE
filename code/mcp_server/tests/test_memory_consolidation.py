import unittest
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

from mcp_server.memory_manager.memory_consolidation import MemoryConsolidationEngine
from mcp_server.memory_manager.memory_manager import MemoryManager
from mcp_server.database.memory_database import initialize_database, get_db_connection, DB_FILE

class TestMemoryConsolidation(unittest.TestCase):

    def setUp(self):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        initialize_database()
        self.memory_manager = MemoryManager()
        self.consolidation_engine = MemoryConsolidationEngine()
        self.session_id = "consolidation_test"

    def tearDown(self):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)

    def test_archive_old_memories(self):
        """Tests that old, low-importance memories are archived."""
        # Add an old memory that should be archived
        self.memory_manager.add_memories(self.session_id, [{
            "entity": "Old News",
            "source_input": "This is some old news.",
            "rehearsal_count": 0,
            "emotional_salience": 0.1
        }])

        # Manually set the timestamp to be old
        conn = get_db_connection()
        old_date = datetime.now() - timedelta(days=40)
        cursor = conn.cursor()
        cursor.execute("UPDATE memories SET timestamp = ?", (old_date.strftime('%Y-%m-%d %H:%M:%S'),))
        conn.commit()
        conn.close()

        # Run the archiving process
        self.consolidation_engine._archive_old_memories(self.session_id)

        # Check that the memory is now archived
        memories = self.memory_manager.get_all_memories(self.session_id)
        self.assertEqual(memories[0]['memory_type'], 'archived')

    def test_contradiction_detection(self):
        """Tests that simple contradictions are detected and flagged."""
        memories = [
            {"entity": "Fact", "source_input": "The weather today is wonderful."},
            {"entity": "Fact", "source_input": "The weather today is not wonderful."}
        ]
        self.memory_manager.add_memories(self.session_id, memories)

        # Run the contradiction detection in isolation
        contradictions_found = self.consolidation_engine._find_contradictions(self.session_id)

        # Manually add them to the database for verification
        conn = get_db_connection()
        if conn and contradictions_found:
            try:
                cursor = conn.cursor()
                for mem1_id, mem2_id, reason in contradictions_found:
                    cursor.execute("INSERT INTO memory_contradictions (memory_id_1, memory_id_2, reason) VALUES (?, ?, ?)", (mem1_id, mem2_id, reason))
                conn.commit()
            finally:
                conn.close()

        # Check the contradictions table
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memory_contradictions")
        contradictions = cursor.fetchall()
        conn.close()

        self.assertEqual(len(contradictions), 1)
        self.assertIn("Simple negation detected", contradictions[0]['reason'])

    def test_merge_similar_memories(self):
        """Tests that very similar memories are merged."""
        memories = [
            {"entity": "Merge", "source_input": "This is a test sentence for merging.", "rehearsal_count": 2},
            {"entity": "Merge", "source_input": "This is a test sentence for merging!", "rehearsal_count": 3}
        ]
        self.memory_manager.add_memories(self.session_id, memories)

        # Run the merge process
        self.consolidation_engine.run_consolidation_cycle(self.session_id)

        # Get memories
        all_memories = self.memory_manager.get_all_memories(self.session_id)

        archived_memories = [m for m in all_memories if m['memory_type'] == 'archived']
        semantic_memories = [m for m in all_memories if m['memory_type'] == 'semantic']

        # The two original memories should be archived
        self.assertEqual(len(archived_memories), 2)
        # There should be one new, merged (semantic) memory
        self.assertEqual(len(semantic_memories), 1)
        # The new memory should have the combined rehearsal count
        self.assertEqual(semantic_memories[0]['rehearsal_count'], 5)

if __name__ == '__main__':
    unittest.main()
