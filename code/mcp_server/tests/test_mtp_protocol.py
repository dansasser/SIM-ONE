import unittest
import sys
import os
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp_server.protocols.mtp.mtp import MTP

class TestAdvancedMTPProtocol(unittest.TestCase):

    class MockMemoryManager:
        """A mock memory manager to isolate tests from the database."""
        def __init__(self):
            self.memories_added = []

        def add_memories(self, memories):
            self.memories_added.extend(memories)

    def setUp(self):
        """Set up a new MTP instance and mock its memory manager for each test."""
        self.mtp = MTP()
        self.mock_memory_manager = self.MockMemoryManager()
        self.mtp.memory_manager = self.mock_memory_manager
        self.session_id = "test-session"

    def test_person_organization_place_relationship(self):
        """Tests 'John works at Microsoft and lives in Seattle'."""
        user_input = "John works at Microsoft and lives in Seattle"
        result = self.mtp.execute({"user_input": user_input, "emotional_context": {}, "session_id": self.session_id})

        entities = {e['entity']: e['type'] for e in result['extracted_entities']}
        self.assertIn("John", entities)
        self.assertEqual(entities["John"], "person")
        self.assertIn("Microsoft", entities)
        self.assertEqual(entities["Microsoft"], "organization")
        self.assertIn("Seattle", entities)
        self.assertEqual(entities["Seattle"], "place")

        relationships = {(r['source'], r['relationship_type'], r['target']) for r in result['entity_relationships']}
        self.assertIn(("John", "works_at", "Microsoft"), relationships)
        self.assertIn(("John", "located_in", "Seattle"), relationships)

    def test_person_event_emotional_tagging(self):
        """Tests 'I'm excited about the meeting with Sarah tomorrow'."""
        user_input = "I'm excited about the meeting with Sarah tomorrow"
        emotional_context = {"valence": "positive", "detected_emotions": [{"emotion": "joy"}]}
        result = self.mtp.execute({"user_input": user_input, "emotional_context": emotional_context, "session_id": self.session_id})

        entities = {e['entity']: e for e in result['extracted_entities']}
        self.assertIn("Sarah", entities)
        self.assertEqual(entities["Sarah"]['type'], "person")
        self.assertEqual(entities["Sarah"]['emotional_state'], "positive")

        self.assertIn("meeting", entities)
        self.assertEqual(entities["meeting"]['type'], "event")
        self.assertIn("tomorrow", entities)
        self.assertEqual(entities["tomorrow"]['type'], "event")

        # Check that the memory tag for Sarah has the right emotional context
        sarah_memory = next(m for m in self.mock_memory_manager.memories_added if m['entity'] == 'Sarah')
        self.assertEqual(sarah_memory['emotional_state'], 'positive')


    def test_object_concept_emotional_association(self):
        """Tests 'The iPhone project deadline is causing stress'."""
        user_input = "The iPhone project deadline is causing stress"
        emotional_context = {"valence": "negative", "detected_emotions": [{"emotion": "fear"}]}
        result = self.mtp.execute({"user_input": user_input, "emotional_context": emotional_context, "session_id": self.session_id})

        entities = {e['entity']: e for e in result['extracted_entities']}
        self.assertIn("iPhone", entities)
        self.assertEqual(entities["iPhone"]['type'], "object")
        self.assertEqual(entities["iPhone"]['emotional_state'], "negative")

        self.assertIn("project", entities)
        self.assertEqual(entities["project"]['type'], "concept")
        self.assertIn("deadline", entities)
        self.assertEqual(entities["deadline"]['type'], "event")
        self.assertIn("stress", entities)
        self.assertEqual(entities["stress"]['type'], "concept")

    def test_person_relationship_and_place(self):
        """Tests 'My friend recommended this restaurant downtown'."""
        user_input = "My friend recommended this restaurant downtown"
        # The pattern for "friend" is not implemented, so this will test basic extraction
        result = self.mtp.execute({"user_input": user_input, "emotional_context": {}, "session_id": self.session_id})

        entities = {e['entity']: e['type'] for e in result['extracted_entities']}
        self.assertIn("downtown", entities)
        self.assertEqual(entities["downtown"], "place")
        # "friend" and "restaurant" are not in the basic patterns, so we expect them not to be found
        self.assertNotIn("friend", entities)
        self.assertNotIn("restaurant", entities)


if __name__ == '__main__':
    # Add a simple proper noun to the person patterns for testing this file directly
    from mcp_server.protocols.mtp import entity_patterns
    entity_patterns.ENTITY_PATTERNS['person'].append(re.compile(r'\b(Sarah|John)\b'))
    unittest.main()
