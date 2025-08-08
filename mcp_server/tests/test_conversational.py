import unittest
from fastapi.testclient import TestClient
from mcp_server.main import app

class TestConversationalProtocols(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_conversational_workflow(self):
        test_request = {
            "protocol_names": [
                "EmotionalStateLayerProtocol",
                "MemoryTaggerProtocol"
            ],
            "initial_data": {
                "user_input": "This is a great day, I love working with Jules at Google."
            },
            "coordination_mode": "Parallel"
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)

        response_json = response.json()
        results = response_json.get("results", {})

        # Check ESL results
        esl_result = results.get("EmotionalStateLayerProtocol", {})
        self.assertEqual(esl_result.get("emotional_state"), "positive")

        # Check MTP results
        mtp_result = results.get("MemoryTaggerProtocol", {})
        new_entities = mtp_result.get("newly_tagged_entities", [])
        self.assertIn("Jules", new_entities)
        self.assertIn("Google", new_entities)

if __name__ == "__main__":
    unittest.main()
