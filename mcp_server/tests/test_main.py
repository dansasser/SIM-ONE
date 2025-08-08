import unittest
import json
from fastapi.testclient import TestClient
from mcp_server.main import app

class TestApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "mCP Server is running."})

    def test_list_protocols(self):
        response = self.client.get("/protocols")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ReasoningAndExplanationProtocol", response.json())

    def test_list_templates(self):
        response = self.client.get("/templates")
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("analyze_only", response_json)
        self.assertIn("full_reasoning", response_json)

    def test_execute_sequential_workflow(self):
        test_request = {
            "protocol_names": ["ReasoningAndExplanationProtocol"],
            "initial_data": { "facts": ["a"], "rules": [[["a"], "b"]] }
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("ReasoningAndExplanationProtocol", response_json["results"])
        self.assertEqual(response_json["results"]["ReasoningAndExplanationProtocol"]["conclusions"], ["b"])

    def test_execute_parallel_workflow(self):
        test_request = {
            "protocol_names": ["ValidationAndVerificationProtocol", "EmotionalStateLayerProtocol"],
            "coordination_mode": "Parallel",
            "initial_data": { "user_input": "This is great.", "rules": [] }
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("ValidationAndVerificationProtocol", response_json["results"])
        self.assertIn("EmotionalStateLayerProtocol", response_json["results"])
        self.assertEqual(response_json["results"]["ValidationAndVerificationProtocol"]["validation_status"], "success")
        self.assertEqual(response_json["results"]["EmotionalStateLayerProtocol"]["emotional_state"], "positive")

    def test_execute_hybrid_workflow(self):
        test_request = {
            "template_name": "full_reasoning",
            "initial_data": { "facts": ["a"], "rules": [[["a"], "b"]] }
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("SummarizerProtocol", response_json["results"])
        self.assertIn("[Mock Summary]", response_json["results"]["SummarizerProtocol"]["summary"])

    def test_conversational_workflow(self):
        # Step 1: Send a message and get a session ID
        request1 = {
            "template_name": "analyze_only",
            "initial_data": { "user_input": "My name is Jules." }
        }
        response1 = self.client.post("/execute", json=request1)
        self.assertEqual(response1.status_code, 200)
        response1_json = response1.json()
        session_id = response1_json["session_id"]
        self.assertIsNotNone(session_id)
        self.assertIn("Jules", response1_json["results"]["MemoryTaggerProtocol"]["newly_tagged_entities"])

        # Step 2: Send another message in the same session
        request2 = {
            "session_id": session_id,
            "template_name": "analyze_only",
            "initial_data": { "user_input": "I am happy." }
        }
        response2 = self.client.post("/execute", json=request2)
        self.assertEqual(response2.status_code, 200)
        response2_json = response2.json()
        # Check that "Jules" is not a *new* entity
        self.assertEqual(len(response2_json["results"]["MemoryTaggerProtocol"]["newly_tagged_entities"]), 0)
        self.assertEqual(response2_json["results"]["EmotionalStateLayerProtocol"]["emotional_state"], "positive")

    def test_websocket_execution(self):
        with self.client.websocket_connect("/ws/execute") as websocket:
            test_request = {
                "protocol_names": ["ReasoningAndExplanationProtocol", "SummarizerProtocol"],
                "initial_data": { "facts": ["c"], "rules": [[["c"], "d"]] }
            }
            websocket.send_json(test_request)

            rep_data = websocket.receive_json()
            self.assertEqual(rep_data["protocol"], "ReasoningAndExplanationProtocol")

            sp_data = websocket.receive_json()
            self.assertEqual(sp_data["protocol"], "SummarizerProtocol")

            completion_data = websocket.receive_json()
            self.assertEqual(completion_data["status"], "complete")

if __name__ == "__main__":
    unittest.main()
