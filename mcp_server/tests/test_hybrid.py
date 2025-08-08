import unittest
from fastapi.testclient import TestClient
from mcp_server.main import app

class TestHybridWorkflow(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_execute_hybrid_workflow(self):
        test_request = {
            "protocol_names": [
                "ReasoningAndExplanationProtocol",
                "SummarizerProtocol"
            ],
            "initial_data": {
                "facts": ["has_feathers", "flies", "lays_eggs"],
                "rules": [
                    [["has_feathers"], "is_bird"],
                    [["flies", "is_bird"], "is_flying_bird"],
                    [["is_bird", "lays_eggs"], "is_oviparous_bird"]
                ]
            },
            "coordination_mode": "Sequential"
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)

        response_json = response.json()

        self.assertIn("session_id", response_json)
        self.assertIn("results", response_json)

        self.assertIn("ReasoningAndExplanationProtocol", response_json["results"])
        self.assertIn("SummarizerProtocol", response_json["results"])

        rep_conclusions = response_json["results"]["ReasoningAndExplanationProtocol"]["conclusions"]
        self.assertIn("is_bird", rep_conclusions)

        sp_result = response_json["results"]["SummarizerProtocol"]
        self.assertEqual(sp_result["status"], "success")
        self.assertIn("[Mock Summary]", sp_result["summary"])

if __name__ == "__main__":
    unittest.main()
