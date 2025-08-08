import unittest
from fastapi.testclient import TestClient
from mcp_server.main import app

class TestParallelExecution(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_execute_parallel_workflow(self):
        test_request = {
            "protocol_names": [
                "ReasoningAndExplanationProtocol",
                "ValidationAndVerificationProtocol"
            ],
            "initial_data": {
                "facts": ["has_feathers", "flies", "lays_eggs"],
                "rules": [
                    [["has_feathers"], "is_bird"],
                    [["flies", "is_bird"], "is_flying_bird"],
                    [["is_bird", "lays_eggs"], "is_oviparous_bird"]
                ]
            },
            "coordination_mode": "Parallel"
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)

        response_json = response.json()

        self.assertIn("session_id", response_json)
        self.assertIn("results", response_json)
        self.assertIn("resource_usage", response_json)

        self.assertIn("ReasoningAndExplanationProtocol", response_json["results"])
        self.assertIn("ValidationAndVerificationProtocol", response_json["results"])

        rep_conclusions = response_json["results"]["ReasoningAndExplanationProtocol"]["conclusions"]
        self.assertIn("is_bird", rep_conclusions)

        vvp_status = response_json["results"]["ValidationAndVerificationProtocol"]["validation_status"]
        self.assertEqual(vvp_status, "success")

        self.assertIn("ReasoningAndExplanationProtocol", response_json["resource_usage"])
        self.assertIn("ValidationAndVerificationProtocol", response_json["resource_usage"])

if __name__ == "__main__":
    unittest.main()
