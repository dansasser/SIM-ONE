import unittest
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

    def test_execute_workflow(self):
        test_request = {
            "protocol_names": ["ReasoningAndExplanationProtocol"],
            "initial_data": {
                "facts": ["has_feathers", "flies", "lays_eggs"],
                "rules": [
                    [["has_feathers"], "is_bird"],
                    [["flies", "is_bird"], "is_flying_bird"],
                    [["is_bird", "lays_eggs"], "is_oviparous_bird"]
                ]
            }
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)

        response_json = response.json()
        self.assertIn("ReasoningAndExplanationProtocol", response_json)

        conclusions = response_json["ReasoningAndExplanationProtocol"]["conclusions"]
        self.assertIn("is_bird", conclusions)
        self.assertIn("is_flying_bird", conclusions)
        self.assertIn("is_oviparous_bird", conclusions)

if __name__ == "__main__":
    unittest.main()
