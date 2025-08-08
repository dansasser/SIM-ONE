import unittest
from fastapi.testclient import TestClient
from mcp_server.main import app

class TestWorkflowTemplates(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_list_templates(self):
        response = self.client.get("/templates")
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("analyze_only", response_json)
        self.assertIn("full_reasoning", response_json)

    def test_execute_with_template(self):
        test_request = {
            "template_name": "full_reasoning",
            "initial_data": {
                "facts": ["has_feathers"],
                "rules": [[["has_feathers"], "is_bird"]]
            }
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)

        response_json = response.json()
        self.assertIn("results", response_json)

        # Check that the correct protocols from the template were run
        self.assertIn("ReasoningAndExplanationProtocol", response_json["results"])
        self.assertIn("SummarizerProtocol", response_json["results"])

    def test_execute_with_invalid_template(self):
        test_request = {
            "template_name": "non_existent_template",
            "initial_data": {}
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200) # The endpoint itself doesn't fail
        response_json = response.json()
        self.assertIn("error", response_json)
        self.assertEqual(response_json["error"], "Template 'non_existent_template' not found.")

if __name__ == "__main__":
    unittest.main()
