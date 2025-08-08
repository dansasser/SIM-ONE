import unittest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from mcp_server.main import app

class TestMCP(unittest.TestCase):
    """The final, consolidated test suite for the mCP Server."""

    def setUp(self):
        self.client = TestClient(app)

    def test_01_basic_endpoints(self):
        """Tests the status, protocols, and templates endpoints."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        response = self.client.get("/protocols")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ReasoningAndExplanationProtocol", response.json())

        response = self.client.get("/templates")
        self.assertEqual(response.status_code, 200)
        self.assertIn("writing_team", response.json())

    def test_02_simple_sequential_workflow(self):
        """Tests a simple sequential workflow."""
        test_request = {
            "protocol_names": ["ReasoningAndExplanationProtocol"],
            "initial_data": {"facts": ["a"], "rules": [[["a"], "b"]]}
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIsNone(response_json.get("error"))
        self.assertIn("ReasoningAndExplanationProtocol", response_json["results"])
        self.assertIn("conclusions", response_json["results"]["ReasoningAndExplanationProtocol"]["result"])

    def test_03_parallel_workflow(self):
        """Tests a parallel workflow."""
        test_request = {
            "protocol_names": ["ValidationAndVerificationProtocol", "EmotionalStateLayerProtocol"],
            "coordination_mode": "Parallel",
            "initial_data": {"user_input": "This is great.", "rules": []}
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIsNone(response_json.get("error"))
        self.assertIn("ValidationAndVerificationProtocol", response_json["results"])
        self.assertIn("EmotionalStateLayerProtocol", response_json["results"])

    @patch('mcp_server.tools.rag_tools.google_search')
    @patch('mcp_server.tools.rag_tools.view_text_website')
    def test_04_writing_team_workflow(self, mock_view_text, mock_google_search):
        """Tests the full multi-agent pipeline with mocked RAG."""
        mock_google_search.return_value = "http://mock.url"
        mock_view_text.return_value = "Mocked web content."

        test_request = {
            "template_name": "writing_team",
            "initial_data": {"topic": "AI Ethics"}
        }
        response = self.client.post("/execute", json=test_request)

        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIsNone(response_json.get("error"), f"Workflow returned an error: {response_json.get('error')}")

        results = response_json["results"]
        self.assertIn("IdeatorProtocol", results)
        self.assertIn("DrafterProtocol", results)
        self.assertIn("CriticProtocol", results)
        self.assertIn("RevisorProtocol", results)
        self.assertIn("SummarizerProtocol", results)
        self.assertIn("summary", results["SummarizerProtocol"]["result"])

if __name__ == "__main__":
    unittest.main()
