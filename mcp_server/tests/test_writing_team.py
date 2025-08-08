import unittest
from fastapi.testclient import TestClient
from mcp_server.main import app

class TestWritingTeamWorkflow(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_full_writing_team_workflow(self):
        """
        Tests the full multi-agent pipeline from topic to summary.
        This is an end-to-end test of the most complex workflow.

        NOTE: This test will be slow as it orchestrates multiple (mocked) LLM calls.
        It also expects the RAG components to fail gracefully.
        """
        test_request = {
            "template_name": "writing_team",
            "initial_data": {
                "topic": "The benefits of a protocol-driven architecture for AI agents"
            }
        }
        response = self.client.post("/execute", json=test_request)
        self.assertEqual(response.status_code, 200)

        response_json = response.json()
        results = response_json.get("results", {})

        # Check that the final context contains output from all key protocols
        self.assertIn("IdeatorProtocol", results)
        self.assertIn("DrafterProtocol", results)
        self.assertIn("CriticProtocol", results)
        self.assertIn("RevisorProtocol", results)
        self.assertIn("SummarizerProtocol", results)

        # Check the output of the first agent
        self.assertIn("ideas", results["IdeatorProtocol"])

        # Check the output of the drafter
        self.assertIn("draft_text", results["DrafterProtocol"])

        # Check the output of the critic
        self.assertIn("feedback", results["CriticProtocol"])

        # Check the output of the revisor
        self.assertIn("revised_draft_text", results["RevisorProtocol"])

        # Check the final output of the summarizer
        self.assertIn("summary", results["SummarizerProtocol"])
        self.assertEqual(results["SummarizerProtocol"]["status"], "success")

        # A simple check to see if the summary is not empty
        self.assertTrue(len(results["SummarizerProtocol"]["summary"]) > 0)

if __name__ == "__main__":
    unittest.main()
