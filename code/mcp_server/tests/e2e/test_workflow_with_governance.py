import unittest
import os
from fastapi.testclient import TestClient

from mcp_server.main import app
from mcp_server.security import key_manager


class TestWorkflowWithGovernance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize API keys for authenticated access
        os.environ["VALID_API_KEYS"] = "e2e-admin-key,e2e-user-key"
        key_manager.initialize_api_keys()

    @classmethod
    def tearDownClass(cls):
        # Clean up generated key store
        if os.path.exists(key_manager.API_KEYS_FILE):
            os.remove(key_manager.API_KEYS_FILE)

    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"X-API-Key": "e2e-user-key"}

    def test_execute_rep_esl_mtp_and_governance_summary(self):
        payload = {
            "protocol_names": [
                "ReasoningAndExplanationProtocol",
                "EmotionalStateLayerProtocol",
                "MemoryTaggerProtocol",
            ],
            "initial_data": {
                # REP inputs
                "facts": ["Socrates is a man", "All men are mortal"],
                "rules": [
                    [["Socrates is a man", "All men are mortal"], "Socrates is mortal"]
                ],
                # Shared user input
                "user_input": "We are thrilled about the team's success and promotion at work.",
            },
        }

        resp = self.client.post("/execute", headers=self.headers, json=payload)
        self.assertEqual(resp.status_code, 200, msg=resp.text)
        body = resp.json()

        # Ensure governance summary is present
        self.assertIn("governance_summary", body)
        summary = body["governance_summary"]
        self.assertIn("quality_scores", summary)
        self.assertIn("is_coherent", summary)

        # Governance details also appear in results.governance
        self.assertIn("results", body)
        self.assertIn("governance", body["results"])  # aggregated diagnostics
        gov = body["results"]["governance"]
        self.assertIn("quality", gov)
        # Coherence should be evaluated since REP, ESL, MTP ran
        self.assertIn("coherence", gov)


if __name__ == "__main__":
    unittest.main()

