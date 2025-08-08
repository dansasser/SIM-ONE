import unittest
import json
from fastapi.testclient import TestClient
from mcp_server.main import app

class TestStreamingApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_websocket_execution(self):
        with self.client.websocket_connect("/ws/execute") as websocket:
            test_request = {
                "protocol_names": [
                    "ReasoningAndExplanationProtocol",
                    "SummarizerProtocol"
                ],
                "initial_data": {
                    "facts": ["has_feathers"],
                    "rules": [[["has_feathers"], "is_bird"]]
                },
                "coordination_mode": "Sequential" # Note: streaming is always sequential
            }
            websocket.send_json(test_request)

            # Receive REP result
            rep_data = websocket.receive_json()
            self.assertEqual(rep_data["protocol"], "ReasoningAndExplanationProtocol")
            self.assertIn("conclusions", rep_data["result"])
            self.assertEqual(rep_data["result"]["conclusions"], ["is_bird"])

            # Receive SP result
            sp_data = websocket.receive_json()
            self.assertEqual(sp_data["protocol"], "SummarizerProtocol")
            self.assertIn("summary", sp_data["result"])
            self.assertIn("[Mock Summary]", sp_data["result"]["summary"])

            # Receive completion message
            completion_data = websocket.receive_json()
            self.assertEqual(completion_data["status"], "complete")
            self.assertIn("session_id", completion_data)

if __name__ == "__main__":
    unittest.main()
