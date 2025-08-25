import unittest
import json
import logging
from io import StringIO
from fastapi.testclient import TestClient

from mcp_server.main import app
from mcp_server.logging_config import setup_logging, JsonFormatter

class TestProductionSetup(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Tests the basic /health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_health_detailed_endpoint(self):
        """Tests the /health/detailed endpoint."""
        response = self.client.get("/health/detailed")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("services", data)
        self.assertIn("database", data["services"])
        self.assertIn("redis", data["services"])
        # In the test environment, redis connects to fakeredis, so it should be "ok"
        self.assertEqual(data["services"]["redis"], "ok")

    def test_json_logging(self):
        """Tests that the logging output is in structured JSON format."""
        # Create a logger and a string stream to capture output
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setFormatter(JsonFormatter())

        logger = logging.getLogger("test_json_logger")
        logger.setLevel(logging.INFO)
        logger.addHandler(ch)

        # Log a message
        msg = "This is a test log message."
        logger.info(msg)

        # Get the captured log
        log_contents = log_capture_string.getvalue()
        log_capture_string.close()

        # Verify it's valid JSON and contains the message
        try:
            log_json = json.loads(log_contents)
            self.assertEqual(log_json["message"], msg)
            self.assertEqual(log_json["level"], "INFO")
        except json.JSONDecodeError:
            self.fail("Logging output is not valid JSON.")

if __name__ == '__main__':
    unittest.main()
