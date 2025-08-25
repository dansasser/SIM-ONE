import unittest
import json
import logging
import os
from io import StringIO
from fastapi.testclient import TestClient

from mcp_server.main import app
from mcp_server.logging_config import setup_logging, JsonFormatter

from mcp_server.security import key_manager

class TestProductionSetup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['VALID_API_KEYS'] = 'admin-key,user-key'
        key_manager.initialize_api_keys()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(key_manager.API_KEYS_FILE):
            os.remove(key_manager.API_KEYS_FILE)

    def setUp(self):
        self.client = TestClient(app)
        self.admin_headers = {"X-API-Key": "admin-key"}
        self.user_headers = {"X-API-Key": "user-key"}

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

    def test_metrics_endpoint(self):
        """Tests the /metrics endpoint, including RBAC."""
        # Test without auth
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 403)

        # Test with non-admin user
        response = self.client.get("/metrics", headers=self.user_headers)
        self.assertEqual(response.status_code, 403)

        # Test with admin user
        response = self.client.get("/metrics", headers=self.admin_headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("system", data)
        self.assertIn("application", data)
        self.assertIn("cpu_percent", data["system"])
        self.assertIn("memory_stats", data["application"])

    def test_startup_script_validation(self):
        """Tests that the production startup script validates environment variables."""
        import subprocess

        # Test case 1: Fails when a required variable is missing
        env = os.environ.copy()
        if "VALID_API_KEYS" in env:
            del env["VALID_API_KEYS"]

        process = subprocess.run(
            ['./run_production.sh'],
            cwd='.',
            capture_output=True,
            text=True,
            env=env
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Error: Required environment variable VALID_API_KEYS is not set.", process.stdout)

        # Test case 2: Fails for missing OpenAI key when backend is 'openai'
        env = os.environ.copy()
        env['NEURAL_ENGINE_BACKEND'] = 'openai'
        if "OPENAI_API_KEY" in env:
            del env["OPENAI_API_KEY"]

        process = subprocess.run(
            ['./run_production.sh'],
            cwd='.',
            capture_output=True,
            text=True,
            env=env
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("OPENAI_API_KEY is not set", process.stdout)

if __name__ == '__main__':
    unittest.main()
