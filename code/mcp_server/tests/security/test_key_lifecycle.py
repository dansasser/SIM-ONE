import unittest
import os
from fastapi.testclient import TestClient

from mcp_server.main import app
from mcp_server.security import key_manager


class TestApiKeyLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['VALID_API_KEYS'] = 'admin-key-init'
        key_manager.initialize_api_keys()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(key_manager.API_KEYS_FILE):
            os.remove(key_manager.API_KEYS_FILE)

    def setUp(self):
        self.client = TestClient(app)
        self.admin_headers = {"X-API-Key": "admin-key-init"}
        self.user_headers = {"X-API-Key": "user-key-nonadmin"}

        # Create a non-admin user to verify RBAC denies
        key_manager.add_api_key("user-key-nonadmin", "user", "test_user")

    def test_admin_list_create_delete_api_keys(self):
        # Non-admin should be denied
        r = self.client.get("/admin/api-keys", headers=self.user_headers)
        self.assertEqual(r.status_code, 403)

        # Admin can list
        r = self.client.get("/admin/api-keys", headers=self.admin_headers)
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

        # Admin can create
        body = {"api_key": "temp-key", "role": "read-only", "user_id": "temp_user"}
        r = self.client.post("/admin/api-keys", headers=self.admin_headers, json=body)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))

        # Admin can delete by user_id
        r = self.client.delete("/admin/api-keys/temp_user", headers=self.admin_headers)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("success"))


if __name__ == '__main__':
    unittest.main()

