import unittest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from mcp_server.main import app
from mcp_server.security import key_manager

class TestEndpointAuthentication(unittest.TestCase):
    """
    Test endpoint authentication and authorization.
    Validates that all protected endpoints require proper API keys
    and implement correct role-based access control.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test API keys with different roles."""
        # Create test API keys for different roles
        os.environ['VALID_API_KEYS'] = 'test-admin-key,test-user-key,test-readonly-key'
        key_manager.initialize_api_keys()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test API keys."""
        if os.path.exists(key_manager.API_KEYS_FILE):
            os.remove(key_manager.API_KEYS_FILE)
    
    def setUp(self):
        self.client = TestClient(app)
        
        # Test headers for different roles
        self.admin_headers = {"X-API-Key": "test-admin-key"}
        self.user_headers = {"X-API-Key": "test-user-key"} 
        self.readonly_headers = {"X-API-Key": "test-readonly-key"}
        self.invalid_headers = {"X-API-Key": "invalid-key"}
        self.no_headers = {}
    
    def test_public_endpoints_no_auth_required(self):
        """Test that public endpoints don't require authentication."""
        public_endpoints = ["/", "/health"]
        
        for endpoint in public_endpoints:
            response = self.client.get(endpoint)
            self.assertNotEqual(response.status_code, 401, 
                              f"Public endpoint {endpoint} should not require auth")
            self.assertNotEqual(response.status_code, 403, 
                              f"Public endpoint {endpoint} should not require auth")
    
    def test_protected_endpoints_require_api_key(self):
        """Test that protected endpoints reject requests without API keys."""
        protected_endpoints = [
            ("/protocols", "GET"),
            ("/templates", "GET"), 
            ("/session/test-session", "GET"),
            ("/metrics", "GET")
        ]
        
        for endpoint, method in protected_endpoints:
            if method == "GET":
                response = self.client.get(endpoint, headers=self.no_headers)
            elif method == "POST":
                response = self.client.post(endpoint, headers=self.no_headers, json={})
            
            self.assertIn(response.status_code, [401, 403, 422], 
                         f"Protected endpoint {method} {endpoint} should require authentication")
    
    def test_protected_endpoints_reject_invalid_api_key(self):
        """Test that protected endpoints reject invalid API keys."""
        protected_endpoints = [
            ("/protocols", "GET"),
            ("/templates", "GET"),
            ("/session/test-session", "GET")
        ]
        
        for endpoint, method in protected_endpoints:
            if method == "GET":
                response = self.client.get(endpoint, headers=self.invalid_headers)
            elif method == "POST":
                response = self.client.post(endpoint, headers=self.invalid_headers, json={})
            
            self.assertEqual(response.status_code, 403, 
                           f"Endpoint {method} {endpoint} should reject invalid API key")
    
    def test_role_based_access_control(self):
        """Test that endpoints enforce proper role-based access control."""
        
        # Test endpoints that should be accessible to all authenticated roles
        all_roles_endpoints = [
            ("/protocols", "GET", [self.admin_headers, self.user_headers, self.readonly_headers]),
            ("/templates", "GET", [self.admin_headers, self.user_headers, self.readonly_headers])
        ]
        
        for endpoint, method, allowed_headers in all_roles_endpoints:
            for headers in allowed_headers:
                if method == "GET":
                    response = self.client.get(endpoint, headers=headers)
                
                self.assertNotIn(response.status_code, [401, 403], 
                               f"Endpoint {method} {endpoint} should be accessible to this role")
        
        # Test admin-only endpoints
        admin_only_endpoints = [
            ("/metrics", "GET")
        ]
        
        for endpoint, method in admin_only_endpoints:
            # Admin should have access
            if method == "GET":
                response = self.client.get(endpoint, headers=self.admin_headers)
            self.assertNotIn(response.status_code, [401, 403], 
                           f"Admin should have access to {method} {endpoint}")
            
            # Non-admin roles should be denied
            for headers in [self.user_headers, self.readonly_headers]:
                if method == "GET":
                    response = self.client.get(endpoint, headers=headers)
                self.assertEqual(response.status_code, 403, 
                               f"Non-admin should be denied access to {method} {endpoint}")
    
    def test_execute_endpoint_authentication(self):
        """Test authentication for the main execute endpoint."""
        execute_payload = {
            "protocol_names": ["ReasoningAndExplanationProtocol"],
            "initial_data": {"facts": ["test fact"]}
        }
        
        # Should reject without API key
        response = self.client.post("/execute", json=execute_payload)
        self.assertIn(response.status_code, [401, 403, 422])
        
        # Should reject invalid API key  
        response = self.client.post("/execute", headers=self.invalid_headers, json=execute_payload)
        self.assertEqual(response.status_code, 403)
        
        # Should accept valid API key (admin and user, but not read-only)
        for headers in [self.admin_headers, self.user_headers]:
            response = self.client.post("/execute", headers=headers, json=execute_payload)
            # Note: May return other errors due to missing services, but should not be auth errors
            self.assertNotIn(response.status_code, [401, 403])
    
    def test_session_endpoint_authorization(self):
        """Test that session endpoints enforce proper user isolation."""
        # This test may need to be enhanced based on actual session implementation
        session_endpoint = "/session/test-session-id"
        
        # Should require authentication
        response = self.client.get(session_endpoint)
        self.assertIn(response.status_code, [401, 403, 422])
        
        # Should reject invalid API key
        response = self.client.get(session_endpoint, headers=self.invalid_headers)
        self.assertEqual(response.status_code, 403)
        
        # Valid API key should not return auth error (may return 404 or other business logic errors)
        response = self.client.get(session_endpoint, headers=self.user_headers)
        self.assertNotIn(response.status_code, [401, 403])
    
    def test_rate_limiting_integration(self):
        """Test that rate limiting works with authentication."""
        # This test verifies that rate limiting doesn't interfere with auth
        # Make a few requests to ensure rate limiting doesn't break auth
        
        for _ in range(3):
            response = self.client.get("/protocols", headers=self.user_headers)
            # Should not fail due to auth issues (may hit rate limit, that's OK)
            if response.status_code in [401, 403]:
                self.fail("Rate limiting should not break authentication")

if __name__ == '__main__':
    unittest.main()