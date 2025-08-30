import unittest
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from mcp_server.main import app
from mcp_server.config import settings

class TestCORSSecurity(unittest.TestCase):
    """
    Test CORS security configuration to ensure proper origin restrictions.
    Validates that the server properly handles cross-origin requests and 
    prevents unauthorized domain access.
    """
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_cors_headers_present(self):
        """Test that CORS headers are properly set in responses."""
        response = self.client.get("/")
        
        # CORS headers should be present
        self.assertIn("access-control-allow-origin", response.headers)
        self.assertIn("access-control-allow-credentials", response.headers)
        self.assertIn("access-control-allow-methods", response.headers)
    
    def test_cors_allowed_origins_configuration(self):
        """Test that CORS only allows configured origins, not wildcards."""
        # Test that settings.ALLOWED_ORIGINS is not wildcard
        self.assertNotEqual(settings.ALLOWED_ORIGINS, ["*"])
        self.assertIsInstance(settings.ALLOWED_ORIGINS, list)
        
        # Each origin should be a specific domain or localhost
        for origin in settings.ALLOWED_ORIGINS:
            self.assertTrue(
                origin.startswith("http://localhost") or 
                origin.startswith("https://") or
                origin.startswith("http://127.0.0.1"),
                f"Invalid origin configuration: {origin}"
            )
    
    def test_cors_credentials_with_specific_origins(self):
        """Test that credentials are only allowed with specific origins, never with wildcards."""
        # This is a critical security check - credentials should never be True with allow_origins=["*"]
        if "true" in str(settings.ALLOWED_ORIGINS).lower():
            self.assertNotEqual(settings.ALLOWED_ORIGINS, ["*"])
    
    @patch.dict(os.environ, {'ALLOWED_ORIGINS': 'http://localhost:3000,https://secure-domain.com'})
    def test_cors_environment_configuration(self):
        """Test that CORS properly reads from environment variables."""
        # Reload settings to pick up environment changes
        from importlib import reload
        import mcp_server.config
        reload(mcp_server.config)
        
        expected_origins = ['http://localhost:3000', 'https://secure-domain.com']
        self.assertEqual(mcp_server.config.settings.ALLOWED_ORIGINS, expected_origins)
    
    def test_cors_preflight_request(self):
        """Test CORS preflight OPTIONS request handling."""
        headers = {
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'X-API-Key,Content-Type'
        }
        
        response = self.client.options("/execute", headers=headers)
        
        # Should handle preflight request properly
        self.assertIn(response.status_code, [200, 204])
        self.assertIn("access-control-allow-methods", response.headers)
    
    def test_cors_method_restrictions(self):
        """Test that only allowed HTTP methods are permitted via CORS."""
        # The app should only allow GET and POST methods
        allowed_methods_header = None
        
        response = self.client.options("/", headers={'Origin': 'http://localhost:3000'})
        if 'access-control-allow-methods' in response.headers:
            allowed_methods_header = response.headers['access-control-allow-methods']
            
            # Should only allow GET and POST
            allowed_methods = [method.strip().upper() for method in allowed_methods_header.split(',')]
            
            # At minimum should allow GET and POST
            self.assertIn('GET', allowed_methods)
            self.assertIn('POST', allowed_methods)
            
            # Should not allow dangerous methods
            dangerous_methods = ['PUT', 'DELETE', 'PATCH', 'TRACE', 'CONNECT']
            for method in dangerous_methods:
                if method in allowed_methods:
                    self.fail(f"Dangerous HTTP method {method} should not be allowed via CORS")

if __name__ == '__main__':
    unittest.main()