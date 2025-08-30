import unittest
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from mcp_server.main import app
from mcp_server.security import key_manager

class TestCognitiveProtocolsSecurity(unittest.TestCase):
    """
    Test that cognitive protocol execution maintains security while preserving
    the SIM-ONE Framework's cognitive governance capabilities.
    
    Ensures that security measures don't break the core cognitive functionality
    that makes the SIM-ONE Framework revolutionary.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with proper API keys."""
        os.environ['VALID_API_KEYS'] = 'test-admin-key,test-user-key,test-readonly-key'
        key_manager.initialize_api_keys()
    
    @classmethod  
    def tearDownClass(cls):
        """Clean up test API keys."""
        if os.path.exists(key_manager.API_KEYS_FILE):
            os.remove(key_manager.API_KEYS_FILE)
    
    def setUp(self):
        self.client = TestClient(app)
        self.admin_headers = {"X-API-Key": "test-admin-key"}
        self.user_headers = {"X-API-Key": "test-user-key"}
        self.readonly_headers = {"X-API-Key": "test-readonly-key"}
        self.invalid_headers = {"X-API-Key": "invalid-key"}
    
    def test_protocol_discovery_with_auth(self):
        """Test that protocol discovery works with proper authentication."""
        # Should fail without auth
        response = self.client.get("/protocols")
        self.assertIn(response.status_code, [401, 403, 422])
        
        # Should work with proper auth for all roles
        for headers in [self.admin_headers, self.user_headers, self.readonly_headers]:
            response = self.client.get("/protocols", headers=headers)
            self.assertNotIn(response.status_code, [401, 403])
            
            if response.status_code == 200:
                protocols = response.json()
                self.assertIsInstance(protocols, dict)
                
                # Should contain core SIM-ONE protocols
                expected_protocols = [
                    "ReasoningAndExplanationProtocol",
                    "EmotionalStateLayerProtocol", 
                    "MemoryTaggerProtocol"
                ]
                
                for protocol in expected_protocols:
                    self.assertIn(protocol, protocols,
                                f"Core cognitive protocol {protocol} should be discoverable")
    
    def test_workflow_templates_with_auth(self):
        """Test that workflow templates are accessible with authentication."""
        # Should fail without auth
        response = self.client.get("/templates")
        self.assertIn(response.status_code, [401, 403, 422])
        
        # Should work with proper auth
        for headers in [self.admin_headers, self.user_headers, self.readonly_headers]:
            response = self.client.get("/templates", headers=headers)
            self.assertNotIn(response.status_code, [401, 403])
            
            if response.status_code == 200:
                templates = response.json()
                self.assertIsInstance(templates, dict)
                
                # Should contain core workflow templates
                expected_templates = ["analyze_only", "full_reasoning", "writing_team"]
                
                for template_name in expected_templates:
                    if template_name in templates:
                        template = templates[template_name]
                        self.assertIn("protocols", template)
                        self.assertIsInstance(template["protocols"], list)
    
    def test_cognitive_workflow_execution_security(self):
        """Test that cognitive workflow execution maintains security."""
        # Test individual protocol execution
        protocol_payload = {
            "protocol_names": ["ReasoningAndExplanationProtocol"],
            "initial_data": {"facts": ["Test security integration with cognitive protocols"]}
        }
        
        # Should fail without authentication
        response = self.client.post("/execute", json=protocol_payload)
        self.assertIn(response.status_code, [401, 403, 422])
        
        # Should fail with invalid key
        response = self.client.post("/execute", headers=self.invalid_headers, json=protocol_payload)
        self.assertEqual(response.status_code, 403)
        
        # Should work for authorized roles (admin and user, not read-only for execution)
        for headers in [self.admin_headers, self.user_headers]:
            response = self.client.post("/execute", headers=headers, json=protocol_payload)
            
            # Should not fail due to authentication (may fail due to missing services)
            self.assertNotIn(response.status_code, [401, 403])
            
            # If successful, should return proper workflow response structure
            if response.status_code == 200:
                data = response.json()
                self.assertIn("session_id", data)
                self.assertIn("results", data)
                self.assertIn("execution_time_ms", data)
    
    def test_workflow_template_execution_security(self):
        """Test that workflow template execution maintains security."""
        template_payload = {
            "template_name": "analyze_only",
            "initial_data": {"text": "Test cognitive governance with security"}
        }
        
        # Should require authentication
        response = self.client.post("/execute", json=template_payload)
        self.assertIn(response.status_code, [401, 403, 422])
        
        # Should work with proper auth
        for headers in [self.admin_headers, self.user_headers]:
            response = self.client.post("/execute", headers=headers, json=template_payload)
            self.assertNotIn(response.status_code, [401, 403])
    
    def test_session_management_security_integration(self):
        """Test that session management maintains security while preserving cognitive continuity."""
        # Test session creation through workflow execution
        session_payload = {
            "protocol_names": ["EmotionalStateLayerProtocol"],
            "initial_data": {"text": "Test emotional analysis with security"},
            "session_id": "test-security-session"
        }
        
        # Execute workflow to create session
        response = self.client.post("/execute", headers=self.user_headers, json=session_payload)
        
        if response.status_code == 200:
            data = response.json()
            session_id = data.get("session_id")
            
            # Test session access
            session_response = self.client.get(f"/session/{session_id}", headers=self.user_headers)
            self.assertNotIn(session_response.status_code, [401, 403])
            
            # Should not allow access without proper auth
            unauth_response = self.client.get(f"/session/{session_id}")
            self.assertIn(unauth_response.status_code, [401, 403, 422])
    
    def test_cognitive_governance_engine_security(self):
        """Test that cognitive governance engine functions maintain security."""
        # Test that governance validation doesn't bypass security
        governance_payload = {
            "protocol_names": ["ReasoningAndExplanationProtocol", "EmotionalStateLayerProtocol"],
            "coordination_mode": "Sequential",
            "initial_data": {
                "facts": ["Test cognitive governance security"],
                "text": "Ensure governance engine maintains security"
            }
        }
        
        # Complex workflows should still require authentication
        response = self.client.post("/execute", json=governance_payload)
        self.assertIn(response.status_code, [401, 403, 422])
        
        # Should work with auth
        response = self.client.post("/execute", headers=self.admin_headers, json=governance_payload)
        self.assertNotIn(response.status_code, [401, 403])
    
    def test_memory_management_security(self):
        """Test that memory management maintains security isolation."""
        # Test memory operations require proper authentication
        memory_payload = {
            "template_name": "full_reasoning",
            "initial_data": {
                "facts": ["Test memory security"],
                "context": "Validate memory isolation with authentication"
            }
        }
        
        # Should require auth
        response = self.client.post("/execute", json=memory_payload)
        self.assertIn(response.status_code, [401, 403, 422])
        
        # Should work with proper auth
        response = self.client.post("/execute", headers=self.user_headers, json=memory_payload)
        self.assertNotIn(response.status_code, [401, 403])
    
    def test_input_validation_cognitive_data(self):
        """Test that input validation works for cognitive protocol data."""
        # Test malformed cognitive protocol requests
        malformed_payloads = [
            {"protocol_names": [], "initial_data": {}},  # Empty protocol list
            {"protocol_names": ["NonexistentProtocol"], "initial_data": {}},  # Invalid protocol
            {"initial_data": {"facts": ["test"]}},  # Missing protocol specification
            {"protocol_names": ["ReasoningAndExplanationProtocol"]},  # Missing initial_data
        ]
        
        for payload in malformed_payloads:
            response = self.client.post("/execute", headers=self.user_headers, json=payload)
            
            # Should handle malformed input gracefully (400/422, not 500)
            if response.status_code >= 500:
                self.fail(f"Server error with payload {payload} - should handle gracefully")
    
    def test_rate_limiting_cognitive_workflows(self):
        """Test that rate limiting applies to cognitive workflow execution."""
        # Test that cognitive operations are subject to rate limiting
        payload = {
            "protocol_names": ["ReasoningAndExplanationProtocol"],
            "initial_data": {"facts": ["Rate limiting test"]}
        }
        
        # Make multiple rapid requests
        responses = []
        for i in range(25):  # Attempt to exceed rate limit
            response = self.client.post("/execute", headers=self.user_headers, json=payload)
            responses.append(response.status_code)
            
            if response.status_code == 429:  # Rate limited
                # Rate limiting should work without breaking cognitive functionality
                break
        
        # At least some requests should succeed (cognitive functionality preserved)
        successful_requests = [r for r in responses if r not in [429, 401, 403, 500]]
        self.assertGreater(len(successful_requests), 0, 
                          "Rate limiting should not completely block cognitive functionality")
    
    def test_five_laws_compliance_with_security(self):
        """Test that security implementation maintains SIM-ONE Five Laws compliance."""
        
        # Law 2: Cognitive Governance - Security processes should be governed
        governance_test_payload = {
            "template_name": "full_reasoning", 
            "initial_data": {"facts": ["Test governance compliance"]}
        }
        
        response = self.client.post("/execute", headers=self.admin_headers, json=governance_test_payload)
        
        # Security should not break cognitive governance
        if response.status_code == 200:
            data = response.json()
            # Should maintain governance structure
            self.assertIn("results", data)
            self.assertIn("session_id", data)
            
        # Law 5: Deterministic Reliability - Security behavior should be consistent
        # Multiple identical requests should have consistent auth behavior
        auth_responses = []
        for _ in range(3):
            response = self.client.get("/protocols", headers=self.user_headers)
            auth_responses.append(response.status_code)
        
        # All auth responses should be consistent
        self.assertEqual(len(set(auth_responses)), 1, 
                        "Authentication behavior should be deterministic")

if __name__ == '__main__':
    unittest.main()