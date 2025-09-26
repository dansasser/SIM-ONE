import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient


class TestSummarizerOpenAIBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure backend as openai and set a dummy key before importing app
        os.environ['NEURAL_ENGINE_BACKEND'] = 'openai'
        os.environ['OPENAI_API_KEY'] = 'dummy'

        # Import app after env set
        from mcp_server.main import app
        cls.app = app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        for k in ['NEURAL_ENGINE_BACKEND','OPENAI_API_KEY']:
            os.environ.pop(k, None)

    @patch('mcp_server.neural_engine.neural_engine.OpenAI')
    def test_summarizer_returns_bullets_json(self, MockOpenAI):
        # Mock OpenAI to return a JSON bullets payload
        mock_instance = MockOpenAI.return_value

        class Message:
            def __init__(self, content): self.content = content
        class Choice:
            def __init__(self, content): self.message = Message(content)
        class Response:
            def __init__(self, content): self.choices = [Choice(content)]

        content = '{"bullets":["a","b","c","d","e"]}'
        mock_instance.chat.completions.create.return_value = Response(content)

        # Provide an initial draft so SP has content to summarize
        body = {
            "protocol_names": ["SummarizerProtocol"],
            "initial_data": {"DrafterProtocol": {"draft_text": "Text about SIM-ONE governance."}}
        }
        # Provide an API key header
        headers = {"X-API-Key": "test-user-key"}

        # Depending on auth setup, we may need to initialize keys; allow non-403
        r = self.client.post("/execute", headers=headers, json=body)
        self.assertNotIn(r.status_code, [401,403], msg=f"Auth failed: {r.text}")
        data = r.json()
        # Ensure bullets present under results
        results = data.get('results', {})
        sp = results.get('SummarizerProtocol') or results.get('SP') or {}
        # Either top-level bullets or inside result dict
        has_bullets = False
        if isinstance(sp, dict):
            if 'bullets' in sp:
                has_bullets = True
            elif 'result' in sp and isinstance(sp['result'], dict) and 'bullets' in sp['result']:
                has_bullets = True
        self.assertTrue(has_bullets, f"No bullets found in summarizer result: {sp}")

