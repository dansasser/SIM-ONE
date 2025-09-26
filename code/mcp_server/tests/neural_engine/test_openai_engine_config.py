import os
import importlib
import unittest


class TestOpenAIEngineConfig(unittest.TestCase):
    def setUp(self):
        os.environ['OPENAI_API_KEY'] = 'test-key'
        os.environ['OPENAI_MODEL'] = 'gpt-4o-mini'
        os.environ['OPENAI_TIMEOUT_SECONDS'] = '25'
        os.environ['OPENAI_MAX_RETRIES'] = '3'
        os.environ['OPENAI_JSON_ONLY'] = 'true'

        # Reload config and engine module to pick new env
        self.cfg = importlib.import_module('mcp_server.config')
        importlib.reload(self.cfg)
        self.ne = importlib.import_module('mcp_server.neural_engine.neural_engine')
        importlib.reload(self.ne)

    def tearDown(self):
        for k in ['OPENAI_API_KEY','OPENAI_MODEL','OPENAI_TIMEOUT_SECONDS','OPENAI_MAX_RETRIES','OPENAI_JSON_ONLY']:
            os.environ.pop(k, None)

    def test_engine_reads_env(self):
        # Patch OpenAI client to avoid real calls
        class Dummy:
            def __init__(self, *a, **kw): pass
        self.ne.OpenAI = Dummy
        if hasattr(self.ne, 'AsyncOpenAI'):
            self.ne.AsyncOpenAI = Dummy

        eng = self.ne.OpenAIEngine()
        self.assertEqual(eng.model, 'gpt-4o-mini')
        self.assertEqual(eng.timeout, 25)
        self.assertEqual(eng.max_retries, 3)
        self.assertTrue(eng.json_only)

