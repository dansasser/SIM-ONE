import unittest
import os
import importlib
import asyncio


class TestGovernancePolicy(unittest.TestCase):
    def setUp(self):
        # Enforce coherence requirement for this test
        os.environ['GOV_REQUIRE_COHERENCE'] = 'true'
        # Reload config and orchestration to pick up new env
        self.config = importlib.import_module('mcp_server.config')
        importlib.reload(self.config)
        self.orch_mod = importlib.import_module('mcp_server.orchestration_engine.orchestration_engine')
        importlib.reload(self.orch_mod)
        self.pm_mod = importlib.import_module('mcp_server.protocol_manager.protocol_manager')
        importlib.reload(self.pm_mod)
        self.rm_mod = importlib.import_module('mcp_server.resource_manager.resource_manager')
        importlib.reload(self.rm_mod)
        self.mm_mod = importlib.import_module('mcp_server.memory_manager.memory_manager')
        importlib.reload(self.mm_mod)

    def test_abort_on_incoherence_after_retry(self):
        ProtocolManager = self.pm_mod.ProtocolManager
        ResourceManager = self.rm_mod.ResourceManager
        MemoryManager = self.mm_mod.MemoryManager
        OrchestrationEngine = self.orch_mod.OrchestrationEngine

        pm = ProtocolManager()
        rm = ResourceManager()
        mm = MemoryManager()
        engine = OrchestrationEngine(pm, rm, mm)

        # Craft a sequential workflow that will produce incoherence across REP (positive) and ESL (negative)
        workflow = [
            {"step": "ReasoningAndExplanationProtocol"},
            {"step": "EmotionalStateLayerProtocol"},
            {"step": "MemoryTaggerProtocol"},
        ]
        ctx = {
            "facts": ["Socrates is a man", "All men are mortal"],
            "rules": [[
                ["Socrates is a man", "All men are mortal"],
                "Socrates is mortal"
            ]],
            # Positive message (for REP conclusion sentiment detection)
            "user_input": "This is an excellent and great success with a promotion.",
        }

        # ESL should be negative to force mismatch
        # Provide input indicating anger to trigger negative valence in ESL
        ctx_esl = {**ctx, "user_input": "I am very angry and upset about this."}

        # Execute: the engine passes the same context dict; to bias ESL we will
        # update context just before ESL via a small monkeypatch on engine._execute_protocol
        original_exec = engine._execute_protocol

        async def patched_exec(name, data):
            if name == 'EmotionalStateLayerProtocol':
                # Use negative input on ESL call
                return await original_exec(name, ctx_esl)
            return await original_exec(name, data)

        engine._execute_protocol = patched_exec  # monkeypatch for test

        loop = asyncio.new_event_loop()
        try:
            final = loop.run_until_complete(engine.execute_workflow(workflow, ctx))
        finally:
            loop.close()

        # With GOV_REQUIRE_COHERENCE=true and mismatch, engine should abort
        self.assertIn('error', final)
        self.assertIn('incoherent', final['error'].lower())


if __name__ == '__main__':
    unittest.main()

