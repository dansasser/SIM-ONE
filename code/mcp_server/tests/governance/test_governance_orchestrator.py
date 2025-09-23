import unittest

from mcp_server.cognitive_governance_engine.governance_orchestrator import GovernanceOrchestrator


class TestGovernanceOrchestrator(unittest.TestCase):
    def test_quality_and_coherence_aggregation(self):
        orch = GovernanceOrchestrator()

        # Build a minimal context that mirrors orchestration storage
        ctx = {}

        # Step 1: REP result with positive conclusion and good confidence
        rep_result = {
            "conclusions": ["The team achieved great success"],
            "validation": {"confidence_score": 0.9},
        }
        ctx["ReasoningAndExplanationProtocol"] = {"result": rep_result}
        delta_rep = orch.evaluate_step(
            "ReasoningAndExplanationProtocol",
            {"user_input": "Did the team succeed?"},
            rep_result,
            ctx,
        )
        self.assertIn("quality", delta_rep)
        self.assertIn("REP", orch.state["quality"])  # shorthand mapping
        self.assertGreaterEqual(orch.state["quality"]["REP"]["quality_score"], 0.5)

        # Step 2: ESL with positive valence
        esl_result = {"valence": "positive", "detected_emotions": [{"emotion": "joy", "intensity": 0.8}]}
        ctx["EmotionalStateLayerProtocol"] = {"result": esl_result}
        delta_esl = orch.evaluate_step(
            "EmotionalStateLayerProtocol",
            {"user_input": "We're thrilled with the outcome."},
            esl_result,
            ctx,
        )
        self.assertIn("quality", delta_esl)

        # Step 3: MTP with entity that aligns positively
        mtp_result = {"extracted_entities": [{"entity": "promotion", "type": "event"}]}
        ctx["MemoryTaggerProtocol"] = {"result": mtp_result}
        delta_mtp = orch.evaluate_step(
            "MemoryTaggerProtocol",
            {"user_input": "The promotion was announced."},
            mtp_result,
            ctx,
        )
        self.assertIn("quality", delta_mtp)

        # Coherence should be computed when all three are present
        self.assertIsNotNone(orch.state.get("coherence"))
        self.assertIn("is_coherent", orch.state["coherence"])
        self.assertTrue(orch.state["coherence"]["is_coherent"])  # alignment is positive

    def test_incoherence_detection_on_mismatch(self):
        orch = GovernanceOrchestrator()

        ctx = {}

        # REP declares a strongly positive conclusion
        rep_result = {
            "conclusions": ["This is an excellent and great success"],
            "validation": {"confidence_score": 0.9},
        }
        ctx["ReasoningAndExplanationProtocol"] = {"result": rep_result}
        orch.evaluate_step("ReasoningAndExplanationProtocol", {"user_input": "Did we succeed?"}, rep_result, ctx)

        # ESL reports negative valence -> mismatch with REP's positive tone
        esl_result = {"valence": "negative", "detected_emotions": [{"emotion": "anger", "intensity": 0.8}]}
        ctx["EmotionalStateLayerProtocol"] = {"result": esl_result}
        orch.evaluate_step("EmotionalStateLayerProtocol", {"user_input": "I am angry about this"}, esl_result, ctx)

        # MTP includes an entity mapped to positive in ENTITY_EMOTION_MAP ('promotion')
        mtp_result = {"extracted_entities": [{"entity": "promotion", "type": "event"}]}
        ctx["MemoryTaggerProtocol"] = {"result": mtp_result}
        orch.evaluate_step("MemoryTaggerProtocol", {"user_input": "We got a promotion"}, mtp_result, ctx)

        coherence = orch.state.get("coherence")
        self.assertIsNotNone(coherence)
        self.assertIn("is_coherent", coherence)
        self.assertFalse(coherence["is_coherent"])  # mismatch should trigger incoherence
        # Optional: verify reasons mention a mismatch
        self.assertTrue(any("mismatch" in r.lower() for r in coherence.get("reasons", [])))


if __name__ == "__main__":
    unittest.main()
