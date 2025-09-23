import logging
from typing import Dict, Any, Optional

from .quality_assurance.quality_scorer import QualityScorer
from .coherence_validator.coherence_checker import run_coherence_checks
from mcp_server.config import settings

logger = logging.getLogger(__name__)


class GovernanceOrchestrator:
    """
    Lightweight governance orchestrator.

    - Scores per-protocol quality using QualityScorer.
    - When REP, ESL, and MTP are available, computes a coherence report.
    - Aggregates diagnostics in an internal report structure that callers may
      attach to workflow context (e.g., context["governance"]).
    """

    def __init__(self) -> None:
        self.quality = QualityScorer()
        self.state: Dict[str, Any] = {
            "quality": {},
            "coherence": None,
            "warnings": [],
            "actions": [],
            "adjustments": {},
        }

    def _map_protocol_name(self, protocol_name: str) -> str:
        mapping = {
            "ReasoningAndExplanationProtocol": "REP",
            "EmotionalStateLayerProtocol": "ESL",
            "MemoryTaggerProtocol": "MTP",
        }
        return mapping.get(protocol_name, protocol_name)

    def _adapt_rep_output(self, rep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize REP output to the structure expected by coherence checks.

        The coherence checker expects keys like 'conclusion' and 'confidence'.
        The implemented REP returns 'conclusions' (list) and 'validation' /
        'confidence_scores'. We adapt conservatively.
        """
        if not isinstance(rep_result, dict):
            return {"conclusion": str(rep_result), "confidence": 0.7}

        conclusion: str = ""
        if isinstance(rep_result.get("conclusions"), list) and rep_result["conclusions"]:
            # choose the last derived conclusion
            conclusion = str(rep_result["conclusions"][-1])
        elif isinstance(rep_result.get("conclusion"), str):
            conclusion = rep_result.get("conclusion")

        confidence = 0.7
        # Prefer validation confidence score if present
        validation = rep_result.get("validation")
        if isinstance(validation, dict) and isinstance(validation.get("confidence_score"), (int, float)):
            confidence = float(validation["confidence_score"])
        else:
            # Fall back to averaging confidence_scores if available
            cs = rep_result.get("confidence_scores")
            if isinstance(cs, dict) and cs:
                vals = [float(v) for v in cs.values() if isinstance(v, (int, float))]
                if vals:
                    confidence = sum(vals) / len(vals)

        return {"conclusion": conclusion, "confidence": confidence}

    def _extract_protocol_output(self, context: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
        """Get inner protocol 'result' if present in context."""
        entry = context.get(name)
        if isinstance(entry, dict) and isinstance(entry.get("result"), dict):
            return entry["result"]
        return None

    def evaluate_step(
        self,
        protocol_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        session_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a single protocol step and update governance diagnostics.

        Returns a minimal delta report for this step. Full aggregated state is
        available in self.state.
        """
        if not settings.GOV_ENABLE:
            return {"protocol": self._map_protocol_name(protocol_name)}

        short = self._map_protocol_name(protocol_name)
        user_input = ""
        # Prefer canonical 'user_input' if present
        if isinstance(input_data, dict):
            user_input = str(input_data.get("user_input", ""))

        # Score quality for this protocol
        try:
            q = self.quality.score_output(short, output_data if isinstance(output_data, dict) else {}, user_input)
            self.state["quality"][short] = q
            if q.get("quality_score", 1.0) < settings.GOV_MIN_QUALITY:
                self.state["warnings"].append(f"{short}: quality below threshold")
        except Exception as e:
            logger.warning(f"Quality scoring failed for {protocol_name}: {e}")

        # Compute coherence when we have REP, ESL, and MTP
        rep_raw = self._extract_protocol_output(session_ctx, "ReasoningAndExplanationProtocol")
        esl_raw = self._extract_protocol_output(session_ctx, "EmotionalStateLayerProtocol")
        mtp_raw = self._extract_protocol_output(session_ctx, "MemoryTaggerProtocol")

        if rep_raw and esl_raw and mtp_raw:
            try:
                rep_norm = self._adapt_rep_output(rep_raw)
                coherence = run_coherence_checks(rep_norm, esl_raw, mtp_raw)
                self.state["coherence"] = coherence
                if not coherence.get("is_coherent", True):
                    self.state["warnings"].extend(coherence.get("reasons", []))
            except Exception as e:
                logger.warning(f"Coherence check failed: {e}")

        # Minimal delta for this step
        return {
            "protocol": short,
            "quality": self.state["quality"].get(short),
            "coherence": self.state.get("coherence"),
        }
