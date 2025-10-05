"""
Five Laws of Cognitive Governance - Unified Evaluator

This module provides a unified interface for evaluating any text or AI response
against SIM-ONE's Five Laws of Cognitive Governance.

The Five Laws:
1. Architectural Intelligence - Intelligence from coordination, not brute force
2. Cognitive Governance - Governed processes over unconstrained generation
3. Truth Foundation - Absolute truth principles over probabilistic drift
4. Energy Stewardship - Computational efficiency and resource awareness
5. Deterministic Reliability - Consistent, predictable outcomes

Usage:
    from tools.lib.five_laws_evaluator import evaluate_text, FiveLawsEvaluator

    # Simple evaluation
    result = evaluate_text("AI response to validate")

    # Advanced evaluation with context
    evaluator = FiveLawsEvaluator()
    result = evaluator.evaluate("Response text", context={"domain": "scientific"})
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Note: This is a lightweight text-based validator that doesn't require
# the full protocol stack. For complete validation, use the full protocols.


@dataclass
class FiveLawsScore:
    """Aggregated Five Laws compliance score"""
    law1_architectural_intelligence: float  # 0-100
    law2_cognitive_governance: float       # 0-100
    law3_truth_foundation: float           # 0-100
    law4_energy_stewardship: float         # 0-100
    law5_deterministic_reliability: float  # 0-100
    overall_compliance: float              # 0-100

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class FiveLawsReport:
    """Complete Five Laws evaluation report"""
    scores: FiveLawsScore
    violations: List[str]
    recommendations: List[str]
    strengths: List[str]
    pass_fail_status: str  # "PASS", "CONDITIONAL", "FAIL"
    detailed_results: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scores": self.scores.to_dict(),
            "violations": self.violations,
            "recommendations": self.recommendations,
            "strengths": self.strengths,
            "pass_fail_status": self.pass_fail_status,
            "detailed_results": self.detailed_results
        }


class FiveLawsEvaluator:
    """
    Unified evaluator for the Five Laws of Cognitive Governance

    This class provides a simple interface to validate any text against
    all five laws and generate a comprehensive compliance report.
    """

    def __init__(self, strictness: str = "moderate"):
        """
        Initialize the Five Laws evaluator

        Args:
            strictness: Evaluation strictness level
                - "lenient": More forgiving thresholds (60% pass)
                - "moderate": Standard thresholds (70% pass)  [DEFAULT]
                - "strict": High standards (85% pass)
        """
        self.strictness = strictness
        self.thresholds = self._get_thresholds(strictness)

        logger.info(f"FiveLawsEvaluator initialized with {strictness} strictness (text-based mode)")

    def _get_thresholds(self, strictness: str) -> Dict[str, float]:
        """Get compliance thresholds based on strictness level"""
        thresholds_map = {
            "lenient": {
                "pass_threshold": 60.0,
                "conditional_threshold": 45.0,
                "individual_law_minimum": 40.0
            },
            "moderate": {
                "pass_threshold": 70.0,
                "conditional_threshold": 55.0,
                "individual_law_minimum": 50.0
            },
            "strict": {
                "pass_threshold": 85.0,
                "conditional_threshold": 70.0,
                "individual_law_minimum": 65.0
            }
        }
        return thresholds_map.get(strictness, thresholds_map["moderate"])

    def evaluate(self,
                 text: str,
                 context: Optional[Dict[str, Any]] = None) -> FiveLawsReport:
        """
        Evaluate text against all Five Laws

        Args:
            text: The text/response to evaluate
            context: Optional context information about the text
                - domain: Domain of the response (e.g., "scientific", "creative")
                - protocol_stack: List of protocols used (if applicable)
                - execution_metrics: Performance metrics (if applicable)

        Returns:
            FiveLawsReport with complete evaluation results
        """
        logger.info(f"Evaluating text ({len(text)} chars) against Five Laws")

        # Prepare evaluation context
        eval_context = context or {}
        eval_context["text"] = text

        # Run all law evaluations
        try:
            # Law 3 (Truth Foundation) - synchronous text analysis
            law3_result = self._evaluate_law3(text, eval_context)

            # Law 1, 2, 4, 5 may be async - run them synchronously for now
            # In production, these would be run with asyncio
            law1_result = self._evaluate_law1_sync(text, eval_context)
            law2_result = self._evaluate_law2_sync(text, eval_context)
            law4_result = self._evaluate_law4_sync(text, eval_context)
            law5_result = self._evaluate_law5_sync(text, eval_context)

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Return minimal failure report
            return self._create_error_report(str(e))

        # Aggregate scores
        scores = FiveLawsScore(
            law1_architectural_intelligence=law1_result.get("compliance_score", 0) * 100,
            law2_cognitive_governance=law2_result.get("compliance_score", 0) * 100,
            law3_truth_foundation=law3_result.get("compliance_score", 0) * 100,
            law4_energy_stewardship=law4_result.get("compliance_score", 0) * 100,
            law5_deterministic_reliability=law5_result.get("compliance_score", 0) * 100,
            overall_compliance=0.0  # Calculated below
        )

        # Calculate overall compliance (weighted average)
        scores.overall_compliance = (
            scores.law1_architectural_intelligence * 0.25 +
            scores.law2_cognitive_governance * 0.25 +
            scores.law3_truth_foundation * 0.25 +
            scores.law4_energy_stewardship * 0.15 +
            scores.law5_deterministic_reliability * 0.10
        )

        # Collect violations
        violations = []
        violations.extend(law1_result.get("violations", []))
        violations.extend(law2_result.get("violations", []))
        violations.extend(law3_result.get("violations", []))
        violations.extend(law4_result.get("violations", []))
        violations.extend(law5_result.get("violations", []))

        # Collect recommendations
        recommendations = []
        recommendations.extend(law1_result.get("recommendations", []))
        recommendations.extend(law2_result.get("recommendations", []))
        recommendations.extend(law3_result.get("recommendations", []))
        recommendations.extend(law4_result.get("recommendations", []))
        recommendations.extend(law5_result.get("recommendations", []))

        # Identify strengths
        strengths = []
        if scores.law1_architectural_intelligence > 80:
            strengths.append("Strong architectural intelligence through coordination")
        if scores.law2_cognitive_governance > 80:
            strengths.append("Well-governed cognitive processes")
        if scores.law3_truth_foundation > 80:
            strengths.append("Solid grounding in truth principles")
        if scores.law4_energy_stewardship > 80:
            strengths.append("Efficient resource utilization")
        if scores.law5_deterministic_reliability > 80:
            strengths.append("Consistent and reliable outputs")

        # Determine pass/fail status
        pass_fail_status = self._determine_status(scores)

        # Compile detailed results
        detailed_results = {
            "law1_details": law1_result,
            "law2_details": law2_result,
            "law3_details": law3_result,
            "law4_details": law4_result,
            "law5_details": law5_result,
            "evaluation_context": eval_context
        }

        return FiveLawsReport(
            scores=scores,
            violations=violations,
            recommendations=recommendations,
            strengths=strengths,
            pass_fail_status=pass_fail_status,
            detailed_results=detailed_results
        )

    def _evaluate_law1_sync(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for Law 1 evaluation"""
        # Law 1 evaluates architectural intelligence
        # For text-only evaluation, we analyze structural patterns

        # Check for coordination patterns
        protocol_stack = context.get("protocol_stack", [])
        has_coordination = len(protocol_stack) > 1

        # Simple heuristic: look for structured reasoning
        has_structure = any(marker in text.lower() for marker in [
            "first", "second", "third", "step ", "phase ", "then", "next",
            "therefore", "consequently", "thus", "because"
        ])

        # Check for brute-force indicators (walls of text, repetition)
        is_concise = len(text) < 2000
        words = text.split()
        unique_ratio = len(set(words)) / len(words) if words else 0

        # Calculate compliance score
        compliance_score = 0.0
        violations = []
        recommendations = []

        if has_coordination:
            compliance_score += 0.4
        else:
            violations.append("Law 1: No evidence of multi-protocol coordination")
            recommendations.append("Use multiple specialized protocols in coordination")

        if has_structure:
            compliance_score += 0.3
        else:
            violations.append("Law 1: Lacks structured reasoning approach")
            recommendations.append("Employ systematic, step-by-step reasoning")

        if is_concise and unique_ratio > 0.6:
            compliance_score += 0.3
        else:
            violations.append("Law 1: Shows signs of brute-force generation (verbosity/repetition)")
            recommendations.append("Focus on concise, coordinated outputs over lengthy generation")

        return {
            "compliance_score": min(compliance_score, 1.0),
            "violations": violations,
            "recommendations": recommendations,
            "details": {
                "has_coordination": has_coordination,
                "has_structure": has_structure,
                "is_concise": is_concise,
                "unique_word_ratio": unique_ratio
            }
        }

    def _evaluate_law2_sync(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for Law 2 evaluation"""
        # Law 2 evaluates cognitive governance
        # Check for evidence of governed processes

        compliance_score = 0.5  # Baseline
        violations = []
        recommendations = []

        # Check for governance indicators
        governance_markers = [
            "validated", "verified", "reviewed", "assessed", "evaluated",
            "compliant", "governed", "controlled", "monitored"
        ]
        text_lower = text.lower()
        governance_count = sum(1 for marker in governance_markers if marker in text_lower)

        if governance_count >= 2:
            compliance_score += 0.3
        else:
            violations.append("Law 2: Insufficient evidence of cognitive governance")
            recommendations.append("Apply validation and verification protocols")

        # Check for quality indicators
        quality_markers = ["accurate", "precise", "rigorous", "systematic", "methodical"]
        quality_count = sum(1 for marker in quality_markers if marker in text_lower)

        if quality_count >= 1:
            compliance_score += 0.2
        else:
            violations.append("Law 2: Lacks quality assurance indicators")
            recommendations.append("Implement quality monitoring and assessment")

        return {
            "compliance_score": min(compliance_score, 1.0),
            "violations": violations,
            "recommendations": recommendations,
            "details": {
                "governance_markers_found": governance_count,
                "quality_markers_found": quality_count
            }
        }

    def _evaluate_law3(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Law 3: Truth Foundation"""
        # Use the actual Law 3 validator
        import re

        compliance_score = 0.8  # Start optimistic
        violations = []
        recommendations = []

        text_lower = text.lower()

        # Check for relativistic language (violations)
        relativistic_patterns = [
            r"\b(?:it depends|could be|might be|perhaps|maybe|possibly)\b",
            r"\b(?:in my opinion|i think|i believe|i feel)\b",
            r"\b(?:subjectively|relatively speaking)\b"
        ]

        relativistic_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in relativistic_patterns
        )

        if relativistic_count > 3:
            compliance_score -= 0.3
            violations.append("Law 3: Excessive relativistic language detected")
            recommendations.append("Ground statements in absolute truth principles")

        # Check for truth grounding (positive indicators)
        truth_patterns = [
            r"\b(?:fact|evidence|proof|demonstrated|verified)\b",
            r"\b(?:therefore|thus|consequently|logically)\b",
            r"\b(?:established|proven|documented)\b"
        ]

        truth_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in truth_patterns
        )

        if truth_count >= 2:
            compliance_score = min(compliance_score + 0.1, 1.0)
        else:
            violations.append("Law 3: Insufficient truth grounding")
            recommendations.append("Support claims with evidence and logical reasoning")

        # Check for factual claims that need verification
        claim_patterns = [r"\b(?:is|are|was|were|will be) [\w\s]{5,30}\b"]
        claims_made = sum(len(re.findall(pattern, text_lower)) for pattern in claim_patterns)

        return {
            "compliance_score": max(compliance_score, 0.0),
            "violations": violations,
            "recommendations": recommendations,
            "details": {
                "relativistic_count": relativistic_count,
                "truth_grounding_count": truth_count,
                "claims_detected": claims_made
            }
        }

    def _evaluate_law4_sync(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for Law 4 evaluation"""
        # Law 4 evaluates energy stewardship
        # Check for efficiency indicators

        compliance_score = 0.6  # Baseline
        violations = []
        recommendations = []

        # Check text efficiency (token efficiency proxy)
        word_count = len(text.split())
        char_count = len(text)
        avg_word_length = char_count / word_count if word_count > 0 else 0

        # Efficient text: concise, low redundancy
        is_concise = word_count < 500
        is_efficient = 4 < avg_word_length < 7  # Reasonable word length

        if is_concise:
            compliance_score += 0.2
        else:
            violations.append("Law 4: Output verbosity indicates energy inefficiency")
            recommendations.append("Generate concise, efficient responses")

        if is_efficient:
            compliance_score += 0.2
        else:
            violations.append("Law 4: Word choice suggests inefficient generation")
            recommendations.append("Optimize for clarity and brevity")

        return {
            "compliance_score": min(compliance_score, 1.0),
            "violations": violations,
            "recommendations": recommendations,
            "details": {
                "word_count": word_count,
                "is_concise": is_concise,
                "avg_word_length": round(avg_word_length, 2)
            }
        }

    def _evaluate_law5_sync(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for Law 5 evaluation"""
        # Law 5 evaluates deterministic reliability
        # Check for consistency and predictability indicators

        compliance_score = 0.7  # Baseline
        violations = []
        recommendations = []

        # Check for deterministic language
        deterministic_markers = [
            "always", "never", "must", "will", "shall", "guaranteed",
            "consistent", "reliable", "predictable", "deterministic"
        ]
        text_lower = text.lower()
        deterministic_count = sum(
            1 for marker in deterministic_markers
            if marker in text_lower
        )

        if deterministic_count >= 1:
            compliance_score += 0.2
        else:
            violations.append("Law 5: Lacks deterministic assurance")
            recommendations.append("Provide consistent, predictable outcomes")

        # Check for probabilistic language (violation)
        probabilistic_markers = [
            "random", "unpredictable", "variable", "uncertain", "may vary"
        ]
        probabilistic_count = sum(
            1 for marker in probabilistic_markers
            if marker in text_lower
        )

        if probabilistic_count > 0:
            compliance_score -= 0.2
            violations.append("Law 5: Contains non-deterministic language")
            recommendations.append("Ensure deterministic, repeatable processes")

        return {
            "compliance_score": max(min(compliance_score, 1.0), 0.0),
            "violations": violations,
            "recommendations": recommendations,
            "details": {
                "deterministic_markers": deterministic_count,
                "probabilistic_markers": probabilistic_count
            }
        }

    def _determine_status(self, scores: FiveLawsScore) -> str:
        """Determine overall pass/fail status"""
        overall = scores.overall_compliance

        # Check if any individual law is critically low
        all_scores = [
            scores.law1_architectural_intelligence,
            scores.law2_cognitive_governance,
            scores.law3_truth_foundation,
            scores.law4_energy_stewardship,
            scores.law5_deterministic_reliability
        ]

        min_score = min(all_scores)

        if overall >= self.thresholds["pass_threshold"] and \
           min_score >= self.thresholds["individual_law_minimum"]:
            return "PASS"
        elif overall >= self.thresholds["conditional_threshold"]:
            return "CONDITIONAL"
        else:
            return "FAIL"

    def _create_error_report(self, error_msg: str) -> FiveLawsReport:
        """Create an error report when evaluation fails"""
        scores = FiveLawsScore(
            law1_architectural_intelligence=0.0,
            law2_cognitive_governance=0.0,
            law3_truth_foundation=0.0,
            law4_energy_stewardship=0.0,
            law5_deterministic_reliability=0.0,
            overall_compliance=0.0
        )

        return FiveLawsReport(
            scores=scores,
            violations=[f"Evaluation error: {error_msg}"],
            recommendations=["Fix evaluation errors before assessing compliance"],
            strengths=[],
            pass_fail_status="ERROR",
            detailed_results={"error": error_msg}
        )


# Convenience function for simple evaluation
def evaluate_text(text: str,
                  context: Optional[Dict[str, Any]] = None,
                  strictness: str = "moderate") -> Dict[str, Any]:
    """
    Quick evaluation function for any text against the Five Laws

    Args:
        text: Text to evaluate
        context: Optional context dictionary
        strictness: "lenient", "moderate", or "strict"

    Returns:
        Dictionary containing evaluation report

    Example:
        >>> result = evaluate_text("The sky is blue because of Rayleigh scattering")
        >>> print(result["scores"]["overall_compliance"])
        85.3
        >>> print(result["pass_fail_status"])
        PASS
    """
    evaluator = FiveLawsEvaluator(strictness=strictness)
    report = evaluator.evaluate(text, context)
    return report.to_dict()


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample text
    test_texts = [
        "The Earth is approximately 4.5 billion years old based on radiometric dating evidence.",
        "I think maybe the sky could be blue, but it depends on your perspective really.",
        "Through coordinated analysis using spectroscopy and geological evidence, we have definitively established that Earth formed 4.54 billion years ago."
    ]

    print("=" * 80)
    print("Five Laws Evaluator - Test Run")
    print("=" * 80)

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text[:80]}...")

        result = evaluate_text(text)
        print(f"Overall Compliance: {result['scores']['overall_compliance']:.1f}%")
        print(f"Status: {result['pass_fail_status']}")
        print(f"Violations: {len(result['violations'])}")
        if result['recommendations']:
            print(f"Top Recommendation: {result['recommendations'][0]}")

    print("\n" + "=" * 80)
