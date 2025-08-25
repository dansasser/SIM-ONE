import logging
from typing import List, Set, Tuple, Dict, Any, Optional
from enum import Enum
import re

logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"

class ConfidenceLevel(Enum):
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

class ReasoningStep:
    def __init__(self, step_type: ReasoningType, premises: List[str],
                 conclusion: str, confidence: float, explanation: str):
        self.step_type = step_type
        self.premises = premises
        self.conclusion = conclusion
        self.confidence = confidence
        self.explanation = explanation

class AdvancedREP:
    """
     Advanced Reasoning and Explanation Protocol with multiple reasoning types,
     confidence scoring, and logical validation.
     """
    def __init__(self):
        self.reasoning_chain: List[ReasoningStep] = []
        self.known_facts: Set[str] = set()
        self.derived_facts: Dict[str, float] = {} # fact -> confidence

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
         Execute advanced reasoning with multiple reasoning types.
         """
        logger.info("Starting advanced REP execution")
        # Initialize
        self.reasoning_chain = []
        self.known_facts = set(data.get("facts", []))
        self.derived_facts = {} # Get reasoning context
        rules = data.get("rules", [])
        reasoning_type = data.get("reasoning_type", "deductive")
        context = data.get("context", "")
        # Execute reasoning based on type
        if reasoning_type == "deductive":
            results = self._deductive_reasoning(rules)
        elif reasoning_type == "inductive":
            results = self._inductive_reasoning(data.get("observations", []))
        elif reasoning_type == "abductive":
            results = self._abductive_reasoning(data.get("observations", []),
                                                data.get("hypotheses", []))
        elif reasoning_type == "analogical":
            results = self._analogical_reasoning(data.get("source_case", {}),
                                                 data.get("target_case", {}))
        elif reasoning_type == "causal":
            results = self._causal_reasoning(data.get("events", []))
        else:
            # Auto-detect best reasoning type
            results = self._auto_reasoning(data)
        # Validate reasoning chain
        validation_results = self._validate_reasoning_chain()
        return {
            "reasoning_type": reasoning_type,
            "conclusions": results,
            "confidence_scores": self.derived_facts,
            "reasoning_chain": [self._step_to_dict(step) for step in self.reasoning_chain],
            "validation": validation_results,
            "explanation": self._generate_explanation()
        }

    def _deductive_reasoning(self, rules: List[Tuple]) -> List[str]:
        """
         Implement deductive reasoning with modus ponens and modus tollens.
         """
        logger.info("Executing deductive reasoning")
        new_conclusions = []
        # Forward chaining with confidence
        changed = True
        while changed:
            changed = False
            for premises, conclusion in rules:
                if conclusion not in self.derived_facts:
                    # Check if all premises are satisfied
                    premise_confidences = []
                    all_premises_satisfied = True
                    for premise in premises:
                        if premise in self.known_facts:
                            premise_confidences.append(1.0)
                        elif premise in self.derived_facts:
                            premise_confidences.append(self.derived_facts[premise])
                        else:
                            all_premises_satisfied = False
                            break
                    if all_premises_satisfied:
                        # Calculate conclusion confidence (minimum of premises)
                        confidence = min(premise_confidences) * 0.95 # Slight degradation
                        self.derived_facts[conclusion] = confidence
                        new_conclusions.append(conclusion)
                        # Add reasoning step
                        step = ReasoningStep(
                            ReasoningType.DEDUCTIVE,
                            premises,
                            conclusion,
                            confidence,
                            f"Deductive inference: {' AND '.join(premises)} → {conclusion}"
                        )
                        self.reasoning_chain.append(step)
                        changed = True
                        logger.debug(f"Deduced: {conclusion} (confidence: {confidence:.2f})")
        return new_conclusions

    def _inductive_reasoning(self, observations: List[str]) -> List[str]:
        """
         Implement inductive reasoning to find patterns and generalizations.
         """
        logger.info("Executing inductive reasoning")
        patterns = []
        # Simple pattern detection
        if len(observations) >= 3:
            # Look for common patterns
            common_words = self._find_common_elements(observations)
            for word in common_words:
                if len([obs for obs in observations if word in obs]) >= len(observations) * 0.7:
                    pattern = f"Pattern detected: Most observations contain '{word}'"
                    confidence = len([obs for obs in observations if word in obs]) / len(observations)
                    self.derived_facts[pattern] = confidence
                    patterns.append(pattern)
                    step = ReasoningStep(
                        ReasoningType.INDUCTIVE,
                        observations,
                        pattern,
                        confidence,
                        f"Inductive generalization from {len(observations)} observations"
                    )
                    self.reasoning_chain.append(step)
        return patterns

    def _abductive_reasoning(self, observations: List[str], hypotheses: List[str]) -> List[str]:
        """
         Implement abductive reasoning to find best explanations.
         """
        logger.info("Executing abductive reasoning")
        best_explanations = []
        # Score hypotheses based on explanatory power
        hypothesis_scores = {}
        for hypothesis in hypotheses:
            score = 0
            explained_observations = 0
            for observation in observations:
                # Simple relevance scoring
                if self._calculate_relevance(hypothesis, observation) > 0.5:
                    explained_observations += 1
                    score += 1
            if explained_observations > 0:
                # Normalize by simplicity (shorter hypotheses preferred)
                simplicity_bonus = 1.0 / (len(hypothesis.split()) / 10 + 1)
                final_score = (score / len(observations)) * simplicity_bonus
                hypothesis_scores[hypothesis] = final_score
        # Select best explanations
        sorted_hypotheses = sorted(hypothesis_scores.items(), key=lambda x: x[1], reverse=True)
        for hypothesis, score in sorted_hypotheses[:3]: # Top 3 explanations
            if score > 0.3: # Minimum threshold
                self.derived_facts[f"Best explanation: {hypothesis}"] = score
                best_explanations.append(hypothesis)
                step = ReasoningStep(
                    ReasoningType.ABDUCTIVE,
                    observations,
                    f"Best explanation: {hypothesis}", score,
                    f"Abductive inference: {hypothesis} explains {score:.1%} of observations"
                )
                self.reasoning_chain.append(step)
        return best_explanations

    def _analogical_reasoning(self, source_case: Dict, target_case: Dict) -> List[str]:
        """
         Implement analogical reasoning between cases.
         """
        logger.info("Executing analogical reasoning")
        analogies = []
        if not source_case or not target_case:
            return analogies
        # Find structural similarities
        source_features = set(source_case.keys())
        target_features = set(target_case.keys())
        common_features = source_features.intersection(target_features)
        if len(common_features) >= 2:
            similarity_score = len(common_features) / len(source_features.union(target_features))
            # Generate analogical inference
            for feature in source_features - target_features:
                if feature in source_case:
                    analogy = f"By analogy: {target_case.get('name', 'target')} likely has {feature} = {source_case[feature]}"
                    confidence = similarity_score * 0.8 # Analogies are inherently uncertain
                    self.derived_facts[analogy] = confidence
                    analogies.append(analogy)
                    step = ReasoningStep(
                        ReasoningType.ANALOGICAL,
                        [f"Source: {source_case}", f"Target: {target_case}"],
                        analogy,
                        confidence,
                        f"Analogical inference based on {len(common_features)} shared features"
                    )
                    self.reasoning_chain.append(step)
        return analogies

    def _causal_reasoning(self, events: List[Dict]) -> List[str]:
        """
         Implement causal reasoning to identify cause-effect relationships. """
        logger.info("Executing causal reasoning")
        causal_relationships = []
        # Sort events by timestamp if available
        sorted_events = sorted(events, key=lambda x: x.get('timestamp', 0))
        # Look for temporal patterns
        for i in range(len(sorted_events) - 1):
            event_a = sorted_events[i]
            event_b = sorted_events[i + 1]
            # Simple causal inference based on temporal order and correlation
            if self._calculate_correlation(event_a, event_b) > 0.6:
                causal_relation = f"Causal relationship: {event_a.get('description', 'Event A')} → {event_b.get('description', 'Event B')}"
                confidence = 0.7 # Causal inference is moderately confident
                self.derived_facts[causal_relation] = confidence
                causal_relationships.append(causal_relation)
                step = ReasoningStep(
                    ReasoningType.CAUSAL,
                    [str(event_a), str(event_b)],
                    causal_relation,
                    confidence,
                    f"Causal inference based on temporal order and correlation"
                )
                self.reasoning_chain.append(step)
        return causal_relationships

    def _auto_reasoning(self, data: Dict[str, Any]) -> List[str]:
        """
         Automatically select the best reasoning type based on input data.
         """
        logger.info("Auto-selecting reasoning type")
        # Determine best reasoning approach
        if "rules" in data and data["rules"]:
            return self._deductive_reasoning(data["rules"])
        elif "observations" in data and len(data["observations"]) >= 3:
            return self._inductive_reasoning(data["observations"])
        elif "hypotheses" in data and "observations" in data:
            return self._abductive_reasoning(data["observations"], data["hypotheses"])
        else:
            # Default to simple fact processing
            return list(self.known_facts)

    def _validate_reasoning_chain(self) -> Dict[str, Any]:
        """
         Validate the logical consistency of the reasoning chain. """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "confidence_score": 1.0
        }
        # Check for contradictions
        conclusions = [step.conclusion for step in self.reasoning_chain]
        for i, conclusion_a in enumerate(conclusions):
            for j, conclusion_b in enumerate(conclusions[i+1:], i+1):
                if self._are_contradictory(conclusion_a, conclusion_b):
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(f"Contradiction: {conclusion_a} vs {conclusion_b}")
        # Calculate overall confidence
        if self.reasoning_chain:
            avg_confidence = sum(step.confidence for step in self.reasoning_chain) / len(self.reasoning_chain)
            validation_results["confidence_score"] = avg_confidence
        return validation_results

    def _generate_explanation(self) -> str:
        """
         Generate human-readable explanation of the reasoning process.
         """
        if not self.reasoning_chain:
            return "No reasoning steps performed."
        explanation = f"Reasoning process with {len(self.reasoning_chain)} steps:\n"
        for i, step in enumerate(self.reasoning_chain, 1):
            explanation += f"{i}. {step.explanation} (confidence: {step.confidence:.2f})\n"
        return explanation

    # Helper methods
    def _find_common_elements(self, observations: List[str]) -> Set[str]:
        """Find common words/elements across observations."""
        if not observations:
            return set()
        word_sets = [set(obs.lower().split()) for obs in observations]
        return set.intersection(*word_sets)

    def _calculate_relevance(self, hypothesis: str, observation: str) -> float:
        """Calculate relevance score between hypothesis and observation."""
        hyp_words = set(hypothesis.lower().split())
        obs_words = set(observation.lower().split())
        if not hyp_words or not obs_words:
            return 0.0
        intersection = hyp_words.intersection(obs_words)
        union = hyp_words.union(obs_words)
        return len(intersection) / len(union)

    def _calculate_correlation(self, event_a: Dict, event_b: Dict) -> float:
        """Calculate correlation between two events."""
        # Simple correlation based on shared attributes
        a_attrs = set(str(v) for v in event_a.values())
        b_attrs = set(str(v) for v in event_b.values())
        if not a_attrs or not b_attrs:
            return 0.0
        intersection = a_attrs.intersection(b_attrs)
        union = a_attrs.union(b_attrs)
        return len(intersection) / len(union)

    def _are_contradictory(self, statement_a: str, statement_b: str) -> bool:
        """Check if two statements are contradictory."""
        # Simple contradiction detection
        negation_words = ["not", "no", "never", "none", "nothing"]
        a_words = statement_a.lower().split()
        b_words = statement_b.lower().split()
        # Check for explicit negations
        a_has_negation = any(word in negation_words for word in a_words)
        b_has_negation = any(word in negation_words for word in b_words)
        if a_has_negation != b_has_negation:
            # One has negation, other doesn't - check for similar content
            a_content = [word for word in a_words if word not in negation_words]
            b_content = [word for word in b_words if word not in negation_words]
            common_words = set(a_content).intersection(set(b_content))
            return len(common_words) >= 2
        return False

    def _step_to_dict(self, step: ReasoningStep) -> Dict[str, Any]:
        """Convert reasoning step to dictionary for JSON serialization."""
        return {
            "type": step.step_type.value,
            "premises": step.premises,
            "conclusion": step.conclusion,
            "confidence": step.confidence,
            "explanation": step.explanation
        }

# Maintain backward compatibility
class REP(AdvancedREP):
    """Backward compatible wrapper for AdvancedREP."""
    pass

if __name__ == '__main__':
    # Test the enhanced REP protocol
    logging.basicConfig(level=logging.INFO)
    rep = AdvancedREP()
    # Test deductive reasoning
    test_data = {
        "reasoning_type": "deductive",
        "facts": ["Socrates is a man", "All men are mortal"],
        "rules": [
            (["Socrates is a man", "All men are mortal"], "Socrates is mortal"),
            (["Socrates is mortal"], "Socrates will die")
        ]
    }
    result = rep.execute(test_data)
    print("Deductive Reasoning Test:")
    print(f"Conclusions: {result['conclusions']}")
    print(f"Confidence Scores: {result['confidence_scores']}")
    print(f"Explanation: {result['explanation']}")
