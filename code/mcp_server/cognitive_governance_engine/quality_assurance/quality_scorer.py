from typing import Dict, Any

from . import completeness_validator
from . import relevance_analyzer
from .quality_monitor import QualityMonitor

class QualityScorer:
    """
    Orchestrates quality checks and scores the output of a given protocol.
    """

    def __init__(self):
        self.quality_monitor = QualityMonitor()
        # Define weights for different quality components
        self.weights = {
            "completeness": 0.6,
            "relevance": 0.4
        }

    def score_output(self, protocol_name: str, protocol_output: Dict[str, Any], user_input: str = "") -> Dict[str, Any]:
        """
        Calculates a comprehensive quality score for a protocol's output.

        Args:
            protocol_name: The name of the protocol (e.g., "REP", "ESL").
            protocol_output: The output dictionary from the protocol.
            user_input: The original user input to the system.

        Returns:
            A dictionary containing the overall quality score and its components.
        """
        if not protocol_output:
            return {"quality_score": 0.0, "reason": "Protocol output is empty."}

        # 1. Completeness Check
        completeness_check = {}
        if protocol_name == "REP":
            completeness_check = completeness_validator.validate_rep_completeness(protocol_output)
        elif protocol_name == "ESL":
            completeness_check = completeness_validator.validate_esl_completeness(protocol_output)
        elif protocol_name == "MTP":
            completeness_check = completeness_validator.validate_mtp_completeness(protocol_output, user_input)
        else:
            # For unknown protocols, assume completeness
            completeness_check = {"score": 1.0, "reason": "No completeness validator for this protocol."}

        # 2. Relevance Check
        relevance_check = relevance_analyzer.analyze_relevance(protocol_output, user_input)

        # 3. Calculate weighted overall score
        completeness_score = completeness_check.get('score', 0.0)
        relevance_score = relevance_check.get('score', 0.0)

        overall_score = (completeness_score * self.weights['completeness']) + \
                        (relevance_score * self.weights['relevance'])

        # 4. Record the score in the monitor
        self.quality_monitor.add_score(protocol_name, overall_score)

        return {
            "quality_score": round(overall_score, 2),
            "components": {
                "completeness": {"score": completeness_score, "reason": completeness_check.get('reason')},
                "relevance": {"score": relevance_score, "reason": relevance_check.get('reason')}
            }
        }

if __name__ == '__main__':
    # Example Usage
    scorer = QualityScorer()

    # Test Case 1: High-quality REP output
    rep_output_good = {"premises": ["All men are mortal", "Socrates is a man"], "conclusion": "Socrates is mortal"}
    user_input1 = "Is Socrates mortal?"
    quality1 = scorer.score_output("REP", rep_output_good, user_input1)
    print("--- Test Case 1: High-quality REP ---")
    import json
    print(json.dumps(quality1, indent=2))
    assert quality1['quality_score'] > 0.8

    # Test Case 2: Low-quality MTP output (incomplete and irrelevant)
    mtp_output_bad = {"extracted_entities": []} # Incomplete for a long input
    user_input2 = "John works at Microsoft in Seattle, a major city in the US."
    # To make it irrelevant, let's assume the output was about something else
    # (though our relevance checker is based on the output dict itself, so we can't easily fake this here)
    # The incompleteness will be the main driver of the low score.
    quality2 = scorer.score_output("MTP", mtp_output_bad, user_input2)
    print("\n--- Test Case 2: Low-quality MTP ---")
    print(json.dumps(quality2, indent=2))
    assert quality2['quality_score'] < 0.5

    # Test Case 3: Check if monitor tracks scores
    _ = scorer.score_output("REP", rep_output_good, user_input1) # Score again
    rep_history = scorer.quality_monitor.get_all_history().get("REP", [])
    print(f"\nREP Score History: {rep_history}")
    assert len(rep_history) == 2
    assert rep_history[0] == quality1['quality_score']
    print("Monitor test passed.")
