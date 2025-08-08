import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ESL:
    """
    A simple, rule-based implementation of the Emotional State Layer (ESL).
    """

    def __init__(self):
        self.positive_keywords = {"happy", "joy", "good", "great", "excellent", "love", "thanks", "wonderful"}
        self.negative_keywords = {"sad", "angry", "bad", "terrible", "hate", "problem", "error", "frustrated"}

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the emotional tone of the user's input.

        Args:
            data: The input data, expected to contain 'user_input'.

        Returns:
            A dictionary with the detected emotional state.
        """
        user_input = data.get("user_input", "").lower()
        logger.info(f"ESL: Analyzing input: '{user_input[:50]}...'")

        if not user_input:
            return {"emotional_state": "unknown", "reason": "No user_input provided."}

        # Simple keyword matching
        positive_score = sum(1 for word in self.positive_keywords if word in user_input)
        negative_score = sum(1 for word in self.negative_keywords if word in user_input)

        if positive_score > negative_score:
            state = "positive"
        elif negative_score > positive_score:
            state = "negative"
        else:
            state = "neutral"

        logger.info(f"ESL: Detected emotional state as '{state}'.")
        return {"emotional_state": state, "positive_score": positive_score, "negative_score": negative_score}

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    esl_protocol = ESL()

    # --- Test Cases ---
    print("--- Testing ESL Protocol ---")

    test1 = {"user_input": "This is a great and wonderful day!"}
    result1 = esl_protocol.execute(test1)
    print(f"Input: '{test1['user_input']}' -> State: {result1['emotional_state']}")
    assert result1['emotional_state'] == 'positive'

    test2 = {"user_input": "I have a problem, this is terrible."}
    result2 = esl_protocol.execute(test2)
    print(f"Input: '{test2['user_input']}' -> State: {result2['emotional_state']}")
    assert result2['emotional_state'] == 'negative'

    test3 = {"user_input": "The sky is blue."}
    result3 = esl_protocol.execute(test3)
    print(f"Input: '{test3['user_input']}' -> State: {result3['emotional_state']}")
    assert result3['emotional_state'] == 'neutral'

    test4 = {"user_input": "This is a good day, but I have a problem."}
    result4 = esl_protocol.execute(test4)
    print(f"Input: '{test4['user_input']}' -> State: {result4['emotional_state']}")
    assert result4['emotional_state'] == 'neutral'
