import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ESL:
    """
    An enhanced implementation of the Emotional State Layer (ESL).
    """

    def __init__(self):
        self.positive_keywords = {"happy", "joy", "good", "great", "excellent", "love", "thanks", "wonderful"}
        self.negative_keywords = {"sad", "angry", "bad", "terrible", "hate", "problem", "error", "frustrated"}

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the emotional tone of the user's input and provides a salience score.

        Args:
            data: The input data, expected to contain 'user_input'.

        Returns:
            A dictionary with the detected emotional state and its salience.
        """
        user_input = data.get("user_input", "").lower()
        logger.info(f"ESL: Analyzing input: '{user_input[:50]}...'")

        if not user_input:
            return {"emotional_state": "unknown", "salience": 0, "reason": "No user_input provided."}

        positive_score = sum(1 for word in self.positive_keywords if word in user_input)
        negative_score = sum(1 for word in self.negative_keywords if word in user_input)

        if positive_score > negative_score:
            state = "positive"
        elif negative_score > positive_score:
            state = "negative"
        else:
            state = "neutral"

        # Salience is a measure of the emotional intensity.
        # For this simple implementation, it's the magnitude of the emotion.
        salience = abs(positive_score - negative_score)

        logger.info(f"ESL: Detected state: '{state}', Salience: {salience}")
        return {"emotional_state": state, "salience": salience}

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    esl_protocol = ESL()

    print("--- Testing Enhanced ESL Protocol ---")

    test1 = {"user_input": "This is a great and wonderful day!"}
    result1 = esl_protocol.execute(test1)
    print(f"Input: '{test1['user_input']}' -> Result: {result1}")
    assert result1['emotional_state'] == 'positive' and result1['salience'] == 2

    test2 = {"user_input": "I have a problem, this is terrible."}
    result2 = esl_protocol.execute(test2)
    print(f"Input: '{test2['user_input']}' -> Result: {result2}")
    assert result2['emotional_state'] == 'negative' and result2['salience'] == 2

    test3 = {"user_input": "This is a good day, but I have a problem."}
    result3 = esl_protocol.execute(test3)
    print(f"Input: '{test3['user_input']}' -> Result: {result3}")
    assert result3['emotional_state'] == 'neutral' and result3['salience'] == 0
